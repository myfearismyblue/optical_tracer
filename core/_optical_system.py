__all__ = ['IOpticalSystem', 'OpticalSystem']

from abc import abstractmethod, ABC
from copy import copy
from math import asin, pi, sin
from typing import Dict, List
from warnings import warn

from ._config import *
from ._exceptions import (ComponentCollisionException,
                          VectorOutOfComponentException,
                          TotalInnerReflectionException,
                          NoIntersectionWarning)
from ._optical_component import (Material,
                                 OpticalComponent,
                                 DefaultOpticalComponent,
                                 IComponentCollision,
                                 ComponentCollision,
                                 Layer,)
from ._rays import Vector


class IOpticalSystem(ABC):
    """
    Interface which any OpticalSystem has to have in order to:
    1. add optical component to itself
    2. trace the given vector through itself
    3. store traced vector as a beam in self.rays
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def rays(self) -> Dict[int, List[Vector]]:
        """
        Traced through optical system beams.
         :return Dict[beam_id: List[node_vector0, node_vector1...]]"""
        ...

    @property
    @abstractmethod
    def components(self) -> List[OpticalComponent]:
        """
        Optical components the optical system composed of.
        :return The list of all components added to system"""
        ...

    @property
    @abstractmethod
    def DEFAULT_CLS_MEDIUM(self) -> Material:
        """ The default substance in which optical components are exist should be known"""
        ...

    @abstractmethod
    def trace(self, vector: Vector):
        ...

    @abstractmethod
    def add_component(self, *, component: OpticalComponent) -> None:
        ...


class OpticalSystem(IOpticalSystem):
    """
    Entire system. Responses for tracing a give vector between components, store the beam; adding components to itself
    """

    # default material in which optical components are exist
    DEFAULT_CLS_MEDIUM: Material = Material(name="Air", transmittance=0, refractive_index=1)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        if not isinstance(val, str):
            raise TypeError(f'Provided name couldnt be treated as a string: {type(val)}')
        self._name = val

    @property
    def rays(self):
        """"""
        return self._rays

    @rays.setter
    def rays(self, value: Dict[int, List[Vector]]):
        self._rays = value  # TODO: make conscious data checking here

    @property
    def components(self) -> List[OpticalComponent]:
        return self._components

    @components.setter
    def components(self, val: List[OpticalComponent]) -> None:
        # TODO: make actual data check here
        self._components = val

    def __init__(self, *, name='Unnamed', default_medium: Material = DEFAULT_CLS_MEDIUM):
        self._name = name
        self._components: List[OpticalComponent] = []
        self._rays: Dict[int, List[Vector]] = {}
        self.default_background_component: DefaultOpticalComponent = \
            self._init_default_background_component(default_medium=default_medium)

    def _init_default_background_component(self, *, default_medium: Material) -> DefaultOpticalComponent:
        """Inits an instance of an optical component -  a special component which negates entire  optical layers"""
        default_component = DefaultOpticalComponent(name="default medium")
        default_component.material = default_medium
        self._add_and_compose_default_layers(default_component)
        return default_component

    def _add_and_compose_default_layers(self, default_component):
        default_layers = self._compose_default_layers()
        default_component.delete_all_layers()
        [default_component.add_layer(layer=layer) for layer in default_layers]

    def _compose_default_layers(self):
        """Composes a list of layers which have opposite sides of all layers in opt system"""
        ret = []
        for component in self.components:
            for layer in component.get_layers():
                ret.append(layer.reverted_layer())
        return ret

    def add_component(self, *, component: OpticalComponent) -> None:
        collision: IComponentCollision = self._check_component_collision(component=component)
        if collision.is_occurred:
            raise ComponentCollisionException(f'Adding component is impossible because of collision. '
                                              f'Details: {collision.details}')
        self.components.append(component)
        self._add_and_compose_default_layers(self.default_background_component)

    def _check_component_collision(self, *, component: OpticalComponent) -> IComponentCollision:
        warn('NOT IMPLEMENTED COLLISION CHECK')
        collision = ComponentCollision()
        collision.is_occurred = False
        collision.details = 'Mock'
        return collision

    def _add_initial_vector(self, *, initial_vector: Vector) -> None:
        """Adds only initial vector of a beam which is to trace."""
        self.rays[id(initial_vector)] = [initial_vector]

    def _append_to_beam(self, *, initial_vector: Vector, node_vector: Vector) -> None:
        """Adds node-vector to a beam, initiated by initial vector"""
        self.rays[id(initial_vector)].append(node_vector)

    def _get_containing_component(self, *, vector: Vector) -> OpticalComponent:
        """Return the component of system which contains given vector or raises VectorOutOfComponentException"""
        for component in self.components:
            if component.check_if_vector_is_inside(vector=vector):
                return component
        raise VectorOutOfComponentException(f'Vector is out of any component: {vector}')

    @staticmethod
    def _get_refract_angle(*, vector_angle: float, normal_angle: float,
                           prev_index: float, next_index: float) -> float:
        """
        Implements Snell's law.
        :param vector_angle: vector's global angle to optical axis [0, 2*pi)
        :param normal_angle: angle of  normal at the point of intersection to optical axis [0, pi)
        :param prev_index: index of medium vector is leaving
        :param next_index: index of medium vector is arriving to
        :return: vector's global angle after transition to the new medium (to the z-axis)
        Raises TotalInnerReflectionException if asin couldn't be calculated
        """
        assert 0 <= vector_angle < 2 * pi  # only clear data in the class
        assert 0 <= normal_angle < pi
        assert prev_index and next_index

        alpha = vector_angle - normal_angle  # local angle of incidence
        assert alpha != pi / 2 and alpha != 3 * pi / 2  # assuming vector isn't tangental to boundary

        try:
            beta = asin(prev_index / next_index * sin(alpha)) % (2 * pi)
        except ValueError as e:
            if 'math domain error' not in e.args:  # if not wrong asin in OpticalSystem._get_refract_angle()
                raise e
            raise TotalInnerReflectionException from e

        # if vector and normal are contrdirected asin(sin(x)) doesn't give x, so make some addition
        beta = pi - beta if pi / 2 < abs(alpha) < 3 * pi / 2 else beta

        ret = (normal_angle + beta) % (2 * pi)  # expecting output in [0, 360)
        return ret

    @staticmethod
    def _get_reflect_angle(*, vector_angle: float, normal_angle: float):
        """
        Implements specular reflection.
        :param vector_angle: vector's global angle to optical axis [0, 2*pi)
        :param normal_angle: angle of  normal at the point of intersection to optical axis [0, pi)
        :return: vector's global angle after reflection (to the z-axis)
        """
        assert 0 <= vector_angle < 2 * pi  # only clear data in the class
        assert 0 <= normal_angle < pi

        alpha = vector_angle - normal_angle  # local angle of incidence
        assert alpha != pi / 2 and alpha != 3 * pi / 2  # assuming vector isn't tangental to boundary
        beta = pi - alpha
        ret = (normal_angle + beta) % (2 * pi)  # expecting output in [0, 360)
        return ret

    def _get_containing_component_or_default(self, *, vector: Vector) -> OpticalComponent:
        """Returns thc component of system which contains given vector or returns default background"""
        try:
            return self._get_containing_component(vector=vector)
        except VectorOutOfComponentException:
            return self.default_background_component

    def _refract(self, *, vector: Vector, layer: Layer, prev_index: float, next_index: float) -> Vector:
        if NO_REFRACTION:
            return vector

        normal_angle = layer.get_normal_angle(point=vector.initial_point)
        refracted_vector = copy(vector)
        refracted_vector.theta = self._get_refract_angle(vector_angle=vector.theta, normal_angle=normal_angle,
                                                         prev_index=prev_index, next_index=next_index)
        return refracted_vector

    def _reflect(self, *, vector: Vector, layer: Layer) -> Vector:
        if N0_REFLECTION:
            return vector

        normal_angle = layer.get_normal_angle(point=vector.initial_point)
        reflected_vector = copy(vector)
        reflected_vector.theta = self._get_reflect_angle(vector_angle=vector.theta, normal_angle=normal_angle)
        return reflected_vector

    def trace(self, vector: Vector):
        """Traces vector through the whole optical system"""
        # beginning of loop:
        #   finds component or background where vector is located
        #   gets equation for the vector's line
        #   finds closest intersections with components in system
        #       if no such point: end of the loop
        #   creates a new instance of Vector at the point of this intersection
        #   gets a refracted angle
        #   creates a new instance of refractored Vector
        #   propagate vector to a boundary of the component
        # end of iteration
        current_vector = initial_vector = vector
        self._add_initial_vector(initial_vector=initial_vector)
        current_component = self._get_containing_component_or_default(vector=current_vector)
        while True:
            try:  # FIXME:  make this outer func.
                current_vector, intersection_layer = current_component.propagate_vector(input_vector=current_vector,
                                                                                        components=self.components)
            except NoIntersectionWarning:
                if DEBUG:
                    print(f'Tracing is finished for vector: {self.rays[id(initial_vector)][0]}. '
                          f'Last point is {self.rays[id(initial_vector)][-1].initial_point}')
                return list(self.rays.values())[0]
            prev_index = current_component.material.refractive_index
            current_component = self._get_containing_component_or_default(vector=current_vector)
            next_index = current_component.material.refractive_index

            try:
                current_vector = self._refract(vector=current_vector, layer=intersection_layer, prev_index=prev_index,
                                               next_index=next_index)
            except TotalInnerReflectionException as e:
                if DEBUG:
                    warn(f'\nTotal internal reflection is occurred for '
                         f'{self.rays[id(initial_vector)][0]}.')
                current_vector = self._reflect(vector=current_vector, layer=intersection_layer)

            self._append_to_beam(initial_vector=initial_vector, node_vector=current_vector)
