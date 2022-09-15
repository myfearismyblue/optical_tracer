__all__ = ['ComponentCollision',
           'DefaultOpticalComponent',
           'OpticalComponent',
           'Layer',
           'Material',
           'Side',
           'reversed_side', ]

import ctypes as ct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from math import atan, pi
from typing import Callable,Optional, List, Tuple
from warnings import warn

from scipy.misc import derivative
from scipy.optimize import fsolve
import numpy as np

from ._config import *
from ._exceptions import (NoIntersectionWarning,
                          UnspecifiedFieldException,
                          VectorNotOnBoundaryException,
                          VectorOutOfComponentException)
from ._rays import Point, Vector, get_distance, ICheckable, BaseCheckStrategy


class IComponentCollision(ABC):
    """The type of optical components' collision intersection"""

    @property
    @abstractmethod
    def is_occurred(self) -> bool:
        ...

    @property
    @abstractmethod
    def details(self):
        # TODO: consider realisation about this
        ...


class ComponentCollision(IComponentCollision):
    """Dataclass of collision between two components in one optsystem"""
    # FIXME: this is just a stub. Rewrite it's behaviour

    def __init__(self):
        self._is_occurred = None
        self._details = None

    @property
    def is_occurred(self) -> bool:
        return self._is_occurred

    @is_occurred.setter
    def is_occurred(self, val):
        self._is_occurred = val

    @property
    def details(self) -> bool:
        return self._details

    @details.setter
    def details(self, val):
        self._details = val


@dataclass
class Material:
    """
    Medium where energy vector propagates.
    :param name, transmittance, refractive_index
    """
    __slots__ = '_name', '_transmittance', '_refractive_index'
    name: str
    transmittance: float  # light absorption in % while tracing 1 sm of thickness
    refractive_index: float

    def __post_init__(self):
        self._name: str = self.name
        self._transmittance: float = self.transmittance
        self._refractive_index: float = self.refractive_index

    @classmethod
    def add_standart_material(cls, **kwargs):
        Material.standart_materials = Material(**kwargs)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def transmittance(self):
        return self._transmittance

    @transmittance.setter
    def transmittance(self, val):
        self._transmittance = val

    @property
    def refractive_index(self):
        return self._refractive_index

    @refractive_index.setter
    def refractive_index(self, val):
        self._refractive_index = val


class Side(Enum):
    RIGHT = auto()
    LEFT = auto()

    def __str__(self):
        """Returns only the last word: LEFT or RIGHT"""
        return super().__str__().split('.')[-1]

    @staticmethod
    def from_str(side: str):
        """Returns self depending of given string"""
        if not isinstance(side, str):
            raise TypeError(f'Input argument supposed to be a string, but was given {type(side)}')
        tmp = side.lower()
        if tmp == 'right':
            return Side.RIGHT
        elif tmp == 'left':
            return Side.LEFT
        else:
            raise ValueError(f'Was given wrong argument f{side}, supposed to be Left or Right')


def reversed_side(side: Side) -> Side:
    assert isinstance(side, Side)
    return Side.RIGHT if side == Side.LEFT else Side.LEFT


class LayerCheckStrategy(BaseCheckStrategy):
    def validate(self, *args, **kwargs):
        """
        The way in which any Layers object's inputs should be checked
        Check kwarg types are ok
        """
        kwargs = self.validate_name(kwargs)
        kwargs = self.validate_side(kwargs)
        kwargs = self.validate_boundary(kwargs)
        return kwargs

    @staticmethod
    def validate_name(kwargs):
        if not isinstance(kwargs['name'], str):
            raise UnspecifiedFieldException(f'Wrong inputs for Layer name. Given: {kwargs}')
        return kwargs

    @staticmethod
    def validate_boundary(kwargs):
        if not isinstance(kwargs['boundary'], Callable):
            raise UnspecifiedFieldException(f'Wrong inputs for Layer boundary. Given: {kwargs}')
        return kwargs

    @staticmethod
    def validate_side(kwargs):
        if not isinstance(kwargs['side'], Side):
            raise UnspecifiedFieldException(f'Wrong inputs for Layer side. Given: {kwargs}')
        return kwargs


class Layer(ICheckable):
    """
    Each optical component is represented by intersection of layers. Each layer has name, boundary and active side,
    where material supposed to be
    """
    __slots__ = '_name', '_boundary', '_side', '_intersection_points'   # points of curve
                                                                        # between which physical boundary located

    def __str__(self):

        attrs_str = [": ".join((attr[1:], f'{getattr(self, attr[1:])} ')) for attr in self.__slots__]
        return f'{self.__class__.__name__}: {str(attrs_str)}'

    def _validate_inputs(self, *args, **kwargs):
        expected_attrs = ['name', 'boundary', 'side']
        kwargs = LayerCheckStrategy().validate_and_check(cls=Layer, expected_attrs=expected_attrs, *args, **kwargs)
        return kwargs

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, _val: str):
        temp_val = LayerCheckStrategy.validate_name({'name': _val})
        self._name = temp_val

    @property
    def boundary(self) -> Callable:
        return self._boundary

    @boundary.setter
    def boundary(self, _val: Callable):
        temp_val = LayerCheckStrategy.validate_boundary({'boundary': _val})
        self._boundary = temp_val

    @property
    def side(self) -> Side:
        return self._side

    @side.setter
    def side(self, _val: Side):
        temp_val = LayerCheckStrategy.validate_side({'side': _val})
        self._side = temp_val

    @property
    def intersection_points(self):
        return self._intersection_points

    def contains_point(self, *, point: Point) -> bool:
        """
        Checks if input point and material (self.side) are at a same side of boundary (including)
        """
        if point is None:
            return False
        point_right = point.z >= self.boundary(point.y)
        point_left = point.z <= self.boundary(point.y)
        if (point_right and self.side == Side.RIGHT) or (point_left and self.side == Side.LEFT):
            return True
        return False

    def directed_inside(self, *, boundary_vector: Vector) -> bool:
        """Returns True if vector, located on boundary, directed inside the material (layer.side)"""
        # FIXME: Test this
        tangential_angle = self.get_tangential_angle(point=boundary_vector.initial_point)
        vector_directed_left = tangential_angle < boundary_vector.theta < tangential_angle + pi
        layer_is_left = self.side == Side.LEFT
        return vector_directed_left == layer_is_left

    def get_layer_intersection(self, *, vector: Vector) -> Point:
        """
        Returns valid closest intersection of the vector with boundary layer.boundary
        :return: (y, z) coord of intersections with boundary - the closest intersection to the point .
        """
        surface = self.boundary
        try:
            line = vector.get_line_equation()
            equation = lambda y: surface(y) - line(y)  # only for (y,z)-plane
            probable_y_intersections = list(fsolve(equation, np.array(OPT_SYS_DIMENSIONS)))
        except ZeroDivisionError:
            if surface(vector.initial_point.y) is None:  # FIXME: actual behaviour of surface() has to be considered
                raise NoIntersectionWarning
            else:
                probable_y_intersections = [vector.initial_point.y]
        # FIXME: throw input as attr. Think about this
        approved_ys = self._check_probable_intersections(probable_ys=probable_y_intersections, vector=vector)
        if not len(approved_ys):
            raise NoIntersectionWarning
        # [(y, z), .....]
        approved_zs = [surface(y) for y in approved_ys]
        assert len(approved_zs) == len(approved_ys)
        approved_intersections = [Point(x=0, y=item[0], z=item[1]) for item in zip(approved_ys, approved_zs)]
        closest_intersection = _find_closest_intersection(approved_intersections=approved_intersections,
                                                          vector=vector)
        return closest_intersection

    def _check_probable_intersections(self, *, probable_ys: List[float], vector: Vector) -> List[float]:
        """
        Checks for each of a given ys:
            1. point on surface and point on line with the y are converges;
        if so:
            2. check if vector is near the boundary
        if vector is not located at the boundary:
            3. checks if it located inside the material
        if so:
            4. checks if vector is directed to the probable intersection point
        if so:
            approve point
        :param probable_ys: List[float], list of probable y-coords
        :param vector: concrete vector which intersections to be checked
        :return: list of (y, z) pairs
        """
        surface = self.boundary
        try:
            line = vector.get_line_equation()
        except ZeroDivisionError:
            line = lambda y: surface(current_y) if y == vector.initial_point.y else float('inf')

        def _is_converges():
            # check if z-coordinate at the line and at the surface convergate
            difference = abs(surface(current_y) - line(current_y))
            if difference > vector.w_length * NANOMETRE / 4:  # quarter part of wave length
                if DEBUG:
                    warn(f'\nLine and surface difference intersections: {difference}', NoIntersectionWarning)
                # FIXME: check measures meters or milimeters?
                return False
            return True

        def _is_near_boundary():
            # check if initial point of the vector is located on the boundary
            vector_point_difference = get_distance(vector.initial_point, intersection_point)
            if vector_point_difference <= vector.w_length * QUARTER_PART_IN_MM:
                if DEBUG:
                    warn(f'\nVector seems to be close to boundary: difference is {vector_point_difference} mm \n'
                         f'Vector directed to {vector.theta}, material is at the {self.side}')
                return True
            return False

        def _is_in_medium():
            # check if vector is located at appropriate layer.side
            vector_is_righter = self.boundary(vector.initial_point.y) < vector.initial_point.z
            vector_is_lefter = not vector_is_righter
            if vector_is_righter and self.side == Side.LEFT or vector_is_lefter and self.side == Side.RIGHT:
                return False
            return True

        def _is_vector_directed_intersection():
            # check if vector is directed to the intersection
            current_z = intersection_point.z
            vector_z = vector.initial_point.z
            vector_y = vector.initial_point.y

            intersection_quadrant: int  # the quadrant where intersection is located relatively to vector
            if vector_y - current_y >= 0 and vector_z - current_z > 0:
                intersection_quadrant = 3
            elif vector_y - current_y < 0 and vector_z - current_z >= 0:
                intersection_quadrant = 2
            elif vector_y - current_y > 0 and vector_z - current_z <= 0:
                intersection_quadrant = 4
            elif vector_y - current_y <= 0 and vector_z - current_z < 0:
                intersection_quadrant = 1
            else:
                raise AssertionError(f'Something wrong with intersection quadrants')

            vector_directed_quadrant: int  # the quadrant vector directed to
            if pi <= vector.theta <= 3 * pi / 2:
                vector_directed_quadrant = 3
            elif 3 * pi / 2 < vector.theta < 2 * pi:
                vector_directed_quadrant = 4
            elif pi / 2 <= vector.theta < pi:
                vector_directed_quadrant = 2
            elif 0 <= vector.theta < pi / 2:
                vector_directed_quadrant = 1
            else:
                raise AssertionError(f'Something wrong with vectors direction quadrants')

            if intersection_quadrant == vector_directed_quadrant:
                return True
            if DEBUG:
                warn(f'\nSurface "{self.name}" is out of vectors direction: '
                     f'theta={vector.theta:.3f}, '
                     f'intersection at (y,z)=({current_y:.3f}, {surface(current_y):.3f})', NoIntersectionWarning)
            return False

        approved_ys = []
        for current_y in probable_ys:
            intersection_point = Point(x=0, y=current_y, z=surface(current_y))
            if not _is_converges():
                continue
            if _is_near_boundary():
                continue
            if not _is_in_medium():
                continue
            if not _is_vector_directed_intersection():
                continue
            approved_ys.append(current_y)
        return approved_ys

    def get_normal_angle(self, *, point: Point) -> float:
        """
        Returns angle in radians of normal line to the surface at the point of intersection.
        Uses scipy.misc.derivative
        """
        y: float = point.y
        surf_equation: Callable = self.boundary
        normal_angle: float = ((3 / 2 * pi - atan(-1 / derivative(surf_equation, y, dx=TOLL))) % pi)
        assert 0 <= normal_angle < pi
        return normal_angle

    def get_tangential_angle(self, *, point: Point) -> float:
        """Returns angle in radians of tangential line to the surface at the point of intersection."""
        # FIXME: test this
        normal_angle = self.get_normal_angle(point=point)
        if 0 <= normal_angle < pi / 2:
            tangential_angle = normal_angle + pi / 2
        else:  # pi/2 <= normal_angle < pi
            tangential_angle = normal_angle - pi / 2
        assert 0 <= tangential_angle < pi
        return tangential_angle

    def reverted_layer(self):
        """
        Returns object of Layer cls which has an opposite side to the current layer and sets the name to the new layer
        """
        opposite_side = reversed_side(self.side)
        boundary = self.boundary
        opposite_name = f'opposite {self.name}'
        return Layer(boundary=boundary, side=opposite_side, name=opposite_name)


def _find_closest_intersection(*, approved_intersections: List[Point], vector: Vector) -> Point:
    """
    In the list of points finds the closest point to vector
    """
    if not len(approved_intersections):
        raise NoIntersectionWarning('Nothing to be closest, approved layer''s intersections is empty')
    min_distance = float('inf')
    cand = None
    for point in approved_intersections:
        current_distance = get_distance(point, vector.initial_point)
        if current_distance < min_distance:
            min_distance = current_distance
            cand = point
    return cand


class OpticalComponent:
    """
    Intersection of layers which are to bound optical material
    """

    def __init__(self, *, name: str, dimensions: Tuple[float] = OPT_SYS_DIMENSIONS):
        self._name: str = name
        self._layers: Optional[List[Layer]] = []
        self._material: Optional[Material] = None
        self._dimensions = dimensions  # FIXME: check inputs here

    def __repr__(self):
        return f'{self.name}, {super().__repr__()}'

    def add_layer(self, *, layer: Layer):
        self._layers.append(layer)

    def delete_all_layers(self):
        self._layers = []

    def get_layers(self):
        return self._layers

    @property
    def material(self):
        if self._material is None:
            raise UnspecifiedFieldException(f'Material is not specified: {self._material}')
        return self._material

    @material.setter
    def material(self, _val: Material):
        self._material = _val

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, _val: str):
        self._name = _val

    def check_if_point_is_inside(self, *, point: Point) -> bool:
        """Returns true if the given point is in the boundaries (including) of all layers of the component"""
        return all(layer.contains_point(point=point) for layer in self._layers)

    def find_boundary_layer(self, *, vector: Vector) -> Layer:
        """Returns a layer on boundary of which vector's initial point is located, or raises exception"""
        for layer in self._layers:
            layer_x = vector_x = vector.initial_point.x
            layer_y = vector_y = vector.initial_point.y
            layer_z = layer.boundary(layer_y)  # FIXME: x here isn't being supported yet
            vector_point_difference = get_distance(Point(x=layer_x, y=layer_y, z=layer_z), vector.initial_point)
            vector_is_on_current_bound = vector_point_difference <= vector.w_length * NANOMETRE / 4
            if vector_is_on_current_bound:
                return layer
        raise VectorNotOnBoundaryException(f'Tried to get boundary for {vector}, '
                                           f'but the vector is not at any of layers'' boundaries')

    def check_if_vector_on_boundary(self, *, vector: Vector) -> bool:
        """Check if the given vector is located at the boundary"""
        try:
            _ = self.find_boundary_layer(vector=vector)
            return True
        except VectorNotOnBoundaryException:
            return False

    def check_if_directed_inside(self, *, vector: Vector) -> bool:
        """Returns true if given vector, which is located on boundary, directed inside the component"""
        bound_layer = self.find_boundary_layer(vector=vector)
        return bound_layer.directed_inside(boundary_vector=vector)

    def check_if_vector_is_inside(self, *, vector: Vector) -> bool:
        """
        Returns true if given vector's initial point is inside all layers of the component or
        true if the given point is at the boundary and vector directed inside the component
        """
        if self.check_if_vector_on_boundary(vector=vector):
            # check direction
            return self.check_if_directed_inside(vector=vector)
        else:
            return self.check_if_point_is_inside(point=vector.initial_point)

    def _get_component_intersection(self, *, vector: Vector) -> Tuple[Layer, Point]:
        """
        Returns the tuple (layer, point) of vector's intersection with the component as a minimum of distances
        to layers' intersections only if vector is in the component.
        """
        found_intersections = {}
        if not self.check_if_point_is_inside(point=vector.initial_point):
            raise VectorOutOfComponentException(f'Tried to find intersection of {vector} with {self}. '
                                                f'But seems that vector is located out of this component.')
        for layer in self._layers:
            try:
                intersection_point: Optional[Point] = layer.get_layer_intersection(vector=vector)
            except NoIntersectionWarning:
                intersection_point = None

            intersection_point_is_inside = self.check_if_point_is_inside(point=intersection_point)
            if intersection_point_is_inside:
                found_intersections[id(layer)] = intersection_point
        if all(point is None for point in found_intersections.values()):
            raise NoIntersectionWarning
        closest_point = _find_closest_intersection(approved_intersections=found_intersections.values(),
                                                   vector=vector)

        closest_layer_id = None
        for k, v in found_intersections.items():
            closest_layer_id = k if v == closest_point else closest_layer_id
        assert closest_layer_id is not None, 'Closest point is found, but layer is not'
        closest_layer = ct.cast(closest_layer_id, ct.py_object).value

        return closest_layer, closest_point

    def propagate_vector(self, *, input_vector: Vector, components=None) -> Tuple[Vector, Layer]:
        """
        Creates a new instance if vector due propagation of input vector in a component. Traces this vector
        to a boundary of the component, but do not refract it
        """
        # get intersection
        intersection = intersection_layer, intersection_point = self._get_component_intersection(vector=input_vector)
        # TODO: check existence of intersection
        destination_distance = get_distance(input_vector.initial_point, intersection_point)
        attenuation = self._calculate_attenuation(distance=destination_distance)
        assert 0 <= attenuation <= 1
        attenuated_lum = input_vector.lum - attenuation * input_vector.lum
        output_theta = input_vector.theta
        output_psi = input_vector.psi
        output_vector = Vector(initial_point=intersection_point, lum=attenuated_lum, w_length=input_vector.w_length,
                               theta=output_theta, psi=output_psi)
        return output_vector, intersection_layer

    def _calculate_attenuation(self, distance: float) -> float:
        """Gets distance in mm and returns attenuation"""
        return 1 - (1 - self.material.transmittance * PERCENT) ** (distance * MILLIMETRE / CENTIMETRE)


class DefaultOpticalComponent(OpticalComponent):
    """Special cls for default background component with overloaded methods"""

    def check_if_point_is_inside(self, *, point: Point, components: List[OpticalComponent]) -> bool:
        """Returns true if the given point is not in any of components in opt system"""
        return not any(component.check_if_point_is_inside(point=point) for component in components)

    def check_if_vector_is_inside(self, *, vector: Vector, components: List[OpticalComponent]) -> bool:
        """Returns true if the given vector is not in any of components in opt system"""
        return not any(component.check_if_vector_is_inside(vector=vector) for component in components)

    def _get_component_intersection(self, *, vector: Vector, components: List[OpticalComponent]) -> Tuple[Layer, Point]:
        found_intersections = {}
        # if not self.check_if_point_is_inside(point=vector.initial_point, components=components):
        #     raise VectorOutOfComponentException
        for component in components:  # use optsystem components to check if points are inside
            for layer in self.get_layers():
                try:
                    intersection_point: Optional[Point] = layer.get_layer_intersection(vector=vector)
                except NoIntersectionWarning:
                    intersection_point = None

                intersection_point_is_inside = component.check_if_point_is_inside(point=intersection_point)
                if intersection_point_is_inside:
                    found_intersections[id(layer)] = intersection_point
        if all(point is None for point in found_intersections.values()):
            raise NoIntersectionWarning
        closest_point = _find_closest_intersection(approved_intersections=found_intersections.values(),
                                                   vector=vector)

        closest_layer_id = None
        for k, v in found_intersections.items():
            closest_layer_id = k if v == closest_point else closest_layer_id
        assert closest_layer_id is not None, 'Closest point is found, but layer is not'
        closest_layer = ct.cast(closest_layer_id, ct.py_object).value

        return closest_layer, closest_point

    def propagate_vector(self, *, input_vector: Vector, components: List[OpticalComponent]) -> Tuple[Vector, Layer]:
        """Stub"""
        intersection = intersection_layer, intersection_point = \
            self._get_component_intersection(vector=input_vector, components=components)
        # TODO: check existence of intersection
        destination_distance = get_distance(input_vector.initial_point, intersection_point)
        attenuation = self.material.transmittance * PERCENT / CENTIMETRE * destination_distance * MILLIMETRE
        assert 0 <= attenuation <= 1
        attenuated_lum = input_vector.lum - attenuation * input_vector.lum
        output_theta = input_vector.theta
        output_psi = input_vector.psi
        output_vector = Vector(initial_point=intersection_point, lum=attenuated_lum, w_length=input_vector.w_length,
                               theta=output_theta, psi=output_psi)
        return output_vector, intersection_layer