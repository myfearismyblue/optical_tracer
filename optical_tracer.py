from abc import ABC, abstractmethod
import ctypes as ct
from dataclasses import dataclass, field
from functools import wraps
from enum import auto, Enum
from math import asin, atan, pi, sin, sqrt, tan
from typing import Callable, Dict, Optional, List, Tuple, Union
from warnings import warn

from scipy.misc import derivative
from scipy.optimize import fsolve
import numpy as np

OPT_SYS_DIMENSIONS = (-100, 100)
QUARTER_PART_IN_MM = 10 ** (-6) / 4     # used in expressions like 555 nm * 10 ** (-6) / 4 to represent tolerance
TOLL = 10 ** -3                         # to use in scipy functions


def kwargs_only(cls):
    """Auxiliary func to make class initions only with keyword args"""

    @wraps(cls)
    def call(**kwargs):
        return cls(**kwargs)

    return call


class VectorOutOfComponentWarning(Warning):
    """Raises then coords of a vector are out of optical component which it was given"""
    pass


class NoIntersectionWarning(Warning):
    """Raises then vector doesn't intersect any surface"""
    pass


class ObjectKeyWordsMismatch(Warning):
    """Raises when __init__ gets unexpected **kwargs"""
    pass


class UnspecifiedFieldException(Exception):
    """Raises when object's field hasn't been set correctly"""
    pass


class ICheckable(ABC):
    """Interface for object with necessity of input vars' check"""

    def __init__(self, *args, **kwargs):
        self._check_inputs(*args, **kwargs)
        self._throw_inputs(*args, **kwargs)

    @abstractmethod
    def _check_inputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def _throw_inputs(self, *args, **kwargs):
        pass


class BaseCheckStrategy(ABC):
    """Base strat class for object input check"""

    @abstractmethod
    def check(self, *args, **kwargs):
        ...


class PointCheckStrategy(BaseCheckStrategy):
    """The way in which any Point object's inputs should be checked"""  # FIXME: Add concrete conditions

    def check(self, *args, **kwargs):
        expected_coords_names = [name[1:] for name in Point.__slots__ if name.startswith('_')]
        for coord in kwargs.items():
            kwargs[coord[0]] = float(coord[1])

        if not all(coord in kwargs for coord in expected_coords_names):
            print(f'Not enough args. Should exists {expected_coords_names}')

        if not all(coord in expected_coords_names for coord in kwargs):
            warn(f'Wrong keyword in : {kwargs}', ObjectKeyWordsMismatch)


class VectorCheckStrategy(BaseCheckStrategy):
    """The way in which any Vector object's inputs should be checked"""  # FIXME: Add concrete conditions

    def check(self, *args, **kwargs):
        raise NotImplementedError


@kwargs_only
class Point(ICheckable):
    """
    Just a point w/ cartesian coordinates in an optical system, where z is an optical axis
    x: float            # (x, y, z) - ordered triplet, where  (x, y) - sagittal plane, (y, z) - meridonial plane
    y: float            # of an optical system
    z: float            # z - optical axis
    """
    __slots__ = '_x', '_y', '_z'

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = val

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, val):
        self._z = val

    def set_coords(self, **coords: Union[float, int]):  # FIXME: check inputs
        tmp_coords = coords.get('x'), coords.get('y'), coords.get('z')
        self_coords = self._x, self._y, self._z
        self_coords = [self_coord if tmp_coord is None else tmp_coord
                       for tmp_coord, self_coord in zip(tmp_coords, self_coords)]

    def get_coords(self, coords: str) -> Dict[str, Union[float, int]]:  # FIXME: check inputs
        """
        :argument Use string as input like point.get_coords('yx').
        :returns dict like {'y': 0, 'x': 0.5}
        """
        return {coord: getattr(self, '_' + coord) for coord in coords}

    def get_distance(self, point) -> float:
        """
        Returns distance to the point
        :param point: particular point with .x, .y, .z attrs
        :return: distance to the particular point
        """
        if all((hasattr(point, 'x'), hasattr(point, 'y'), hasattr(point, 'z'))):
            return sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2 + (self.z - point.z) ** 2)
        raise UnspecifiedFieldException

    def _check_inputs(self, *args, **kwargs):
        PointCheckStrategy().check(*args, **kwargs)

    def _throw_inputs(self, *args, **kwargs):  # FIXME: fix this shit
        self._x: float = kwargs['x']
        self._y: float = kwargs['y']
        self._z: float = kwargs['z']

    def __repr__(self):
        return f'{self.__class__}, x = {self.x}, y = {self.y}, z = {self.z}'


@kwargs_only
@dataclass
class Vector:
    """
    A simple vector with defined energy (luminance) in it
    initial_point: Point
    lum: float              # luminance
    w_length: float         # wave length, default 555nm green light
    theta: float            # angle between optical axis and the vector in (z,y) plane positive in CCW direction
    psi: float              # angle between optical axis and the vector in (z,x) plane positive in CCW direction
    y
    ^    ^ x
    |   /
    | /
    + ------------> z
    """
    __slots__ = '_initial_point', '_lum', '_w_length', '_theta', '_psi'
    initial_point: Point
    lum: float
    w_length: float
    theta: float
    psi: float

    def __post_init__(self):
        self._initial_point: Point = self.initial_point
        self._lum: float = self.lum
        self._w_length: float = self.w_length
        self._theta: float = self.theta
        self._psi: float = self.psi

    @property
    def direction(self) -> Dict[str, float]:
        return {'theta': self._theta, 'psi': self._psi}

    @direction.setter
    def direction(self, values: Dict[str, float]):
        tmp_theta, tmp_psi = values.get('theta'), values.get('psi')
        self._theta = self._theta if tmp_theta is None else tmp_theta
        self._psi = self._psi if tmp_psi is None else tmp_psi

    @property
    def initial_point(self) -> Point:
        return self._initial_point

    @initial_point.setter
    def initial_point(self, value: Point):
        self._initial_point = value

    @property
    def lum(self) -> Union[int, float]:
        return self._lum

    @lum.setter
    def lum(self, value: Union[int, float]):
        self._lum = value

    @property
    def w_length(self) -> Union[int, float]:
        return self._w_length

    @w_length.setter
    def w_length(self, value: Union[int, float]):
        self._w_length = value

    @property
    def theta(self) -> Union[int, float]:
        return self._theta

    @theta.setter
    def theta(self, value: Union[int, float]):
        self._theta = value

    @property
    def psi(self) -> Union[int, float]:
        return self._psi

    @psi.setter
    def psi(self, value: Union[int, float]):
        self._psi = value

    def get_line_equation(self, repr=False) -> Callable:
        """Returns callable - equation of a line in z = f(y), where z is an optical axis"""
        A = 1 / tan(self.theta)
        B = self.initial_point.z - self.initial_point.y / tan(self.theta)
        print(f'{A}*y + {B}') if repr else None
        return lambda y: A * y + B


@kwargs_only
@dataclass
class Material:
    """
    Medium where energy vector propagates.
    :param name, transparency, refractive_index
    """
    __slots__ = '_name', '_transparency', '_refractive_index'
    name: str
    transparency: float
    refractive_index: float

    def __post_init__(self):
        self._name: str = self.name
        self._transparency: float = self.transparency
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
    def transparency(self):
        return self._transparency

    @transparency.setter
    def transparency(self, val):
        self._transparency = val

    @property
    def refractive_index(self):
        return self._refractive_index

    @refractive_index.setter
    def refractive_index(self, val):
        self._refractive_index = val


class Side(Enum):
    RIGHT = auto()
    LEFT = auto()


@kwargs_only
@dataclass
class Layer:
    """
    Each optical component is represented by intersection of layers. Each layer has name, boundary and active side,
    where material supposed to be
    """
    __slots__ = '_name', '_boundary', '_side'
    name: str
    boundary: Callable
    side: Side = field(default=Side.RIGHT)

    def __post_init__(self):
        self._name = self.name
        self._boundary = self.boundary
        self._side = self.side

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, _val: str):
        self._name = _val

    @property
    def boundary(self) -> Callable:
        return self._boundary

    @boundary.setter
    def boundary(self, _val: Callable):
        self._boundary = _val

    @property
    def side(self) -> Side:
        return self._side

    @side.setter
    def side(self, _val: Side):
        self._side = _val

    def contains_point(self, *, point: Point) -> bool:
        """
        Checks if input point and material (self.side) are at a same side of boundary
        """
        point_right = point.z > self.boundary(point.y)
        point_left = not point_right
        if (point_right and self.side == Side.RIGHT) or (point_left and self.side == Side.LEFT):
            return True
        return False

    def get_layer_intersection(self, *, vector: Vector) -> Point:
        """
        Returns valid closest intersection of the vector with boundary layer.boundary
        :return: (y, z) coord of intersections with boundary - the closest intersection to the point .
        """
        line = vector.get_line_equation()
        surface = self.boundary
        equation = lambda y: surface(y) - line(y)  # only for (y,z)-plane
        probable_y_intersections = list(fsolve(equation, np.array(OPT_SYS_DIMENSIONS)))
                                                        # FIXME: throw input as attr. Think about this
        approved_ys = _check_probable_intersections(probable_ys=probable_y_intersections,
                                                    layer=self,
                                                    vector=vector)
        # [(y, z), .....]
        approved_zs = [surface(y) for y in approved_ys]
        assert len(approved_zs) == len(approved_ys)
        approved_intersections = [Point(x=0, y=item[0], z=item[1]) for item in zip(approved_ys, approved_zs)]
        closest_intersection = _find_closest_intersection(approved_intersections=approved_intersections,
                                                          vector=vector)
        return closest_intersection


def _check_probable_intersections(*, probable_ys: List[float], layer: Layer, vector: Vector) -> List[float]:
    """
    Checks if each of given ys is:
        1. at the semiplane to which vector is directed;
        2. point on surface and point on line with the y are convergates.
    :param probable_ys: List[float], list of probable y-coords
    :param layer: a layer with concrete boundary
    :param vector: concrete vector which intersections to be checked
    :return: list of (y, z) pairs
    """
    surface = layer.boundary
    line = vector.get_line_equation()
    approved_ys = []
    for current_y in probable_ys:
        # get z-coord of intersection on surface
        z_surf_intersection = surface(current_y)
        # get z-coord of intersection on line
        z_line_intersection = line(current_y)
        # is this the same point?
        difference = z_surf_intersection - z_line_intersection

        if difference > vector.w_length * QUARTER_PART_IN_MM:  # quarter part of wave length
            warn(f'\nLine and surface difference intersections: {difference}', NoIntersectionWarning)
            # FIXME: check measures meters or milimeters?
            continue

        # check if vector is directed to the intersection
        vector_directed_left = pi / 2 <= vector.theta <= 3 * pi / 2
        intersection_is_righter = z_surf_intersection > vector.initial_point.z
        if intersection_is_righter == vector_directed_left:
            warn(f'\nSurface "{layer.name}" is out of vectors direction: '
                 f'theta={vector.theta:.3f}, '
                 f'intersection at (y,z)=({current_y:.3f}, {z_surf_intersection:.3f})', NoIntersectionWarning)
            continue

        # check if initial point of the vector is located on the boundary
        intersection_point = Point(x=0, y=current_y, z=z_surf_intersection)
        vector_difference = vector.initial_point.get_distance(intersection_point)
        if vector_difference <= vector.w_length * QUARTER_PART_IN_MM:
            material_at_the_left = layer.side==Side.LEFT
            warn(f'\nVector seems to be close to boundary: difference is {vector_difference} mm \n'
                 f'Vector directed to {vector.theta}, material is at the {layer.side}')
            if vector_directed_left == material_at_the_left:
                continue

        approved_ys.append(current_y)
    return approved_ys


def _find_closest_intersection(*, approved_intersections: List[Point], vector: Vector) -> Point:
    """
    In the list of points finds the closest point to vector
    """
    min_distance = float('inf')
    cand = None
    for point in approved_intersections:
        current_distance = point.get_distance(vector.initial_point)
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

    def add_layer(self, *, new_layer: Layer):
        self._layers.append(new_layer)

    @property
    def material(self):
        if self._material is None:
            raise UnspecifiedFieldException
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

    def _check_if_point_is_inside(self, *, point: Point) -> bool:
        return all((layer.contains_point(point=point) for layer in self._layers))

    def _get_component_intersection(self, *, vector: Vector) -> Tuple[Layer, Point]:
        """
        Returns the tuple (layer, point) of vector intersection with the component as a minimum of distances
        to layers' intersections.
        """
        found_intersections = {}
        for layer in self._layers:
            found_intersections[id(layer)] = layer.get_layer_intersection(vector=vector)
        if all(point is None for point in found_intersections.values()):
            raise VectorOutOfComponentWarning
        closest_point = _find_closest_intersection(approved_intersections=found_intersections.values(),
                                                   vector=vector)
        for k, v in found_intersections.items():
            closest_layer_id = k if v == closest_point else None
        assert closest_layer_id is not None, 'Closest point is found, but layer is not'
        closest_layer = ct.cast(closest_layer_id, ct.py_object).value
        return closest_layer, closest_point



    @staticmethod
    def _get_normal_angle(*, intersection: Tuple[Layer, Point]) -> float:
        """
        Returns angle in radians of normal line to the surface at the point of intersection.
        Uses scipy.misc.derivative
        """
        y: float = intersection[1].y
        surf_equation: Callable = intersection[0].boundary
        normal_angle: float = ((3/2*pi - atan(-1/derivative(surf_equation, y, dx=TOLL))) % pi)
        assert 0 <= normal_angle < pi
        return normal_angle

    def _propagate_vector(self, *, vector: Vector, layer:  Layer, ):
        pass

    @staticmethod
    def _get_refract_angle(*, vector_angle: float, normal_angle: float,
                           refractive_index1: float, refractive_index2: float) -> float:
        """
        Implements Snell's law.
        :param vector_angle: vector's global angle to optical axis [0, 2*pi)
        :param normal_angle: angle of  normal at the point of intersection to optical axis [0, pi)
        :param refractive_index1: index of medium vector is leaving
        :param refractive_index2: index of medium vector is arriving to
        :return: vector's global angle after transition to the new medium (to the z-axis)
        """
        assert 0 <= vector_angle < 2 * pi  # only clear data in the class
        assert 0 <= normal_angle < pi
        assert refractive_index1 and refractive_index2

        alpha = vector_angle - normal_angle  # local angle of incidence
        assert alpha != pi / 2 and alpha != 3 * pi / 2  # assuming vector isn't tangental to boundary
        beta = asin(refractive_index1 / refractive_index2 * sin(alpha)) % (2 * pi)
        # if vector and normal are contrdirected asin(sin(x)) doesn't give x, so make some addition
        beta = pi - beta if pi / 2 < alpha < 3 * pi / 2 else beta

        ret = (normal_angle + beta) % (2 * pi)  # expecting output in [0, 360)
        return ret

    def trace_vector_on_layer(self, *, vector: Vector, layer: Layer):  # FIXME: add return annotation
        # get intersection
        # if exists do propogate
        # find tangent and normal at the point of intersection
        # do some magic to find next media
        # refract vector
        raise NotImplementedError


class OpticalSystem:
    """
    Entire system. Responses for propagating vector between components
    """
    def __init__(self):
        self._components: List[OpticalComponent] = []

    def add_component(self, *, component):
        self._components.append(component)


def main():
    first_lense = OpticalComponent(name='first lense')
    first_lense.material = Material(name='Glass', transparency=0.9, refractive_index=1.5)
    parabolic_l = Layer(name='parabolic',
                        boundary=lambda y: y ** 2 / 10 ,
                        side=Side.RIGHT,
                        )
    first_lense.add_layer(new_layer=parabolic_l)
    plane_l = Layer(name='plane',
                    boundary=lambda y: 10,
                    side=Side.LEFT,
                    )
    first_lense.add_layer(new_layer=plane_l)

    opt_sys = OpticalSystem()
    opt_sys.add_component(component=first_lense)

    second_lense = OpticalComponent(name='second lense')
    second_lense.material = Material(name='Glass', transparency=0.9, refractive_index=1.5)

    plane_sec = Layer(name='plane_sec',boundary=lambda y: 20,side=Side.RIGHT)
    second_lense.add_layer(new_layer=plane_sec)

    parabolic_sec = Layer(name='parabolic', boundary=lambda y: 30-y ** 2 / 10 , side=Side.LEFT)
    second_lense.add_layer(new_layer=parabolic_sec)
    opt_sys.add_component(component=second_lense)
    v = Vector(initial_point=Point(x=0, y=0.01, z=0), lum=1, w_length=555, theta=0.03+pi, psi=0)
    print(plane_l.get_layer_intersection(vector=v))





if __name__ == '__main__':
    main()
