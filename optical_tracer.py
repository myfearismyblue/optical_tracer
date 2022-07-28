from abc import ABC, abstractmethod
import ctypes as ct
from copy import copy
from dataclasses import dataclass
from functools import wraps
from enum import auto, Enum
from math import asin, atan, pi, sin, sqrt, tan, isnan
from typing import Iterable, Callable, Dict, Optional, List, Tuple, Union
from warnings import warn

from scipy.misc import derivative
from scipy.optimize import fsolve
import numpy as np

DEBUG = False
OPT_SYS_DIMENSIONS = (-100, 100)
OPTICAL_RANGE = (380, 780)  # in nanometers
QUARTER_PART_IN_MM = 10 ** (-6) / 4  # used in expressions like 555 nm * 10 ** (-6) / 4 to represent tolerance
TOLL = 10 ** -3  # to use in scipy functions
PERCENT = 0.01
METRE = 1
CENTIMETRE = METRE * 10 ** -2
MILLIMETRE = METRE * 10 ** -3
NANOMETRE = METRE * 10 ** -9
N0_REFLECTION = NO_REFRACTION = False and DEBUG


def kwargs_only(cls):
    """Auxiliary func to make class initions only with keyword args"""

    @wraps(cls)
    def call(**kwargs):
        return cls(**kwargs)

    return call


class DomainBaseException(Exception):
    """Base class for optical domain exceptions"""


class VectorOutOfComponentException(DomainBaseException):
    """Raises then coords of a vector are out of optical component which it was given"""
    pass


class VectorNotOnBoundaryException(DomainBaseException):
    """Raises then vector is supposed to be on the boundary of layer, but it is not"""
    pass


class NoIntersectionWarning(DomainBaseException):
    """Raises then vector doesn't intersect any surface"""
    pass


class TotalInnerReflectionException(DomainBaseException):
    """Raises when refraction couldn't be provided"""
    pass


class ComponentCollisionException(DomainBaseException):
    """Raises when two components has intersection"""
    pass


class ObjectKeyWordsMismatchException(Warning):
    """Raises when __init__ gets unexpected **kwargs"""
    pass


class UnspecifiedFieldException(Exception):
    """Raises when object's field hasn't been set correctly"""
    pass


class ICheckable(ABC):
    """
    Interface for object with necessity of input vars' validation.
    All vars to validate are to be forwarded to attrs with name like '_val', which are defined in slots of a concrete cls
    """

    def __init__(self, *args, **kwargs):
        kwargs = self._validate_inputs(*args, **kwargs)
        self._throw_inputs(*args, **kwargs)

    @abstractmethod
    def _validate_inputs(self, *args, **kwargs):
        pass

    def _throw_inputs(self, *args, **kwargs):
        """Throwing only for attrs starts with _underscore"""
        [setattr(self, attr, kwargs[attr[1:]]) if attr.startswith('_') else setattr(self, attr, kwargs[attr])
         for attr in self.__slots__]


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


class BaseCheckStrategy(ABC):
    """Base strat class for object input check"""

    def validate_and_check(self, cls, expected_attrs, *args, **kwargs):
        """
        The way in which any object inputs should be checked
        1. Ensure __slots__ has all attrs which are considered to be properties and all attrs in slot are to be checked
        2. Ensure kwargs and __slots__ have the same attrs
        3. Make abstract validation
        """
        self._ensure___slots__ok(cls, expected_attrs)
        self._check_kwarg_completeness(cls, kwargs)
        kwargs = self.validate(*args, **kwargs)
        return kwargs

    @staticmethod
    def _ensure___slots__ok(cls, expected_attrs):
        """Inner assertion to ensure validation to be done for all __slots__ """
        assert all(
            (f'{sl_attr[1:]}' if sl_attr.startswith('_') else sl_attr in expected_attrs for sl_attr in cls.__slots__))
        assert all((f'_{exp_attr}' in cls.__slots__ for exp_attr in expected_attrs))

    @staticmethod
    def _check_kwarg_completeness(cls, kwargs):
        """
        Checks if it's enough kwargs and if it's more than needed.
        Raises  ObjectKeyWordsMismatchException if not enough, or if there are some extra kwargs
        """

        expected_kwargs_names = [kw[1:] if kw.startswith('_') else kw for kw in cls.__slots__]

        if not all(coord in kwargs for coord in expected_kwargs_names):
            raise ObjectKeyWordsMismatchException(f'Not enough args. Should exists {expected_kwargs_names},'
                                                  f' but was given {kwargs}')

        if not all(coord in expected_kwargs_names for coord in kwargs):
            raise ObjectKeyWordsMismatchException(f'Wrong keyword in : {kwargs}, should be {expected_kwargs_names}')

    @abstractmethod
    def validate(self, *args, **kwargs):
        ...


class PointCheckStrategy(BaseCheckStrategy):
    """
    The way in which any Point object's inputs should be checked
    Make all coords float
    """

    @staticmethod
    def _make_kwagrs_float(kwargs):
        for coord in kwargs.items():
            temp = float(coord[1])
            if temp in (float('inf'), float('-inf')) or isnan(temp):
                raise ValueError(f'Not allowed points at infinity: {coord}')
            kwargs[coord[0]] = temp
        return kwargs

    def validate(self, *args, **kwargs):
        kwargs = self._make_kwagrs_float(kwargs)
        return kwargs


class VectorCheckStrategy(BaseCheckStrategy):
    """
    The way in which any Vector object's inputs should be checked
    Initial point should be instance of Point cls otherwise rises UnspecifiedFieldException
    Make luminance float and ensure it is not negative
    Warn if wave length is out of optical range, and ensure it is not negative, and make it float
    Div angles to 2*pi and make them float
    """

    @staticmethod
    def validate_initial_point(kwargs):
        if not isinstance(kwargs.get('initial_point'), Point):
            _ = 'initial_point'
            raise UnspecifiedFieldException(f'initial_point kwarg has to be type Point, '
                                            f'but was given {type(kwargs.get(_))}')
        return kwargs

    @staticmethod
    def validate_luminance(kwargs):
        temp = float(kwargs.get('lum'))
        if temp in (float('inf'), float('-inf')) or isnan(temp):
            raise ValueError(f'Not allowed luminance infinity: {temp}')
        elif temp < 0:
            raise UnspecifiedFieldException(f'Luminance should be not negative')
        kwargs['lum'] = temp
        return kwargs

    @staticmethod
    def validate_w_length(kwargs):
        temp = float(kwargs.get('w_length'))
        if temp in (float('inf'), float('-inf')) or isnan(temp):
            raise ValueError(f'Not allowed wave length infinity: {temp}')
        elif temp < 0:
            raise UnspecifiedFieldException(f'Wave length  should be not negative')
        if not (OPTICAL_RANGE[0] <= float(kwargs.get('w_length')) <= OPTICAL_RANGE[1]):
            warn('Wave length is out of optical range')
        kwargs['w_length'] = temp
        return kwargs

    @staticmethod
    def validate_angles(kwargs):
        for kw in kwargs:
            if kw in ['theta', 'psi']:
                temp = float(kwargs[kw])
                if temp in (float('inf'), float('-inf')) or isnan(temp):
                    raise ValueError(f'Not allowed angle: {kw}: {temp}')
                kwargs[kw] = temp % (2 * pi)
        return kwargs

    def validate(self, *args, **kwargs):
        kwargs = self.validate_initial_point(kwargs)
        kwargs = self.validate_luminance(kwargs)
        kwargs = self.validate_w_length(kwargs)
        kwargs = self.validate_angles(kwargs)
        return kwargs


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

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def set_coords(self, **coords):
        coords = self._validate_inputs(**coords)
        [setattr(self, key, coords[key[1:]]) for key in self.__slots__ if key.startswith('_')]

    def get_coords(self, coords: str) -> Dict[str, Union[float, int]]:
        """
        :argument Use string as input like point.get_coords('yx').
        :returns dict like {'y': 0, 'x': 0.5}
        """
        if not coords or not all(('_' + str(coord) in self.__slots__ for coord in coords)):
            raise UnspecifiedFieldException(
                f'Input coords mismatches. Was given {[c for c in coords]} '
                f'but needed any of {[c[1:] if c.startswith("_") else c for c in self.__slots__]}')
        return {str(coord): getattr(self, '_' + str(coord)) for coord in coords}

    def _validate_inputs(self, *args, **kwargs):
        expected_attrs = ['x', 'y', 'z']
        kwargs = PointCheckStrategy().validate_and_check(cls=Point, expected_attrs=expected_attrs, *args, **kwargs)
        return kwargs

    def __eq__(self, other):
        if isinstance(other, Point) and (other.x, other.y, other.z == self.x, self.y, self.z):
            return True
        return False

    def __repr__(self):
        return f'{self.__class__.__name__}: x = {self.x}, y = {self.y}, z = {self.z}'


def get_distance(point1: Point, point2: Point) -> float:
    """
    Returns distance between points
    :param point1: particular point with .x, .y, .z attrs
    :param point2: particular point with .x, .y, .z attrs
    :return: distance to the particular point
    """
    if not all((isinstance(point1, Point), isinstance(point2, Point))):
        raise UnspecifiedFieldException(f'Should be given Point cls, but was given: {type(point1), type(point2)}')
    return sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)


class Vector(ICheckable):
    """
    A simple vector with defined energy (luminance) in it
    initial_point: Point
    lum: float              # luminance
    w_length: float         # wave length, default 555nm green light
    theta: float            # angle between optical axis and the vector in (z,y) plane positive in CCW direction
    psi: float              # angle between optical axis and the vector in (z,x) plane positive in CCW direction
    y
    ^   ^ x
    |  /
    | /
    + ------------> z
    """
    __slots__ = '_initial_point', '_lum', '_w_length', '_theta', '_psi'

    def _validate_inputs(self, *args, **kwargs):
        expected_attrs = ['initial_point', 'lum', 'w_length', 'theta', 'psi']
        kwargs = VectorCheckStrategy().validate_and_check(cls=Vector, expected_attrs=expected_attrs, *args, **kwargs)
        return kwargs

    @property
    def direction(self) -> Dict[str, float]:
        return {'theta': self._theta, 'psi': self._psi}

    @direction.setter
    def direction(self, values: Dict[str, float]):
        self._theta, self._psi = VectorCheckStrategy.validate_angles(values)

    @property
    def initial_point(self) -> Point:
        return self._initial_point

    @initial_point.setter
    def initial_point(self, value: Point):
        temp_val = VectorCheckStrategy.validate_initial_point({'initial_point': value})
        self._initial_point = temp_val['initial_point']

    @property
    def lum(self) -> Union[int, float]:
        return self._lum

    @lum.setter
    def lum(self, value: Union[int, float]):
        temp_val = VectorCheckStrategy.validate_luminance({'lum': value})
        self._lum = temp_val['lum']

    @property
    def w_length(self) -> Union[int, float]:
        return self._w_length

    @w_length.setter
    def w_length(self, value: Union[int, float]):
        temp_val = VectorCheckStrategy.validate_w_length({'w_length': value})
        self._w_length = temp_val['w_length']

    @property
    def theta(self) -> Union[int, float]:
        return self._theta

    @theta.setter
    def theta(self, value: Union[int, float]):
        temp_val = VectorCheckStrategy.validate_angles({'theta': value})
        self._theta = temp_val['theta']

    @property
    def psi(self) -> Union[int, float]:
        return self._psi

    @psi.setter
    def psi(self, value: Union[int, float]):
        temp_val = VectorCheckStrategy.validate_angles({'psi': value})
        self._psi = temp_val['psi']

    def get_line_equation(self, verbose=False) -> Callable:
        """Returns callable - equation of a line in z = f(y), where z is an optical axis"""
        slope = 1 / tan(self.theta)
        intercept = self.initial_point.z - self.initial_point.y / tan(self.theta)
        if DEBUG:
            print(f'{slope}*y + {intercept}') if verbose else None
        return lambda y: slope * y + intercept

    @staticmethod
    def calculate_angles(*, slope, deg=False):
        """Returns and inclination between optical axis (z-axis) and a line, which slope is given"""
        slope = float(slope)
        if isnan(slope):
            raise ValueError('NaN is not a allowed as input')
        deg = bool(deg)
        try:
            theta = atan(1 / slope) % (pi)
        except ZeroDivisionError:
            theta = pi / 2
        theta = theta * 180 / pi if deg else theta
        print(f'theta is {theta} degs' if deg else f'theta is {theta} rads')
        return theta

    def __repr__(self):
        return f'{self.__class__.__name__}: ({self.initial_point}), {self.lum}, {self.w_length}, {self.theta}, {self.psi}'


@kwargs_only
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


def reversed_side(side: Side) -> Side:
    assert isinstance(side, Side)
    return Side.RIGHT if side == Side.LEFT else Side.LEFT


class Layer(ICheckable):
    """
    Each optical component is represented by intersection of layers. Each layer has name, boundary and active side,
    where material supposed to be
    """
    __slots__ = '_name', '_boundary', '_side'

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


class IOpticalSystem(ABC):
    """
    Interface which any OpticalSystem has to have in order to:
    1. add optical component to itself
    2. trace the given vector through itself
    3. store traced vector as a beam in self.vectors
    """

    @property
    @abstractmethod
    def vectors(self) -> Dict[int, List[Vector]]:
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
    @classmethod
    @abstractmethod
    def DEFAULT_CLS_MEDIUM(cls) -> Material:
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
    def vectors(self):
        return self._vectors

    @vectors.setter
    def vectors(self, value: Dict[int, List[Vector]]):
        self._vectors = value  # TODO: make conscious data checking here

    @property
    def components(self) -> List[OpticalComponent]:
        return self._components

    @components.setter
    def components(self, val: List[OpticalComponent]) -> None:
        # TODO: make actual data check here
        self._components = val

    def __init__(self, *, default_medium: Material = DEFAULT_CLS_MEDIUM):
        self._components: List[OpticalComponent] = []
        self._vectors: Dict[int, List[Vector]] = {}
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
        self.vectors[id(initial_vector)] = [initial_vector]

    def _append_to_beam(self, *, initial_vector: Vector, node_vector: Vector) -> None:
        """Adds node-vector to a beam, initiated by initial vector"""
        self.vectors[id(initial_vector)].append(node_vector)

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
                    print(f'Tracing is finished for vector: {self.vectors[id(initial_vector)][0]}. '
                          f'Last point is {self.vectors[id(initial_vector)][-1].initial_point}')
                return list(self.vectors.values())[0]
            prev_index = current_component.material.refractive_index
            current_component = self._get_containing_component_or_default(vector=current_vector)
            next_index = current_component.material.refractive_index

            try:
                current_vector = self._refract(vector=current_vector, layer=intersection_layer, prev_index=prev_index,
                                               next_index=next_index)
            except TotalInnerReflectionException as e:
                if DEBUG:
                    warn(f'\nTotal internal reflection is occurred for '
                         f'{self.vectors[id(initial_vector)][0]}.')
                current_vector = self._reflect(vector=current_vector, layer=intersection_layer)

            self._append_to_beam(initial_vector=initial_vector, node_vector=current_vector)


class IOpticalSystemBuilder(ABC):
    """Interface for any OptSys builder"""

    @property
    @abstractmethod
    def optical_system(self):
        ...

    @property
    @abstractmethod
    def vectors(self):
        ...

    @abstractmethod
    def reset(self):
        """Clear all components from optical system"""
        ...

    @abstractmethod
    def trace(self, *, vector: Vector):
        """Wrapping around optical_system.trace()"""
        ...

    @abstractmethod
    def add_components(self, components: Union[OpticalComponent, Iterable[OpticalComponent]]):
        """Appends given optical component or components to the system"""
        ...

    @abstractmethod
    def create_component(self, *, name: str, dimensions: Tuple, layers: Iterable[Layer],
                         material: Material) -> OpticalComponent:
        ...

    @abstractmethod
    def create_layer(self, *, name: str, boundary: Callable, side: Side) -> Layer:
        ...

    @abstractmethod
    def create_material(self, *, name: str, transmittance: float, refractive_index: float) -> Material:
        ...

    @abstractmethod
    def create_vector(self, *, initial_point: Point, lum: float, w_length: float, theta: float, psi: float) -> Vector:
        ...

    @abstractmethod
    def create_point(self, *args, **kwargs):
        ...


class OpticalSystemBuilder(IOpticalSystemBuilder):
    """The concrete builder of optical systems"""

    def __init__(self):
        self._optical_system = None
        self._vectors = []

    @property
    def optical_system(self):
        if not isinstance(self._optical_system, OpticalSystem):
            raise UnspecifiedFieldException(f'Optical system currently hasn''t been initialised properly. '
                                            'Use builder.reset() to create a new optical system.')
        return self._optical_system

    @optical_system.setter
    def optical_system(self, obj: OpticalSystem):
        if not isinstance(obj, OpticalSystem):
            raise UnspecifiedFieldException(f'Wrong argument type. '
                                            f'Supposed to be OpticalSystem, but was given: {type(obj)}')
        self._optical_system = obj

    @property
    def vectors(self):
        return self._vectors

    @vectors.setter
    def vectors(self, vectors: Union[Vector, Iterable[Vector]]):
        if isinstance(vectors, Vector):
            self._vectors.append(vectors)
        elif isinstance(vectors, Iterable):
            self._vectors.extend(vectors)
        else:
            raise UnspecifiedFieldException(f'Wrong argument type. '
                                            f'Supposed to be Vector or iterable of Vectors, but was given: {type(vectors)}')

    def reset(self, *, default_medium: Material = OpticalSystem.DEFAULT_CLS_MEDIUM):
        if not isinstance(default_medium, Material):
            raise UnspecifiedFieldException(f'Wrong argument type. '
                                            f'Supposed to be Material, but was given: {type(default_medium)}')
        new_opt_sys = OpticalSystem(default_medium=default_medium)
        self.optical_system = new_opt_sys
        self.vectors = list()

    def trace(self, *, vectors: Union[Vector, Iterable[Vector]]):
        opt_sys = self.optical_system
        if isinstance(vectors, Vector):
            opt_sys.trace(vector=vectors)
        elif isinstance(vectors, Iterable):
            for v in vectors:
                opt_sys.trace(vector=v)
        else:
            raise UnspecifiedFieldException(f'Wrong argument type. '
                                            f'Supposed to be Vector or iterable of Vectors, but was given: {type(vectors)}')

    def trace_all(self):
        opt_sys = self.optical_system
        self.trace(vectors=self.vectors)

    def add_components(self, components: Union[OpticalComponent, Iterable[OpticalComponent]]):
        """Checks if adding is ok and adds component to optical_system.components"""
        if isinstance(components, OpticalComponent):
            self.optical_system.add_component(component=components)
        elif isinstance(components, Iterable):
            for c in components:
                self.optical_system.add_component(component=c)
        else:
            raise UnspecifiedFieldException(f'Wrong argument type. '
                                            f'Supposed to be OpticalComponent or iterable of OpticalComponents, '
                                            f'but was given: {type(components)}')

    def create_component(self, *, name, dimensions=OPT_SYS_DIMENSIONS, layers, material) -> OpticalComponent:

        # TODO. make this aggregation through interface
        # TODO: make creation through builder
        new_component = OpticalComponent(name=name, dimensions=dimensions)
        new_component.material = material
        for l in layers:
            new_component.add_layer(layer=l)
        return new_component

    def create_layer(self, *, name: str, boundary: Callable, side: Side) -> Layer:
        new_layer = Layer(name=name, boundary=boundary, side=side)
        return new_layer

    def create_material(self, *, name: str, transmittance: float, refractive_index: float) -> Material:
        new_material = Material(name=name, transmittance=transmittance, refractive_index=refractive_index)
        return new_material

    def create_vector(self, *, initial_point: Point, lum: float, w_length: float, theta: float, psi: float) -> Vector:
        new_vector = Vector(initial_point=initial_point, lum=lum, w_length=w_length, theta=theta, psi=psi)
        return new_vector

    def create_point(self, *args, **kwargs):
        """
        Gets the first tuple of floats of len==3 in positional args: (x: float, y: float, z:float),
        or keyword arguments for each dimension and returns Point cls object
        """
        for arg in args:
            if len(arg) == 3 and all((isinstance(el, (int, float)) for el in arg)):
                coords = {'x': arg[0], 'y': arg[1], 'z': arg[2]}
                return Point(**coords)

        if all((dim in kwargs for dim in ['x', 'y', 'z'])):
            coords = {dim: kwargs[dim] for dim in ['x', 'y', 'z']}  # filter kwargs with certain keys
            return Point(**coords)

        raise UnspecifiedFieldException('Wrong arguments. Supposed to be Tuple[float, float, float] in args or '
                                        '{''x'': float, ''y'': float, ''z'': float} in kwargs,  but was given')


def main():
    def create_first_medium():
        first_left_bound = Layer(boundary=lambda y: 0 - y ** 2 / 400, side=Side.LEFT, name='First-left bound')  #
        # first_right_bound = Layer(boundary=lambda y: 100 + y ** 2 / 400, side=Side.LEFT, name='First-right bound')
        first_material = Material(name='Glass', transmittance=0.9, refractive_index=1.1)
        first_medium = OpticalComponent(name='First')
        first_medium.add_layer(layer=first_left_bound)
        # first_medium.add_layer(layer=first_right_bound)
        first_medium.material = first_material
        return first_medium

    def create_second_medium():
        second_left_bound = Layer(boundary=lambda y: 100 + y ** 2 / 400, side=Side.RIGHT, name='Second-left bound')
        second_right_bound = Layer(boundary=lambda y: 200 + y ** 2 / 400, side=Side.LEFT, name='Second-right bound')
        second_material = Material(name='Glass', transmittance=0.9, refractive_index=1.2)
        second_medium = OpticalComponent(name='Second')
        second_medium.add_layer(layer=second_left_bound)
        second_medium.add_layer(layer=second_right_bound)
        second_medium.material = second_material
        return second_medium

    def create_third_medium():
        third_left_bound = Layer(boundary=lambda y: 200 + y ** 2 / 400, side=Side.RIGHT, name='Third-left bound')
        third_right_bound = Layer(boundary=lambda y: 300 + y ** 2 / 400, side=Side.LEFT, name='Third-right bound')
        third_material = Material(name='Glass', transmittance=0.9, refractive_index=1.3)
        third_medium = OpticalComponent(name='Third')
        third_medium.add_layer(layer=third_left_bound)
        third_medium.add_layer(layer=third_right_bound)
        third_medium.material = third_material
        return third_medium

    def create_fourth_medium():
        fourth_left_bound = Layer(boundary=lambda y: 300 + y ** 2 / 400, side=Side.RIGHT, name='Fourth-left bound')
        fourth_material = Material(name='Glass', transmittance=0.9, refractive_index=1.4)
        fourth_medium = OpticalComponent(name='Fourth')
        fourth_medium.add_layer(layer=fourth_left_bound)
        fourth_medium.material = fourth_material
        return fourth_medium

    opt_sys = OpticalSystem()
    first_medium, second_medium, third_medium, fourth_medium = (medium for medium in (create_first_medium(),
                                                                                      create_second_medium(),
                                                                                      create_third_medium(),
                                                                                      create_fourth_medium()
                                                                                      )
                                                                )
    [opt_sys.add_component(component=med) for med in (first_medium, second_medium, third_medium, fourth_medium)]
    in_point = Point(x=0, y=50, z=250)
    resolution = 10  # vectors per circle
    for theta in range(
            int(2 * pi * resolution + 2 * pi * 1 / resolution)):  # 2 * pi * 1/resolution addition to make compleete circle
        if 52 <= theta < 53 and True:  #
            v = Vector(initial_point=in_point, lum=1, w_length=555, theta=theta / resolution, psi=0)
            opt_sys.trace(vector=v)
    print(opt_sys.vectors)
    return opt_sys


if __name__ == '__main__':
    main()
