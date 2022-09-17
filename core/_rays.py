__all__ = ['Point', 'Vector', 'get_distance']

from abc import ABC, abstractmethod
from math import sqrt, tan, isnan, pi, atan
from typing import Union, Dict, Callable
from warnings import warn

from ._config import DEBUG, OPTICAL_RANGE
from ._exceptions import ObjectKeyWordsMismatchException, UnspecifiedFieldException


class ICheckable(ABC):
    """
    Interface for object with necessity of input vars' validation.
    All vars to validate are to be forwarded to attrs with name like '_val'
    """

    def __init__(self, *args, **kwargs):
        kwargs = self._validate_inputs(*args, **kwargs)
        self._throw_inputs(*args, **kwargs)

    @abstractmethod
    def _validate_inputs(self, *args, **kwargs):
        ...

    def _throw_inputs(self, *args, **kwargs):
        """
        Throwing attrs adding _underscore if it hasn't it,
        if there is a special slots which are not to be set with expected kwargs, then set it with None
        """
        [setattr(self, attr, kwargs[attr]) if attr.startswith('_')  # if kwarg already starts with _ - throw it
         else setattr(self, ''.join(('_', attr)), kwargs[attr]) for attr in kwargs.keys()]  # else add _ as prefix

        extra_slots = set(self.__slots__) - set((item if item.startswith('_') else
                                                 ''.join(('_', item)) for item in kwargs.keys()))
        [setattr(self, item, None) for item in extra_slots]


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
        self._check_kwarg_completeness(cls, kwargs, expected_attrs)
        kwargs = self.validate(*args, **kwargs)
        return kwargs

    @staticmethod
    def _ensure___slots__ok(cls, expected_attrs):
        """Inner assertion to ensure that all expected inputs in slots """
        assert all((f'_{exp_attr}' in cls.__slots__ for exp_attr in expected_attrs))

    @staticmethod
    def _check_kwarg_completeness(cls, kwargs, expected_attrs):
        """
        Checks if it's enough kwargs and if it's more than needed.
        Raises  ObjectKeyWordsMismatchException if not enough, or if there are some extra kwargs
        """

        expected_kwargs_names = [kw[1:] if kw.startswith('_') else kw for kw in expected_attrs]

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
            if isnan(temp):
                raise ValueError(f'Not allowed Nan: {coord}')
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

