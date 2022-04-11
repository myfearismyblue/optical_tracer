from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import atan, pi
from typing import Callable, Dict, Optional, Tuple, Union


class VectorOutOfComponentError(Exception):
    """Raises then coords of a vector are out of optical component which it was given"""
    pass


class Point:
    """
    Just a point w/ cartesian coordinates in an optical system, where z is an optical axis
    x: float            # (x, y, z) - ordered triplet, where  (x, y) - sagittal plane, (y, z) - meridonial plane
    y: float            # of an optical system
    z: float            # z - optical axis
    """
    __slots__ = '_x', '_y', '_z'

    def __init__(self, *, x: Union[float, int], y: Union[float, int], z: Union[float, int]):
        self._x: float = x
        self._y: float = y
        self._z: float = z

    def __str__(self):
        res = str(self.__class__)
        for attr in self.__slots__:
            res = "".join((res, attr, ' = ', str(getattr(self, attr)), ', '))
        return res

    def set_coords(self, **coords:  Union[float, int]):                                             # FIXME: check inputs
        tmp_x, tmp_y, tmp_z = coords.get('x'), coords.get('y'), coords.get('z')
        self._x = self._x if tmp_x is None else tmp_x
        self._y = self._y if tmp_y is None else tmp_y
        self._z = self._z if tmp_z is None else tmp_z

    def get_coords(self, coords: str) -> Dict[str, Union[float, int]]:                               # FIXME: check inputs
        """
        :argument Use string as input like point.get_coords('yx').
        :returns dict like {'y': 0, 'x': 0.5}
        """
        return {coord: getattr(self, '_'+coord) for coord in coords}


class Vector:
    """
    A simple vector with defined energy (luminance) in it
    initial_point: Point
    lum: float              # luminance
    w_length: float         # wave length, default 555nm green light
    tetha: float            # angle between optical axis and the vector in (z,y) plane positive in CCW direction
    psi: float              # angle between optical axis and the vector in (z,x) plane positive in CCW direction
    y
    ^    ^ x
    |   /
    | /
    + ------------> z
    """
    __slots__ = '_initial_point', '_lum', '_w_length', '_theta', '_psi'

    def __init__(self, *, initial_point: Point, lum: float, w_length: float, theta: float, psi: float):
        self._initial_point = initial_point
        self._lum = lum
        self._w_length = w_length
        self._theta = theta
        self._psi = psi

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
    def initial_point(self,value: Point):
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


p = Point(x=0,y=1,z=2)
v = Vector(initial_point=p, w_length=555, lum=1, theta=0, psi=0)
v.direction = {'theta': 0.4, 'psi': 0.1}
print(v.direction)


@dataclass
class Material:
    """Medium where energy vector propagates"""
    transparency: float
    refractive_index: float


# class Boundaries(ABC):
#     """Put here your equations constraining optical components surfaces in F(x, y, z)"""
#     @abstractmethod
#     def boundaries_check(self, x: Union[float, int], y: Union[float, int], z: Union[float, int]) -> bool:
#         pass


class OpticalComponent(ABC):
    """Material with boundaries which are to constrain material"""
    def __init__(self, material: Material):
        self.material = material

    @abstractmethod
    def __is_in_boundaries(self, point: Point) -> bool:
        """
        Place here equations constraining optical components surfaces like x ** 2 + y ** 2 + z ** 2  <= 0
        Return True if the point is in the boundaries of the optical component.
        """
        pass

    @abstractmethod
    def __get_intersection(self, vector: Vector) -> Point:
        """Returns point where vector line intersects with  boundaries of a cimponent"""
        pass

    def trace_vector(self, vector: Vector) -> Vector:
        """Traces the vector to a boundary of the component"""

        # ensure about the vector is in boundaries
        # get a direction
        # find intersection between the direction line and surface
        # create vector on the surface

        if not self.__is_in_boundaries(self, vector):
            raise VectorOutOfComponentError()
        direction_angles: Dict[str, float] = vector.direction
        outer_point = self.__get_intersection(vector)
        return outer_point


class OpticalSystem:
    """Singleton of entire system"""
    def __init__(self):
        self.optical_components = dict()            # FIXME: maybe id:instance ?
        self.component_distances = dict()           # FIXME: what about component's geometry? distance to which point?

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__()
        else:
            return cls.instance

    def add_component(self):
        raise NotImplementedError

    def delete_component(self):
        raise NotImplementedError


def main():
    ...


if __name__ == '__main__':
    main()