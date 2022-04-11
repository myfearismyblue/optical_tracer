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




p = Point(x=0,y=0,z=0)
p.set_coords(y=2, f=2)
print(p.get_coords('yx'))


@dataclass
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
    __slots__ = '_initial_point', '_lum', '_w_length', '_tetha', '_psi'
    _initial_point: Point
    _lum: float             # luminance
    _w_length: float        # wave length
    _tetha: float           # in rads.angle between optical axis and the vector in (z,y) plane positive in CCW direction
    _psi: float             # in rads.angle between optical axis and the vector in (z,x) plane positive in CCW direction

    @property
    def direction(self) -> Tuple[float, float]:
        return self._tetha, self.psi

    @property
    def initial_point(self) -> Point:
        return self._initial_point

    @property
    def lum(self) -> Union[int, float]:
        return self._lum

    @property
    def w_length(self) -> Union[int, float]:
        return self._w_length

    @property
    def tetha(self) -> Union[int, float]:
        return self._tetha

    @property
    def psi(self) -> Union[int, float]:
        return self._psi





@dataclass
class Material:
    """Medium where energy vector propagates"""
    transparency: float
    refractive_index: float


class Boundaries(ABC):
    """Put here your equations constraining optical components surfaces in F(x, y, z)"""
    @abstractmethod
    def boundaries_check(self, x: Union[float, int], y: Union[float, int], z: Union[float, int]) -> bool:
        pass


class OpticalComponent:
    """Material with concrete boundaries which are to constrain material"""
    def __init__(self, material: Material, boundaries: Boundaries):
        self.material = material
        self._boundaries = boundaries.boundaries_check

    def __is_in_boundaries(self, point: Point) -> bool:
        """Return True if the point is in the boundaries of the optical component"""
        return self._boundaries(point)

    def trace_vector(self, vector: Vector) -> Vector:
        """Traces the vector to a boundary of the component"""

        # ensure about the vector is in boundaries
        # get a direction
        # find intersection between the direction line and surface
        # create vector on the surface

        if not self.__is_in_boundaries(self, vector):
            raise VectorOutOfComponentError()

        direction_angles: Tuple[Union[int, float]] = vector.direction

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