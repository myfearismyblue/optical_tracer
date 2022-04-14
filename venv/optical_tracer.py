from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from enum import auto, Enum
from math import atan, pi
from typing import Callable, ClassVar, Dict, Optional, List, Tuple, Union

from scipy.optimize import fsolve


def kwargs_only(cls):
    """Auxiliary func to make class initions only with keyword args"""
    @wraps(cls)
    def call(**kwargs):
        return cls(**kwargs)

    return call


class VectorOutOfComponentError(Exception):
    """Raises then coords of a vector are out of optical component which it was given"""
    pass

@kwargs_only
@dataclass
class Point:
    """
    Just a point w/ cartesian coordinates in an optical system, where z is an optical axis
    x: float            # (x, y, z) - ordered triplet, where  (x, y) - sagittal plane, (y, z) - meridonial plane
    y: float            # of an optical system
    z: float            # z - optical axis
    """
    __slots__ = '_x', '_y', '_z'
    x: float
    y: float
    z: float

    def __post_init__(self):
        self._x: float = self.x
        self._y: float = self.y
        self._z: float = self.z

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

    def set_coords(self, **coords:  Union[float, int]):                                             # FIXME: check inputs
        tmp_coords = coords.get('x'), coords.get('y'), coords.get('z')
        self_coords = self._x, self._y, self._z
        self_coords = [self_coord if tmp_coord is None else tmp_coord
                                                            for tmp_coord, self_coord in zip(tmp_coords,self_coords)]



    def get_coords(self, coords: str) -> Dict[str, Union[float, int]]:                               # FIXME: check inputs
        """
        :argument Use string as input like point.get_coords('yx').
        :returns dict like {'y': 0, 'x': 0.5}
        """
        return {coord: getattr(self, '_'+coord) for coord in coords}


@kwargs_only
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


m = Material(name='Glass', transparency=0.9, refractive_index=0.8)
m.add_standart_material(name='Air', transparency=1, refractive_index=1)
print(Material.standart_materials)

class OpticalComponent(ABC):
    """Material with boundaries which are to constrain material"""
    def __init__(self, material: Material):
        self.material = material

    def __get_self_position(self):
        """
        Ask an optical system about the components self position.
        :returns Absolute offset of (x=0, y=0, z=0) point of the component relativly (x=0, y=0, z=0) of the system
        """
        raise NotImplementedError

    @abstractmethod
    def __is_in_boundaries(self, point: Point) -> bool:
        """
        Place here equations constraining optical components surfaces like x ** 2 + y ** 2 + z ** 2  <= 0
        Return True if the point is in the boundaries of the optical component.
        """
        pass

    @abstractmethod
    def __get_intersection(self, vector: Vector) -> Point:
        """Returns point where vector line intersects boundaries of a component"""
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


class Side(Enum):
    RIGHT = auto()
    LEFT = auto()

@kwargs_only
@dataclass
class Layer:
    """
    Whole optycal system represented by conception of layers. Each medium with surfaces is not component in the system,
    but it's a layer of nothing and medium devided by surface. And the opticals system is the superposition of layers.
    """
    __slots__ = '_name', '_material', '_boundary', '_side'
    name: str
    material: Material
    boundary: Callable
    side: Side = field(default=Side.RIGHT)

    def __post_init__(self):
        self._name = self.name
        self._material = self.material
        self._boundary = self.boundary
        self._side = self.side

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, _val: str):
        self._name = _val

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, _val: Material):
        self._material = _val

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, _val: Callable):
        self._boundary = _val

    @property
    def side(self):
        return self._side

    @side.setter
    def side(self, _val: Side):
        self._side = _val


class OpticalSystem:
    """Singleton of entire system"""

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
            return cls.instance

        else:
            return cls.instance

    def __init__(self):
        self._layers = Layer(name='Only Air',
                             material=Material(name='Air', transparency=1, refractive_index=1),
                             boundary = lambda y: float('inf')
                             )




os = OpticalSystem()
print(os._layers)


def main():
    ...


if __name__ == '__main__':
    main()