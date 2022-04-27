from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from enum import auto, Enum
from math import acos, asin, atan, pi, sqrt, tan
from typing import Callable, ClassVar, Dict, Optional, List, Tuple, Union
from warnings import warn

from scipy.optimize import fsolve
import numpy as np

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
    """Raises then vector doesn't inresect any surface"""
    pass

class ObjectKeyWordsMissmatch(Warning):
    """Raises when __init__ gets unexpexted **kwargs"""
    pass


class ICheckable(ABC):
    """Interface for object with necessity of imput vars' check"""
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
    """The way in which any Point object's inputs should be checked"""          #FIXME: Add concrete conditions
    def check(self, *args, **kwargs):
        expected_coords_names = [name[1:] for name in Point.__slots__ if name.startswith('_')]
        for coord in kwargs.items():
            kwargs[coord[0]] = float(coord[1])

        if not all (coord in kwargs for coord in expected_coords_names):
            print(f'Not enough args. Should exists {expected_coords_names}')


        if not all(coord in expected_coords_names for coord in kwargs):
            warn(f'Wrong keyword in : {kwargs}', ObjectKeyWordsMissmatch)


class VectorCheckStrategy(BaseCheckStrategy):
    """The way in which any Vector object's inputs should be checked"""          #FIXME: Add concrete conditions
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

    def _check_inputs(self, *args, **kwargs):
        PointCheckStrategy().check(*args, **kwargs)

    def _throw_inputs(self, *args, **kwargs):       #FIXME: fix this shit
        self._x: float = kwargs['x']
        self._y: float = kwargs['y']
        self._z: float = kwargs['z']


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

    def get_line_equation(self, repr=False) -> Callable:
        """Returns callable - equation of a line in z = f(y), where z is an optical axis"""
        A = 1 / tan(self.theta)
        B = self.initial_point.z - self.initial_point.y / tan(self.theta)
        print(f'{A}*y + {B}') if repr else None
        return lambda y : A * y + B

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
    """
    Singleton of entire system
    :param dimentions : List[float] aproximate the most dimentions in (x, y) plane in mm. Defaule [-100, 100]
    """
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self, dimentions: List[float] = [-100, 100]):
        self._layers = [Layer(name='Only Air',
                             material=Material(name='Air', transparency=1, refractive_index=1),
                             boundary=lambda y: float('inf')        # no intersection point w/ any of lines at plane
                             )
                        ]
        self._dimentions = dimentions                                                # FIXME: check inputs here

    def add_layer(self, *, name: str, material: Material, boundary: Callable, side: Side):
        new_layer = Layer(name=name, material=material, boundary=boundary, side=side)
        self._layers.append(new_layer)

    def _check_probable_intersections(self, *, probable_ys: List[float], layer: Layer, vector: Vector):
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
        for y in probable_ys:
            # get z-coord of intersection on surface
            z_surf_intersection = surface(y)
            # get z-coord of intersection on line
            z_line_intersection = line(y)
            # is this the same point?
            difference = z_surf_intersection - z_line_intersection

            if difference > vector.w_length * 10 ** (-6) / 4:                            # quarter part of wave length
                warn(f'Line and surface difference intersections: {difference}', NoIntersectionWarning)
                # FIXME: check measures meters or milimeters?
                continue

            vector_directed_left = pi / 2 <= vector.theta <= 3 * pi / 2
            intersection_is_righter = z_surf_intersection > vector.initial_point.z
            if intersection_is_righter == vector_directed_left :
                # checks if vector is directed to the intersection
                warn(f'Surface is out of vectors direction: '
                     f'theta={vector.theta:.3f}, '
                     f'intersection at (y,z)=({y:.3f}, {z_surf_intersection:.3f})', NoIntersectionWarning)
                continue
            approved_ys.append(y)
        return approved_ys

    def _get_intersection(self, *, vector: Vector, layer: Layer) -> List[Tuple[float]]:
        """
        Returns valid intersections of the vector with boundary layer.boundary
        :return: List of tuples
        """
        line = vector.get_line_equation()
        surface = layer.boundary
        equation = lambda y: surface(y) - line(y)                                    # only for (y,z)-plane
        probable_y_intersections = list(fsolve(equation, np.array(self._dimentions)))
        approved_ys = self._check_probable_intersections(probable_ys=probable_y_intersections,
                                                         layer=layer,
                                                         vector=vector)
        # [(y, z), .....]
        approved_zs = [surface(y) for y in approved_ys]
        return list(zip(approved_ys, approved_zs))




    def trace_vector_on_layer(self, *, vector: Vector, layer: Layer):                 # FIXME: add return annotation
        raise NotImplementedError



def main():
    opt_s = OpticalSystem()
    opt_s.add_layer(name='Glass parabolic',
                    material=Material(name='Glass', transparency=0.9, refractive_index=0.75),
                    boundary=lambda y: y ** 2 + 100,
                    side=Side.RIGHT,
                    )
    opt_s.add_layer(name='Glass plane',
                    material=Material(name='Glass', transparency=0.9, refractive_index=0.75),
                    boundary=lambda y: 120,
                    side=Side.LEFT,
                    )

    v = Vector(initial_point=Point(q=0, x=0, y=0, z=0), lum=1, w_length=555, theta=0.03, psi=0)
    print(*opt_s._get_intersection(vector=v, layer=opt_s._layers[-2]), sep='\n')


if __name__ == '__main__':
    main()