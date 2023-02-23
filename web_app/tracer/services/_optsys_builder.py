__all__ = ['IOpticalSystemBuilder',
           'OpticalSystemBuilder']

from abc import ABC, abstractmethod
from typing import Optional, List, Union, Iterable, Tuple, Callable
from warnings import warn

from core import (OpticalSystem,
                  Vector,
                  OpticalComponent,
                  Layer,
                  Material,
                  Side,
                  Point,
                  IOpticalSystem,
                  UnspecifiedFieldException,
                  OPT_SYS_DIMENSIONS)
from ._infrastructural_exceptions import EmptyBoundaryEquationSyntaxError, WrongBoundaryEquationSyntaxError


class IOpticalSystemBuilder(ABC):
    """Interface for any OptSys builder"""

    @property
    @abstractmethod
    def optical_system(self):
        ...

    @abstractmethod
    def set_optical_system_name(self, name: str):
        """Sets the name of inited optical system"""

    @property
    @abstractmethod
    def vectors(self):
        """Initial vectors which is to be traced with self.trace_all()"""
        ...

    @abstractmethod
    def reset(self, optical_system: Optional[OpticalSystem], vectors: List[Optional[Vector]]):
        """Resets builder instance with the provided optical system and vectors list."""
        ...

    @abstractmethod
    def trace(self, *, vector: Vector):
        """Wrapping around optical_system.trace()"""
        ...

    @abstractmethod
    def trace_all(self):
        """Uses vectors in self.vectors and traces them"""
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

    @staticmethod
    @abstractmethod
    def create_side(*, side: str) -> Side:
        ...

    @staticmethod
    @abstractmethod
    def create_boundary_callable(*, equation: str) -> Callable:
        ...

    @abstractmethod
    def create_vector(self, *, initial_point: Point, lum: float, w_length: float, theta: float, psi: float) -> Vector:
        ...

    @abstractmethod
    def create_point(self, *args, **kwargs):
        ...


class OpticalSystemBuilder(IOpticalSystemBuilder):
    """The concrete builder of optical systems"""

    def __init__(self, *, optical_system: Optional[IOpticalSystem] = None):
        self._vectors = []
        if optical_system is None:
            self.reset()
            warn(f'{self.__class__}: while initing builder OpticalSystem hadn''t been provided. '
                 'Empty optical system has been created.')
        else:
            self._optical_system = optical_system

    @property
    def optical_system(self):
        if not isinstance(self._optical_system, OpticalSystem):
            raise UnspecifiedFieldException(f'Optical system currently hasn''t been initialised properly. '
                                            'Use builder.reset() to create a new optical system.')
        return self._optical_system

    @optical_system.setter
    def optical_system(self, obj: IOpticalSystem):
        if not isinstance(obj, OpticalSystem):
            raise UnspecifiedFieldException(f'Wrong argument type. '
                                            f'Supposed to be OpticalSystem, but was given: {type(obj)}')
        self._optical_system = obj

    def set_optical_system_name(self, name: str):
        self.optical_system.name = name

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

    def reset(self, *, optical_system: Optional[OpticalSystem] = OpticalSystem(),
              vectors: List[Optional[Vector]] = list()):
        """
        Resets builder instance with the provided optical system and vectors list.
        If args are not given, sets optical system empty with default values, sets vectors as empty list
        """
        self.optical_system = optical_system
        self.vectors = vectors

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

    @staticmethod
    def create_side(*, side: str) -> Side:
        """Returns Side object depending on the give string Left or Right"""
        return Side.from_str(side)

    @staticmethod
    def create_boundary_callable(*, equation: str) -> Callable:
        """Gets input as string, validates it and returns callable object"""
        math_variable_name = 'y'  # y is an independent variable, z = f(y) z - optical axes
        chars_allowed = "".join(("+-*/0123456789.,() ", math_variable_name))

        def _validate(equation: str) -> None:
            """
            Validates the given equation using set difference between input and allowed chars
            @param equation: supposed to be a string containing the name of math variable, digits, comma, dot, and signs
            @return: bool
            """
            assert isinstance(equation, str), f'Wrong equation type while validating in ' \
                                              f'builder.create_boundary_callable. Was given {type(equation)}, ' \
                                              f'but should be a string'
            if not len(equation):
                raise EmptyBoundaryEquationSyntaxError(f'Equation field is empty.')
            not_allowed_input = set(equation) - set(chars_allowed)
            if not_allowed_input:
                raise WrongBoundaryEquationSyntaxError(f'Equation contains not allowed chars: {not_allowed_input}')

        def _prepare_to_eval(equation: str) -> str:
            """
            Make input string.
            Replace decimal separator
            Make sure there is no ellipsises
            """

            def _remove_ellipsis():
                """Shrinks all consecutive dots to a single dot"""
                nonlocal equation
                if len(equation) < 2:
                    return
                equation_lst = list(equation)
                idx = 0
                while idx < len(equation_lst) - 1:
                    if equation_lst[idx] in ['.', ''] and equation_lst[idx + 1] in ['.', '']:
                        equation_lst[idx + 1] = ''
                    idx += 1
                equation = ''.join(equation_lst)
                return

            equation = str(equation)
            equation.replace(',', '.')
            _remove_ellipsis()
            return equation

        _validate(equation)
        equation = _prepare_to_eval(equation)
        try:
            func = lambda y: eval(equation)
        except SyntaxError as e:
            raise WrongBoundaryEquationSyntaxError from e
        return func

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
