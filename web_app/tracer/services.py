from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import sqrt, atan, pi
from typing import Tuple, Callable, List, Dict, Iterable, Union, Optional
from warnings import warn

import dill
from django.forms import forms

from optical_tracer import Side, Layer, Material, OpticalComponent, OpticalSystem, Vector, Point, \
    UnspecifiedFieldException, OPT_SYS_DIMENSIONS, IOpticalSystem
from .forms import AddComponentForm, ChooseOpticalSystemForm
from .models import AxisView, BoundaryView, BeamView, VectorView, SideView, OpticalSystemView


class UserInfrastructuralBaseException(Exception):
    """The base for django-models exceptions"""
    ...


class UnregisteredContextException(UserInfrastructuralBaseException):
    """Raises when unknown context requested to be prepared"""
    ...


class WrongBoundaryEquationSyntaxError(UserInfrastructuralBaseException):
    """Raises when wrong boundary equation syntax is used while form's submitting"""
    ...


class EmptyBoundaryEquationSyntaxError(UserInfrastructuralBaseException):
    """Raises when empty boundary equation syntax is used while form's submitting"""
    ...


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
        math_variable_name = 'y'    # y is an independent variable, z = f(y) z - optical axes
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


@dataclass
class ContextRequest:
    """DTO for requesting different contexts for a view from GraphService"""
    contexts_list: List[str]  # list of certain contexts which are supposed to be drawn with GraphService
    graph_info: Dict  # common info for GraphService like dimensions of canvas etc. To be specified


@dataclass
class Context:
    """DTO as a response to be forwarded to view.py"""
    name: str  # context name. Specified in ContextRegistry
    value: Dict


class InitGraphServiceBaseStrategy(ABC):
    """The base class of initing graph service depending on different given contexts"""

    def __init__(self, contexts_request: ContextRequest, graph_service):
        self._contexts_request = contexts_request
        self._graph_service = graph_service

    @abstractmethod
    def init_graph_service(self):
        """ Do various stuff before preparing contexts"""
        ...


class CanvasInitGraphServiceStrategy(InitGraphServiceBaseStrategy):
    """
    Responsible for correct initing of graph service if 'canvas_context' is demanded.
    Occasionally it should be done anyway if we want something to be drawn on a canvas
    """

    def init_graph_service(self):
        contexts_request = self._contexts_request  # just make synonym
        self._graph_service._graph_objects = {}
        self._graph_service._canvas_dimensions = (contexts_request.graph_info['canvas_width'],
                                                  contexts_request.graph_info['canvas_height'])

        # offset of entire optical system relatively to (0, 0) canvas point which is upper-left corner
        self._graph_service._offset = self._graph_service._canvas_dimensions[0] // 3, \
                                      self._graph_service._canvas_dimensions[1] // 3
        self._graph_service._scale = contexts_request.graph_info['scale']

        # ranges in which components to be drawn relatively to OPTICAL_SYSTEM_OFFSET in pixels here
        # coordinates are upside-down because of reversion of vertical axis
        self._graph_service._height_draw_ranges = self._graph_service._offset[1] - \
                                                  self._graph_service._canvas_dimensions[1], \
                                                  self._graph_service._offset[1]
        self._graph_service._width_draw_ranges = -self._graph_service._offset[0], \
                                                 self._graph_service._canvas_dimensions[0] - \
                                                 self._graph_service._offset[0]


class BoundariesInitGraphServiceStrategy(InitGraphServiceBaseStrategy):
    """Responsible for correct initing of graph service if optical component should be drawn"""

    def init_graph_service(self):
        self._graph_service.push_layers_to_db()


class AxisInitGraphServiceStrategy(InitGraphServiceBaseStrategy):
    """Responsible for correct initing of graph service if axis should be drawn"""

    def init_graph_service(self):
        """Using inner properties of service calculates AxisView model and pushes them to db"""
        self._graph_service.push_axes_to_db()


class BeamsInitGraphServiceStrategy(InitGraphServiceBaseStrategy):
    """Responsible for correct initing of graph service if optical rays should be drawn"""

    def init_graph_service(self):
        self._graph_service.push_beams_to_db()


class OpticalSystemInitGraphServiceStrategy(InitGraphServiceBaseStrategy):

    def init_graph_service(self):
        pass


class PrepareContextBaseStrategy(ABC):
    """Base class for different strategies preparing various objects' context for template """

    @abstractmethod
    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        ...

    @staticmethod
    def _stringify_points_for_template(points: Iterable) -> str:
        """Prepares coords to be forwarded in a <svg> <polyline points="x0, y0 x1, y1 x2, y2..." >"""
        res = ''
        for p in points:
            if hasattr(p, 'x0'):  # TODO: Refactor this shit
                current_element = f'{p.x0}, {p.y0}'
            else:
                current_element = f'{p[0]}, {p[1]}'
            res = ' '.join((res, current_element))  # _space_ here to separate x1, y1_space_x2, y2
        return res


class BoundariesPrepareContextStrategy(PrepareContextBaseStrategy):
    """
    Describes the way in which template context for drawning OpticalComponent boundaries, material etc. should be created
    """
    __context_name = 'boundaries_context'

    def __init__(self):
        self.context = Context(name=self.__context_name, value={})

    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        """
        Fetches all boundaries from models.BoundaryView. For each one fetches points to draw from model.BoundaryPoint.
        Stringifys them and appends to context to be forwarded to template
        """

        for boundary in BoundaryView.objects.all():
            # here layer_points is the dict which is give in kwargs directly from the GraphService
            # TODO here is to consider behaviour of GraphService in cases where data is fetching from db or getting directly
            # from domain model. I suppose sometimes it is needed to retrieve data like layers, axis etc from db,
            # but sometimes is should be calculated in a runtime. Bad idea to hardcode 'layer_points' literal directly
            layer_points = kwargs['layer_points']
            points = self._stringify_points_for_template(layer_points[boundary.memory_id])
            self.context.value[boundary.memory_id] = points
        return self.context


class AxisPrepareContextStrategy(PrepareContextBaseStrategy):
    """Describes the way in which template context for drawning optical axis should be created"""
    __context_name = 'axis_context'

    def __init__(self):
        self.context = Context(name=self.__context_name, value={})

    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        for axis in AxisView.objects.all():
            if axis.direction in ['down', 'right']:
                points = f'{axis.x0}, {axis.y0} {axis.x1}, {axis.y1}'
            elif axis.direction in ['up', 'left']:
                points = f'{axis.x1}, {axis.y1} {axis.x0}, {axis.y0}'
            else:
                raise ValueError(f'Wrong direction type in {axis}: {axis.direction}')
            self.context.value[axis.memory_id] = points
        return self.context


class BeamsPrepareContextStrategy(PrepareContextBaseStrategy):
    """Describes the way in which template context for drawning rays should be created"""
    __context_name = 'beams_context'

    def __init__(self):
        self.context = Context(name=self.__context_name, value={})

    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        """
        Fetches beams from models.BeamView. For each beam gets it's points from models.VectorView.
        Stringifys them and appends to context to be forwarded to template
        """
        for beam in BeamView.objects.all():
            points = self._stringify_points_for_template(VectorView.objects.filter(beam=beam.pk))
            self.context.value[beam.memory_id] = points
        return self.context


class CanvasPrepareContextStrategy(PrepareContextBaseStrategy):
    __context_name = 'canvas_context'

    def __init__(self):
        self.context = Context(name=self.__context_name, value={})

    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        self.context.value = context_request.graph_info
        return self.context


class OpticalSystemPrepareContextStrategy(PrepareContextBaseStrategy):
    """"""  # FIXME: make some docstring after certain implementation will be clear
    __context_name = 'opt_sys_context'

    def __init__(self):
        self.context = Context(name=self.__context_name, value={})

    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        raise NotImplementedError


@dataclass
class ContextRegistry:
    # FIXME: link this with ContextRequest
    __registered_contexts = {'canvas_context': {'prepare_context': CanvasPrepareContextStrategy,
                                                'init_graph_service': CanvasInitGraphServiceStrategy},
                             'boundaries_context': {'prepare_context': BoundariesPrepareContextStrategy,
                                                    'init_graph_service': BoundariesInitGraphServiceStrategy},
                             'axis_context': {'prepare_context': AxisPrepareContextStrategy,
                                              'init_graph_service': AxisInitGraphServiceStrategy},
                             'beams_context': {'prepare_context': BeamsPrepareContextStrategy,
                                               'init_graph_service': BeamsInitGraphServiceStrategy},
                             'opt_sys_context': {'prepare_context': OpticalSystemPrepareContextStrategy,
                                                 'init_graph_service': OpticalSystemInitGraphServiceStrategy}, }

    def get_prepare_strategy(self, context_name: str) -> PrepareContextBaseStrategy:
        if str(context_name) not in self.__registered_contexts:
            raise UnregisteredContextException(f'Requested context is unknown: {context_name}. '
                                               f'Registered contexts: {self.__registered_contexts}')
        return self.__registered_contexts[context_name]['prepare_context']

    def get_init_strategy(self, context_name: str) -> InitGraphServiceBaseStrategy:
        if str(context_name) not in self.__registered_contexts:
            raise UnregisteredContextException(f'Requested context is unknown: {context_name}. '
                                               f'Registered contexts: {self.__registered_contexts}')
        return self.__registered_contexts[context_name]['init_graph_service']

    def get_registered_contexts(self):
        return tuple(self.__registered_contexts.keys())

    def get_context_name(self, context_name: str):
        """Returns a name of context if context is registered"""
        if str(context_name) not in self.__registered_contexts:
            raise UnregisteredContextException(f'Requested context is unknown: {context_name}. '
                                               f'Registered contexts: {self.__registered_contexts}')
        return str(context_name)


class IGraphService(ABC):
    @abstractmethod
    def fetch_optical_system(self):  # TODO: add contract here
        """Initing concrete optical system"""
        ...

    @abstractmethod
    def prepare_contexts(self, contexts_request: ContextRequest) -> List[Context]:
        """Using optical system make some contexts to be thrown in template"""
        ...


def push_sides_to_db_if_not_exist() -> None:
    """Sets boundary sides - left and right - to db"""

    def _reset_sides():
        SideView.objects.create(side='Left')
        SideView.objects.create(side='Right')

    query = SideView.objects.all()
    stored_values = []
    [stored_values.append(obj.side) for obj in query]
    if len(query) != 2 or stored_values not in (['Left', 'Right'], ['Right', 'Left']):
        _reset_sides()


class GraphService(IGraphService):
    """Responsible for """

    def __init__(self, contexts_request: ContextRequest, optical_system: OpticalSystem = None):
        """
        Check if the given contexts are valid, checks if the given optical system is valid,
        if optical system is not provided than creates new one with builder,
        clears off db of unnecessary data of previous sessions,
        and make appropriate initions according to the given context reuest list
        @param contexts_request: A ContextRequest which contains info for canvas graphing and info about things that shold be drawn
        @param optical_system: If not given than an empty one will be created
        """

        self._check_context_registered(contexts_request)
        if not isinstance(optical_system, OpticalSystem):
            optical_system = OpticalSystemBuilder().optical_system
        self._optical_system = optical_system

        # apply various initing configurations depending on what should be drawn
        self._clear_db()
        for item in contexts_request.contexts_list:
            itemInitGraphServiceStrategy: InitGraphServiceBaseStrategy = ContextRegistry().get_init_strategy(item)
            itemInitGraphServiceStrategy(contexts_request, self).init_graph_service()

    @property
    def optical_system(self):
        if not isinstance(self._optical_system, OpticalSystem):
            raise UnspecifiedFieldException(f'Service hasn''t instanced any optical system yet. '
                                            'Use fetch_optical_system')
        return self._optical_system

    def prepare_contexts(self, contexts_request: ContextRequest) -> Dict[str, Dict]:
        """
        Gets contexts request, checks all contexts are valid, and using appropriate strategies prepares all contexts to
        be given to controller
        """

        def _convert_context_format(context_list: List[Context]) -> Dict[str, Dict]:
            """Final preparation from list of contexts to kwarg dict, which is to be given to django render func"""
            converted_context = {}
            for current_context in context_list:
                converted_context[current_context.name] = current_context.value
            return converted_context

        self._check_context_registered(contexts_request)
        contexts = []
        for item in contexts_request.contexts_list:
            itemPrepareStrategy: PrepareContextBaseStrategy = ContextRegistry().get_prepare_strategy(item)
            prepared_context = itemPrepareStrategy().prepare(contexts_request, layer_points=self._graph_objects)
            contexts.append(prepared_context)

        merged_context: Dict[str, Dict] = _convert_context_format(contexts)
        return merged_context

    @staticmethod
    def _clear_db():
        django_models_to_clear = [BoundaryView, AxisView, BeamView]  # not SideView
        [cls.objects.all().delete() for cls in django_models_to_clear]

    def fetch_optical_system(self) -> OpticalSystem:
        """Retrieve somehow an optical system"""
        return self._create_hardcoded_optical_system()

    @staticmethod
    def _check_context_registered(contexts_request: ContextRequest):
        """Make sure all contexts are valid and regisetred in ContextRegistry cls"""
        registered_contexts = ContextRegistry().get_registered_contexts()
        if not all(cont in registered_contexts for cont in contexts_request.contexts_list):
            unknown_context = list(set(contexts_request.contexts_list).difference(registered_contexts))
            raise UnregisteredContextException(f'Requested context is unknown: {unknown_context}. '
                                               f'Registered contexts: {registered_contexts}')

    @staticmethod
    def _create_hardcoded_optical_system(name: str = 'Hardcoded Optical System') -> OpticalSystem:
        """Uses builder to creates an Optical System"""
        builder = OpticalSystemBuilder()

        first_comp_right_boundary = builder.create_layer(boundary=lambda y: 0 - y ** 2 / 400, side=Side.LEFT,
                                                         name='First-right bound')
        first_comp_mat = builder.create_material(name='Glass', transmittance=0.9, refractive_index=1.1)
        first_lense = builder.create_component(name='First lense',
                                               layers=[first_comp_right_boundary],
                                               material=first_comp_mat)

        sec_comp_left_boundary = builder.create_layer(boundary=lambda y: 100 + y ** 2 / 400, side=Side.RIGHT,
                                                      name='Second-left bound')
        sec_comp_sec_layer = builder.create_layer(boundary=lambda y: 200 + y ** 2 / 400, side=Side.LEFT,
                                                  name='Second-right bound')
        sec_comp_mat = builder.create_material(name='Glass', transmittance=0.9, refractive_index=1.2)
        second_lense = builder.create_component(name='Second lense',
                                                layers=[sec_comp_left_boundary, sec_comp_sec_layer],
                                                material=sec_comp_mat)

        thrd_comp_left_boundary = builder.create_layer(boundary=lambda y: 200 + y ** 2 / 400, side=Side.RIGHT,
                                                       name='Third-left bound')
        thrd_comp_right_boundary = builder.create_layer(boundary=lambda y: 300 + y ** 2 / 400, side=Side.LEFT,
                                                        name='Third-right bound')
        thrd_comp_mat = builder.create_material(name='Glass', transmittance=0.9, refractive_index=1.3)
        third_lense = builder.create_component(name='Third lense',
                                               layers=[thrd_comp_left_boundary, thrd_comp_right_boundary],
                                               material=thrd_comp_mat)

        fifth_comp_left_boundary = builder.create_layer(boundary=lambda y: 300 + y ** 2 / 400, side=Side.RIGHT,
                                                        name='Third-left bound')
        fifth_comp_right_boundary = builder.create_layer(boundary=lambda y: 400 + y ** 2 / 400, side=Side.LEFT,
                                                         name='Third-right bound')
        fifth_comp_mat = builder.create_material(name='Glass', transmittance=0.9, refractive_index=1.4)
        fifth_lense = builder.create_component(name='Third lense',
                                               layers=[fifth_comp_left_boundary, fifth_comp_right_boundary],
                                               material=fifth_comp_mat)

        fourth_comp_left_boundary = builder.create_layer(boundary=lambda y: 400 + y ** 2 / 400, side=Side.RIGHT,
                                                         name='Fourth-left bound')
        fourth_comp_mat = builder.create_material(name='Glass', transmittance=0.9, refractive_index=1.5)
        fourth_lense = builder.create_component(name='Fourth lense',
                                                layers=[fourth_comp_left_boundary, ],
                                                material=fourth_comp_mat)

        builder.reset()
        builder.set_optical_system_name(name=name)

        builder.add_components(components=(first_lense, second_lense, third_lense, fourth_lense, fifth_lense))

        initial_point = builder.create_point(x=0, y=50, z=0)

        resolution = 10  # rays per circle
        for theta in range(
                int(2 * pi * resolution + 2 * pi * 1 / resolution)):  # 2 * pi * 1/resolution addition to make compleete circle
            if True:  # 52 <= theta < 53 and
                v = builder.create_vector(initial_point=initial_point, lum=1, w_length=555, theta=theta / resolution,
                                          psi=0)
                builder.vectors.append(v)

        builder.trace_all()
        return builder.optical_system

    @staticmethod
    def _create_empty_optical_system(name: str = 'Empty Optical System') -> OpticalSystem:
        builder = OpticalSystemBuilder()
        builder.reset(name=name)
        return builder.optical_system

    def _push_optical_system_to_db(self):
        opt_sys = {'opt_sys_serial': dill.dumps(self.optical_system),
                   'name': self.optical_system.name}
        OpticalSystemView.objects.create(**opt_sys)

    def push_layers_to_db(self):
        """
        Fetches layers from optical system and pushes them to db.
        For each layers' curve calculates points it consists of and pushes them to db.
        """
        layers = self._fetch_optical_components_layers()
        for layer in layers:
            layer_view = self._append_layer_to_db(layer)
            points: List[Tuple[int, int]] = self._calculate_layer_points(layer)
            for p in points:
                self._graph_objects[id(layer)] = points
                # self._append_point_to_db(p, layer_view)

    def push_axes_to_db(self):
        """Calculates position of optical axes and pushes them to db"""
        axes = self._calculate_axes()
        self._append_axes_to_db(axes)

    def push_beams_to_db(self):
        beams = self._fetch_beams()
        self._append_beams_to_db(beams)

    def _fetch_optical_components_layers(self) -> List[Layer]:
        res = []
        try:
            for component in self.optical_system.components:
                [res.append(l) for l in component._layers]
        except UnspecifiedFieldException:
            pass  # nothing here. suppose self.optical_system is None
        return res

    @staticmethod
    def _append_layer_to_db(layer):
        boundary_serial = dill.dumps(layer)
        current_side = SideView.objects.get(side='Left') if layer.side == Side.LEFT else \
            SideView.objects.get(side='Right')
        layer_view = BoundaryView(name=layer.name, side=current_side, memory_id=id(layer),
                                  boundary_serial=boundary_serial)
        layer_view.save()
        return layer_view

    def _calculate_layer_points(self, layer: Layer) -> List[Tuple[int, int]]:
        """
        Retrieves points of a given layer from optical system.
        return: list of points, represented by tuples of (x0, y0)). Coordinates are in pixels
        """

        def _calculate_points_of_boundary_to_draw(boundary_func: Callable, step: int = 1) -> List[Tuple[int, int]]:
            """
            Gets a callable func of a boundary and calculates points
            which the boundary is consisted of with given step in pixels
            """
            assert isinstance(boundary_func, Callable), f'Wrong call: {boundary_func}'
            ys_in_mm = (el * self._scale for el in range(self._height_draw_ranges[0], self._height_draw_ranges[1], step))
            zs_in_mm = (boundary_func(y) for y in ys_in_mm)
            points = list(self._convert_opticalcoords_to_canvascoords(z, y)
                          for z, y in zip(zs_in_mm, ys_in_mm))
            return points

        boundary_points = _calculate_points_of_boundary_to_draw(layer.boundary)
        return boundary_points

    def _fetch_boundaries(self) -> List[Callable]:
        """Returns all boundaries of all layers in a whole optical system as a list of callables"""
        res = []
        for comp in self._optical_system.components:
            for l in comp._layers:
                res.append(l.boundary)
        return res

    def _calculate_axes(self) -> Tuple[Dict, Dict]:
        abscissa = {'direction': 'right',
                    'name': 'abscissa',
                    'x0': 0,
                    'y0': 0 + self._offset[1],
                    'x1': self._canvas_dimensions[0],
                    'y1': self._offset[1],
                    'memory_id': 0,
                    }
        abscissa['memory_id'] = id(abscissa)

        ordinate = {'direction': 'up',
                    'name': 'ordinate',
                    'x0': self._offset[0],
                    'y0': 0,
                    'x1': self._offset[0],
                    'y1': self._canvas_dimensions[1],
                    'memory_id': 0,
                    }
        ordinate['memory_id'] = id(ordinate)
        return abscissa, ordinate

    def _fetch_beams(self) -> Dict[int, List[Tuple[float, float]]]:
        """Fetches traced beams from optical system and prepares them to be forwarded to db"""

        def _get_point_from_vector(vector: Vector) -> Tuple[float, float]:
            return vector.initial_point.z, vector.initial_point.y

        def _get_canvas_vector_intersection(vector: Vector) -> Tuple[int, int]:
            """For a given vector and canvas returns intersection of this vector with canvas borders in pixels"""

            left_border_z_mm, upper_border_y_mm = self._convert_canvascoords_to_optical(0, 0)
            right_border_z_mm, bottom_border_y_mm = self._convert_canvascoords_to_optical(*self._canvas_dimensions)
            try:
                vector_equation: Callable = vector.get_line_equation()  # z=f(y)
                if 0 < vector.theta < pi:
                    upper_border_z_mm = vector_equation(upper_border_y_mm)
                    opt_z, opt_y = upper_border_z_mm, upper_border_y_mm
                elif pi < vector.theta < 2 * pi:
                    bottom_border_z_mm = vector_equation(bottom_border_y_mm)
                    opt_z, opt_y = bottom_border_z_mm, bottom_border_y_mm
                else:
                    raise AssertionError(f'Cannot find intersection with canvas for: {vector}')

            except ZeroDivisionError:
                if vector.theta == pi:
                    opt_z, opt_y = left_border_z_mm, vector.initial_point.y
                elif vector.theta == 0:
                    opt_z, opt_y = right_border_z_mm, vector.initial_point.y
                else:
                    raise AssertionError(f'Cannot find intersection with canvas for: {vector}')

            canvas_z, canvas_y = self._convert_opticalcoords_to_canvascoords(opt_z, opt_y)
            return canvas_z, canvas_y

        beams = dict()
        try:
            tmp_beams = self.optical_system.rays  # {beam_id: [Vectors]}
            for id, vector_list in tmp_beams.items():
                beams[id] = []
                for vector in vector_list:
                    optical_system_point = _get_point_from_vector(vector)  # Tuple[float, float]
                    canvas_point = self._convert_opticalcoords_to_canvascoords(optical_system_point[0],
                                                                               optical_system_point[1])
                    beams[id].append(canvas_point)
                last_point = _get_canvas_vector_intersection(vector_list[-1])
                beams[id].append(last_point)
        except UnspecifiedFieldException:
            pass  # suppsoe self.optical_system is None so return empty dict
        return beams

    def _append_point_to_db(self, point: Tuple[int, int], model_layer) -> None:
        """Gets a point (tuple of x0, y0) and an instance of a model of layer on boundary of which point is located.
        Checks and creates an object of such point
        """
        assert len(point) == 2, f'Wrong line format: {point}'
        assert all((isinstance(coord, int) for coord in point)), f'Coords of line must be integers, ' \
                                                                 f'but was given {[type(coord) for coord in point]}'
        BoundaryPoint.objects.create(x0=point[0], y0=point[1], line=model_layer)

    def _append_axes_to_db(self, axes: Tuple) -> None:
        for axis in axes:
            AxisView.objects.create(**axis)

    @staticmethod
    def _append_beams_to_db(beams: Dict[int, List[Tuple[float, float]]]) -> None:
        for beam_id, beam_points in beams.items():
            model_beam = BeamView(memory_id=beam_id)
            model_beam.save()
            for point in beam_points:
                VectorView.objects.create(beam=model_beam, x0=point[0], y0=point[1])


    def _convert_opticalcoords_to_canvascoords(self, opt_absciss: float, opt_ordinate: float) -> Tuple[int, int]:
        """ Maps optical coords in mm (opt_absciss, opt_ordinate) to a canvas coords in pix
            scale - in pixels per mm
            returns: tuple of canvas (abscissa, ordinate)
        """
        scale: float = self._scale
        absciss_offset: int = self._offset[0]
        ordinate_offset: int = self._offset[1]
        canvas_abscissa = int(opt_absciss * scale + absciss_offset)
        canvas_ordinate = int(ordinate_offset - opt_ordinate * scale)  # minus because of canvas ordinate directed down
        return canvas_abscissa, canvas_ordinate

    def _convert_canvascoords_to_optical(self, canvas_abscissa: int, canvas_ordinate: int) -> Tuple[float, float]:
        """ Returns real coordinates in mm
        scale - in pixels per mm
        """
        scale: float = self._scale
        absciss_offset: int = self._offset[0]
        ordinate_offset: int = self._offset[1]
        opt_absciss = (canvas_abscissa - absciss_offset) / scale
        opt_ordinate = (ordinate_offset - canvas_ordinate) / scale
        return opt_absciss, opt_ordinate

    @staticmethod
    def _transform_line_representation(x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int, float, int]:
        """
        Responsible for transformation of lines from two-points (x0, y0) (x1, y1) representation to offset-length-angle
        representation
        x0: int, y0: int, x1: int, y1: int - coords in pixels on a browser canvas
        return: transition_absciss, transition_ordinate - offset of a line, angle in radians CCW - positive, length - in
        pixels

        """
        assert x0 != x1 or y0 != y1, f'Line is singular: {(x0, y0), (x1, y1)}'
        transition_absciss = x0
        transition_ordinate = y0
        length = int(sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))  # TODO: check if float is ok for template
        try:
            angle = atan((y1 - y0) / (x1 - x0))
        except ZeroDivisionError:
            angle = pi / 2 if y1 > y0 else -pi / 2

        return transition_absciss, transition_ordinate, angle, length


class FormHandleBaseStrategy(ABC):
    """
    Abstract cls to perform different form handle strategies. Each strategy gets certain form,
    performs handling of certain data type and returns name of a context in ContextRequest which is to be given to graph
    service in order to prepare this context"""

    @property
    def optical_system_id(self):
        return self._optical_system_id

    @optical_system_id.setter
    def optical_system_id(self, opt_sys_id):
        if not isinstance(opt_sys_id, (int, type(None))):  # for 3.0<=python<=3.9
            raise TypeError(f'Wrong id for inner optical system in FormHandleService: {opt_sys_id}')
        opt_sys_id = (None if opt_sys_id is None else int(opt_sys_id))
        self._optical_system_id = opt_sys_id

    @abstractmethod
    def handle(self, form_instance: forms.Form) -> ContextRequest:
        ...


def fetch_optical_system_by_id(*, id: Optional[int]) -> IOpticalSystem:
    """
    Returns optical system fetching it from db using given id as pk.
    If pk is None or invalid returns an empty optical system
    """
    if id is None or not isinstance(id, int):
        warn(f'Fetching optical system is unsuccessful. Given type(id): {type(id)}. '
             f'An empty optical system has been created')
        return OpticalSystemBuilder().optical_system
    try:
        modelOpticalSystemView = OpticalSystemView.objects.get(pk=id)
    except OpticalSystemView.DoesNotExist:
        warn(f'Optical system with the given pk is not found. pk={id}. An empty optical system has been created')
        return OpticalSystemBuilder().optical_system

    optical_system = dill.loads(modelOpticalSystemView.opt_sys_serial)
    assert isinstance(optical_system, OpticalSystem), f'Fetched from DB: {optical_system}'
    return optical_system


class AddComponentFormHandleService(FormHandleBaseStrategy):
    """Responsible for handling django form AddComponentForm and forwarding it to domain model.
    An optical system pk should be give while instance the cls, in which component is going to be added.
    If opt sys is not given, the new one will be created"""

    def __init__(self, *, opt_sys_id: Optional[int] = None) -> None:
        self.optical_system_id = opt_sys_id
        optical_system: IOpticalSystem = fetch_optical_system_by_id(
            id=opt_sys_id)  # empty optical system if id is invalid
        self._builder: IOpticalSystemBuilder = OpticalSystemBuilder(optical_system=optical_system)

    def handle(self, form_instance: AddComponentForm):
        if not isinstance(form_instance, AddComponentForm):
            raise TypeError(f'Wrong type of argument for this type of handler.'
                            f'Should be AddComponentForm form, but was given {type(form_instance)}')
        if form_instance.is_valid():
            self._create_and_pull_new_component(form_instance.cleaned_data)

        return ContextRegistry().get_context_name('opt_sys_context')

    @property
    def builder(self) -> IOpticalSystemBuilder:
        """Cls uses OpticalSystemBuilder to handle with domain model"""
        if not isinstance(self._builder, IOpticalSystemBuilder):
            raise UnspecifiedFieldException(f'Optical system builder hasn''t been initialised properly. ')
        return self._builder

    def _create_and_pull_new_component(self, cleaned_data: Dict):
        """
        Creates a new optical component via cleaned data from form.
        After creating pushes the component to builder.optical_system and traces vectors
        After that updates optical system in db
        """
        try:
            new_component: OpticalComponent = self._compose_new_component(cleaned_data)
            self.builder.add_components(components=new_component)
        except WrongBoundaryEquationSyntaxError as e:
            warn(f'WrongBoundaryEquationSyntaxError is occurred. No component is added. {e.args}')
            pass  # if any equation is wrong just do nothing with optical system
        self.builder.trace_all()
        self._update_optical_system()

    def _compose_new_component(self, cleaned_data: Dict) -> OpticalComponent:
        """Gets data from form.cleaned_data and uses OpticalSystemBuilder to compose a new OpticalComponent object."""
        layers = [] # layers to build the component
        try:
            first_layer_side = self.builder.create_side(side=str(cleaned_data['first_layer_side']))
            first_layer_boundary: Callable = self.builder.create_boundary_callable(
                equation=cleaned_data['first_layer_equation'])
            first_new_layer = self.builder.create_layer(name=cleaned_data['first_layer_name'],
                                                        side=first_layer_side,
                                                        boundary=first_layer_boundary)
            layers.append(first_new_layer)
        except EmptyBoundaryEquationSyntaxError:
            pass  # if equation is empty just do nothing

        try:
            second_layer_side = self.builder.create_side(side=str(cleaned_data['second_layer_side']))
            second_layer_boundary: Callable = self.builder.create_boundary_callable(
                equation=cleaned_data['second_layer_equation'])
            second_new_layer = self.builder.create_layer(name=cleaned_data['second_layer_name'],
                                                         side=second_layer_side,
                                                         boundary=second_layer_boundary)
            layers.append(second_new_layer)
        except EmptyBoundaryEquationSyntaxError:
            pass  # if equation is empty just do nothing with optical system

        current_material = self.builder.create_material(name=cleaned_data['material_name'],
                                                        transmittance=cleaned_data['transmittance'],
                                                        refractive_index=cleaned_data['index'])
        new_component = self.builder.create_component(name=cleaned_data['component_name'],
                                                      layers=layers,
                                                      material=current_material)
        return new_component

    def _update_optical_system(self) -> None:
        """
        Gets OpticalSystemView object with self.optical_system_id as pk.
        If not exist in db, creates an empty model object
        Serializes the current optical system which is in self.builder.optical_system.
        Sets serialized optical system and it's name to that object and saves it.
        @return: None
        """
        try:
            current_opt_sys_model = OpticalSystemView.objects.get(pk=self.optical_system_id)
        except OpticalSystemView.DoesNotExist:
            current_opt_sys_model = OpticalSystemView()

        current_opt_sys_model.opt_sys_serial = dill.dumps(self.builder.optical_system)
        current_opt_sys_model.name = self.builder.optical_system.name
        current_opt_sys_model.save()


class ChooseOpticalSystemFormHandleService(FormHandleBaseStrategy):
    """Responsible for handling form of optical system choice"""

    def __init__(self, *, opt_sys_id: Optional[int] = None) -> None:
        self.optical_system_id = opt_sys_id
        optical_system = fetch_optical_system_by_id(id=opt_sys_id)
        self._builder: IOpticalSystemBuilder = OpticalSystemBuilder(optical_system=optical_system)

    @property
    def builder(self) -> IOpticalSystemBuilder:
        """Cls uses OpticalSystemBuilder to handle with domain model"""
        if not isinstance(self._builder, IOpticalSystemBuilder):
            raise UnspecifiedFieldException(f'Optical system builder hasn''t been initialised properly. ')
        return self._builder

    def handle(self, form_instance: ChooseOpticalSystemForm):
        formChooseOpticalSystem = form_instance
        if formChooseOpticalSystem.is_valid:
            modelOpticalSystemView = formChooseOpticalSystem.cleaned_data['optical_system']
            self.optical_system_id = modelOpticalSystemView.pk
            name = modelOpticalSystemView.name
            optical_system = dill.loads(modelOpticalSystemView.opt_sys_serial)
            self.builder.reset(optical_system=optical_system)
            self.builder.set_optical_system_name(name=name)
