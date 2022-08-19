from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import sqrt, atan, pi
from typing import Tuple, Callable, List, Dict, Iterable, Union

import dill

from optical_tracer import Side, Layer, Material, OpticalComponent, OpticalSystem, Vector, Point, \
    UnspecifiedFieldException, OPT_SYS_DIMENSIONS
from .models import AxisView, BoundaryView, BeamView, VectorView, SideView, OpticalSystemView


class UserInfrastructuralBaseException(Exception):
    """The base for django-models exceptions"""
    ...


class UnregisteredContextException(UserInfrastructuralBaseException):
    """Raises when unknown context requested to be prepared"""
    ...


class IOpticalSystemBuilder(ABC):
    """Interface for any OptSys builder"""

    @property
    @abstractmethod
    def optical_system(self):
        ...

    @property
    @abstractmethod
    def vectors(self):
        """Initial vectors which is to be traced with self.trace_all()"""
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

    def reset(self, *, name='Unnamed OptSys', default_medium: Material = OpticalSystem.DEFAULT_CLS_MEDIUM):
        if not isinstance(default_medium, Material):
            raise UnspecifiedFieldException(f'Wrong argument type. '
                                            f'Supposed to be Material, but was given: {type(default_medium)}')
        new_opt_sys = OpticalSystem(name=name, default_medium=default_medium)
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

    @staticmethod
    def create_side(*, side: str) -> Side:
        """Returns Side object depending on the give string Left or Right"""
        return Side.from_str(side)

    @staticmethod
    def create_boundary_callable(*, equation: str) -> Callable:
        """Gets input as string, validates it and returns callable object"""
        def _validate(equation):
            # FIXME: !!!!
            return True

        if _validate(equation):
            return lambda y: eval(equation)

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
    name: str   # context name. Specified in ContextRegistry
    value: Dict


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


class FormsPrepareContextStrategy(PrepareContextBaseStrategy):
    """Make preparations of forms' content after submition"""
    pass


@dataclass
class ContextRegistry:
    # FIXME: link this with ContextRequest
    __registered_contexts = {'canvas_context': CanvasPrepareContextStrategy,
                             'boundaries_context': BoundariesPrepareContextStrategy,
                             'axis_context': AxisPrepareContextStrategy,
                             'beams_context': BeamsPrepareContextStrategy}

    def get_prepare_strategy(self, context_name: str) -> PrepareContextBaseStrategy:
        if context_name not in self.__registered_contexts:
            raise UnregisteredContextException(f'Requested context is unknown: {context_name}. '
                                              f'Registered contexts: {self.__registered_contexts}')
        return self.__registered_contexts[context_name]

    def get_registered_contexts(self):
        return tuple(self.__registered_contexts.keys())


class IGraphService(ABC):
    @abstractmethod
    def fetch_optical_system(self):  # TODO: add contract here
        """Initing concrete optical system"""
        ...

    @abstractmethod
    def prepare_contexts(self, contexts_request: ContextRequest) -> List[Context]:
        """Using optical system make some contexts to be thrown in template"""
        ...


class GraphService(IGraphService):  # FIXME: looks like a godclass. split it with responsibilities
    CANVAS_WIDTH = 1600  # px
    CANVAS_HEIGHT = 1200  # px
    SCALE = 1  # mm/px
    OPTICAL_SYSTEM_OFFSET = (+1 * CANVAS_WIDTH // 3, +1 * CANVAS_HEIGHT // 3)  # in pixels here

    def __init__(self, contexts_request: ContextRequest):
        self._canvas_dimensions = contexts_request.graph_info['canvas_width'], contexts_request.graph_info[
            'canvas_height']

        # offset of entire optical system relatively to (0, 0) canvas point which is upper-left corner
        self._offset = self._canvas_dimensions[0] // 3, self._canvas_dimensions[1] // 3
        self._scale = contexts_request.graph_info['scale']
        self._optical_system = None
        self._graph_objects = {}

        # ranges in which components to be drawn relatively to OPTICAL_SYSTEM_OFFSET in pixels here
        # coordinates are upside-down because of reversion of vertical axis
        self._height_draw_ranges = self._offset[1] - self._canvas_dimensions[1], self._offset[1]
        self._width_draw_ranges = -self._offset[0], self._canvas_dimensions[0] - self._offset[0]

    @property
    def optical_system(self):
        if not isinstance(self._optical_system, OpticalSystem):
            raise UnspecifiedFieldException(f'Service hasn''t instanced any optical system yet. '
                                            'Use fetch_optical_system')
        return self._optical_system

    def prepare_contexts(self, contexts_request: ContextRequest) -> Dict[str, Dict]:
        """
        Reshapes context format from List[Context] to {Context1.name: Context1.value, Context2.name: Context2.value...}
        """
        def _check_context_registered(contexts_request: ContextRequest):
            """Make sure all contexts are valid and regisetred in ContextRegistry cls"""
            registered_contexts = ContextRegistry().get_registered_contexts()
            if not all(cont in registered_contexts for cont in contexts_request.contexts_list):
                unknown_context = list(set(contexts_request.contexts_list).difference(registered_contexts))
                raise UnregisteredContextException(f'Requested context is unknown: {unknown_context}. '
                                                  f'Registered contexts: {registered_contexts}')

        def _convert_context_format(context_list: List[Context]) -> Dict[str, Dict]:
            """Final preparation from list of contexts to kwarg dict, which is to be given to django render func"""
            converted_context = {}
            for current_context in context_list:
                converted_context[current_context.name] = current_context.value
            return converted_context

        _check_context_registered(contexts_request)
        contexts = []
        for item in contexts_request.contexts_list:
            itemPrepareStrategy: PrepareContextBaseStrategy = ContextRegistry().get_prepare_strategy(item)
            cont = itemPrepareStrategy().prepare(contexts_request, layer_points=self._graph_objects)
            contexts.append(cont)

        merged_context = _convert_context_format(contexts)
        return merged_context

    def make_initials(self):
        """Forwarding all objects to Django """
        self._clear_db()
        self._optical_system = self.fetch_optical_system(name='Hardcoded OptSys')
        # temporary dont do this because this responsibility
        # i'm going to replace to controller may be
        self._push_sides_to_db()
        self._push_layers_to_db()
        self._push_axes_to_db()
        self._push_beams_to_db()

    @staticmethod
    def _clear_db():
        BoundaryView.objects.all().delete()  # on_delete=models.CASCADE for models.BoundaryPoint
        AxisView.objects.all().delete()
        BeamView.objects.all().delete()

    @staticmethod
    def fetch_optical_system(name):
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

        builder.reset(name=name)

        builder.add_components(components=(first_lense, second_lense, third_lense, fourth_lense, fifth_lense))

        initial_point = builder.create_point(x=0, y=50, z=0)

        resolution = 10  # vectors per circle
        for theta in range(
                int(2 * pi * resolution + 2 * pi * 1 / resolution)):  # 2 * pi * 1/resolution addition to make compleete circle
            if True:  # 52 <= theta < 53 and
                v = builder.create_vector(initial_point=initial_point, lum=1, w_length=555, theta=theta / resolution,
                                          psi=0)
                builder.vectors.append(v)

        builder.trace_all()

        return builder.optical_system

    def _push_optical_system_to_db(self):
        opt_sys_serial = dill.dumps(self.optical_system)
        OpticalSystemView.objects.create()


    def _push_sides_to_db(self):
        """Sets boundary sides - left and right - to db"""
        SideView.objects.create(side='Left')
        SideView.objects.create(side='Right')

    def _push_layers_to_db(self):
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

    def _push_axes_to_db(self):
        """Calculates position of optical axes and pushes them to db"""
        axes = self._calculate_axes()
        self._append_axes_to_db(axes)

    def _push_beams_to_db(self):
        beams = self._fetch_beams()
        self._append_beams_to_db(beams)

    def _fetch_optical_components_layers(self) -> List[Layer]:
        res = []
        for component in self._optical_system.components:
            [res.append(l) for l in component._layers]
        return res

    @staticmethod
    def _append_layer_to_db(layer):
        boundary_serial = dill.dumps(layer)
        current_side = SideView.objects.filter(side='Left')[0] if layer.side==Side.LEFT else SideView.objects.filter(side='Right')[0]
        layer_view = BoundaryView(name=layer.name, side=current_side, memory_id=id(layer),boundary_serial=boundary_serial)
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
            ys_in_mm = (el * self.SCALE for el in range(self._height_draw_ranges[0], self._height_draw_ranges[1], step))
            zs_in_mm = (boundary_func(y) for y in ys_in_mm)
            points = list(self._convert_opticalcoords_to_canvascoords(z, y, scale=self.SCALE,
                                                                      absciss_offset=self.OPTICAL_SYSTEM_OFFSET[0],
                                                                      ordinate_offset=self.OPTICAL_SYSTEM_OFFSET[1])
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
                    'x1': self.CANVAS_WIDTH,
                    'y1': self._offset[1],
                    'memory_id': 0,
                    }
        abscissa['memory_id'] = id(abscissa)

        ordinate = {'direction': 'up',
                    'name': 'ordinate',
                    'x0': self._offset[0],
                    'y0': 0,
                    'x1': self._offset[0],
                    'y1': self.CANVAS_HEIGHT,
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

        tmp_beams = self._optical_system.vectors  # {beam_id: [Vectors]}
        beams = dict()
        for id, vector_list in tmp_beams.items():
            beams[id] = []
            for vector in vector_list:
                optical_system_point = _get_point_from_vector(vector)  # Tuple[float, float]
                canvas_point = self._convert_opticalcoords_to_canvascoords(optical_system_point[0],
                                                                           optical_system_point[1])
                beams[id].append(canvas_point)
            last_point = _get_canvas_vector_intersection(vector_list[-1])
            beams[id].append(last_point)
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

    @staticmethod
    def _convert_opticalcoords_to_canvascoords(opt_absciss: float, opt_ordinate: float, scale: float = SCALE,
                                               absciss_offset: int = OPTICAL_SYSTEM_OFFSET[0],
                                               ordinate_offset: int = OPTICAL_SYSTEM_OFFSET[1]) -> Tuple[int, int]:
        """ Maps optical coords in mm (opt_absciss, opt_ordinate) to a canvas coords in pix
            scale - in pixels per mm
            returns: tuple of canvas (abscissa, ordinate)
        """
        canvas_abscissa = int(opt_absciss * scale + absciss_offset)
        canvas_ordinate = int(ordinate_offset - opt_ordinate * scale)  # minus because of canvas ordinate directed down
        return canvas_abscissa, canvas_ordinate

    @staticmethod
    def _convert_canvascoords_to_optical(canvas_abscissa: int, canvas_ordinate: int, *, scale: float = SCALE,
                                         absciss_offset: int = OPTICAL_SYSTEM_OFFSET[0],
                                         ordinate_offset: int = OPTICAL_SYSTEM_OFFSET[1]) -> Tuple[float, float]:
        """ Returns real coordinates in mm
        scale - in pixels per mm
        """
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
