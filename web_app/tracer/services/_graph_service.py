__all__ = ['push_sides_to_db_if_not_exist',
           'GraphService']

from abc import ABC, abstractmethod
from math import pi, sqrt, atan
from typing import List, Dict, Callable, Tuple

import dill

from core import OpticalSystem, Side, UnspecifiedFieldException, Layer, Vector
from ._infrastructural_exceptions import UnregisteredContextException
from ._context_registry import ContextRegistry
from ._context_stratagies import PrepareContextBaseStrategy
from ._initing_stratagies import InitGraphServiceBaseStrategy
from ._optsys_builder import OpticalSystemBuilder
from ._context_requests import ContextRequest, Context
from ..models import SideView, OpticalSystemView, BoundaryView, BeamView, AxisView, VectorView


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
        contexts = []  # just a simple list, which afterwards will be squashed to dict
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

