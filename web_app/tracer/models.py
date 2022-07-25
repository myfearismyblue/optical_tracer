from dataclasses import dataclass
from math import sqrt, atan, pi
from typing import Tuple, Callable, List, Dict, Iterable

import os
import sys

sys.path.append(os.path.abspath('..'))  # magic to make optical_tracer to be found

from django.db import models
from optical_tracer import Side, Layer, Material, OpticalComponent, OpticalSystem, Vector, Point




@dataclass
class ContextRequest:
    """Interface for requesting different contexts for a view from GraphService"""
    contexts_list: List[str]
    graph_info: Dict


class BoundaryView(models.Model):
    name = models.CharField(max_length=50, verbose_name='Имя')
    side = models.CharField(max_length=10, verbose_name='Сторона')
    memory_id = models.BigIntegerField()

    def __str__(self):
        return f'Граница: {self.name}, сторона: {self.side}'

    class Meta:
        verbose_name = 'Граница'
        verbose_name_plural = 'Границы'
        ordering = ['pk']


class AxisView(models.Model):
    name = models.CharField(max_length=15, default=None)
    x0 = models.IntegerField(default=None)
    y0 = models.IntegerField(default=None)
    x1 = models.IntegerField(default=None)
    y1 = models.IntegerField(default=None)
    direction = models.CharField(max_length=10, default=None)
    memory_id = models.BigIntegerField()

    class Meta:
        verbose_name = 'Ось'
        verbose_name_plural = 'Оси'
        ordering = ['name']


class BeamView(models.Model):
    memory_id = models.IntegerField(default=None)


class VectorView(models.Model):
    x0 = models.IntegerField(default=None)
    y0 = models.IntegerField(default=None)
    beam = models.ForeignKey(BeamView, on_delete=models.CASCADE)


class GraphService:  # FIXME: looks like a godclass. split it with responsibilities
    CANVAS_WIDTH = 1600  # px
    CANVAS_HEIGHT = 1200  # px
    SCALE = 1  # mm/px
    OPTICAL_SYSTEM_OFFSET = (+1 * CANVAS_WIDTH // 3, +1 * CANVAS_HEIGHT // 3)  # in pixels here

    def __init__(self, contexts_request: ContextRequest):
        self._canvas_dimensions = contexts_request.graph_info['canvas_width'], contexts_request.graph_info['canvas_height']

        # offset of entire optical system relatively to (0, 0) canvas point which is upper-left corner
        self._offset = self._canvas_dimensions[0] // 3, self._canvas_dimensions[1] // 3
        self._scale = contexts_request.graph_info['scale']
        self._optical_system = None
        self._graph_objects = {}

        # ranges in which components to be drawn relatively to OPTICAL_SYSTEM_OFFSET in pixels here
        # coordinates are upside-down because of reversion of vertical axis
        self._height_draw_ranges = self._offset[1] - self._canvas_dimensions[1], self._offset[1]
        self._width_draw_ranges = -self._offset[0], self._canvas_dimensions[0] - self._offset[0]

    def prepare_contexts(self, contexts_request: ContextRequest) -> Dict[str, Dict]:  # {'context_name': {id: 'conext',
                                                                                      #                   id2: 'context2}
                                                                                      #    ...        }
        def _stringify_points_for_template(pts: Iterable) -> str:
            """Prepares coords to be forwarded in a <svg> <polyline points="x0, y0 x1, y1 x2, y2..." >"""
            res = ''
            for p in pts:
                if hasattr(p, 'x0'):  # TODO: Refactor this shit
                    current_element = f'{p.x0}, {p.y0}'
                else:
                    current_element = f'{p[0]}, {p[1]}'
                res = ' '.join((res, current_element))  # _space_ here to separate x1, y1_space_x2, y2
            return res

        def prepare_context_boundaries(lines_points_context: Dict[int, str] = dict()) -> Dict[int, str]:
            """Fetches all boundaries from models.BoundaryView. For each one fetches points to draw from model.BoundaryPoint.
            Stringifys them and appends to context to be forwarded to template"""
            for line in BoundaryView.objects.all():
                points = _stringify_points_for_template(self._graph_objects[line.memory_id])
                lines_points_context[line.memory_id] = points
            return lines_points_context

        def prepare_context_axes(axis_points_context: Dict[int, str] = dict()) -> Dict[int, str]:
            """Fetches axes from models.AxisView and prepares context to render"""
            for axis in AxisView.objects.all():
                if axis.direction in ['down', 'right']:
                    points = f'{axis.x0}, {axis.y0} {axis.x1}, {axis.y1}'
                elif axis.direction in ['up', 'left']:
                    points = f'{axis.x1}, {axis.y1} {axis.x0}, {axis.y0}'
                else:
                    raise ValueError(f'Wrong direction type in {axis}: {axis.direction}')
                axis_points_context[axis.memory_id] = points
            return axis_points_context

        def prepare_context_beams(beams_points_context: Dict[int, str] = dict()) -> Dict[int, str]:
            """
            Fetches beams from models.BeamView. For each beam gets it's points from models.VectorView.
            Stringifys them and appends to context to be forwarded to template
            """
            for beam in BeamView.objects.all():
                points = _stringify_points_for_template(VectorView.objects.filter(beam=beam.pk))
                beams_points_context[beam.memory_id] = points
            return beams_points_context

        for item in contexts_request.contexts_list:
            ...

        context = []
        return context

    def make_initials(self):
        """Forwarding all objects to Django """
        self._clear_db()
        self._optical_system = self.build_optical_system()
        self._push_layers_to_db()
        self._push_axes_to_db()
        self._push_beams_to_db()

    @staticmethod
    def _clear_db():
        BoundaryView.objects.all().delete()  # on_delete=models.CASCADE for models.BoundaryPoint
        AxisView.objects.all().delete()
        BeamView.objects.all().delete()

    @staticmethod
    def build_optical_system():
        """Creates an Optical System"""

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
            if True:  # 52 <= theta < 53 and
                v = Vector(initial_point=in_point, lum=1, w_length=555, theta=theta / resolution, psi=0)
                opt_sys.trace(vector=v)
        return opt_sys

    def _push_layers_to_db(self):
        """
        Fetches layers from optical system and pushes them to db.
        For each layers' curve calculates points it consists of and pushes them to db.
        """
        layers = self._fetch_optical_components_layers()
        for layer in layers:
            layer_view = self._append_layer_to_db(layer)
            points = self._calculate_layer_points(layer)
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

    def _append_layer_to_db(self, layer):
        layer_view = BoundaryView(name=layer.name, side=layer.side, memory_id=id(layer))
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
        Checks and creates an object
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
