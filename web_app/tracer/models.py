from math import sqrt, atan, pi
from typing import Tuple, Callable, List

from django.db import models

import os
import sys

sys.path.append(os.path.abspath('..'))

from optical_tracer import Side, Layer, Material, OpticalComponent, OpticalSystem


class Line(models.Model):
    x0 = models.IntegerField(verbose_name='x0')
    y0 = models.IntegerField(verbose_name='y0')
    x1 = models.IntegerField(verbose_name='x1')
    y1 = models.IntegerField(verbose_name='y1')

    def __str__(self):
        return f'Линия: Точка 1: {(self.x0, self.y0)}, Точка 2: {(self.x1, self.y1)}'

    class Meta:
        verbose_name = 'Линия'
        verbose_name_plural = 'Линии'
        ordering = ['pk']


class Point(models.Model):
    x0 = models.IntegerField(verbose_name='x0')
    y0 = models.IntegerField(verbose_name='y0')


    def __str__(self):
        return f'Точкa : {(self.x0, self.y0)}'



    class Meta:
        verbose_name = 'Точка'
        verbose_name_plural = 'Точки'
        ordering = ['pk']


class LineDBAppender:  # FIXME: looks like a godclass. split it with responsibilities
    CANVAS_WIDTH = 800
    CANVAS_HEIGHT = 600
    SCALE = 1  # mm/px
    # offset of entire optical system relatively to (0, 0) canvas point which is upper-left corner
    OPTICAL_SYSTEM_OFFSET = (+1 * CANVAS_WIDTH // 3, +1 * CANVAS_HEIGHT // 3)  # in pixels here

    # ranges in which components to be drawn relatively to OPTICAL_SYSTEM_OFFSET in pixels here
    # coordinates are upside-down because of reversion of vertical axis
    BOUNDARY_DRAW_RANGES = (OPTICAL_SYSTEM_OFFSET[1] - CANVAS_HEIGHT, OPTICAL_SYSTEM_OFFSET[1])

    def __new__(cls, opt_system):
        cls._optical_system = opt_system
        self = super().__new__(cls)
        return self

    @staticmethod
    def _init_optical_system():
        """Creates an Optical System which is composed of three parallel layers and five optical media"""

        def create_first_medium():
            first_left_bound = Layer(boundary=lambda y: 0 + y ** 2 / 300, side=Side.RIGHT, name='First-left bound')    #
            first_right_bound = Layer(boundary=lambda y: 10 + y ** 2 / 300, side=Side.LEFT, name='First-right bound')
            first_material = Material(name='Glass', transmittance=0.9, refractive_index=1.1)
            first_medium = OpticalComponent(name='First')
            first_medium.add_layer(layer=first_left_bound)
            first_medium.add_layer(layer=first_right_bound)
            first_medium.material = first_material
            return first_medium

        def create_second_medium():
            second_left_bound = Layer(boundary=lambda y: 10 + y ** 2 / 300, side=Side.RIGHT, name='Second-left bound')
            second_right_bound = Layer(boundary=lambda y: 20 + y ** 2 / 300, side=Side.LEFT, name='Second-right bound')
            second_material = Material(name='Glass', transmittance=0.9, refractive_index=1.2)
            second_medium = OpticalComponent(name='Second')
            second_medium.add_layer(layer=second_left_bound)
            second_medium.add_layer(layer=second_right_bound)
            second_medium.material = second_material
            return second_medium

        def create_third_medium():
            third_left_bound = Layer(boundary=lambda y: 20 + y ** 2 / 300, side=Side.RIGHT, name='Third-left bound')
            third_right_bound = Layer(boundary=lambda y: 30 + y ** 2 / 300, side=Side.LEFT, name='Third-right bound')
            third_material = Material(name='Glass', transmittance=0.9, refractive_index=1.3)
            third_medium = OpticalComponent(name='Third')
            third_medium.add_layer(layer=third_left_bound)
            third_medium.add_layer(layer=third_right_bound)
            third_medium.material = third_material
            return third_medium

        def create_fourth_medium():
            fourth_left_bound = Layer(boundary=lambda y: 30 + y ** 2 / 300, side=Side.RIGHT, name='Fourth-left bound')
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
        return opt_sys

    @classmethod
    def do_test(cls):
        """Temporary func for debugging """
        lineDBAppender = LineDBAppender(LineDBAppender._init_optical_system())
        points = cls.fetch_optical_components_points()
        for p in points:
            cls.append_point_to_db(p)

    @classmethod
    def append_point_to_db(cls, point: Tuple[int, int]) -> None:
        """Gets a point (tuple of x0, y0), checks and creates an object"""
        assert len(point) == 2, f'Wrong line format: {point}'
        assert all((isinstance(coord, int) for coord in point)), f'Coords of line must be integers, '\
                                                                 f'but was given {[type(coord) for coord in point]}'
        Point.objects.create(x0=point[0], y0=point[1])

    @classmethod
    def fetch_optical_components_points(cls) -> List[Tuple[int, int]]:
        """
        Retrieves points of components to draw from optical system.
        return: list of points, represented by tuples of (x0, y0)). Coordinates are in pixels
        """

        def _fetch_boundaries() -> List[Callable]:
            """Returns all boundaries of all layers in a whole optical system as a list of callables"""
            res = []
            for comp in cls._optical_system._components:
                for l in comp._layers:
                    res.append(l.boundary)
            return res

        def _calculate_lines_of_boundary_to_draw(boundary_func: Callable) -> List[Tuple[int, int]]:
            """Gets a callable func of a boundary and calculates lines which the boundary is consisted of"""
            assert isinstance(boundary_func, Callable), f'Wrong call: {boundary_func}'
            ys_in_mm = (el * cls.SCALE for el in range(cls.BOUNDARY_DRAW_RANGES[0], cls.BOUNDARY_DRAW_RANGES[1], 5))
            zs_in_mm = (boundary_func(y) for y in ys_in_mm)
            points = list(cls.convert_opticalcoords_to_canvascoords(z, y, scale=cls.SCALE,
                                                                    absciss_offset=cls.OPTICAL_SYSTEM_OFFSET[0],
                                                                    ordinate_offset=cls.OPTICAL_SYSTEM_OFFSET[1])
                          for z, y in zip(zs_in_mm, ys_in_mm))
            return points


        boundaries = _fetch_boundaries()
        all_lines_together = _calculate_lines_of_boundary_to_draw(boundaries[0])

        # for boundary in boundaries:
        #     current_lines = _calculate_lines_of_boundary_to_draw(boundary)
        #     all_lines_together.extend(current_lines)
        #     print(f'Lines added: {len(current_lines)}')
        return all_lines_together

    @classmethod
    def convert_opticalcoords_to_canvascoords(cls, opt_absciss: float, opt_ordinate: float, scale: float = SCALE,
                                              absciss_offset: int =OPTICAL_SYSTEM_OFFSET[0],
                                              ordinate_offset: int =OPTICAL_SYSTEM_OFFSET[1]) -> Tuple[int, int]:
        """ Maps optical coords in mm (opt_absciss, opt_ordinate) to a canvas coords in pix
            scale - in pixels per mm
            returns: tuple of canvas (abscissa, ordinate)
        """
        canvas_abscissa = int(opt_absciss * scale + absciss_offset)
        canvas_ordinate = int(ordinate_offset - opt_ordinate * scale)   # minus because of canvas ordinate directed down
        return canvas_abscissa, canvas_ordinate


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
