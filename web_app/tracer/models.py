from math import sqrt, atan, pi
from typing import Tuple

from django.db import models


class Line(models.Model):
    length = models.IntegerField(verbose_name='Длина')
    angle = models.FloatField(verbose_name='Угол')
    transition_absciss = models.IntegerField(verbose_name='Абсцисса смещения')
    transition_ordinate = models.IntegerField(verbose_name='Ордината смещения')

    def __str__(self):
        return f'Линия: длина {self.length}, поворот {self.angle}rad, ' \
               f'смещение {self.transition_absciss, self.transition_ordinate}'

    class Meta:
        verbose_name = 'Линия'
        verbose_name_plural = 'Линии'
        ordering = ['pk']


class LineDBAppender:
    CANVAS_WIDTH = 800
    CANVAS_HEIGHT = 600

    # offset of entire optical system relatively to (0, 0) canvas point which is upper-left corner
    OPTICAL_SYSTEM_OFFSET = (+1 * CANVAS_WIDTH // 3, +1 * CANVAS_HEIGHT // 3)  # in pixels here

    # ranges in which components to be drawn relatively to OPTICAL_SYSTEM_OFFSET in pixels here
    # coordinates are upside-down because of reversion of vertical axis
    BOUNDARY_DRAW_RANGES = (OPTICAL_SYSTEM_OFFSET[1] - CANVAS_HEIGHT, OPTICAL_SYSTEM_OFFSET[1])

    @staticmethod
    def do_test():
        l_template = LineDBAppender._transform_line_representation(10, 100, 10, 300)
        test_line = Line.objects.create(transition_absciss=l_template[0], transition_ordinate=l_template[1],
                 angle=l_template[2], length=l_template[3])
        print(f'{test_line} is created.')

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
        length = int(sqrt((x1 - x0)**2 + (y1 - y0)**2))             # TODO: check if float is ok for template
        try:
            angle = atan((y1 - y0) / (x1 - x0))
        except ZeroDivisionError:
            angle = pi / 2 if y1 > y0 else -pi /2

        return transition_absciss, transition_ordinate, angle, length


