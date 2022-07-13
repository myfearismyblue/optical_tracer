from typing import Dict

from django.db.models import QuerySet
from django.http import HttpResponse
from django.shortcuts import render

from .models import Boundary, Grapher, Point, Axis


def index(request):
    def _stringify_points_for_template(pts: QuerySet) -> str:
        """Prepares coords to be forwarded in a <svg> <polyline points="x0, y0 x1, y1 x2, y2..." >"""
        res = ''
        for p in pts:
            current_element = f'{p.x0}, {p.y0}'
            res = ' '.join((res, current_element))      # _space_ here to separate x1, y1_space_x2, y2
        return res

    def prepare_context_boundaries(lines_points_context: Dict[int, str] = dict()) -> Dict[int, str]:
        """Fetches all boundaries from models.Boundary. For each one fetches points to draw from model.Point. Stringifys
        them and appends to context to be forwarded to template"""
        for line in Boundary.objects.all():
            points = _stringify_points_for_template(Point.objects.filter(line=line.pk))
            lines_points_context[line.memory_id] = points
        return lines_points_context

    def prepare_context_axes(lines_points_context: Dict[int, str] = dict()) -> Dict[int, str]:
        """Fetches axes from models.Axis and prepares context to render"""
        for axis in Axis.objects.all():
            points = f'{axis.x0}, {axis.y0} {axis.x1}, {axis.y1}'
            lines_points_context[axis.memory_id] = points
        return lines_points_context
    Grapher.make_initials()
    lines_points_context = prepare_context_boundaries()
    lines_points_context = prepare_context_axes(lines_points_context)
    return render(request, 'tracer/tracer.html', {'lines_points': lines_points_context,
                                                  'canvas_width': Grapher.CANVAS_WIDTH,
                                                  'canvas_height': Grapher.CANVAS_HEIGHT})
