from django.db.models import QuerySet
from django.http import HttpResponse
from django.shortcuts import render

from .models import Boundary, Grapher, Point


def index(request):
    def _stringify_points_for_template(pts: QuerySet) -> str:
        res = ''
        for p in pts:
            current_element = f'{p.x0}, {p.y0}'
            res = ' '.join((res, current_element))
        return res

    Grapher.make_initials()
    lines_points_context = dict()
    for line in Boundary.objects.all():
        points = _stringify_points_for_template(Point.objects.filter(line=line.pk))
        lines_points_context[line.pk] = points
    return render(request, 'tracer/tracer.html', {'lines_points': lines_points_context,
                                                  'canvas_width': Grapher.CANVAS_WIDTH,
                                                  'canvas_height': Grapher.CANVAS_HEIGHT})
