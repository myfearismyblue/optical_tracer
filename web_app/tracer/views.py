from django.db.models import QuerySet
from django.http import HttpResponse
from django.shortcuts import render

from .models import Line, PointsDBAppender, Point


def index(request):
    def _stringify_points_for_template(pts: QuerySet) -> str:
        res = ''
        for p in pts:
            current_element = f'{p.x0}, {p.y0}'
            res = ' '.join((res, current_element))
        return res

    PointsDBAppender.do_test()
    lines_points_context = dict()
    for line in Line.objects.all():
        points = _stringify_points_for_template(Point.objects.filter(line=line.pk))
        lines_points_context[line.pk] = points
    return render(request, 'tracer/tracer.html', {'lines_points': lines_points_context,
                                                  'canvas_width': PointsDBAppender.CANVAS_WIDTH,
                                                  'canvas_height': PointsDBAppender.CANVAS_HEIGHT})
