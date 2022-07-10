from django.db.models import QuerySet
from django.http import HttpResponse
from django.shortcuts import render

from .models import Line, LineDBAppender, Point


def index(request):
    def _prepare_points_for_temaplate(pts: QuerySet) -> str:
        res = ''
        for p in pts:
            current_element = f'{p.x0}, {p.y0}'
            res = ' '.join((res, current_element))
        return res

    LineDBAppender.do_test()
    points = _prepare_points_for_temaplate(Point.objects.all())
    return render(request, 'tracer/tracer.html', {'polyline_points': points,
                                                  'canvas_width': LineDBAppender.CANVAS_WIDTH,
                                                  'canvas_height': LineDBAppender.CANVAS_HEIGHT})
