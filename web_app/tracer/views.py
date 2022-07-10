from django.http import HttpResponse
from django.shortcuts import render

from .models import Line, LineDBAppender


def index(request):
    LineDBAppender.do_test()
    lines = Line.objects.all()
    return render(request, 'tracer/tracer.html', {'lines': lines,
                                                  'canvas_width': LineDBAppender.CANVAS_WIDTH,
                                                  'canvas_height': LineDBAppender.CANVAS_HEIGHT})
