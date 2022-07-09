from django.http import HttpResponse
from django.shortcuts import render

from .models import Line, LineDBAppender


def index(request):
    lines = Line.objects.all()
    LineDBAppender.do_test()
    return render(request, 'tracer/tracer.html', {'lines': lines})
