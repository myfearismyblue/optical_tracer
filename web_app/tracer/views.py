from django.http import HttpResponse
from django.shortcuts import render

from .models import Line


def index(request):
    lines = Line.objects.all()
    return render(request, 'tracer/tracer.html', {'lines': lines})
