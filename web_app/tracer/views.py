from typing import Dict, Iterable

from django.db.models import QuerySet
from django.http import HttpResponse
from django.shortcuts import render

from .models import BoundaryView, GraphService, AxisView, BeamView, VectorView, ContextRequest


def index(request):


    graph_info = {'canvas_width': 1600,
                  'canvas_height': 1200,
                  'scale': 1,
                  }
    contexts_list = ['canvas_context', 'lines_points', 'axis_points', 'beams_points']
    contexts_request = ContextRequest(contexts_list=contexts_list, graph_info=graph_info)
    gr_service = GraphService(contexts_request=contexts_request)
    gr_service.make_initials()

    context = gr_service.prepare_contexts(contexts_request)


    return render(request, 'tracer/tracer.html', context)
