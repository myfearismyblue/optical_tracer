from typing import Dict

from django.shortcuts import render

from .forms import AddLayer
from .services import ContextRequest, GraphService


def index(request):

    graph_info = {'canvas_width': 1600,
                  'canvas_height': 1200,
                  'scale': 1,
                  }
    contexts_list = ['canvas_context', 'boundaries_context', 'beams_context', 'axis_context']
    contexts_request = ContextRequest(contexts_list=contexts_list, graph_info=graph_info)

    gr_service = GraphService(contexts_request=contexts_request)
    gr_service.make_initials()

    context: Dict = gr_service.prepare_contexts(contexts_request)


    if request.method == 'POST':
        form = AddLayer(request.POST)
        if form.is_valid():
            print(form.cleaned_data)
    else:
        form = AddLayer()

        context = {**context, 'form': form.as_p}

    return render(request, 'tracer/tracer.html',context)
