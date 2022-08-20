from typing import Dict, Callable

from django.shortcuts import render

from .forms import AddComponent
from .services import ContextRequest, GraphService, OpticalSystemBuilder, FormHandleService


def index(request):
    if request.method == 'POST':
        formAddComponent = AddComponent(request.POST)
        if formAddComponent.is_valid():
            form_handler = FormHandleService(optical_system=None)       # TODO: consider the way of fetching opt_sys
            print(form_handler.pull_new_component(formAddComponent.cleaned_data))
    else:
        formAddComponent = AddComponent()

    graph_info = {'canvas_width': 1600,
                  'canvas_height': 1200,
                  'scale': 1,
                  }
    contexts_list = ['canvas_context', 'boundaries_context', 'beams_context', 'axis_context']
    contexts_request = ContextRequest(contexts_list=contexts_list, graph_info=graph_info)

    gr_service = GraphService(contexts_request=contexts_request)
    gr_service.make_initials()

    context: Dict = gr_service.prepare_contexts(contexts_request)

    context = {**context, 'form': formAddComponent.as_p}

    return render(request, 'tracer/tracer.html', context)
