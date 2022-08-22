from typing import Dict, Callable

from django.shortcuts import render

from .forms import AddComponent, ChooseOpticalSystem
from .services import ContextRequest, GraphService, FormHandleBaseStrategy, AddComponentFormHandleService, \
    ChooseOpticalSystemFormHandleService


def index(request):
    contexts_list = ['canvas_context', 'boundaries_context', 'beams_context', 'axis_context']
    formAddComponent = AddComponent(request.POST or None)
    formChooseOpticalSystem = ChooseOpticalSystem(request.POST or None)
    if request.method == 'POST':
        # FIXME: make a convenient way of choosing strategy of form handling
        if formAddComponent.is_valid():
            # TODO: consider the way of fetching opt_sys
            form_handler: FormHandleBaseStrategy = AddComponentFormHandleService(optical_system = None)
            form_handler.handle(formAddComponent)
        if formChooseOpticalSystem.is_valid():
            form_handler: FormHandleBaseStrategy = ChooseOpticalSystemFormHandleService()
            form_handler.handle(formChooseOpticalSystem)


    graph_info = {'canvas_width': 1600,
                  'canvas_height': 1200,
                  'scale': 1,
                  }

    contexts_request = ContextRequest(contexts_list=contexts_list, graph_info=graph_info)

    gr_service = GraphService(contexts_request=contexts_request)
    gr_service.make_initials()

    context: Dict = gr_service.prepare_contexts(contexts_request)

    context = {**context, 'formChooseOpticalSystem': formChooseOpticalSystem, 'formAddComponent': formAddComponent}

    return render(request, 'tracer/tracer.html', context)
