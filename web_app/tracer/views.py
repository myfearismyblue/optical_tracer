from typing import Dict, Callable

from django.shortcuts import render

from .forms import AddComponent, ChooseOpticalSystem
from .services import ContextRequest, GraphService, FormHandleBaseStrategy, AddComponentFormHandleService, \
    ChooseOpticalSystemFormHandleService


def index(request):
    graph_info = {'canvas_width': 1600,
                  'canvas_height': 1200,
                  'scale': 1,
                  }
    contexts_list = ['canvas_context', 'boundaries_context', 'beams_context', 'axis_context']
    contexts_request = ContextRequest(contexts_list=contexts_list, graph_info=graph_info)
    formAddComponent = AddComponent(request.POST or None)
    formChooseOpticalSystem = ChooseOpticalSystem(request.POST or None)
    if request.method == 'POST':
        # FIXME: make a convenient way of choosing strategy of form handling
        if formAddComponent.is_valid():
            # TODO: consider the way of fetching opt_sys
            form_handler: FormHandleBaseStrategy = AddComponentFormHandleService(optical_system=None)
            form_handler.handle(formAddComponent)
        if formChooseOpticalSystem.is_valid():
            form_handler: FormHandleBaseStrategy = ChooseOpticalSystemFormHandleService()
            form_handler.handle(formChooseOpticalSystem)

        gr_service = GraphService(contexts_request=contexts_request,
                                  optical_system=form_handler.builder.optical_system)

        gr_service._push_sides_to_db()
        gr_service._push_layers_to_db()
        gr_service._push_beams_to_db()


    else:
        gr_service = GraphService(contexts_request=contexts_request)

              # FIXME: make strategy initials here
        gr_service._push_sides_to_db()

    context: Dict = gr_service.prepare_contexts(contexts_request)



    context = {**context, 'formChooseOpticalSystem': formChooseOpticalSystem, 'formAddComponent': formAddComponent}

    return render(request, 'tracer/tracer.html', context)
