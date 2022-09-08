from typing import Dict, Callable

from django import forms
from django.shortcuts import render

from .forms import AddComponentForm, ChooseOpticalSystemForm
from .services import ContextRequest, GraphService, FormHandleBaseStrategy, AddComponentFormHandleService, \
    ChooseOpticalSystemFormHandleService, push_sides_to_db_if_not_exist


def index(request):
    graph_info = {'canvas_width': 1600,
                  'canvas_height': 1200,
                  'scale': 1,
                  }
    contexts_list = ['canvas_context', 'axis_context']
    add_component_form: forms.Form = AddComponentForm(request.POST or None)
    choose_optical_system_form: forms.Form = ChooseOpticalSystemForm(request.POST or None)
    if request.method == 'POST':
        # FIXME: make a convenient way of choosing strategy of form handling
        if add_component_form.is_valid():
            # TODO: consider the way of fetching opt_sys
            form_handler: FormHandleBaseStrategy = AddComponentFormHandleService(optical_system=None)
            form_handler.handle(add_component_form)
        if choose_optical_system_form.is_valid():
            form_handler: FormHandleBaseStrategy = ChooseOpticalSystemFormHandleService()
            form_handler.handle(choose_optical_system_form)
        contexts_list.extend(['boundaries_context', 'beams_context'])
        contexts_request = ContextRequest(contexts_list=contexts_list, graph_info=graph_info)
        push_sides_to_db_if_not_exist()     # sides are needed in add optical component form
        gr_service = GraphService(contexts_request=contexts_request,
                                  optical_system=form_handler.builder.optical_system)
    else:
        contexts_request = ContextRequest(contexts_list=contexts_list, graph_info=graph_info)
        gr_service = GraphService(contexts_request=contexts_request)

    context: Dict = gr_service.prepare_contexts(contexts_request)
    context = {**context, 'formChooseOpticalSystem': choose_optical_system_form, 'addComponentForm': add_component_form}

    return render(request, 'tracer/tracer.html', context)
