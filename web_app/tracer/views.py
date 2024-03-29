from typing import Dict,  Optional

from django import forms
from django.shortcuts import render

from .forms import AddComponentForm, ChooseOpticalSystemForm
from .services import (AddComponentFormHandleService,
                                                    ChooseOpticalSystemFormHandleService,
                                                    ContextRequest,
                                                    GraphService,
                                                    FormHandleBaseStrategy,
                                                    push_sides_to_db_if_not_exist,
                                                    fetch_optical_system_by_id,)


def index(request):
    # FIXME: suppose there is a better way of such handling
    try:        # check out if a user had previously chosen or built opt sys
        current_opt_sys_id: Optional[int] = int(request.COOKIES.get('current_opt_sys_id'))
    except ValueError:
        current_opt_sys_id = None

    if request.method == 'POST':
        add_component_form: forms.Form = AddComponentForm(request.POST)
        choose_optical_system_form: forms.Form = ChooseOpticalSystemForm(request.POST)

        # FIXME: make a convenient way of choosing strategy of form handling
        if add_component_form.is_valid():
            form_handler: FormHandleBaseStrategy = AddComponentFormHandleService(opt_sys_id=current_opt_sys_id)
            form_handler.handle(add_component_form)
            # if this form is submitted, let instance next one in a correct way
            choose_optical_system_form: forms.Form = ChooseOpticalSystemForm()
        if choose_optical_system_form.is_valid():
            form_handler: FormHandleBaseStrategy = ChooseOpticalSystemFormHandleService(opt_sys_id=current_opt_sys_id)
            form_handler.handle(choose_optical_system_form)
            # if this form is submitted, let instance previous one in a correct way
            add_component_form: forms.Form = AddComponentForm()
        push_sides_to_db_if_not_exist()     # sides are needed in add optical component form
        current_optical_system = form_handler.builder.optical_system
        current_opt_sys_id = form_handler.optical_system_id
    else:
        add_component_form: forms.Form = AddComponentForm()
        choose_optical_system_form: forms.Form = ChooseOpticalSystemForm()
        current_optical_system = fetch_optical_system_by_id(id=current_opt_sys_id)  # None if id is invalid or None

    graph_info = {'canvas_width': 1600,
                  'canvas_height': 1200,
                  'scale': 1,
                  }
    contexts_list = ['canvas_context', 'axis_context', 'boundaries_context', 'beams_context']
    contexts_request = ContextRequest(contexts_list=contexts_list, graph_info=graph_info)
    gr_service = GraphService(contexts_request=contexts_request,
                              optical_system=current_optical_system)

    context: Dict = gr_service.prepare_contexts(contexts_request)
    context = {**context,
               'choose_optical_system_form': choose_optical_system_form,
               'add_component_form': add_component_form}
    response = render(request, 'tracer/tracer.html', context)
    response.set_cookie('current_opt_sys_id', current_opt_sys_id)
    return response

