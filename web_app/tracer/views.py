from typing import Dict, Callable

from django.shortcuts import render

from .forms import AddLayer
from .services import ContextRequest, GraphService, OpticalSystemBuilder


def index(request):

    if request.method == 'POST':
        formAddLayer = AddLayer(request.POST)
        if formAddLayer.is_valid():
            # print(formAddLayer.cleaned_data)
            builder = OpticalSystemBuilder()
            # builder.optical_system = fetch_optical_system()
            new_layer_data = formAddLayer.cleaned_data
            new_layer_side = builder.create_side(side=str(new_layer_data['side']))
            new_layer_boundary: Callable = builder.create_boundary_callable(equation=new_layer_data['equation'])   # FIXME: Make alot validations here
            new_layer = builder.create_layer(name=new_layer_data['name'],
                                             side=new_layer_side,
                                             boundary=new_layer_boundary)
            print(new_layer)

            # fetch current optical system
            # build a new layer from AddLayer form
            # append new layer to the optical system
            # do context prepare
    else:
        formAddLayer = AddLayer()

    graph_info = {'canvas_width': 1600,
                  'canvas_height': 1200,
                  'scale': 1,
                  }
    contexts_list = ['canvas_context', 'boundaries_context', 'beams_context', 'axis_context']
    contexts_request = ContextRequest(contexts_list=contexts_list, graph_info=graph_info)

    gr_service = GraphService(contexts_request=contexts_request)
    gr_service.make_initials()

    context: Dict = gr_service.prepare_contexts(contexts_request)

    context = {**context, 'form': formAddLayer.as_p}

    return render(request, 'tracer/tracer.html', context)
