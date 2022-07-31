from typing import Dict,  List

from django.shortcuts import render

from .models import BoundaryView, GraphService, AxisView, BeamView, VectorView, ContextRequest, Context


def index(request):

    def convert_context_format(context_list: List[Context]) -> Dict[str, Dict]:
        """
        Reshapes context format from List[Context] to {Context1.name: Context1.value, Context2.name: Context2.value...}
        """
        converted_context = {}
        for current_context in context_list:
            converted_context[current_context.name] = current_context.value

        return converted_context


    graph_info = {'canvas_width': 1600,
                  'canvas_height': 1200,
                  'scale': 1,
                  }
    contexts_list = ['canvas_context', 'boundaries_context', 'beams_context', 'axis_context']
    contexts_request = ContextRequest(contexts_list=contexts_list, graph_info=graph_info)

    gr_service = GraphService(contexts_request=contexts_request)
    gr_service.make_initials()

    contexts: List[Context] = gr_service.prepare_contexts(contexts_request)     # Context.name, Context.value: Dict
    merged_context = convert_context_format(contexts)

    return render(request, 'tracer/tracer.html', merged_context)
