from typing import Dict

from django.db.models import QuerySet
from django.http import HttpResponse
from django.shortcuts import render

from .models import Boundary, Grapher, BoundaryPoint, Axis, Beam, BeamVector


def index(request):
    def _stringify_points_for_template(pts: QuerySet) -> str:
        """Prepares coords to be forwarded in a <svg> <polyline points="x0, y0 x1, y1 x2, y2..." >"""
        res = ''
        for p in pts:
            current_element = f'{p.x0}, {p.y0}'
            res = ' '.join((res, current_element))      # _space_ here to separate x1, y1_space_x2, y2
        return res

    def prepare_context_boundaries(lines_points_context: Dict[int, str] = dict()) -> Dict[int, str]:
        """Fetches all boundaries from models.Boundary. For each one fetches points to draw from model.BoundaryPoint.
        Stringifys them and appends to context to be forwarded to template"""
        for line in Boundary.objects.all():
            points = _stringify_points_for_template(BoundaryPoint.objects.filter(line=line.pk))
            lines_points_context[line.memory_id] = points
        return lines_points_context

    def prepare_context_axes(axis_points_context: Dict[int, str] = dict()) -> Dict[int, str]:
        """Fetches axes from models.Axis and prepares context to render"""
        for axis in Axis.objects.all():
            if axis.direction in ['down', 'right']:
                points = f'{axis.x0}, {axis.y0} {axis.x1}, {axis.y1}'
            elif axis.direction in ['up', 'left']:
                points = f'{axis.x1}, {axis.y1} {axis.x0}, {axis.y0}'
            else:
                raise ValueError(f'Wrong direction type in {axis}: {axis.direction}')
            axis_points_context[axis.memory_id] = points
        return axis_points_context

    def prepare_context_beams(beams_points_context: Dict[int, str] = dict()) -> Dict[int, str]:
        """
        Fetches beams from models.Beam. For each beam gets it's points from models.BeamVector.
        Stringifys them and appends to context to be forwarded to template
        """
        for beam in Beam.objects.all():
            points = _stringify_points_for_template(BeamVector.objects.filter(beam=beam.pk))
            beams_points_context[beam.memory_id] = points
        return beams_points_context


    Grapher.make_initials()
    lines_points_context = prepare_context_boundaries()
    axis_points_context = prepare_context_axes()
    beams_points_context = prepare_context_beams()
    print(beams_points_context)
    return render(request, 'tracer/tracer.html', {'lines_points': lines_points_context,
                                                  'axis_points': axis_points_context,
                                                  'beams_points': beams_points_context,
                                                  'canvas_width': Grapher.CANVAS_WIDTH,
                                                  'canvas_height': Grapher.CANVAS_HEIGHT})
