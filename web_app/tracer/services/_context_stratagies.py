__all__ = ['BoundariesPrepareContextStrategy',
           'AxisPrepareContextStrategy',
           'BeamsPrepareContextStrategy',
           'CanvasPrepareContextStrategy',
           'OpticalSystemPrepareContextStrategy']

from abc import ABC, abstractmethod
from typing import Iterable

from ..models import BoundaryView, AxisView, BeamView, VectorView
from ._context_requests import ContextRequest, Context


class PrepareContextBaseStrategy(ABC):
    """Base class for different strategies preparing various objects' context for template """

    @abstractmethod
    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        ...

    @staticmethod
    def _stringify_points_for_template(points: Iterable) -> str:
        """
        Prepares coords [(x0, y0), (y1, y1), ...] to be forwarded in a <svg> <polyline points="x0, y0 x1, y1 x2, y2...">
        """
        res = ''
        for p in points:
            if hasattr(p, 'x0'):  # TODO: Refactor this shit
                current_element = f'{p.x0}, {p.y0}'
            else:
                current_element = f'{p[0]}, {p[1]}'
            res = ' '.join((res, current_element))  # _space_ here to separate x1, y1_space_x2, y2
        return res


class BoundariesPrepareContextStrategy(PrepareContextBaseStrategy):
    """
    Describes the way in which template context for drawning OpticalComponent boundaries, material etc. should be created
    """
    __context_name = 'boundaries_context'

    def __init__(self):
        self.context = Context(name=self.__context_name, value={})

    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        """
        Fetches all boundaries from models.BoundaryView. For each one fetches points to draw from model.BoundaryPoint.
        Stringifys them and appends to context to be forwarded to template
        """

        for boundary in BoundaryView.objects.all():
            # here layer_points is the dict which is give in kwargs directly from the GraphService
            # TODO here is to consider behaviour of GraphService in cases where data is fetching from db or getting directly
            # from domain model. I suppose sometimes it is needed to retrieve data like layers, axis etc from db,
            # but sometimes is should be calculated in a runtime. Bad idea to hardcode 'layer_points' literal directly
            layer_points = kwargs['layer_points']
            points = self._stringify_points_for_template(layer_points[boundary.memory_id])
            self.context.value[boundary.memory_id] = points
        return self.context


class AxisPrepareContextStrategy(PrepareContextBaseStrategy):
    """Describes the way in which template context for drawning optical axis should be created"""
    __context_name = 'axis_context'

    def __init__(self):
        self.context = Context(name=self.__context_name, value={})

    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        for axis in AxisView.objects.all():
            if axis.direction in ['down', 'right']:
                points = f'{axis.x0}, {axis.y0} {axis.x1}, {axis.y1}'
            elif axis.direction in ['up', 'left']:
                points = f'{axis.x1}, {axis.y1} {axis.x0}, {axis.y0}'
            else:
                raise ValueError(f'Wrong direction type in {axis}: {axis.direction}')
            self.context.value[axis.memory_id] = points
        return self.context


class BeamsPrepareContextStrategy(PrepareContextBaseStrategy):
    """Describes the way in which template context for drawning rays should be created"""
    __context_name = 'beams_context'

    def __init__(self):
        self.context = Context(name=self.__context_name, value={})

    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        """
        Fetches beams from models.BeamView. For each beam gets it's points from models.VectorView.
        Stringifys them and appends to context to be forwarded to template
        """
        for beam in BeamView.objects.all():
            points = self._stringify_points_for_template(VectorView.objects.filter(beam=beam.pk))
            self.context.value[beam.memory_id] = points
        return self.context


class CanvasPrepareContextStrategy(PrepareContextBaseStrategy):
    __context_name = 'canvas_context'

    def __init__(self):
        self.context = Context(name=self.__context_name, value={})

    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        self.context.value = context_request.graph_info
        return self.context


class OpticalSystemPrepareContextStrategy(PrepareContextBaseStrategy):
    """"""  # FIXME: make some docstring after certain implementation will be clear
    __context_name = 'opt_sys_context'

    def __init__(self):
        self.context = Context(name=self.__context_name, value={})

    def prepare(self, context_request: ContextRequest, **kwargs) -> Context:
        raise NotImplementedError

