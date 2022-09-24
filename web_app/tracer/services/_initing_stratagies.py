__all__ = ['CanvasInitGraphServiceStrategy',
           'BoundariesInitGraphServiceStrategy',
           'AxisInitGraphServiceStrategy',
           'BeamsInitGraphServiceStrategy',
           'OpticalSystemInitGraphServiceStrategy']

from abc import ABC, abstractmethod

from ._context_requests import ContextRequest


class InitGraphServiceBaseStrategy(ABC):
    """The base class of initing graph service depending on different given contexts"""

    def __init__(self, contexts_request: ContextRequest, graph_service):
        self._contexts_request = contexts_request
        self._graph_service = graph_service

    @abstractmethod
    def init_graph_service(self):
        """ Do various stuff before preparing contexts"""
        ...


class CanvasInitGraphServiceStrategy(InitGraphServiceBaseStrategy):
    """
    Responsible for correct initing of graph service if 'canvas_context' is demanded.
    Occasionally it should be done anyway if we want something to be drawn on a canvas
    """

    def init_graph_service(self):
        contexts_request = self._contexts_request  # just make synonym
        self._graph_service._graph_objects = {}
        self._graph_service._canvas_dimensions = (contexts_request.graph_info['canvas_width'],
                                                  contexts_request.graph_info['canvas_height'])

        # offset of entire optical system relatively to (0, 0) canvas point which is upper-left corner
        self._graph_service._offset = self._graph_service._canvas_dimensions[0] // 3, \
                                      self._graph_service._canvas_dimensions[1] // 3
        self._graph_service._scale = contexts_request.graph_info['scale']

        # ranges in which components to be drawn relatively to OPTICAL_SYSTEM_OFFSET in pixels here
        # coordinates are upside-down because of reversion of vertical axis
        self._graph_service._height_draw_ranges = self._graph_service._offset[1] - \
                                                  self._graph_service._canvas_dimensions[1], \
                                                  self._graph_service._offset[1]
        self._graph_service._width_draw_ranges = -self._graph_service._offset[0], \
                                                 self._graph_service._canvas_dimensions[0] - \
                                                 self._graph_service._offset[0]


class BoundariesInitGraphServiceStrategy(InitGraphServiceBaseStrategy):
    """Responsible for correct initing of graph service if optical component should be drawn"""

    def init_graph_service(self):
        self._graph_service.push_layers_to_db()


class AxisInitGraphServiceStrategy(InitGraphServiceBaseStrategy):
    """Responsible for correct initing of graph service if axis should be drawn"""

    def init_graph_service(self):
        """Using inner properties of service calculates AxisView model and pushes them to db"""
        self._graph_service.push_axes_to_db()


class BeamsInitGraphServiceStrategy(InitGraphServiceBaseStrategy):
    """Responsible for correct initing of graph service if optical rays should be drawn"""

    def init_graph_service(self):
        self._graph_service.push_beams_to_db()


class OpticalSystemInitGraphServiceStrategy(InitGraphServiceBaseStrategy):

    def init_graph_service(self):
        pass
