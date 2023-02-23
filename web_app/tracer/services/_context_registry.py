__all__ = ['ContextRegistry']

from dataclasses import dataclass

from ._context_stratagies import (PrepareContextBaseStrategy,
                                  CanvasPrepareContextStrategy,
                                  BoundariesPrepareContextStrategy,
                                  AxisPrepareContextStrategy,
                                  BeamsPrepareContextStrategy,
                                  OpticalSystemPrepareContextStrategy)
from ._initing_stratagies import (InitGraphServiceBaseStrategy,
                                  CanvasInitGraphServiceStrategy,
                                  BoundariesInitGraphServiceStrategy,
                                  AxisInitGraphServiceStrategy,
                                  BeamsInitGraphServiceStrategy,
                                  OpticalSystemInitGraphServiceStrategy,
                                  )
from ._infrastructural_exceptions import UnregisteredContextException


@dataclass
class ContextRegistry:
    # FIXME: link this with ContextRequest
    __registered_contexts = {'canvas_context': {'prepare_context': CanvasPrepareContextStrategy,
                                                'init_graph_service': CanvasInitGraphServiceStrategy},
                             'boundaries_context': {'prepare_context': BoundariesPrepareContextStrategy,
                                                    'init_graph_service': BoundariesInitGraphServiceStrategy},
                             'axis_context': {'prepare_context': AxisPrepareContextStrategy,
                                              'init_graph_service': AxisInitGraphServiceStrategy},
                             'beams_context': {'prepare_context': BeamsPrepareContextStrategy,
                                               'init_graph_service': BeamsInitGraphServiceStrategy},
                             'opt_sys_context': {'prepare_context': OpticalSystemPrepareContextStrategy,
                                                 'init_graph_service': OpticalSystemInitGraphServiceStrategy}, }

    def get_prepare_strategy(self, context_name: str) -> PrepareContextBaseStrategy:
        if str(context_name) not in self.__registered_contexts:
            raise UnregisteredContextException(f'Requested context is unknown: {context_name}. '
                                               f'Registered contexts: {self.__registered_contexts}')
        return self.__registered_contexts[context_name]['prepare_context']

    def get_init_strategy(self, context_name: str) -> InitGraphServiceBaseStrategy:
        if str(context_name) not in self.__registered_contexts:
            raise UnregisteredContextException(f'Requested context is unknown: {context_name}. '
                                               f'Registered contexts: {self.__registered_contexts}')
        return self.__registered_contexts[context_name]['init_graph_service']

    def get_registered_contexts(self):
        return tuple(self.__registered_contexts.keys())

    def get_context_name(self, context_name: str):
        """Returns a name of context if context is registered"""
        if str(context_name) not in self.__registered_contexts:
            raise UnregisteredContextException(f'Requested context is unknown: {context_name}. '
                                               f'Registered contexts: {self.__registered_contexts}')
        return str(context_name)

