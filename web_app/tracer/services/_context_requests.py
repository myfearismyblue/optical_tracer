__all__ = ['ContextRequest',
           'Context']

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ContextRequest:
    """DTO for requesting different contexts for a view from GraphService"""
    contexts_list: List[str]  # list of certain contexts which are supposed to be drawn with GraphService
    graph_info: Dict  # common info for GraphService like dimensions of canvas etc. To be specified


@dataclass
class Context:
    """DTO as a response to be forwarded to view.py"""
    name: str  # context name. Specified in ContextRegistry
    value: Dict
