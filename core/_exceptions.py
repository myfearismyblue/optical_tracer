__all__ = ['VectorOutOfComponentException',
           'VectorNotOnBoundaryException',
           'NoLayersIntersectionException',
           'NoVectorIntersectionWarning',
           'TotalInnerReflectionException',
           'ComponentCollisionException',
           'ObjectKeyWordsMismatchException',
           'UnspecifiedFieldException',
           'NanPointException',]


class DomainBaseException(Exception):
    """Base class for optical domain exceptions"""


class NanPointException(DomainBaseException):
    """Raises when Nan occures while calculating with Point cls"""
    pass


class VectorOutOfComponentException(DomainBaseException):
    """Raises when coords of a vector are out of optical component which it was given"""
    pass


class VectorNotOnBoundaryException(DomainBaseException):
    """Raises when vector is supposed to be on the boundary of layer, but it is not"""
    pass


class NoVectorIntersectionWarning(DomainBaseException):
    """Raises when vector doesn't intersect any surface"""
    pass


class NoLayersIntersectionException(DomainBaseException):
    """Raises when layers doesn't have any intersections"""
    pass


class TotalInnerReflectionException(DomainBaseException):
    """Raises when refraction couldn't be provided"""
    pass


class ComponentCollisionException(DomainBaseException):
    """Raises when two components has intersection"""
    pass


class ObjectKeyWordsMismatchException(Warning):
    """Raises when __init__ gets unexpected **kwargs"""
    pass


class UnspecifiedFieldException(Exception):
    """Raises when object's field hasn't been set correctly"""
    pass
