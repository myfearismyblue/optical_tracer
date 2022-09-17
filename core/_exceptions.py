__all__ = ['VectorOutOfComponentException',
           'VectorNotOnBoundaryException',
           'NoLayersIntersectionException',
           'NoVectorIntersectionWarning',
           'TotalInnerReflectionException',
           'ComponentCollisionException',
           'ObjectKeyWordsMismatchException',
           'UnspecifiedFieldException', ]


class DomainBaseException(Exception):
    """Base class for optical domain exceptions"""


class VectorOutOfComponentException(DomainBaseException):
    """Raises then coords of a vector are out of optical component which it was given"""
    pass


class VectorNotOnBoundaryException(DomainBaseException):
    """Raises then vector is supposed to be on the boundary of layer, but it is not"""
    pass


class NoVectorIntersectionWarning(DomainBaseException):
    """Raises then vector doesn't intersect any surface"""
    pass


class NoLayersIntersectionException(DomainBaseException):
    """Raises then layers doesn't have any intersections"""
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
