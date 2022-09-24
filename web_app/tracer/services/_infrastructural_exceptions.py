__all__ = ['UnregisteredContextException',
           'WrongBoundaryEquationSyntaxError',
           'EmptyBoundaryEquationSyntaxError']

class UserInfrastructuralBaseException(Exception):
    """The base for django-models exceptions"""
    ...


class UnregisteredContextException(UserInfrastructuralBaseException):
    """Raises when unknown context requested to be prepared"""
    ...


class WrongBoundaryEquationSyntaxError(UserInfrastructuralBaseException):
    """Raises when wrong boundary equation syntax is used while form's submitting"""
    ...


class EmptyBoundaryEquationSyntaxError(UserInfrastructuralBaseException):
    """Raises when empty boundary equation syntax is used while form's submitting"""
    ...



