"""Share the small helpers of the package."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from sklearn.base import clone

from ..components.systems import BaseSystem
from ..errors import ExperimentError, ValidationError
from .types import Source


def build_system(source: Source, arguments: Mapping[str, object]) -> BaseSystem:
    """Return the system of a source with model parameters set.

    It is a copy of a system, or what a factory builds.

    Args:
        source:
            The system, whose copy takes the arguments through `set_params`, or the factory, which takes them as
            keyword arguments.

        arguments:
            The model parameters by name.

    Returns:
        system:
            The system with the model parameters.

    Raises:
        ValidationError:
            A model parameter is not one of the system.

        ExperimentError:
            The factory returns no system.
    """
    if isinstance(source, BaseSystem):
        unknown = sorted(set(arguments) - set(source.get_params(deep=False)))
        if unknown:
            names = ', '.join(sorted(source.get_params(deep=False)))
            error_msg = (
                f'{", ".join(unknown)} is not a model parameter of {type(source).__name__}. Its model parameters are '
                f'{names}, and dotted paths such as `growth.r` name the states and parameters of its components.'
            )
            raise ValidationError(error_msg)
        return clone(source).set_params(**arguments)
    system = source(**arguments)
    if not isinstance(system, BaseSystem):
        error_msg = f'The factory of an experiment returns a system, not {type(system).__name__}.'
        raise ExperimentError(error_msg)
    return system
