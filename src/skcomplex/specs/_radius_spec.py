"""Define the radius spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec

REDUCTIONS = ('sum', 'mean')


@dataclass(frozen=True, eq=False)
class RadiusSpec(BaseSpec):
    """Term of the other agents within a radius of an agent, reduced by a sum or a mean, with a priority.

    Every other agent within `radius` gives the vector `function(agent, other, model)`, and the terms reduce by
    `reduce`. Among several radii, only the radii of the highest priority that hold another agent count.

    Args:
        radius:
            The radius, a number or the name of a parameter of the agent.

        function:
            The term of one other agent, a function of the agent, the other agent, and the model view that returns a
            vector.

        reduce:
            `'sum'` or `'mean'`.

        priority:
            The priority of the radius, higher first.

    Raises:
        ValidationError:
            The reduction is unknown, the function is not callable, or the radius is negative, raised by `prepare`.

    Examples:
        >>> from skcomplex.specs import RadiusSpec
        >>> RadiusSpec(2.0, lambda agent, other, model: (1.0, 0.0), reduce='mean').reduce
        'mean'
    """

    radius: float | str
    function: Callable[..., object]
    reduce: str = 'sum'
    priority: int = 0

    def _check(self) -> None:
        """Raise an error for an unknown reduction, a function that cannot be called, or a negative radius.

        Raises:
            ValidationError:
                The reduction is unknown, the function is not callable, or the radius is negative.
        """
        if self.reduce not in REDUCTIONS:
            error_msg = f'The reduction of a radius is {self.reduce!r}. Use one of {", ".join(REDUCTIONS)}.'
            raise ValidationError(error_msg)
        if not callable(self.function):
            error_msg = f'The function of a radius must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
        if not isinstance(self.radius, str) and not self.radius >= 0:
            error_msg = f'The radius is {self.radius!r}. Give a nonnegative number or the name of a parameter.'
            raise ValidationError(error_msg)
