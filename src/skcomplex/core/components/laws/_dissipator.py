"""Define dissipator laws, the Lindblad terms of a Markovian master equation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import ContinuousTime
from ._base import BaseLaw
from .utils._local_operators import has_operator


class DissipatorLaw(BaseLaw):
    """Lindblad term of a Markovian master equation.

    The term has a collapse operator and a rate. The master equation gains `rate * (L rho L^† - (L^† L rho + rho L^† L)
    / 2)`.

    Args:
        operator:
            The collapse operator `L`, an expression of local operators of the namespaces the law receives.

        rate:
            The non-negative rate, a number or a Sage expression of parameters and time.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, var
        >>> from skcomplex.core.components.laws import DissipatorLaw
        >>> from skcomplex.core.components.laws.utils import local_operator
        >>> minus = local_operator(SR.symbol('q'), SR.symbol('minus'))
        >>> DissipatorLaw(minus, 2 * var('gamma')).prepare().run({'gamma': 0.05}).value_
        (local_operator(q, minus), 0.1)
    """

    law_tag: ClassVar[str] = 'rate'

    def __init__(self, operator: object, rate: object = 1.0, *, label: str = '') -> None:
        super().__init__(label)
        self.operator = operator
        self.rate = rate

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states.

        The dissipator changes the operator state of the whole system.
        """
        return ()

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(ContinuousTime,))

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for a function as operator or rate, or a bad operator or rate.

        It raises it for no local operators, or a rate of another type.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        if any(callable(item) and not isinstance(item, Expression) for item in (self.operator, self.rate)):
            error_msg = (
                'Dissipators need symbolic operators and rates, so neither can be a function. Give the rate as '
                'a `series` of time, or use a `RateLaw` with a function on the states of a classical system.'
            )
            raise ValidationError(error_msg)
        if not has_operator(self.operator):
            error_msg = f'A dissipator takes an operator built from local operators, but got {self.operator!r}.'
            raise ValidationError(error_msg)
        try:
            rate = SR(self.rate)
        except (TypeError, ValueError):
            rate = None
        if rate is None or has_operator(rate):
            error_msg = f'The rate of a dissipator is a number or a symbolic expression, but got {self.rate!r}.'
            raise ValidationError(error_msg)

    def _run(self, inputs: object) -> None:
        """Set `value_` to the operator and the rate at given symbol values.

        The symbols are those they read.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate((self.operator, self.rate), inputs)
