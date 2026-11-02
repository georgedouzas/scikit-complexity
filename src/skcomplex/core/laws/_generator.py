"""Define generator laws, the Hermitian operators that add up to the generator of a linear evolution."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import ContinuousTime
from ._base import BaseLaw
from .utils import has_operator


class GeneratorLaw(BaseLaw):
    """Hermitian operator that adds to the generator `H` of the linear evolution `dx/dt = -i H x` on a tensor product.

    Args:
        operator:
            The Hermitian operator, an expression of local operators of the namespaces the law receives.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, var
        >>> from skcomplex.core.laws import GeneratorLaw
        >>> from skcomplex.core.laws.utils import local_operator
        >>> z = local_operator(SR.symbol('q'), SR.symbol('z'))
        >>> GeneratorLaw(var('h') * z).check().prepare().run({'h': 0.5}).value_
        0.5*local_operator(q, z)
    """

    def __init__(self, operator: object, *, label: str = '') -> None:
        super().__init__(label)
        self.operator = operator

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states, since the Hamiltonian changes the operator state of the whole system."""
        return ()

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(ContinuousTime,))

    def _check(self) -> None:
        """Raise an error for a function as the operator, or an operator without local operators."""
        if callable(self.operator) and not isinstance(self.operator, Expression):
            error_msg = (
                'Generators need symbolic operators, so a generator cannot be a function. Give the coefficient '
                'as a `series` of time, or use a `RateLaw` with a function on the states of a classical system.'
            )
            raise ValidationError(error_msg)
        if not has_operator(self.operator):
            error_msg = f'A generator takes an operator built from local operators, but got {self.operator!r}.'
            raise ValidationError(error_msg)

    def _run(self, inputs: object) -> None:
        """Set `value_` to the operator with its coefficients at given values of the symbols they read.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(self.operator, inputs)
