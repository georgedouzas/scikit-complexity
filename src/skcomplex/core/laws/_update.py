"""Define update laws, the next values of states after one discrete time step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...specs import RequirementSpec, TopologyRequestSpec
from ..spaces import DiscreteTime
from ..types import Target
from ._base import BaseLaw


class UpdateLaw(BaseLaw):
    """Next value of a state after one discrete time step.

    Args:
        target:
            The symbol of a scalar state, the column of symbols of a vector state, or a `TopologyRequestSpec`, whose
            value is the probability that the request fires.

        value:
            A symbolic expression, a `Call`, or a callable `f(t, values)` that gives the next value.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import UpdateLaw
        >>> x = var('x')
        >>> UpdateLaw(x, 4 * x * (1 - x)).check().prepare().run({'x': 0.5}).value_
        1.0
    """

    def __init__(self, target: Target | TopologyRequestSpec, value: object, *, label: str = '') -> None:
        super().__init__(label)
        self.target = target
        self.value = value

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the update sets."""
        return (self.target,)

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _check(self) -> None:
        """Raise an error for a value that is a NumPy array."""
        self._check_array('value')

    def _run(self, inputs: object) -> None:
        """Set `value_` to the next value at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(self.value, inputs)
