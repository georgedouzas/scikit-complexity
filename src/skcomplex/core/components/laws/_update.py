"""Define update laws, the next values of states after one discrete time step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ...errors import ValidationError
from ...specs import RequirementSpec, TopologyRequestSpec
from ..spaces import DiscreteTime
from ..types import Target
from ._base import BaseLaw


class UpdateLaw(BaseLaw):
    """Next value of a state after one discrete time step.

    A state takes one update law, unless each of its update laws declares how it combines with the others. An
    `increment` law gives the state plus a change, and a law with a `source` changes only a state that holds the source
    value and keeps every other value. Several such laws of one state combine in one step by adding their changes,
    `x + sum_k (f_k - x)`, all computed from the states at the start of the step, so increments add and moves from
    distinct source values compose. Two laws with the same source, or a law that declares neither, are an error.

    With an `order`, the members of a group update one after the other, in increasing order of the value of `order` at
    the start of the step, ties in the order of the members, and each member reads the values the members before it
    set. Without one, every member updates at once from the states at the start of the step.

    With a `phase`, the law runs in that phase of the step, which a `PhasesScheduleSpec` of the discrete time orders.
    The phases run one after the other, each reading the states the phases before it set, and the laws of one phase
    update at once. A law without a phase runs in the default phase, first unless the schedule places it.

    Args:
        target:
            The symbol of a scalar state, the column of symbols of a vector state, or a `TopologyRequestSpec`, whose
            value is the probability that the request fires.

        value:
            A symbolic expression, a `Call`, or a callable `f(t, values)` that gives the next value.

        label:
            The name of the law, or the empty string for a law without one.

        increment:
            Whether the value is the state plus a change that adds to the changes of the other update laws of the
            state.

        source:
            The value the law moves the state from, keeping every other value, or `None`.

        order:
            A symbolic expression of the states, parameters, and draws of a member, whose values order the members
            that update one after the other, or `None` for members that update at once.

        phase:
            The name of the phase of the step the law runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The value is a NumPy array, or an increment law has a source.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import UpdateLaw
        >>> x = var('x')
        >>> UpdateLaw(x, 4 * x * (1 - x)).prepare().run({'x': 0.5}).value_
        1.0
        >>> UpdateLaw(x, x + 1, increment=True).increment
        True
        >>> UpdateLaw(x, x + 1, phase='imitate').phase
        'imitate'
    """

    law_tag: ClassVar[str] = 'update'

    def __init__(
        self,
        target: Target | TopologyRequestSpec,
        value: object,
        *,
        label: str = '',
        increment: bool = False,
        source: float | None = None,
        order: object = None,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.value = value
        self.increment = increment
        self.source = source
        self.order = order
        self.phase = phase

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the update sets."""
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for a value that is a NumPy array, or an increment with a source.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.

        Raises:
            ValidationError:
                The value is a NumPy array, or an increment law has a source.
        """
        super()._prepare(context)
        self._check_array('value')
        if self.increment and self.source is not None:
            error_msg = (
                f'The update law {self.label!r} is an increment with the source {self.source!r}. An increment changes '
                'every value, so give it no source.'
            )
            raise ValidationError(error_msg)

    def _run(self, inputs: object) -> None:
        """Set `value_` to the next value at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(self.value, inputs)
