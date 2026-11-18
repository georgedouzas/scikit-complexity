"""Define the law that gives a scalar function of states to minimise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import ClassVar

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ..spaces import ContinuousTime, StaticTime
from ._base import BaseLaw
from ._requirement_spec import RequirementSpec


class ObjectiveLaw(BaseLaw):
    """A law that gives a scalar function of states, which a system minimises.

    The value `U(x)` is a symbolic expression, and the targets are the states the minimum is sought over. The
    objectives of several laws add up into one, and the constraint laws of the system hold at the minimum.

    With `controls`, the law chooses values over time instead, as optimal control does: it names the values of the
    elements that it steers, such as `ObjectiveLaw(s.plant.x**2 + s.plant.u**2, controls=[s.plant.u])`, and the
    run minimises the integral of the value over the run plus the terminal cost at its end, while the rate laws of the
    system move the states. The bounds of a control are the allowed values of its element, such as
    `u: Annotated[float, RealSet.closed(0, 1)] = 0.0`. A cost to maximise, such as a profit, is minimised with its sign
    changed.

    Unlike `ConstraintLaw`, which states a relation that holds, it gives a cost that the system makes least. Choose it
    when the answer is the best state rather than a path in time.

    It appears in many fields. A loaded structure rests at its least potential energy, a household allocates a budget
    to its greatest utility, a model takes the weights of its least loss, and a health authority times a vaccination
    campaign to the least cost of an epidemic.

    Read more in the user guide, under Laws.

    Args:
        value:
            The scalar symbolic expression to minimise.

        targets:
            The symbol of a state, or the symbols of the states, that the minimum is sought over, or `None` with
            controls.

        controls:
            The symbols of the values the law steers over a run, or `None` for a static minimum over `targets`.

        terminal:
            With controls, the cost at the end of the run, a symbolic expression or a number.

        label:
            The name of the law, or the empty string for a law without one.

    Attributes:
        value_:
            The value of the objective at the values of its last run.

    See Also:
        ConstraintLaw: A relation that holds at the minimum.
        LinearProgramLaw: A linear objective with linear constraints, solved as a linear program.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import ObjectiveLaw
        >>> x = var('x')
        >>> float(ObjectiveLaw((x - 2) ** 2, x).prepare().run({'x': 5.0}).value_)
        9.0
        >>> u = var('u')
        >>> ObjectiveLaw(x**2 + u**2, controls=[u]).prepare().targets_
        (u,)
    """

    law_tag: ClassVar[str] = 'objective'

    def __init__(
        self,
        value: object,
        targets: object = None,
        *,
        controls: Sequence[object] | None = None,
        terminal: object = 0,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.value = value
        self.targets = targets
        self.controls = controls
        self.terminal = terminal

    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the minimum is sought over, or the controls."""
        if self.controls is not None:
            return tuple(self.controls)
        if self.targets is None:
            return ()
        if isinstance(self.targets, Sequence) and not isinstance(self.targets, str):
            return tuple(self.targets)
        return (self.targets,)

    def _get_law_requirements(self) -> RequirementSpec:
        """Return the time the law needs, continuous with controls and static otherwise."""
        return RequirementSpec(time=(ContinuousTime,)) if self.controls is not None else self._get_requirements()

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: static time."""
        return RequirementSpec(time=(StaticTime,))

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for an objective that is no expression or varies no state.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.

        Raises:
            ValidationError:
                The value is a function, the law names no target or both targets and controls, or a terminal cost
                without controls.
        """
        super()._prepare(context)
        if callable(self.value) and not isinstance(self.value, Expression):
            error_msg = 'An objective is minimised with its exact gradient, so its value is a symbolic expression.'
            raise ValidationError(error_msg)
        if self.controls is not None and self.targets is not None:
            error_msg = (
                'An objective is minimised over `targets` or steers `controls` over a run, not both, such as '
                '`ObjectiveLaw(cost, controls=[s.plant.u])`.'
            )
            raise ValidationError(error_msg)
        if not self._get_targets():
            error_msg = (
                'An objective names the states it is minimised over in `targets`, or the values it steers in '
                '`controls`.'
            )
            raise ValidationError(error_msg)
        if self.controls is None and not bool(SR(self.terminal).is_trivial_zero()):
            error_msg = 'A terminal cost is the cost at the end of a run, so it needs `controls`.'
            raise ValidationError(error_msg)

    def _run(self, inputs: object) -> None:
        """Set `value_` to the objective at given symbol values.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(SR(self.value), inputs)
