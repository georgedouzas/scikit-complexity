"""Define the law that gives a scalar function of states to minimise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ..spaces._continuous_time import _ContinuousTime
from ..spaces._static_time import _StaticTime
from ._base import BaseLaw
from ._requirement_spec import RequirementSpec


@dataclass(frozen=True)
class ObjectiveLaw(BaseLaw):
    """A law that gives a cost of values, which the system makes least.

    It gives a cost that the system makes least, such as `ObjectiveLaw(s.k * s.spring.x**2 / 2 - s.f * s.spring.x,
    s.spring.x)` inside the interaction of a loaded spring, which finds the stretch `x = f / k` of least energy. The
    costs of several laws add up, and the constraint laws of the system hold at the minimum. A cost linear in its
    targets is solved as a linear program with the constraint laws that read them, at every time of a run, and a
    sequence of costs is made least in order, each minimum held while the next is sought. With `controls`, the law
    steers values over a run instead, and makes least the integral of the cost plus the cost at the end, while the
    rate laws move the values. A cost to make greatest, such as a profit, is made least with its sign changed. A loaded
    structure rests at its least energy, a cell routes its fluxes to its greatest growth, a model takes the weights of
    its least loss, and a health authority times a vaccination campaign at the least cost.

    Read more in the user guide, under Laws.

    Args:
        value:
            The scalar symbolic expression to minimise, or a sequence of them linear in the targets, minimised in
            order.

        targets:
            The symbol of a value, or the symbols of the values, that the minimum is sought over, or `None` with
            controls.

        controls:
            The symbols of the values the law steers over a run, or `None` for a minimum over `targets`. The allowed
            values of a control are those of its element, such as `u: Annotated[float, RealSet.closed(0, 1)] = 0.0`.

        terminal:
            With controls, the cost at the end of the run, a symbolic expression or a number.

        active:
            For a linear value, a condition under which the minimum is sought, such as `biomass > 0`, or `None` for
            always. Where it is false the targets are zero.

        infeasible:
            For a linear value, what a run does when the constraints cannot all hold, `'raise'` to raise an error,
            `'zero'` to set the targets to zero, or `'hold'` to keep their last values, as a culture that stops
            growing when its medium runs out.

        duals:
            For a linear value, whether a run records the dual value of each constraint, under `<path>.dual_<k>` in
            the order of the constraints. The dual value of a constraint is how much the first value changes when the
            right side of the constraint grows by one, such as the price of one more unit of a nutrient of a cell.

        label:
            The name of the law, or the empty string for a law without one.

    See Also:
        ConstraintLaw: States an equation or a bound that holds at the minimum.
        UpdateLaw: Gives the next value after one step, such as one step of a descent.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import ObjectiveLaw
        >>> x, k, f = var('x k f')
        >>> ObjectiveLaw(k * x**2 / 2 - f * x, x).targets
        x
    """

    law_tag: ClassVar[str] = 'objective'

    value: object
    targets: object = None
    _: KW_ONLY
    controls: Sequence[object] | None = None
    terminal: object = 0
    active: object = None
    infeasible: str = 'raise'
    duals: bool = False
    label: str = ''

    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the minimum is sought over, or the controls."""
        if self.controls is not None:
            return tuple(self.controls)
        if self.targets is None:
            return ()
        if isinstance(self.targets, Sequence) and not isinstance(self.targets, str):
            return tuple(self.targets)
        return (self.targets,)

    def _get_values_in_order(self) -> tuple[object, ...]:
        """Return the values to minimise in order, one for a single value."""
        if isinstance(self.value, Sequence) and not isinstance(self.value, str | Expression):
            return tuple(self.value)
        return (self.value,)

    def _get_law_requirements(self) -> RequirementSpec:
        """Return the time the law needs, continuous with controls and static otherwise."""
        return RequirementSpec(time=(_ContinuousTime,)) if self.controls is not None else self._get_requirements()

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: static time."""
        return RequirementSpec(time=(_StaticTime,))

    def _check(self) -> None:
        """Raise an error for an objective that is no expression or varies no state.

        Raises:
            ValidationError:
                The value is a function, the law names no target or both targets and controls, or a terminal cost
                without controls.
        """
        super()._check()
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

    def _get_value(self, inputs: object) -> object:
        """Return the objective at given symbol values.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        return self._evaluate(SR(self.value), inputs)
