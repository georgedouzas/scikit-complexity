"""Define dosing, the input of a drug into a compartment at a constant rate or in a schedule of infusions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from .....core.components.interactions import RateInteraction
from .....core.components.laws.utils import pulse_train
from .....core.errors import ValidationError
from .....core.specs import RoleSpec


class DosingInteraction(RateInteraction):
    """Input of a drug into a state of a target at the rate `dose`, constantly or in a schedule of infusions.

    Without a period the drug enters at the constant rate `dose`. With a period it enters at the rate `dose` for `width`
    time units every `period` from `start`, `count` times or without end. The dose is a function of the namespace,
    which reads the parameters of the target, or a number.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose drug state the dose raises.

        state:
            The name of the state of the target that holds the drug.

        dose:
            A function of the namespace that returns the rate of the input during an infusion, or a number.

        period:
            The time between the starts of infusions, or `None` for a constant input.

        width:
            The duration of every infusion.

        start:
            The time of the first infusion.

        count:
            The number of infusions, or `None` for a schedule that never ends.

    Raises:
        ValidationError:
            The state is not a name, the dose is neither a function nor a number, or the schedule is not positive.

    Examples:
        >>> from skcomplex.core.components.elements import CompartmentElement
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.pharmacokinetics import DosingInteraction
        >>> from skcomplex.core.engine.runtime import simulate
        >>> dosing = DosingInteraction('dosing', target='patient', state='n', dose=2.0, period=10.0, count=3)
        >>> system = System(elements=[CompartmentElement('patient', n=0.0)], interactions=[dosing])
        >>> trajectory = simulate(system, until=40.0, dt=0.5)
        >>> round(float(trajectory.values('patient.n')[-1]), 3)
        6.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'x',
        dose: Callable[[SimpleNamespace], object] | float = 1.0,
        period: float | None = None,
        width: float = 1.0,
        start: float = 0.0,
        count: int | None = None,
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start
        self.count = count

    def _check(self) -> None:
        """Raise an error for a state, a dose, or a schedule the input cannot take.

        Raises:
            ValidationError:
                The state is not a name, the dose is neither a function nor a number, or the schedule is not positive.
        """
        super()._check()
        if not (callable(self.dose) or isinstance(self.dose, int | float)):
            error_msg = f'The dose of {self.label!r} is {self.dose!r}. Give a function of the namespace or a number.'
            raise ValidationError(error_msg)
        if self.period is not None and not (self.period > 0 and self.width > 0):
            error_msg = (
                f'The schedule of {self.label!r} has the period {self.period!r} and the width {self.width!r}. Give '
                'both above 0.'
            )
            raise ValidationError(error_msg)

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the dose, times the schedule of infusions when the input has a period.

        Args:
            namespace:
                The namespace of the interaction, with the namespace of the target and the time `t`.

        Returns:
            rate:
                The rate of the input.
        """
        dose = self._evaluate(self.dose, namespace)
        if self.period is None:
            constant: Expression = dose
            return constant
        schedule: Expression = dose * pulse_train(
            namespace.t,
            self.period,
            self.width,
            start=self.start,
            count=self.count,
        )
        return schedule
