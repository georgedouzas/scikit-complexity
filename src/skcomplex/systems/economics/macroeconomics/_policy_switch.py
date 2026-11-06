"""Policy switch of a central bank, which moves the reserve ratio of the banks from `ratio` to `new_ratio`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import FeedbackInteraction
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec


class PolicySwitchInteraction(FeedbackInteraction[SystemView]):
    """Policy switch of a central bank, which moves the reserve ratio of the banks from `ratio` to `new_ratio`.

    The ratio switches at the time `change_at`.

    Args:
        label:
            The label of the institution.

        rule:
            Unused, the regulator overrides `act`.

        phase:
            The phase the regulator acts in, or `None` for the default phase.

        ratio:
            The reserve ratio before the change.

        new_ratio:
            The reserve ratio after the change.

        change_at:
            The time of the change.

        banks:
            The path of the group of banks.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import PolicySwitchInteraction
        >>> PolicySwitchInteraction('regulator', ratio=0.1, new_ratio=0.2, change_at=100.0).new_ratio
        0.2
    """

    def __init__(
        self,
        label: str,
        *,
        rule: Callable[[SystemView], None] | None = None,
        phase: str | None = None,
        ratio: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        new_ratio: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        change_at: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        banks: str = 'banks',
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.ratio = ratio
        self.new_ratio = new_ratio
        self.change_at = change_at
        self.banks = banks

    def act(self, system: SystemView) -> None:
        """Set the reserve ratio of the banks from the schedule.

        Args:
            system:
                The system view of the regulator.
        """
        before = system.time < system.parameter('change_at')
        ratio = system.parameter('ratio') if before else system.parameter('new_ratio')
        name = f'{self.banks}.reserve_ratio'
        if system.parameters[name] != ratio:
            system.set_parameter(name, ratio)
