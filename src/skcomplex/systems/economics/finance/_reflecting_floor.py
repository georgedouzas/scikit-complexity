"""Lower bound on the sizes of a group at a fraction of their mean before the step, `x = max(x, c <x_{t-1}>)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import MapInteraction
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, RoleSpec
from ._gibrat_firm_view import GibratFirmView


class ReflectingFloorInteraction(MapInteraction):
    """Lower bound on the sizes of a group at a fraction of their mean before the step, `x = max(x, c <x_{t-1}>)`.

    It is a map of the agent form that raises every size below the floor to the floor, the mean of the states
    `previous` of the group times `c`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of firms.

        c:
            The fraction of the previous mean size below which no size falls.

        phase:
            The phase of the step the floor acts in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.finance import ReflectingFloorInteraction
        >>> ReflectingFloorInteraction('floor', target='firms', c=0.5).c
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, state='x', function=self._get_floored, form='agent', phase=phase)
        self.c = c

    def _get_floor(self, system: SystemView) -> float:
        """Return the floor of the step, `c` times the mean size of the group before it, computed once per step."""
        cached = system.storage.get('floor')
        if isinstance(cached, tuple) and cached[0] == system.time:
            return float(cached[1])
        previous = [firm.previous for firm in system.agents('target', GibratFirmView)]
        floor = system.parameter('c') * float(np.mean(previous))
        system.storage['floor'] = (system.time, floor)
        return floor

    def _get_floored(self, agent: GibratFirmView, system: SystemView) -> float:
        """Return the size of a firm raised to the floor, `c` times the mean size of its group before the step."""
        return max(agent.x, self._get_floor(system))
