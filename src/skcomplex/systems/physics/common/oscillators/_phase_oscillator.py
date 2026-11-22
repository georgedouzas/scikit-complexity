"""Define phase oscillators, phases that turn at a natural frequency."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping
from typing import Annotated

from .....core.components.elements import Element, Period
from .....core.components.spaces import BaseSpace


class PhaseOscillatorElement(Element):
    """Phase oscillator, a phase `theta` on the circle with a natural frequency `omega`.

    The phase wraps around the circle, and `PhaseRotationInteraction` turns it at the natural frequency, as for a
    clock, a firefly, or a generator of a power grid. Couplings to other oscillators add to the rate of the phase.

    Args:
        label:
            The label of the oscillator or the group.

        theta:
            The initial phase, or one phase per member.

        omega:
            The natural frequency, or one frequency per member.

        size:
            The number of oscillators, or `None` for one or one per site of the space.

        space:
            The space of the oscillators, or `None` for none.

        on:
            `'edges'` or `'hyperedges'` for one oscillator on every link of a network, or `None` for its sites.

        values:
            Further values by name, or `None`.

    Examples:
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.oscillators import PhaseOscillatorElement, PhaseRotationInteraction
        >>> clock = PhaseOscillatorElement('clock', omega=2.0)
        >>> system = System(elements=[clock], interactions=[PhaseRotationInteraction('turn', target='clock')])
        >>> float(system.prepare().simulate(until=1.0, dt=0.5).get_values('clock.theta')[-1])
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        omega: object = 1.0,
        size: int | None = None,
        space: BaseSpace | None = None,
        on: str | None = None,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space, on=on)
        self.theta = theta
        self.omega = omega
