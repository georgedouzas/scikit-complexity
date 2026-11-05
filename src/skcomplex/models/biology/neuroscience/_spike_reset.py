"""Define spike resets, crossings that reset the potential of a neuron when it reaches a threshold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import CrossingInteraction
from ....specs import ParameterSpec, RoleSpec


class SpikeResetInteraction(CrossingInteraction):
    """Spike of a neuron, which resets its potential when it crosses a threshold upwards and kicks a second state.

    When the potential `v` rises through `threshold`, it is set to `after`, and the state `kicked`, such as a recovery
    variable or a synaptic trace, grows by `kick`. Every spike is an event labelled by the interaction.

    Args:
        label:
            The label of the interaction and of its events.

        target:
            The path of the neuron.

        state:
            The name of the potential.

        threshold:
            The potential of a spike.

        after:
            The potential after a spike.

        kicked:
            The name of the state a spike increments, or `None` for none.

        kick:
            The increment of the kicked state at a spike.

    Raises:
        ValidationError:
            The direction is not -1, 0, or 1, or the guard or a new value is not a function.

    Examples:
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.interactions import RateInteraction
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.biology.neuroscience import SpikeResetInteraction
        >>> drive = RateInteraction('drive', target='neuron', state='n', rate=1.0)
        >>> spike = SpikeResetInteraction('spike', target='neuron', state='n', threshold=1.0, after=0.0)
        >>> system = System(elements=[CompartmentElement('neuron')], interactions=[drive, spike])
        >>> trajectory = Simulation(system, dt=0.25).prepare().run(until=2.1).trajectory_
        >>> [round(event.time, 6) for event in trajectory.events]
        [1.0, 2.0]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'v',
        threshold: Annotated[float, ParameterSpec()] = 1.0,
        after: Annotated[float, ParameterSpec()] = 0.0,
        kicked: str | None = None,
        kick: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            guard=self._get_guard,
            reset=self._get_resets(state, kicked),
            direction=1,
        )
        self.state = state
        self.threshold = threshold
        self.after = after
        self.kicked = kicked
        self.kick = kick

    def _get_guard(self, s: SimpleNamespace) -> Expression:
        """Return the potential above the threshold."""
        return getattr(s.target, self.state) - s.threshold

    def _get_resets(self, state: str, kicked: str | None) -> dict[str, Callable[[SimpleNamespace], object]]:
        """Return the new value of the potential and of the kicked state after a spike."""
        resets: dict[str, Callable[[SimpleNamespace], object]] = {state: lambda s: s.after}
        if kicked is not None:
            resets[kicked] = lambda s: getattr(s.target, kicked) + s.kick
        return resets
