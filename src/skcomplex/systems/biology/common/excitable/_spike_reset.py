"""Define spike resets of the potential of a neuron.

A spike reset is a crossing that resets the potential of a neuron when it reaches a threshold.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, EventLaw
from .....core.types import Role


class SpikeResetInteraction(Interaction):
    """Spike of a neuron that resets its potential when it crosses a threshold upwards.

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
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.common.pharmacokinetics import DosingInteraction
        >>> from skcomplex.systems.biology.neuroscience import SpikeResetInteraction
        >>> drive = DosingInteraction('drive', target='neuron', state='x', dose=1.0)
        >>> spike = SpikeResetInteraction('spike', target='neuron', state='x', threshold=1.0, after=0.0)
        >>> system = System(elements=[Element('neuron', values={'x': 0.0})], interactions=[drive, spike])
        >>> trajectory = system.prepare().simulate(until=2.1, dt=0.25)
        >>> [round(event.time, 6) for event in trajectory.events]
        [1.0, 2.0]
    """

    law_classes = (EventLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'v',
        threshold: float = 1.0,
        after: float = 0.0,
        kicked: str | None = None,
        kick: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.threshold = threshold
        self.after = after
        self.kicked = kicked
        self.kick = kick

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the spike, an event when the potential rises through the threshold.

        Args:
            context:
                The namespace of the neuron, the threshold, the potential after a spike, and the kick.
        """
        super()._prepare(context)
        neuron = context.target
        potential = getattr(neuron, self.state)
        reset = {potential: context.after}
        if self.kicked is not None:
            kicked = getattr(neuron, self.kicked)
            reset[kicked] = kicked + context.kick
        self.laws_: list[BaseLaw] = [EventLaw(potential - context.threshold, reset, 1, label=self.label)]
