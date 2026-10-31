"""Define threshold crossings of a state, which fire members and cascade to their neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ...errors import ValidationError
from ...specs import RoleSpec
from ...types import Array
from ..laws import BaseLaw, DriveLaw, LocalRuleLaw, RelaxationLaw
from ..spaces import BaseNeighborhood, VonNeumannNeighborhood
from ._neighbor import NeighborInteraction

MODES = ('redistribute', 'reset')


class ThresholdInteraction(NeighborInteraction):
    """Firing of the members whose state reaches a threshold, which cascades to their neighbours until all are below it.

    At every step the drive changes the state, and then every member whose state reaches `threshold` fires, all at
    once, until no member does. In the mode `'redistribute'` a firing member loses `threshold` and each of its
    neighbours gains `share`, and in the mode `'reset'` a firing member falls to 0 and each of its neighbours gains
    `share` times the state it fired with. Neighbours outside a fixed boundary take no share, so it leaves the system.
    Every cascade is recorded as an event with its size, the number of firings, its duration, and its area.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group on the nodes or sites.

        state:
            The name of the state of the members.

        threshold:
            The value of the state at which a member fires, fixed during a run.

        share:
            What every neighbour of a firing member gains, an amount, or a fraction of the state in the mode
            `'reset'`, fixed during a run.

        drive:
            A function `f(values, *, random)` of the array of the states and the random generator of the simulation
            that returns the states after the drive of a step, or `None` for no drive.

        over:
            The neighbourhood of the member itself first and its neighbours, or `None` for the von Neumann
            neighbourhood of a square lattice with its centre. A network takes `AdjacencyNeighborhood(center=True)`.

        mode:
            `'redistribute'` or `'reset'`.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the graph of the network.

    Raises:
        ValidationError:
            The mode is unknown, the drive is not callable, or the neighbourhood leaves out the member itself.

    Examples:
        >>> from skcomplex.core.interactions import ThresholdInteraction
        >>> sandpile = ThresholdInteraction('sandpile', nodes='sites', state='height', threshold=4, share=1)
        >>> sandpile.check().mode
        'redistribute'
    """

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        state: str = 'x',
        threshold: float = 1.0,
        share: float = 0.0,
        drive: Callable[..., Array] | None = None,
        over: BaseNeighborhood | None = None,
        mode: str = 'redistribute',
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.nodes = nodes
        self.state = state
        self.threshold = threshold
        self.share = share
        self.drive = drive
        self.over = over
        self.mode = mode

    def _check(self) -> None:
        """Raise an error for an unknown mode, a drive that cannot be called, or a neighbourhood without its centre.

        Raises:
            ValidationError:
                The mode is unknown, the drive is not callable, or the neighbourhood leaves out the member itself.
        """
        super()._check()
        if self.mode not in MODES:
            error_msg = f'The mode of the threshold is {self.mode!r}. Use one of {", ".join(MODES)}.'
            raise ValidationError(error_msg)
        if self.drive is not None and not callable(self.drive):
            error_msg = f'The drive of the threshold must be callable, but it is {self.drive!r}.'
            raise ValidationError(error_msg)
        if self.over is not None and not getattr(self.over, 'center', False):
            error_msg = (
                f'The threshold reads the neighbourhood {self.over!r}, which leaves out the member itself. Give it '
                '`center=True`.'
            )
            raise ValidationError(error_msg)

    def _fire(self, threshold: float, share: float) -> Callable[..., Array]:
        """Return the rule of one firing of every member, from its state first and the states of its neighbours."""
        reset = self.mode == 'reset'

        def fire(value: Array, *around: Array) -> Array:
            gains = sum((np.where(item >= threshold, share * (item if reset else 1.0), 0.0) for item in around), 0.0)
            fired = value >= threshold
            remaining = np.where(fired, 0.0, value) if reset else value - threshold * fired
            return np.asarray(remaining + gains, dtype=float)

        return fire

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the relaxation of the state of the target, with the drive.

        Args:
            namespace:
                The namespace of an edge.

        Returns:
            laws:
                The relaxation of the state.
        """
        value = getattr(namespace.target, self.state)
        threshold, share = self.threshold, self.share
        drive = [] if self.drive is None else [DriveLaw(value, self.drive)]
        over = VonNeumannNeighborhood(center=True) if self.over is None else self.over
        return [
            RelaxationLaw(
                drive=drive,
                relax=[LocalRuleLaw(value, over, self._fire(threshold, share))],
                stable=lambda values: bool((values < threshold).all()),
                activity=lambda values: values >= threshold,
                label=self.label,
            ),
        ]
