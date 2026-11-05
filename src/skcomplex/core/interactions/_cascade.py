"""Define cascades of a state, which drive it and then relax every member at or above a threshold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np

from ...errors import ValidationError
from ...specs import BaseSpec, RoleSpec
from ...types import Array, ElementKind
from ..laws import BaseLaw, CascadeLaw, DriveLaw, LocalRuleLaw
from ..spaces import BaseNeighborhood, LatticeNeighborhood, VonNeumannNeighborhood
from ._neighbor import NeighborInteraction

MODES = ('redistribute', 'reset', 'random')


class CascadeInteraction(NeighborInteraction):
    """Cascade of the members whose state reaches a threshold, which relax and pass shares to their neighbours.

    At every step the drive changes the state, and then every member whose state reaches `threshold` relaxes, all at
    once, until no member does. In the mode `'redistribute'` a relaxing member loses `threshold` and each of its
    neighbours gains `share`. In the mode `'reset'` a relaxing member falls to 0 and each of its neighbours gains
    `share` times the state it relaxed from. In the mode `'random'` a relaxing member loses `threshold` in parcels of
    `share`, and every parcel goes to one of its neighbours drawn at random. Neighbours outside a fixed boundary take no
    share, so it leaves the system, and the mode `'random'` treats every lattice as bounded. Every cascade is recorded
    as an event with its size, the number of relaxations, its duration, and its area.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group on the nodes or sites.

        state:
            The name of the state of the members.

        threshold:
            The value of the state at which a member relaxes, fixed during a run.

        share:
            What a neighbour of a relaxing member gains, an amount, a fraction of the state in the mode `'reset'`,
            or the amount of one parcel in the mode `'random'`, fixed during a run.

        mode:
            `'redistribute'`, `'reset'`, or `'random'`.

        drive:
            A function `f(values, *, random)` of the array of the states and the random generator of the simulation
            that returns the states after the drive of a step, or `None` for no drive.

        neighborhood:
            The neighbourhood of the member itself first and its neighbours, or `None` for the von Neumann
            neighbourhood of a square lattice with its centre. A network takes `AdjacencyNeighborhood(center=True)`,
            and the mode `'random'` takes a lattice neighbourhood.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the graph of the network.

    Raises:
        ValidationError:
            The mode is unknown, the drive is not callable, the neighbourhood leaves out the member itself, or the
            mode `'random'` has no lattice neighbourhood or a threshold that is not a whole number of parcels.

    Examples:
        >>> from skcomplex.core.interactions import CascadeInteraction
        >>> cascade = CascadeInteraction('cascade', target='sites', state='height', threshold=4, share=1)
        >>> cascade.mode
        'redistribute'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (CascadeLaw, DriveLaw, LocalRuleLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',))],
        state: str = 'x',
        threshold: float = 1.0,
        share: float = 0.0,
        mode: str = 'redistribute',
        drive: Callable[..., Array] | None = None,
        neighborhood: BaseNeighborhood | None = None,
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.target = target
        self.state = state
        self.threshold = threshold
        self.share = share
        self.mode = mode
        self.drive = drive
        self.neighborhood = neighborhood

    def _check_random(self) -> None:
        """Raise an error for a random cascade without a lattice neighbourhood or a whole number of parcels."""
        if self.mode != 'random':
            return
        if self.neighborhood is not None and not isinstance(self.neighborhood, LatticeNeighborhood):
            error_msg = (
                f'The random cascade reads the neighbourhood {self.neighborhood!r}. Give it a lattice neighbourhood.'
            )
            raise ValidationError(error_msg)
        parcels = self.threshold / self.share if self.share > 0 else 0.0
        if parcels < 1 or not float(parcels).is_integer():
            error_msg = (
                f'The random cascade passes the threshold {self.threshold} in parcels of {self.share}. Give a positive '
                'share that divides the threshold.'
            )
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an unknown mode, a drive, a neighbourhood, or parcels the cascade cannot take.

        Raises:
            ValidationError:
                The mode is unknown, the drive is not callable, the neighbourhood leaves out the member itself, or the
                mode `'random'` has no lattice neighbourhood or a threshold that is not a whole number of parcels.
        """
        super()._check()
        self._check_variant('mode', self.mode, MODES, (BaseSpec,))
        if self.drive is not None and not callable(self.drive):
            error_msg = f'The drive of the cascade must be callable, but it is {self.drive!r}.'
            raise ValidationError(error_msg)
        if self.neighborhood is not None and not getattr(self.neighborhood, 'center', False):
            error_msg = (
                f'The cascade reads the neighbourhood {self.neighborhood!r}, which leaves out the member itself. Give '
                'it `center=True`.'
            )
            raise ValidationError(error_msg)
        self._check_random()

    def _relax(self, threshold: float, share: float) -> Callable[..., Array]:
        """Return the rule of one relaxation of every member, from its state first and the states of its neighbours."""
        if self.mode not in MODES:
            raise self._get_variant_error(self.mode, 'mode', '_relax')
        reset = self.mode == 'reset'

        def relax(value: Array, *around: Array) -> Array:
            gains = sum((np.where(item >= threshold, share * (item if reset else 1.0), 0.0) for item in around), 0.0)
            relaxing = value >= threshold
            remaining = np.where(relaxing, 0.0, value) if reset else value - threshold * relaxing
            return np.asarray(remaining + gains, dtype=float)

        return relax

    def _scatter(self, neighborhood: LatticeNeighborhood, threshold: float, share: float) -> Callable[..., Array]:
        """Return the drive that passes the parcels of every relaxing member to neighbours drawn at random."""
        steps = np.array([offset for offset in neighborhood.prepare().offsets_ if any(offset)])
        parcels = round(threshold / share)

        def scatter(values: Array, *, random: np.random.Generator) -> Array:
            relaxing = np.argwhere(values >= threshold)
            following = values.copy()
            following[tuple(relaxing.T)] -= threshold
            moves = steps[random.integers(len(steps), size=parcels * len(relaxing))]
            targets = np.repeat(relaxing, parcels, axis=0) + moves
            inside = np.all((targets >= 0) & (targets < values.shape), axis=1)
            np.add.at(following, tuple(targets[inside].T), share)
            return following

        return scatter

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the cascade of the state of the target, with the drive.

        Args:
            namespace:
                The namespace of an edge.

        Returns:
            laws:
                The cascade of the state.
        """
        value = getattr(self._get_member(namespace, 'target'), self.state)
        threshold, share = self.threshold, self.share
        drive = [] if self.drive is None else [DriveLaw(value, self.drive)]
        neighborhood = VonNeumannNeighborhood(center=True) if self.neighborhood is None else self.neighborhood
        if self.mode == 'random' and isinstance(neighborhood, LatticeNeighborhood):
            relax: list[BaseLaw] = [DriveLaw(value, self._scatter(neighborhood, threshold, share))]
        else:
            relax = [LocalRuleLaw(value, neighborhood, self._relax(threshold, share))]
        return [
            CascadeLaw(
                drive=drive,
                relax=relax,
                stable=lambda values: bool((values < threshold).all()),
                activity=lambda values: values >= threshold,
                label=self.label,
            ),
        ]
