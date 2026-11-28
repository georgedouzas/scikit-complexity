"""Define cascades of a state.

A cascade drives a state and then relaxes every member at or above a threshold.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from sage.all__sagemath_symbolics import Expression

from .....core.base import BaseSpec
from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, UpdateLaw
from .....core.components.laws.utils import relax
from .....core.components.spaces import BaseNeighborhood, LatticeNeighborhood, RadiusNeighborhood
from .....core.errors import ValidationError
from .....core.types import Array, Role

MODES = ('redistribute', 'reset', 'random')
NEIGHBORS = ('fixed', 'random')


class TopplingInteraction(Interaction):
    """Cascade of the members whose state reaches a threshold.

    These members relax and pass shares to their neighbours. At every step the drive changes the state, and then every
    member whose state reaches `threshold` relaxes, all at once, until no member does. In the mode `'redistribute'` a
    relaxing member loses `threshold` and each of its neighbours gains `share`. In the mode `'reset'` a relaxing member
    falls to 0 and each of its neighbours gains `share` times the state it relaxed from. In the mode `'random'` a
    relaxing member loses `threshold` in parcels of `share`, and every parcel goes to one of its neighbours drawn at
    random. Neighbours outside a fixed boundary take no share, so it leaves the system, and the mode `'random'` treats
    every lattice as bounded. With `neighbors='random'`, the neighbours of a relaxing member are drawn afresh at every
    relaxation: every parcel of `share` goes to a member drawn uniformly among all members with the probability
    `conservation` and leaves the system otherwise, the random neighbour rule of the mean-field sandpile. Every cascade
    is recorded as an event with its size, the number of relaxations, its duration, and its area.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

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
            A function `f(values, *, random, **kwargs)` of the array of the states, the random generator of the
            simulation, and the other keywords of a rule of an `UpdateLaw`, that returns the states after the drive of
            a step, or `None` for no drive. A cascade with a drive draws numbers.

        neighborhood:
            The neighbourhood of the member itself first and its neighbours, or `None` for the von Neumann
            neighbourhood of a square lattice with its centre. A network takes `AdjacencyNeighborhood(center=True)`,
            and the mode `'random'` takes a lattice neighbourhood.

        neighbors:
            `'fixed'` for the neighbourhood of the space, or `'random'` for members drawn afresh at every relaxation.

        conservation:
            The probability that a parcel sent to random neighbours stays in the system, unused with fixed neighbours.

    Raises:
        ValidationError:
            The mode or the neighbours are unknown, the drive is not callable, the neighbourhood leaves out the member
            itself, the mode `'random'` has no lattice neighbourhood, a threshold is not a whole number of parcels in
            the mode `'random'` or with random neighbours, random neighbours have the mode `'reset'`, or the
            conservation lies outside the unit interval.

    Examples:
        >>> from skcomplex.systems.physics.common.avalanches import TopplingInteraction
        >>> cascade = TopplingInteraction('cascade', target='sites', state='height', threshold=4, share=1)
        >>> cascade.mode
        'redistribute'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (UpdateLaw,)

    def _get_draws(self) -> bool:
        """Return whether the cascade draws.

        It draws by parcels sent at random, or by a drive, which receives the generator.
        """
        return super()._get_draws() or self.mode == 'random' or self.neighbors == 'random' or self.drive is not None

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'x',
        threshold: float = 1.0,
        share: float = 0.0,
        mode: str = 'redistribute',
        drive: Callable[..., Array] | None = None,
        neighborhood: BaseNeighborhood | None = None,
        neighbors: str = 'fixed',
        conservation: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.threshold = threshold
        self.share = share
        self.mode = mode
        self.drive = drive
        self.neighborhood = neighborhood
        self.neighbors = neighbors
        self.conservation = conservation

    def _check_parcels(self) -> None:
        """Raise an error for a threshold that is not a whole number of parcels."""
        parcels = self.threshold / self.share if self.share > 0 else 0.0
        if parcels < 1 or not float(parcels).is_integer():
            error_msg = (
                f'The random cascade passes the threshold {self.threshold} in parcels of {self.share}. Give a positive '
                'share that divides the threshold.'
            )
            raise ValidationError(error_msg)

    def _check_random_neighbors(self) -> None:
        """Raise an error for random neighbours of the mode `'reset'`, a bad conservation, or broken parcels."""
        self._check_variant('neighbors', self.neighbors, NEIGHBORS, (BaseSpec,))
        if self.neighbors != 'random':
            return
        if self.mode == 'reset':
            error_msg = (
                "The cascade sends parcels to random neighbours, which the mode 'reset' does not pass. Use the mode "
                "'redistribute' or 'random'."
            )
            raise ValidationError(error_msg)
        if not 0 <= self.conservation <= 1:
            error_msg = f'The cascade keeps parcels with the probability {self.conservation}. Give one in [0, 1].'
            raise ValidationError(error_msg)
        self._check_parcels()

    def _check_random(self) -> None:
        """Raise an error for a random cascade without lattice neighbours or whole parcels.

        The cascade needs a lattice neighbourhood and a whole number of parcels.
        """
        if self.mode != 'random':
            return
        if self.neighborhood is not None and not isinstance(
            self.neighborhood,
            LatticeNeighborhood | RadiusNeighborhood,
        ):
            error_msg = (
                f'The random cascade reads the neighbourhood {self.neighborhood!r}. Give it a lattice neighbourhood.'
            )
            raise ValidationError(error_msg)
        self._check_parcels()

    def _check_arguments(self) -> None:
        """Raise an error for a mode, drive, neighbourhood, or parcels it cannot take.

        Raises:
            ValidationError:
                The mode or the neighbours are unknown, the drive is not callable, the neighbourhood leaves out the
                member itself, the mode `'random'` has no lattice neighbourhood, the parcels are broken, or random
                neighbours have the mode `'reset'` or a conservation outside the unit interval.
        """
        super()._check_arguments()
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
        self._check_random_neighbors()

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the cascade.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        target = self._get_symbolic(self._get_member(context, 'target'), context.t)
        self.laws_ = self._get_cascade(getattr(target, self.state))

    def _relax(self, threshold: float, share: float) -> Callable[..., Array]:
        """Return the rule of one relaxation of every member.

        The rule reads the state of the member first and the states of its neighbours.
        """
        if self.mode not in MODES:
            raise self._get_variant_error(self.mode, 'mode', '_relax')
        reset = self.mode == 'reset'

        def relax(value: Array, *around: Array, **kwargs: object) -> Array:
            gains = sum((np.where(item >= threshold, share * (item if reset else 1.0), 0.0) for item in around), 0.0)
            relaxing = value >= threshold
            remaining = np.where(relaxing, 0.0, value) if reset else value - threshold * relaxing
            return np.asarray(remaining + gains, dtype=float)

        return relax

    def _scatter(self, neighborhood: BaseNeighborhood, threshold: float, share: float) -> Callable[..., Array]:
        """Return the drive that passes parcels of relaxing members to random neighbours.

        Every relaxing member passes its parcels to neighbours drawn at random, among the offsets of the neighbourhood
        in the dimensions of the lattice.
        """
        parcels = round(threshold / share)

        def scatter(values: Array, *, random: np.random.Generator, **kwargs: object) -> Array:
            offsets = neighborhood._get_offsets(values.ndim) or ()
            steps = np.array([offset for offset in offsets if any(offset)])
            relaxing = np.argwhere(values >= threshold)
            following = values.copy()
            following[tuple(relaxing.T)] -= threshold
            moves = steps[random.integers(len(steps), size=parcels * len(relaxing))]
            targets = np.repeat(relaxing, parcels, axis=0) + moves
            inside = np.all((targets >= 0) & (targets < values.shape), axis=1)
            np.add.at(following, tuple(targets[inside].T), share)
            return following

        return scatter

    def _scatter_anywhere(self, threshold: float, share: float) -> Callable[..., Array]:
        """Return the rule that passes the parcels of relaxing members to members drawn among all.

        Every parcel stays with the probability `conservation` and goes to a member drawn uniformly, and leaves the
        system otherwise.
        """
        parcels, conservation = round(threshold / share), self.conservation

        def scatter(values: Array, *, random: np.random.Generator, **kwargs: object) -> Array:
            following = np.array(values, dtype=float)
            flat = following.reshape(-1)
            relaxing = np.flatnonzero(flat >= threshold)
            flat[relaxing] -= threshold
            kept = random.binomial(parcels * relaxing.size, conservation)
            np.add.at(flat, random.integers(flat.size, size=kept), share)
            return following

        return scatter

    def _get_cascade(self, value: Expression) -> list[BaseLaw]:
        """Return the update of the state of the target, driven and relaxed until stable."""
        threshold, share = self.threshold, self.share
        drive = [] if self.drive is None else [UpdateLaw(value, self.drive, arrays=True, stochastic=True)]
        neighborhood = self.neighborhood
        if neighborhood is None:
            neighborhood = RadiusNeighborhood(1, metric='manhattan', center=True)
        if self.neighbors == 'random':
            rules: list[BaseLaw] = [
                UpdateLaw(value, self._scatter_anywhere(threshold, share), arrays=True, stochastic=True),
            ]
        elif self.mode == 'random':
            rules = [
                UpdateLaw(value, self._scatter(neighborhood, threshold, share), arrays=True, stochastic=True),
            ]
        else:
            rules = [UpdateLaw(value, self._relax(threshold, share), neighborhood=neighborhood)]
        relaxation = relax(
            value,
            rules,
            lambda values: bool((values < threshold).all()),
            drive=drive,
            activity=lambda values: values >= threshold,
        )
        return [UpdateLaw(value, relaxation, label=self.label)]
