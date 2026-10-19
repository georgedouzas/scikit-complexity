"""Self-organized criticality models: sandpiles and forest fires."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, parameter, state
from ....core.model import Drive, Element, LatticeSpace, Law, LocalRule, Neighborhood, Relaxation, Symbols, System
from ....core.runtime import register_model

type Array = npt.NDArray[np.float64]

NEAREST = Neighborhood.von_neumann(center=True)
NEAREST_STEPS = np.array(Neighborhood.von_neumann().offsets)
MANNA_THRESHOLD = 2


def _drop_grain(heights: Array, *, random: np.random.Generator) -> Array:
    """Return the heights with one grain added on a random site."""
    following = heights.copy()
    following[tuple(random.integers(size) for size in heights.shape)] += 1
    return following


def _topple_at_random(heights: Array, *, random: np.random.Generator) -> Array:
    """Return the heights after every site with two or more grains sends two grains to random neighbours."""
    active = np.argwhere(heights >= MANNA_THRESHOLD)
    following = heights.copy()
    following[tuple(active.T)] -= MANNA_THRESHOLD
    targets = (
        np.repeat(active, MANNA_THRESHOLD, axis=0)
        + NEAREST_STEPS[random.integers(len(NEAREST_STEPS), size=2 * len(active))]
    )
    inside = np.all((targets >= 0) & (targets < heights.shape), axis=1)
    np.add.at(following, tuple(targets[inside].T), 1)
    return following


class SandpileCell(Element):
    """Site of the Bak-Tang-Wiesenfeld sandpile, which topples to its four neighbours when it holds four grains.

    Every step drops one grain on a random site, then sites at or above the threshold topple until the pile is
    stable. Grains that topple over the edge leave the pile.

    Args:
        label:
            The label of the site or the lattice group.

        height:
            The initial number of grains.

        threshold:
            The number of grains at which a site topples.

    Examples:
        >>> from skcomplex.models.physics.criticality import SandpileCell
        >>> SandpileCell('site').threshold
        4
    """

    height: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    threshold: int = 4

    def laws(self, s: Symbols) -> list[Law]:
        """Return the drive by single grains and the toppling relaxation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        threshold = self.threshold

        def topple(height: Array, *around: Array) -> Array:
            return height - threshold * (height >= threshold) + sum(item >= threshold for item in around)

        return [
            Relaxation(
                drive=[Drive(s.height, _drop_grain)],
                relax=[LocalRule(s.height, NEAREST, topple)],
                stable=lambda heights: bool((heights < threshold).all()),
                activity=lambda heights: heights >= threshold,
                label='avalanche',
            ),
        ]


class MannaCell(Element):
    """Site of the Manna sandpile, which topples when it holds two grains and sends each to a random neighbour.

    Every step drops one grain on a random site, then all sites with two or more grains topple at once until the pile
    is stable. Grains that topple over the edge leave the pile.

    Args:
        label:
            The label of the site or the lattice group.

        height:
            The initial number of grains.

    Examples:
        >>> from skcomplex.models.physics.criticality import MannaCell
        >>> MannaCell('site').height
        0
    """

    height: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the drive by single grains and the stochastic toppling relaxation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Relaxation(
                drive=[Drive(s.height, _drop_grain)],
                relax=[Drive(s.height, _topple_at_random)],
                stable=lambda heights: bool((heights < MANNA_THRESHOLD).all()),
                activity=lambda heights: heights >= MANNA_THRESHOLD,
                label='avalanche',
            ),
        ]


class Forest(Element):
    """Site of the Drossel-Schwabl forest fire: empty at 0, a tree at 1, or burning at 2.

    Trees grow on empty sites with probability `growth`, catch fire from a burning neighbour or from lightning with
    probability `lightning`, and burning sites become empty.

    Args:
        label:
            The label of the site or the lattice group.

        cover:
            The initial cover, 0, 1, or 2.

        growth:
            The probability that a tree grows on an empty site per step.

        lightning:
            The probability that lightning strikes a tree per step.

    Examples:
        >>> from skcomplex.models.physics.criticality import Forest
        >>> Forest('site').growth
        0.05
    """

    cover: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    growth: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    lightning: float = parameter(default=1e-4, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the stochastic rule of growth, fire, and burning out.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """

        def rule(cover: Array, *around: Array, growth: Any, lightning: Any, random: Array) -> Array:  # noqa: ANN401
            burning = np.any(np.stack(around) == 2, axis=0)  # noqa: PLR2004
            ignites = burning | (random < lightning)
            tree = np.where(ignites, 2.0, 1.0)
            empty = np.where(random < growth, 1.0, 0.0)
            return np.where(cover == 2, 0.0, np.where(cover == 1, tree, empty))  # noqa: PLR2004

        return [LocalRule(s.cover, NEAREST, rule, stochastic=True)]


@register_model(
    field='physics/criticality',
    tags=('lattice', 'avalanches', 'power law'),
    references=('Bak, Tang, and Wiesenfeld 1987', 'Charbonneau 2017'),
    defaults={'until': 20000, 'dt': 100},
)
def sandpile(shape: Sequence[int] = (50, 50), *, random_state: int | np.random.Generator | None = 0) -> System:
    """Return the Bak-Tang-Wiesenfeld sandpile on a lattice with open edges, started from random stable heights.

    Its avalanches are recorded as `avalanche` events with their sizes, the number of topplings, their durations,
    and their areas.

    Args:
        shape:
            The number of rows and columns.

        random_state:
            The seed or generator of the initial heights.

    Returns:
        system:
            The model, with the heights `pile.height`.

    Examples:
        >>> from skcomplex.models.physics.criticality import sandpile
        >>> sandpile((10, 10)).elements[0].size
        100
    """
    heights = np.random.default_rng(random_state).integers(0, 4, size=tuple(shape)).astype(float)
    space = LatticeSpace(tuple(shape), boundary='fixed')
    return System(elements=[SandpileCell.lattice('pile', space, height=heights)])


@register_model(
    field='physics/criticality',
    tags=('lattice', 'avalanches', 'power law', 'stochastic'),
    references=('Manna 1991', 'Jensen 2023'),
    defaults={'until': 10000, 'dt': 50},
)
def manna_sandpile(shape: Sequence[int] = (32, 32), *, random_state: int | np.random.Generator | None = 0) -> System:
    """Return the Manna sandpile, a stochastic sandpile of another universality class than Bak-Tang-Wiesenfeld.

    The pile starts from random stable heights of 0 or 1 grain, and its avalanches are recorded as `avalanche`
    events with their sizes, durations, and areas.

    Args:
        shape:
            The number of rows and columns.

        random_state:
            The seed or generator of the initial heights.

    Returns:
        system:
            The model, with the heights `pile.height`.

    Examples:
        >>> from skcomplex.models.physics.criticality import manna_sandpile
        >>> manna_sandpile((10, 10)).elements[0].size
        100
    """
    heights = np.random.default_rng(random_state).integers(0, MANNA_THRESHOLD, size=tuple(shape)).astype(float)
    space = LatticeSpace(tuple(shape), boundary='fixed')
    return System(elements=[MannaCell.lattice('pile', space, height=heights)])


@register_model(
    field='physics/criticality',
    tags=('lattice', 'fire', 'stochastic', 'power law'),
    references=('Drossel and Schwabl 1992', 'Charbonneau 2017'),
    defaults={'until': 500, 'dt': 1},
)
def forest_fire(
    shape: Sequence[int] = (100, 100),
    *,
    growth: float = 0.05,
    lightning: float = 1e-4,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Drossel-Schwabl forest fire, where slow growth and rare lightning give fires of every size.

    Args:
        shape:
            The number of rows and columns.

        growth:
            The probability that a tree grows on an empty site per step.

        lightning:
            The probability that lightning strikes a tree per step.

        density:
            The initial fraction of sites with a tree.

        random_state:
            The seed or generator of the initial forest.

    Returns:
        system:
            The model, with the covers `forest.cover`.

    Examples:
        >>> from skcomplex.models.physics.criticality import forest_fire
        >>> forest_fire((10, 10)).elements[0].values['growth']
        0.05
    """
    cover = (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)
    space = LatticeSpace(tuple(shape))
    return System(elements=[Forest.lattice('forest', space, cover=cover, growth=growth, lightning=lightning)])
