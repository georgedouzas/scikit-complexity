"""Lattice gases.

In the HPP and FHP gases, particles move along the directions of a lattice and collide.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import BallNeighborhood, LatticeNeighborhood, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import StateSpec, VisualSpec
from ....core.types import Array


class HPPSiteElement(Element):
    """Site of the HPP lattice gas.

    The site has one boolean channel per direction of the square lattice.

    The local rules of the gas turn every head-on pair of particles at a site by a quarter turn, and then move every
    particle to the neighbouring site along its direction. The state `density` holds the number of particles at the
    site.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n0:
            The initial occupation of the channel along the columns, towards higher column indices.

        n1:
            The initial occupation of the channel along the rows, towards higher row indices.

        n2:
            The initial occupation of the channel along the columns, towards lower column indices.

        n3:
            The initial occupation of the channel along the rows, towards lower row indices.

        density:
            The initial number of particles at the site.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.fluids import HPPSiteElement
        >>> HPPSiteElement('site').n0
        0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n0: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        n1: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        n2: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        n3: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        density: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.density = density


DIRECTIONS = 6
CHANNELS = tuple(f'n{direction}' for direction in range(DIRECTIONS))
SITE = LatticeNeighborhood(((0, 0),))
TRIANGULAR = BallNeighborhood(metric='hexagonal')
VELOCITIES = np.column_stack(
    [np.cos(np.arange(DIRECTIONS) * np.pi / 3), np.sin(np.arange(DIRECTIONS) * np.pi / 3)],
)
HEAD_ON = (0b001001, 0b010010, 0b100100)
TRIPLES = (0b010101, 0b101010)
TURN_PROBABILITY = 0.5
DENSITY_VISUAL = VisualSpec('sites', {'states': ('density',), 'low': 0.0, 'high': 3.0})
SQUARE_DIRECTIONS = 4
SQUARE_CHANNELS = tuple(f'n{direction}' for direction in range(SQUARE_DIRECTIONS))
SQUARE_STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0))
ARRIVALS = LatticeNeighborhood(tuple((-row, -column) for row, column in SQUARE_STEPS))
SQUARE_HEAD_ON = {0b0101: 0b1010, 0b1010: 0b0101}


def _rotated(code: int, turns: int) -> int:
    """Return the occupation code of a site with every particle turned.

    Every particle turns by a number of sixths of a turn.
    """
    turns %= DIRECTIONS
    return ((code << turns) | (code >> (DIRECTIONS - turns))) & (2**DIRECTIONS - 1)


def _collisions() -> npt.NDArray[np.intp]:
    """Return the occupation code after a collision.

    The code is given for every code and for both choices of the turn of a pair.
    """
    codes = np.arange(2**DIRECTIONS)
    following = np.stack([codes, codes], axis=1)
    for code in HEAD_ON:
        following[code] = (_rotated(code, 1), _rotated(code, -1))
    for code in TRIPLES:
        following[code] = _rotated(code, 1)
    return following.astype(np.intp)


COLLISIONS = _collisions()


def _count(*channels: Array) -> Array:
    """Return the number of particles at every site."""
    return np.sum(channels, axis=0)


class FHPSiteElement(Element):
    """Site of the FHP-I lattice gas.

    The site has one boolean channel per direction of the triangular lattice.

    The local rules of the gas collide the particles at every site, head-on pairs turning by a sixth of a turn to either
    side with equal probability and symmetric triples turning by a sixth of a turn, and then move every particle to the
    neighbouring site along its direction. Collisions conserve the number of particles and their momentum at every
    site. The state `density` holds the number of particles at the site.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n0:
            The initial occupation of the channel at 0 degrees.

        n1:
            The initial occupation of the channel at 60 degrees.

        n2:
            The initial occupation of the channel at 120 degrees.

        n3:
            The initial occupation of the channel at 180 degrees.

        n4:
            The initial occupation of the channel at 240 degrees.

        n5:
            The initial occupation of the channel at 300 degrees.

        density:
            The initial number of particles at the site.

    Examples:
        >>> from skcomplex.systems.physics.fluids import FHPSiteElement
        >>> FHPSiteElement('site').n0
        0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n0: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        n1: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        n2: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        n3: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        n4: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        n5: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        density: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.n5 = n5
        self.density = density


class FHPLatticeGasSystem(BaseSystem):
    """The FHP-I lattice gas on a periodic triangular lattice.

    A dense disc spreads into the gas.

    Every channel of every site is occupied at random with probability `density`, or `bump` inside a disc at the centre
    of the lattice, so the excess density spreads as a sound wave and then diffuses. The system has the channels
    `gas.n0` to `gas.n5` and the particle numbers `gas.density`.

    Args:
        shape:
            The number of rows and columns of the triangular lattice.

        density:
            The occupation probability of a channel outside the disc.

        bump:
            The occupation probability of a channel inside the disc.

        radius:
            The radius of the disc in lattice spacings, or `None` for a sixth of the smaller side.

        random_state:
            The seed or generator of the initial occupations.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import FHPLatticeGasSystem
        >>> FHPLatticeGasSystem((8, 8)).prepare().system_.elements[0].prepare().size_
        64
    """

    field = 'physics/fluids'
    topics = ('lattice', 'lattice gas', 'stochastic', 'conservation')
    references = ('Frisch, Hasslacher, and Pomeau 1986', 'Dilão 2023')
    until = 200
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        density: float = 0.2,
        bump: float = 0.6,
        radius: float | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.density = density
        self.bump = bump
        self.radius = radius
        self.random_state = random_state

    @staticmethod
    def _collide(*channels: Array, random: Array) -> tuple[Array, ...]:
        """Return the channels after the collisions of FHP-I at every site.

        The collisions are the head-on and symmetric three-particle ones.
        """
        codes = sum(np.asarray(channel, dtype=np.intp) << direction for direction, channel in enumerate(channels))
        following = COLLISIONS[codes, (random < TURN_PROBABILITY).astype(np.intp)]
        return tuple(((following >> direction) & 1).astype(float) for direction in range(DIRECTIONS))

    @staticmethod
    def _propagate(*columns: Array) -> tuple[Array, ...]:
        """Return the channels after every particle moves one site along its direction."""
        return tuple(columns[DIRECTIONS * direction + (direction + 3) % DIRECTIONS] for direction in range(DIRECTIONS))

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        space = LatticeSpace(tuple(self.shape), geometry='triangular').prepare()
        positions = space.positions_
        centre = positions.mean(axis=0)
        size = min(space.shape_) / 6 if self.radius is None else self.radius
        inside = np.linalg.norm(positions - centre, axis=1) < size
        probability = np.where(inside, self.bump, self.density)[:, None]
        occupied = (np.random.default_rng(self.random_state).random((space.size_, DIRECTIONS)) < probability).astype(
            float,
        )
        channels = {name: occupied[:, direction] for direction, name in enumerate(CHANNELS)}
        gas = GroupElement(
            'gas',
            element=FHPSiteElement,
            space=space,
            values={'density': occupied.sum(axis=1), **channels},
            visual=DENSITY_VISUAL,
            on='sites',
        )
        return CompositeSystem(
            self.label,
            elements=[gas],
            interactions=[
                LocalRuleInteraction(
                    'collision',
                    target='gas',
                    state=CHANNELS,
                    rule=self._collide,
                    neighborhood=SITE,
                    stochastic=True,
                ),
                LocalRuleInteraction(
                    'propagation',
                    target='gas',
                    state=CHANNELS,
                    rule=self._propagate,
                    neighborhood=TRIANGULAR,
                ),
                LocalRuleInteraction(
                    'count',
                    target='gas',
                    state='density',
                    rule=_count,
                    neighborhood=SITE,
                    reads=CHANNELS,
                ),
            ],
        )


class HPPLatticeGasSystem(BaseSystem):
    """The HPP lattice gas on a periodic square lattice.

    Its particles move along four directions and collide.

    Every site holds up to four particles, one per direction. Every step the particles first collide, where a site with
    exactly two particles of opposite directions turns them by a quarter turn and every other site keeps its particles,
    and then every particle moves one site along its direction. The rules are deterministic and reversible, and they
    conserve the number of particles and the momentum, and also the momentum along every row and every column, a
    spurious invariant that keeps the HPP gas from isotropic hydrodynamics. Every channel is occupied at random with
    probability `density`, or `bump` inside a disc at the centre. With `full`, as in Problem 4.3 of Lindgren, the
    particles start only in full cells, which hold all four directions, placed at random with probability `density`. The
    system has the channels `gas.n0` to `gas.n3` and the particle numbers `gas.density`.

    Args:
        shape:
            The number of rows and columns.

        density:
            The occupation probability of a channel outside the disc, or of a full cell with `full`.

        bump:
            The occupation probability of a channel inside the disc, unused with `full`.

        radius:
            The radius of the disc in lattice spacings, or `None` for a sixth of the smaller side.

        full:
            Whether the particles start in full cells only.

        random_state:
            The seed or generator of the initial occupations.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import HPPLatticeGasSystem
        >>> HPPLatticeGasSystem((8, 8)).prepare().system_.elements[0].prepare().size_
        64
    """

    field = 'physics/fluids'
    topics = ('lattice', 'lattice gas', 'deterministic', 'conservation', 'reversible')
    references = ('Hardy, de Pazzis, and Pomeau 1976', 'Lindgren 2024')
    until = 200
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        density: float = 0.2,
        bump: float = 0.6,
        radius: float | None = None,
        full: bool = False,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.density = density
        self.bump = bump
        self.radius = radius
        self.full = full
        self.random_state = random_state

    @staticmethod
    def _collide_square(*channels: Array) -> tuple[Array, ...]:
        """Return the HPP channels after every head-on pair turns by a quarter turn.

        This is the only collision.
        """
        codes = np.sum(
            [np.asarray(channel, dtype=np.intp) << direction for direction, channel in enumerate(channels)],
            axis=0,
        )
        following = codes.copy()
        for before, after in SQUARE_HEAD_ON.items():
            following = np.where(codes == before, after, following)
        return tuple(((following >> direction) & 1).astype(float) for direction in range(SQUARE_DIRECTIONS))

    @staticmethod
    def _propagate_square(*columns: Array) -> tuple[Array, ...]:
        """Return the channels of the HPP gas after every particle moves one site.

        Every particle moves along its direction.
        """
        return tuple(columns[SQUARE_DIRECTIONS * direction + direction] for direction in range(SQUARE_DIRECTIONS))

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        space = LatticeSpace(tuple(self.shape), boundary='periodic').prepare()
        rng = np.random.default_rng(self.random_state)
        if self.full:
            occupied = np.repeat((rng.random((space.size_, 1)) < self.density).astype(float), SQUARE_DIRECTIONS, axis=1)
        else:
            rows, columns = np.indices(tuple(self.shape))
            positions = np.column_stack([rows.ravel(), columns.ravel()]).astype(float)
            size = min(self.shape) / 6 if self.radius is None else self.radius
            inside = np.linalg.norm(positions - positions.mean(axis=0), axis=1) < size
            probability = np.where(inside, self.bump, self.density)[:, None]
            occupied = (rng.random((space.size_, SQUARE_DIRECTIONS)) < probability).astype(float)
        channels = {name: occupied[:, direction] for direction, name in enumerate(SQUARE_CHANNELS)}
        gas = GroupElement(
            'gas',
            element=HPPSiteElement,
            space=space,
            values={'density': occupied.sum(axis=1), **channels},
            visual=DENSITY_VISUAL,
            on='sites',
        )
        return CompositeSystem(
            self.label,
            elements=[gas],
            interactions=[
                LocalRuleInteraction(
                    'collision',
                    target='gas',
                    state=SQUARE_CHANNELS,
                    rule=self._collide_square,
                    neighborhood=SITE,
                ),
                LocalRuleInteraction(
                    'propagation',
                    target='gas',
                    state=SQUARE_CHANNELS,
                    rule=self._propagate_square,
                    neighborhood=ARRIVALS,
                ),
                LocalRuleInteraction(
                    'count',
                    target='gas',
                    state='density',
                    rule=_count,
                    neighborhood=SITE,
                    reads=SQUARE_CHANNELS,
                ),
            ],
        )
