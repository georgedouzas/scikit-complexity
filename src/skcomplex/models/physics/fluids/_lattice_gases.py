"""Lattice gases: the FHP gas of particles that move along the six directions of a triangular lattice and collide."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, state
from ....core.components import Element
from ....core.laws import Law, LocalRule, Symbols
from ....core.runtime import register_model
from ....core.space import LatticeSpace, Neighborhood
from ....core.system import System

type Array = npt.NDArray[np.float64]

DIRECTIONS = 6
CHANNELS = tuple(f'n{direction}' for direction in range(DIRECTIONS))
SITE = Neighborhood(((0, 0),))
TRIANGULAR = Neighborhood.triangular()
VELOCITIES = np.column_stack(
    [np.cos(np.arange(DIRECTIONS) * np.pi / 3), np.sin(np.arange(DIRECTIONS) * np.pi / 3)],
)
HEAD_ON = (0b001001, 0b010010, 0b100100)
TRIPLES = (0b010101, 0b101010)
DENSITY_VISUAL = VisualSpec('sites', {'states': ('density',), 'low': 0.0, 'high': 3.0})


def _rotated(code: int, turns: int) -> int:
    """Return the occupation code of a site with every particle turned by a number of sixths of a turn."""
    turns %= DIRECTIONS
    return ((code << turns) | (code >> (DIRECTIONS - turns))) & (2**DIRECTIONS - 1)


def _collisions() -> npt.NDArray[np.intp]:
    """Return the occupation code after a collision, for every code and for both choices of the turn of a pair."""
    codes = np.arange(2**DIRECTIONS)
    following = np.stack([codes, codes], axis=1)
    for code in HEAD_ON:
        following[code] = (_rotated(code, 1), _rotated(code, -1))
    for code in TRIPLES:
        following[code] = _rotated(code, 1)
    return following.astype(np.intp)


COLLISIONS = _collisions()


def _collide(*channels: Array, random: Array) -> tuple[Array, ...]:
    """Return the channels after the head-on and symmetric three-particle collisions of FHP-I at every site."""
    codes = sum(np.asarray(channel, dtype=np.intp) << direction for direction, channel in enumerate(channels))
    following = COLLISIONS[codes, (random < 0.5).astype(np.intp)]  # noqa: PLR2004
    return tuple(((following >> direction) & 1).astype(float) for direction in range(DIRECTIONS))


def _propagate(*columns: Array) -> tuple[Array, ...]:
    """Return the channels after every particle moves one site along its direction."""
    return tuple(columns[DIRECTIONS * direction + (direction + 3) % DIRECTIONS] for direction in range(DIRECTIONS))


def _count(*channels: Array) -> Array:
    """Return the number of particles at every site."""
    return np.sum(channels, axis=0)


class FHPSite(Element):
    """Site of the FHP-I lattice gas, with one boolean channel per direction of the triangular lattice.

    Every step collides the particles at every site, head-on pairs turning by a sixth of a turn to either side with
    equal probability and symmetric triples turning by a sixth of a turn, and then moves every particle to the
    neighbouring site along its direction. Collisions conserve the number of particles and their momentum at every
    site. The state `density` holds the number of particles at the site.

    Args:
        label:
            The label of the site or the lattice group.

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
        >>> from skcomplex.models.physics.fluids import FHPSite
        >>> FHPSite('site').n0
        0
    """

    n0: float = state(default=0, domain=Domain.BOOLEAN)
    n1: float = state(default=0, domain=Domain.BOOLEAN)
    n2: float = state(default=0, domain=Domain.BOOLEAN)
    n3: float = state(default=0, domain=Domain.BOOLEAN)
    n4: float = state(default=0, domain=Domain.BOOLEAN)
    n5: float = state(default=0, domain=Domain.BOOLEAN)
    density: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the collision, the propagation along the six directions, and the count of particles.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        channels = tuple(getattr(s, name) for name in CHANNELS)
        return [
            LocalRule(channels, SITE, _collide, stochastic=True),
            LocalRule(channels, TRIANGULAR, _propagate),
            LocalRule(s.density, SITE, _count, reads=channels),
        ]


@register_model(
    field='physics/fluids',
    tags=('lattice', 'lattice gas', 'stochastic', 'conservation'),
    references=('Frisch, Hasslacher, and Pomeau 1986', 'Dilão 2023'),
    defaults={'until': 200, 'dt': 1},
)
def fhp_lattice_gas(
    shape: Sequence[int] = (64, 64),
    *,
    density: float = 0.2,
    bump: float = 0.6,
    radius: float | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the FHP-I lattice gas on a periodic triangular lattice, with a dense disc that spreads into the gas.

    Every channel of every site is occupied at random with probability `density`, or `bump` inside a disc at the centre
    of the lattice, so the excess density spreads as a sound wave and then diffuses.

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

    Returns:
        system:
            The model, with the channels `gas.n0` to `gas.n5` and the particle numbers `gas.density`.

    Examples:
        >>> from skcomplex.models.physics.fluids import fhp_lattice_gas
        >>> fhp_lattice_gas((8, 8)).elements[0].size
        64
    """
    space = LatticeSpace(tuple(shape), geometry='triangular')
    positions = space.positions
    centre = positions.mean(axis=0)
    size = min(space.dimensions) / 6 if radius is None else radius
    inside = np.linalg.norm(positions - centre, axis=1) < size
    probability = np.where(inside, bump, density)[:, None]
    occupied = (np.random.default_rng(random_state).random((space.size, DIRECTIONS)) < probability).astype(float)
    channels = {name: occupied[:, direction] for direction, name in enumerate(CHANNELS)}
    gas = FHPSite.lattice('gas', space, visual=DENSITY_VISUAL, density=occupied.sum(axis=1), **channels)
    return System(components=[gas])
