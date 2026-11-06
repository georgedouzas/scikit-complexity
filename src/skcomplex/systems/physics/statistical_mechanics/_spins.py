"""Statistical mechanics systems: gases, fluids, and spin systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.components.elements import LatticeElement
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ..._catalog import register_system
from ..common.spins import IsingSamplingInteraction, PottsSamplingInteraction, XYSamplingInteraction
from ._ising_spin import IsingSpinElement
from ._potts_spin import PottsSpinElement
from ._xy_spin import XYSpinElement


def _square(size: int | Sequence[int]) -> tuple[int, ...]:
    """Return the shape of a square lattice of a side, or a given shape."""
    if isinstance(size, Sequence):
        return tuple(int(item) for item in size)
    return (int(size), int(size))


@register_system(
    field='physics/statistical_mechanics',
    tags=('lattice', 'Monte Carlo', 'phase transition', 'spins'),
    references=('Ising 1925', 'Onsager 1944', 'Jensen 2023'),
    defaults={'until': 200, 'dt': 1},
)
def ising(
    size: int | Sequence[int] = 64,
    *,
    temperature: float = 2.0,
    J: float = 1.0,
    h: float = 0.0,
    initial: str = 'random',
    sampler: str = 'metropolis',
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the two-dimensional Ising system on a torus, ordered below the critical temperature `2 / ln(1 + sqrt 2)`.

    One unit of time is one sweep of checkerboard updates.

    Args:
        size:
            The side of the square lattice, or its shape.

        temperature:
            The temperature in units of the coupling, which can change during a run.

        J:
            The coupling between neighbouring spins.

        h:
            The external field.

        initial:
            `'random'` for random spins or `'up'` for all spins up.

        sampler:
            The acceptance rule, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        random_state:
            The seed or generator of the random initial spins.

    Returns:
        system:
            The system, with the spins `spins.spin`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ising
        >>> ising(8, temperature=1.5).elements[0].values['temperature']
        1.5
    """
    shape = _square(size)
    spins = np.ones(shape) if initial == 'up' else np.random.default_rng(random_state).choice([-1.0, 1.0], size=shape)
    space = LatticeSpace(shape)
    group = LatticeElement(
        'spins',
        element=IsingSpinElement,
        space=space,
        values={'spin': spins, 'J': J, 'h': h, 'temperature': temperature},
    )
    return System(elements=[group], interactions=[IsingSamplingInteraction('sampling', target='spins', rule=sampler)])


@register_system(
    field='physics/statistical_mechanics',
    tags=('lattice', 'Monte Carlo', 'phase transition', 'spins'),
    references=('Potts 1952', 'Wu 1982'),
    defaults={'until': 200, 'dt': 1},
)
def potts(
    size: int | Sequence[int] = 64,
    *,
    q: int = 3,
    temperature: float = 1.0,
    J: float = 1.0,
    sampler: str = 'heat-bath',
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the q-state Potts system on a torus, ordered below the critical temperature `1 / ln(1 + sqrt q)`.

    Args:
        size:
            The side of the square lattice, or its shape.

        q:
            The number of states.

        temperature:
            The temperature in units of the coupling.

        J:
            The coupling between neighbouring spins.

        sampler:
            The acceptance rule, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        random_state:
            The seed or generator of the random initial states.

    Returns:
        system:
            The system, with the states `spins.color`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import potts
        >>> potts(8, q=4).interactions[0].q
        4
    """
    shape = _square(size)
    colors = np.random.default_rng(random_state).integers(0, q, size=shape).astype(float)
    space = LatticeSpace(shape)
    group = LatticeElement(
        'spins',
        element=PottsSpinElement,
        space=space,
        values={'color': colors, 'J': J, 'temperature': temperature},
    )
    return System(
        elements=[group],
        interactions=[PottsSamplingInteraction('sampling', target='spins', q=q, rule=sampler)],
    )


@register_system(
    field='physics/statistical_mechanics',
    tags=('lattice', 'Monte Carlo', 'vortices', 'spins'),
    references=('Kosterlitz and Thouless 1973', 'Jensen 2023'),
    defaults={'until': 200, 'dt': 1},
)
def xy(
    size: int | Sequence[int] = 64,
    *,
    temperature: float = 0.5,
    J: float = 1.0,
    width: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the XY system of planar spins on a torus, with vortices that bind below the Kosterlitz-Thouless point.

    Args:
        size:
            The side of the square lattice, or its shape.

        temperature:
            The temperature in units of the coupling.

        J:
            The coupling between neighbouring spins.

        width:
            The largest change of a proposed rotation.

        random_state:
            The seed or generator of the random initial angles.

    Returns:
        system:
            The system, with the angles `spins.theta`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import xy
        >>> xy(8).interactions[0].width
        1.0
    """
    shape = _square(size)
    angles = np.random.default_rng(random_state).uniform(0, 2 * np.pi, size=shape)
    space = LatticeSpace(shape)
    group = LatticeElement(
        'spins',
        element=XYSpinElement,
        space=space,
        values={'theta': angles, 'J': J, 'temperature': temperature},
    )
    return System(elements=[group], interactions=[XYSamplingInteraction('sampling', target='spins', width=width)])
