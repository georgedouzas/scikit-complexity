"""Statistical mechanics models: gases, fluids, and spin models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
from sage.all__sagemath_symbolics import cos

from ....core.base import Domain, parameter, state
from ....core.model import (
    Element,
    Energy,
    LatticeSpace,
    Law,
    Move,
    Neighborhood,
    Sample,
    Sampler,
    Symbols,
    System,
    neighbors,
    where,
)
from ....core.runtime import register_model

NEAREST = Neighborhood.von_neumann()


class IsingSpin(Element):
    """Spin of the Ising model, up at 1 or down at -1, with local energy `-J s sum(neighbours) - h s`.

    Args:
        label:
            The label of the spin or the lattice group.

        spin:
            The initial spin, 1 or -1.

        J:
            The coupling between neighbouring spins.

        h:
            The external field.

        temperature:
            The temperature in units of the coupling.

        sampler:
            The acceptance rule, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import IsingSpin
        >>> IsingSpin('spin').temperature
        2.0
    """

    spin: float = state(default=1.0, domain=Domain.INTEGER)
    J: float = parameter(default=1.0)
    h: float = parameter(default=0.0)
    temperature: float = parameter(default=2.0, domain=Domain.POSITIVE)
    sampler: str = 'metropolis'

    def laws(self, s: Symbols) -> list[Law]:
        """Return the local energy, the spin flip, and the sampler.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        energy = -s.J * s.spin * sum(neighbors(s.spin, NEAREST)) - s.h * s.spin
        return [Sample(Energy(energy, NEAREST), Move.flip(s.spin), Sampler(self.sampler, s.temperature))]


class PottsSpin(Element):
    """Spin of the q-state Potts model, with local energy `-J` times the number of neighbours in the same state.

    Args:
        label:
            The label of the spin or the lattice group.

        color:
            The initial state from 0 to `q - 1`.

        J:
            The coupling between neighbouring spins.

        temperature:
            The temperature in units of the coupling.

        q:
            The number of states.

        sampler:
            The acceptance rule, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import PottsSpin
        >>> PottsSpin('spin').q
        3
    """

    color: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    J: float = parameter(default=1.0)
    temperature: float = parameter(default=1.0, domain=Domain.POSITIVE)
    q: int = 3
    sampler: str = 'heat-bath'

    def laws(self, s: Symbols) -> list[Law]:
        """Return the local energy, the random value move, and the sampler.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        energy = -s.J * sum(where(s.color == item, 1, 0) for item in neighbors(s.color, NEAREST))
        return [
            Sample(
                Energy(energy, NEAREST),
                Move.random_value(s.color, range(self.q)),
                Sampler(self.sampler, s.temperature),
            ),
        ]


class XYSpin(Element):
    """Planar spin of the XY model, an angle with local energy `-J sum(cos(theta - neighbours))`.

    Args:
        label:
            The label of the spin or the lattice group.

        theta:
            The initial angle.

        J:
            The coupling between neighbouring spins.

        temperature:
            The temperature in units of the coupling.

        width:
            The largest change of a proposed rotation.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import XYSpin
        >>> XYSpin('spin').width
        1.0
    """

    theta: float = state(default=0.0, domain=Domain.ANGLE)
    J: float = parameter(default=1.0)
    temperature: float = parameter(default=0.5, domain=Domain.POSITIVE)
    width: float = 1.0

    def laws(self, s: Symbols) -> list[Law]:
        """Return the local energy, the rotation, and the Metropolis sampler.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        energy = -s.J * sum(cos(s.theta - item) for item in neighbors(s.theta, NEAREST))
        return [Sample(Energy(energy, NEAREST), Move.rotate(s.theta, self.width), Sampler('metropolis', s.temperature))]


def _square(size: int | Sequence[int]) -> tuple[int, ...]:
    """Return the shape of a square lattice of a side, or a given shape."""
    if isinstance(size, Sequence):
        return tuple(int(item) for item in size)
    return (int(size), int(size))


@register_model(
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
    """Return the two-dimensional Ising model on a torus, ordered below the critical temperature `2 / ln(1 + sqrt 2)`.

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
            The model, with the spins `spins.spin`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ising
        >>> ising(8, temperature=1.5).elements[0].values['temperature']
        1.5
    """
    shape = _square(size)
    spins = np.ones(shape) if initial == 'up' else np.random.default_rng(random_state).choice([-1.0, 1.0], size=shape)
    space = LatticeSpace(shape)
    group = IsingSpin.lattice('spins', space, spin=spins, J=J, h=h, temperature=temperature, sampler=sampler)
    return System(components=[group])


@register_model(
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
    """Return the q-state Potts model on a torus, ordered below the critical temperature `1 / ln(1 + sqrt q)`.

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
            The model, with the states `spins.color`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import potts
        >>> potts(8, q=4).elements[0].values['q']
        4
    """
    shape = _square(size)
    colors = np.random.default_rng(random_state).integers(0, q, size=shape).astype(float)
    space = LatticeSpace(shape)
    group = PottsSpin.lattice('spins', space, color=colors, J=J, temperature=temperature, q=q, sampler=sampler)
    return System(components=[group])


@register_model(
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
    """Return the XY model of planar spins on a torus, with vortices that bind below the Kosterlitz-Thouless point.

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
            The model, with the angles `spins.theta`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import xy
        >>> xy(8).elements[0].values['width']
        1.0
    """
    shape = _square(size)
    angles = np.random.default_rng(random_state).uniform(0, 2 * np.pi, size=shape)
    space = LatticeSpace(shape)
    group = XYSpin.lattice('spins', space, theta=angles, J=J, temperature=temperature, width=width)
    return System(components=[group])
