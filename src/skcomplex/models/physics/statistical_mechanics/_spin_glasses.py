"""Lattice spin glasses: Ising spins with quenched bond couplings, the Edwards-Anderson and Mattis models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, Expression, ValidationError, VisualSpec, parameter, state
from ....core.components import Element
from ....core.laws import Energy, Law, Move, Sample, Sampler, Symbols, neighbors
from ....core.runtime import register_model
from ....core.space import LatticeSpace, Neighborhood
from ....core.system import Observable, System
from ._fully_connected_spins import SPIN_IMAGE

type Couplings = npt.NDArray[np.floating]

NEAREST = Neighborhood.von_neumann()


def _square(size: int | Sequence[int]) -> tuple[int, int]:
    """Return the shape of a square lattice of a side, or a given shape of two sides."""
    if isinstance(size, Sequence):
        rows, columns = (int(item) for item in size)
        return rows, columns
    return int(size), int(size)


def _bond_states(right: Couplings, down: Couplings) -> dict[str, Couplings]:
    """Return the couplings of every site to its up, left, right, and down neighbours from the bonds right and down."""
    return {
        'J_up': np.roll(down, 1, axis=0),
        'J_left': np.roll(right, 1, axis=1),
        'J_right': right,
        'J_down': down,
    }


def _overlap(member: Symbols) -> Expression:
    """Return the overlap of a spin with its quenched pattern."""
    return member.spin * member.pattern


class BondSpin(Element):
    """Ising spin on a square lattice with quenched couplings to its four neighbours, of local energy `-s sum J_k s_k`.

    The couplings are constant states of every site, so a bond between two sites is stored at both of them with the same
    value. The spins flip by Metropolis Monte Carlo at the temperature `T`, with the energy
    `-sum_<ij> J_ij s_i s_j - h sum_i s_i`.

    Args:
        label:
            The label of the spin or the lattice group.

        spin:
            The initial spin, 1 or -1.

        J_up:
            The coupling to the upper neighbour.

        J_left:
            The coupling to the left neighbour.

        J_right:
            The coupling to the right neighbour.

        J_down:
            The coupling to the lower neighbour.

        h:
            The external field.

        temperature:
            The temperature.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import BondSpin
        >>> BondSpin('spin', J_left=-1.0).J_left
        -1.0
    """

    spin: float = state(default=1.0, domain=Domain.INTEGER)
    J_up: float = state(default=1.0)
    J_left: float = state(default=1.0)
    J_right: float = state(default=1.0)
    J_down: float = state(default=1.0)
    h: float = parameter(default=0.0)
    temperature: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the local energy, the spin flip, and the Metropolis sampler.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        couplings = (s.J_up, s.J_left, s.J_right, s.J_down)
        field = sum(coupling * other for coupling, other in zip(couplings, neighbors(s.spin, NEAREST), strict=True))
        return [
            Sample(Energy(-s.spin * (field + s.h), NEAREST), Move.flip(s.spin), Sampler('metropolis', s.temperature)),
        ]


class MattisSpin(BondSpin):
    """Spin of the Mattis model, a bond spin that also holds its quenched pattern `xi`, with couplings `J xi_i xi_j`.

    Args:
        label:
            The label of the spin or the lattice group.

        spin:
            The initial spin, 1 or -1.

        pattern:
            The quenched pattern `xi`, 1 or -1.

        J_up:
            The coupling to the upper neighbour.

        J_left:
            The coupling to the left neighbour.

        J_right:
            The coupling to the right neighbour.

        J_down:
            The coupling to the lower neighbour.

        h:
            The external field.

        temperature:
            The temperature.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import MattisSpin
        >>> MattisSpin('spin', pattern=-1.0).pattern
        -1.0
    """

    pattern: float = state(default=1.0, domain=Domain.INTEGER)


def _magnetization_observables(system: System) -> list[Observable]:
    """Return the magnetization of the spins."""
    return [Observable('magnetization', group='spins', variable='spin')]


@register_model(
    field='physics/statistical_mechanics',
    tags=('spin glass', 'disorder', 'lattice', 'Monte Carlo', 'spins'),
    references=('Edwards and Anderson 1975', 'Bertin 2021', 'Binder and Young 1986'),
    defaults={'until': 500, 'dt': 5},
    observables=_magnetization_observables,
)
def edwards_anderson(
    size: int | Sequence[int] = 48,
    *,
    temperature: float = 0.8,
    J: float = 1.0,
    h: float = 0.0,
    disorder: str = 'bimodal',
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Edwards-Anderson spin glass on a periodic square lattice, with quenched random bonds.

    Every bond takes the coupling `+J` or `-J` with equal probability, or a Gaussian coupling of mean 0 and standard
    deviation `J`, drawn once. The magnetization stays near zero at every temperature, while at low temperature the
    spins freeze in random directions and the energy per spin approaches the ground-state energy, about `-1.40 J` for
    `+J` and `-J` bonds. One unit of time is one sweep of Metropolis updates.

    Args:
        size:
            The side of the square lattice, or its two sides.

        temperature:
            The temperature, which can change during a run.

        J:
            The scale of the couplings.

        h:
            The external field.

        disorder:
            `'bimodal'` for couplings `+J` and `-J`, or `'gaussian'` for Gaussian couplings.

        random_state:
            The seed or generator of the couplings and of the random initial spins.

    Returns:
        system:
            The model, with the spins `spins.spin`, the couplings `spins.J_right` and `spins.J_down` of the bonds to the
            right and down, and the observable `magnetization`.

    Raises:
        ValidationError:
            The disorder is neither `'bimodal'` nor `'gaussian'`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import edwards_anderson
        >>> sorted(set(edwards_anderson(8).elements[0].values['J_right'].tolist()))
        [-1.0, 1.0]
    """
    shape = _square(size)
    rng = np.random.default_rng(random_state)
    if disorder == 'bimodal':
        right, down = J * rng.choice([-1.0, 1.0], size=(2, *shape))
    elif disorder == 'gaussian':
        right, down = J * rng.standard_normal((2, *shape))
    else:
        error_msg = f"The disorder is 'bimodal' or 'gaussian', not {disorder!r}."
        raise ValidationError(error_msg)
    spins = rng.choice([-1.0, 1.0], size=shape)
    group = BondSpin.lattice(
        'spins',
        LatticeSpace(shape),
        visual=SPIN_IMAGE,
        spin=spins,
        h=h,
        temperature=temperature,
        **_bond_states(right, down),
    )
    return System(components=[group])


def _pattern_observables(system: System) -> list[Observable]:
    """Return the magnetization of the spins and their overlap with the pattern."""
    return [
        Observable('magnetization', group='spins', variable='spin'),
        Observable('overlap', _overlap, group='spins'),
    ]


@register_model(
    field='physics/statistical_mechanics',
    tags=('spin glass', 'disorder', 'lattice', 'Monte Carlo', 'spins'),
    references=('Mattis 1976', 'Bertin 2021'),
    defaults={'until': 300, 'dt': 2},
    observables=_pattern_observables,
)
def mattis_spin_glass(
    size: int | Sequence[int] = 48,
    *,
    temperature: float = 1.5,
    J: float = 1.0,
    h: float = 0.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Mattis spin glass on a periodic square lattice, with couplings `J_ij = J xi_i xi_j` of a pattern `xi`.

    The gauge change `sigma_i = xi_i s_i` maps the model at `h = 0` to the Ising model, so the overlap
    `m_xi = sum_i xi_i s_i / N` orders below the Ising critical temperature `2 J / ln(1 + sqrt 2)` while the
    magnetization averages to zero over the disorder. One unit of time is one sweep of Metropolis updates.

    Args:
        size:
            The side of the square lattice, or its two sides.

        temperature:
            The temperature, which can change during a run.

        J:
            The coupling.

        h:
            The external field.

        random_state:
            The seed or generator of the pattern and of the random initial spins.

    Returns:
        system:
            The model, with the spins `spins.spin`, the pattern `spins.pattern`, and the observables `magnetization`
            and `overlap`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.physics.statistical_mechanics import mattis_spin_glass
        >>> values = mattis_spin_glass(8).elements[0].values
        >>> pattern = values['pattern'].reshape(8, 8)
        >>> bool(np.array_equal(values['J_right'].reshape(8, 8), pattern * np.roll(pattern, -1, axis=1)))
        True
    """
    shape = _square(size)
    rng = np.random.default_rng(random_state)
    pattern = rng.choice([-1.0, 1.0], size=shape)
    right = J * pattern * np.roll(pattern, -1, axis=1)
    down = J * pattern * np.roll(pattern, -1, axis=0)
    spins = rng.choice([-1.0, 1.0], size=shape)
    group = MattisSpin.lattice(
        'spins',
        LatticeSpace(shape),
        visual=VisualSpec('image', {'states': ()}),
        spin=spins,
        pattern=pattern,
        h=h,
        temperature=temperature,
        **_bond_states(right, down),
    )
    return System(components=[group])
