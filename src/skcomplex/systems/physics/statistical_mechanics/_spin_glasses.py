"""Lattice spin glasses: Ising spins with quenched bond couplings, the Edwards-Anderson and Mattis systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import LatticeElement
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, VisualSpec
from ..._catalog import register_system
from ._bond_sampling import BondSamplingInteraction
from ._bond_spin import BondSpinElement
from ._fully_connected_spins import SPIN_IMAGE
from ._mattis_spin import MattisSpinElement


def _square(size: int | Sequence[int]) -> tuple[int, int]:
    """Return the shape of a square lattice of a side, or a given shape of two sides."""
    if isinstance(size, Sequence):
        rows, columns = (int(item) for item in size)
        return rows, columns
    return int(size), int(size)


def _bond_states(
    right: npt.NDArray[np.floating],
    down: npt.NDArray[np.floating],
) -> dict[str, npt.NDArray[np.floating]]:
    """Return the couplings of every site to its up, left, right, and down neighbours from the bonds right and down."""
    return {
        'J_up': np.roll(down, 1, axis=0),
        'J_left': np.roll(right, 1, axis=1),
        'J_right': right,
        'J_down': down,
    }


def _overlap(member: SimpleNamespace) -> Expression:
    """Return the overlap of a spin with its quenched pattern."""
    return member.spin * member.pattern


def _magnetization_observables(system: System) -> list[ObservableSpec]:
    """Return the magnetization of the spins."""
    return [ObservableSpec('magnetization', group='spins', variable='spin')]


@register_system(
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
            The system, with the spins `spins.spin`, the couplings `spins.J_right` and `spins.J_down` of the bonds to
            the right and down, and the observable `magnetization`.

    Raises:
        ValidationError:
            The disorder is neither `'bimodal'` nor `'gaussian'`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import edwards_anderson
        >>> sorted(set(edwards_anderson(8).elements[0].values['J_right'].ravel().tolist()))
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
    group = LatticeElement(
        'spins',
        element=BondSpinElement,
        space=LatticeSpace(shape),
        values={'spin': spins, 'h': h, 'temperature': temperature, **_bond_states(right, down)},
        visual=SPIN_IMAGE,
    )
    return System(elements=[group], interactions=[BondSamplingInteraction('sampling', target='spins')])


def _pattern_observables(system: System) -> list[ObservableSpec]:
    """Return the magnetization of the spins and their overlap with the pattern."""
    return [
        ObservableSpec('magnetization', group='spins', variable='spin'),
        ObservableSpec('overlap', _overlap, group='spins'),
    ]


@register_system(
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

    The gauge change `sigma_i = xi_i s_i` maps the system at `h = 0` to the Ising system, so the overlap
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
            The system, with the spins `spins.spin`, the pattern `spins.pattern`, and the observables `magnetization`
            and `overlap`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.physics.statistical_mechanics import mattis_spin_glass
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
    group = LatticeElement(
        'spins',
        element=MattisSpinElement,
        space=LatticeSpace(shape),
        values={'spin': spins, 'pattern': pattern, 'h': h, 'temperature': temperature, **_bond_states(right, down)},
        visual=VisualSpec('image', {'states': ()}),
    )
    return System(elements=[group], interactions=[BondSamplingInteraction('sampling', target='spins')])
