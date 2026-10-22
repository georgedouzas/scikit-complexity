"""Fully connected spin models with Glauber dynamics: the Curie-Weiss magnet and the Sherrington-Kirkpatrick glass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, Expression, VisualSpec, parameter, state
from ....core.components import Element
from ....core.laws import Jump, Law, Symbols, count, mean, total, where
from ....core.runtime import register_model
from ....core.space import LatticeSpace
from ....core.system import Observable, System

type Array = npt.NDArray[np.float64]

SPIN_IMAGE = VisualSpec('image', {'states': ('spin',), 'low': -1.0, 'high': 1.0})


def _initial_spins(n_spins: int, initial: str, random_state: int | np.random.Generator | None) -> Array:
    """Return all spins up, or random spins, for `initial` set to `'up'` or `'random'`."""
    if initial == 'up':
        return np.ones(n_spins)
    return np.random.default_rng(random_state).choice([-1.0, 1.0], size=n_spins).astype(np.float64)


def _magnetization() -> Observable:
    """Return the observable of the mean spin of the group `spins`."""
    return Observable('magnetization', group='spins', variable='spin')


class FullyConnectedSpin(Element):
    """Ising spin coupled to every other spin, flipped at the Glauber rate `1 / (1 + exp(2 s h / T))`.

    The local field is `h = J (sum_{j != i} s_j) / N + h_ext` of the Curie-Weiss energy `-(J / N) sum_{i<j} s_i s_j`,
    or `h = J sum_j C_ij s_j + h_ext` with a symmetric coupling matrix `C` of zero diagonal. The rate satisfies
    detailed balance with the Boltzmann distribution at the temperature `T`.

    Args:
        label:
            The label of the spin or the group.

        spin:
            The initial spin, 1 or -1.

        J:
            The scale of the couplings.

        h:
            The external field.

        temperature:
            The temperature.

        couplings:
            The coupling matrix `C`, one row and one column per spin, or `None` for the uniform couplings `1 / N`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import FullyConnectedSpin
        >>> FullyConnectedSpin('spin').temperature
        0.5
    """

    spin: float = state(default=1.0, domain=Domain.INTEGER)
    J: float = parameter(default=1.0)
    h: float = parameter(default=0.0)
    temperature: float = parameter(default=0.5, domain=Domain.POSITIVE)
    couplings: npt.ArrayLike | None = None

    def _field(self, s: Symbols) -> Expression:
        """Return the local field of a spin, without its own contribution."""
        if self.couplings is None:
            return s.J * (mean(s.spin) - s.spin / count()) + s.h
        return s.J * total(s.spin, weights=np.asarray(self.couplings, dtype=float)) + s.h

    def laws(self, s: Symbols) -> list[Law]:
        """Return the flips of up spins and of down spins at the Glauber rate.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        rate = 1 / (1 + exp(2 * s.spin * self._field(s) / s.temperature))
        return [Jump(where(s.spin > 0, rate, 0), {s.spin: -2}), Jump(where(s.spin < 0, rate, 0), {s.spin: 2})]


@register_model(
    field='physics/statistical_mechanics',
    tags=('mean field', 'phase transition', 'spins', 'jump process'),
    references=('Bertin 2021', 'Glauber 1963'),
    defaults={'until': 30.0, 'dt': 0.1},
    observables=lambda system: [_magnetization()],
)
def curie_weiss(
    n_spins: int = 500,
    *,
    temperature: float = 0.5,
    J: float = 1.0,
    h: float = 0.0,
    initial: str = 'random',
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the fully connected Ising model with Glauber dynamics, ordered below the critical temperature `T_c = J`.

    The energy is `-(J / N) sum_{i<j} s_i s_j - h sum_i s_i`. In the large system limit the magnetization `m` solves
    `m = tanh((J m + h) / T)`, which has a nonzero root below `T_c` when `h = 0`. One unit of time is one attempted
    flip per spin. The spins are drawn in a row, which orders the drawing only.

    Args:
        n_spins:
            The number of spins.

        temperature:
            The temperature, which can change during a run.

        J:
            The coupling.

        h:
            The external field.

        initial:
            `'random'` for random spins or `'up'` for all spins up.

        random_state:
            The seed or generator of the random initial spins.

    Returns:
        system:
            The model, with the spins `spins.spin` and the observable `magnetization`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import curie_weiss
        >>> curie_weiss(10).elements[0].size
        10
    """
    spins = _initial_spins(n_spins, initial, random_state)
    group = FullyConnectedSpin.lattice(
        'spins',
        LatticeSpace(n_spins),
        visual=SPIN_IMAGE,
        spin=spins,
        J=J,
        h=h,
        temperature=temperature,
    )
    return System(components=[group])


@register_model(
    field='physics/statistical_mechanics',
    tags=('spin glass', 'disorder', 'mean field', 'spins', 'jump process'),
    references=('Sherrington and Kirkpatrick 1975', 'Bertin 2021', 'Mezard, Parisi, and Virasoro 1987'),
    defaults={'until': 50.0, 'dt': 0.5},
    observables=lambda system: [_magnetization()],
)
def sherrington_kirkpatrick(
    n_spins: int = 200,
    *,
    temperature: float = 0.5,
    J: float = 1.0,
    h: float = 0.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Sherrington-Kirkpatrick spin glass with Glauber dynamics, frozen below the temperature `T_c = J`.

    The energy is `-sum_{i<j} J_ij s_i s_j - h sum_i s_i` with quenched Gaussian couplings `J_ij` of mean 0 and
    variance `J^2 / N`, drawn once. Above `T_c` the energy per spin is `-J^2 / (2 T)`, and below it the spins freeze
    in random directions, with no magnetization. The spins are drawn in a row, which orders the drawing only.

    Args:
        n_spins:
            The number of spins.

        temperature:
            The temperature, which can change during a run.

        J:
            The standard deviation of the couplings times the square root of the number of spins.

        h:
            The external field.

        random_state:
            The seed or generator of the couplings and of the random initial spins.

    Returns:
        system:
            The model, with the spins `spins.spin`, the coupling matrix of unit scale in the values of the group, and
            the observable `magnetization`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import sherrington_kirkpatrick
        >>> couplings = sherrington_kirkpatrick(10).elements[0].values['couplings']
        >>> bool((couplings == couplings.T).all()), float(couplings.trace())
        (True, 0.0)
    """
    rng = np.random.default_rng(random_state)
    upper = np.triu(rng.normal(0.0, 1.0 / np.sqrt(n_spins), (n_spins, n_spins)), 1)
    spins = _initial_spins(n_spins, 'random', rng)
    group = FullyConnectedSpin.lattice(
        'spins',
        LatticeSpace(n_spins),
        visual=SPIN_IMAGE,
        spin=spins,
        J=J,
        h=h,
        temperature=temperature,
        couplings=upper + upper.T,
    )
    return System(components=[group])
