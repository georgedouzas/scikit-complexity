"""Fully connected spin systems with Glauber dynamics.

The module covers the Curie-Weiss magnet and the Sherrington-Kirkpatrick glass.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ, Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.components.laws.utils import count, mean, total, where
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array

SPIN_IMAGE = VisualSpec('image', {'states': ('spin',), 'low': -1.0, 'high': 1.0})


def _initial_spins(n_spins: int, initial: str, random_state: int | np.random.Generator | None) -> Array:
    """Return all spins up, or random spins, for `initial` set to `'up'` or `'random'`."""
    if initial == 'up':
        return np.ones(n_spins)
    return np.random.default_rng(random_state).choice([-1.0, 1.0], size=n_spins).astype(np.float64)


def _magnetization() -> ObservableSpec:
    """Return the observable of the mean spin of the group `spins`."""
    return ObservableSpec('magnetization', group='spins', variable='spin')


class FullyConnectedSpinElement(Element):
    """Ising spin coupled to every other spin.

    The spin flips at the Glauber rate `1 / (1 + exp(2 s h / T))`.

    The local field is `h = J (sum_{j != i} s_j) / N + h_ext` of the Curie-Weiss energy `-(J / N) sum_{i<j} s_i s_j`,
    or `h = J sum_j C_ij s_j + h_ext` with a symmetric coupling matrix `C` of zero diagonal. The rate satisfies
    detailed balance with the Boltzmann distribution at the temperature `T`.

    Args:
        label:
            The label of the spin or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.physics.statistical_mechanics import FullyConnectedSpinElement
        >>> FullyConnectedSpinElement('spin').temperature
        0.5
    """

    law_classes = (JumpLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        h: Annotated[float, ParameterSpec()] = 0.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        couplings: npt.ArrayLike | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.spin = spin
        self.J = J
        self.h = h
        self.temperature = temperature
        self.couplings = couplings

    def _field(self, s: SimpleNamespace) -> Expression:
        """Return the local field of a spin, without its own contribution."""
        if self.couplings is None:
            return s.J * (mean(s.spin) - s.spin / count()) + s.h
        return s.J * total(s.spin, weights=np.asarray(self.couplings, dtype=float)) + s.h

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the flips of up spins and of down spins at the Glauber rate.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        rate = 1 / (1 + exp(2 * s.spin * self._field(s) / s.temperature))
        return [JumpLaw(where(s.spin > 0, rate, 0), {s.spin: -2}), JumpLaw(where(s.spin < 0, rate, 0), {s.spin: 2})]


class CurieWeissSystem(BaseSystem):
    """The fully connected Ising system with Glauber dynamics.

    The system orders below the critical temperature `T_c = J`.

    The energy is `-(J / N) sum_{i<j} s_i s_j - h sum_i s_i`. In the large system limit the magnetization `m` solves `m
    = tanh((J m + h) / T)`, which has a nonzero root below `T_c` when `h = 0`. One unit of time is one attempted flip
    per spin. The spins are drawn in a row, which orders the drawing only. The system has the spins `spins.spin` and the
    observable `magnetization`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import CurieWeissSystem
        >>> CurieWeissSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'physics/statistical_mechanics'
    topics = ('mean field', 'phase transition', 'spins', 'jump process')
    references = ('Bertin 2021', 'Glauber 1963')
    until = 30.0
    dt = 0.1
    observables = (_magnetization(),)

    def __init__(
        self,
        n_spins: int = 500,
        *,
        temperature: float = 0.5,
        J: float = 1.0,
        h: float = 0.0,
        initial: str = 'random',
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_spins = n_spins
        self.temperature = temperature
        self.J = J
        self.h = h
        self.initial = initial
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        spins = _initial_spins(self.n_spins, self.initial, self.random_state)
        group = GroupElement(
            'spins',
            element=FullyConnectedSpinElement,
            space=LatticeSpace(self.n_spins),
            values={'spin': spins, 'J': self.J, 'h': self.h, 'temperature': self.temperature},
            visual=SPIN_IMAGE,
            on='sites',
        )
        return System(self.label, elements=[group])


class SherringtonKirkpatrickSystem(BaseSystem):
    """The Sherrington-Kirkpatrick spin glass with Glauber dynamics.

    The glass freezes below the temperature `T_c = J`.

    The energy is `-sum_{i<j} J_ij s_i s_j - h sum_i s_i` with quenched Gaussian couplings `J_ij` of mean 0 and variance
    `J^2 / N`, drawn once. Above `T_c` the energy per spin is `-J^2 / (2 T)`, and below it the spins freeze in random
    directions, with no magnetization. The spins are drawn in a row, which orders the drawing only. The system has the
    spins `spins.spin`, the coupling matrix of unit scale in the values of the group, and the observable
    `magnetization`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import SherringtonKirkpatrickSystem
        >>> couplings = SherringtonKirkpatrickSystem(10).prepare().system_.elements[0].values['couplings']
        >>> bool((couplings == couplings.T).all()), float(couplings.trace())
        (True, 0.0)
    """

    field = 'physics/statistical_mechanics'
    topics = ('spin glass', 'disorder', 'mean field', 'spins', 'jump process')
    references = ('Sherrington and Kirkpatrick 1975', 'Bertin 2021', 'Mezard, Parisi, and Virasoro 1987')
    until = 50.0
    dt = 0.5
    observables = (_magnetization(),)

    def __init__(
        self,
        n_spins: int = 200,
        *,
        temperature: float = 0.5,
        J: float = 1.0,
        h: float = 0.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_spins = n_spins
        self.temperature = temperature
        self.J = J
        self.h = h
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        upper = np.triu(rng.normal(0.0, 1.0 / np.sqrt(self.n_spins), (self.n_spins, self.n_spins)), 1)
        spins = _initial_spins(self.n_spins, 'random', rng)
        group = GroupElement(
            'spins',
            element=FullyConnectedSpinElement,
            space=LatticeSpace(self.n_spins),
            values={
                'spin': spins,
                'J': self.J,
                'h': self.h,
                'temperature': self.temperature,
                'couplings': upper + upper.T,
            },
            visual=SPIN_IMAGE,
            on='sites',
        )
        return System(self.label, elements=[group])
