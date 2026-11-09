"""The random energy system, an exact Boltzmann sum over quenched random energies."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components.base import BasePart
from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import EquationLaw
from ....core.components.laws.utils import log_sum_exp, total
from ....core.components.spaces import StaticTime
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class BoltzmannAverageElement(Element):
    """Log partition function, mean energy, and entropy per spin of a group of levels.

    Args:
        label:
            The label of the element.

        log_partition:
            The initial guess of the log partition function, which the equations replace.

        energy:
            The initial guess of the mean energy, which the equations replace.

        entropy:
            The initial guess of the entropy per spin, which the equations replace.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import BoltzmannAverageElement
        >>> BoltzmannAverageElement('ensemble').entropy
        0.0
    """

    def __init__(
        self,
        label: str,
        log_partition: Annotated[float, StateSpec()] = 0.0,
        energy: Annotated[float, StateSpec()] = 0.0,
        entropy: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.log_partition = log_partition
        self.energy = energy
        self.entropy = entropy


class EnergyLevelElement(Element):
    """Energy level of a configuration, held fixed.

    Args:
        label:
            The label of the level or the group.

        energy:
            The energy of the level.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import EnergyLevelElement
        >>> EnergyLevelElement('level', energy=-1.5).energy
        -1.5
    """

    def __init__(self, label: str, energy: Annotated[float, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.energy = energy


class BoltzmannSumInteraction(Interaction):
    """Boltzmann sums over a group of levels.

    The sums give the log partition function, mean energy, and entropy per spin.

    The log partition function is `ln Z = log sum_j exp(-E_j / T)`, the mean energy is
    `U = sum_j E_j exp(-E_j / T - ln Z)`, and the entropy per spin is `(ln Z + U / T) / N`. The sums reduce in log
    space, so they do not overflow at any temperature.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element that holds the averages.

        levels:
            The path of the group of levels.

        temperature:
            The temperature.

        n_spins:
            The number of spins the entropy is divided by.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import BoltzmannSumInteraction
        >>> BoltzmannSumInteraction('sums', target='ensemble', levels='levels', temperature=0.5).temperature
        0.5
    """

    law_classes = (EquationLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        levels: Annotated[str, RoleSpec()],
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        n_spins: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.levels = levels
        self.temperature = temperature
        self.n_spins = n_spins

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the log partition function, mean energy, and entropy per spin equations.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The three equations.
        """
        weight = -s.levels.energy / s.temperature
        log_partition = log_sum_exp(weight, over=s.levels)
        energy = total(s.levels.energy * exp(weight - log_partition), over=s.levels)
        entropy = (s.target.log_partition + s.target.energy / s.temperature) / s.n_spins
        return [
            EquationLaw(s.target.log_partition == log_partition, target=s.target.log_partition),
            EquationLaw(s.target.energy == energy, target=s.target.energy),
            EquationLaw(s.target.entropy == entropy, target=s.target.entropy),
        ]


class RandomEnergySystem(BaseSystem):
    """The random energy system.

    It has `2^N` quenched Gaussian energies of variance `N / 2` at a temperature.

    The entropy per spin follows `ln 2 - 1 / (4 T^2)` above the glass temperature `T_g = 1 / (2 sqrt(ln 2))` and freezes
    below it, where the Boltzmann weight falls on the few lowest levels (Bertin 2021 §1.5.3). The energies are drawn
    from `random_state`, so a disorder average sweeps it with `sweep`. The system has the states
    `ensemble.log_partition`, `ensemble.energy`, and `ensemble.entropy`.

    Args:
        n_spins:
            The number of spins `N`.

        temperature:
            The temperature.

        random_state:
            The seed or generator of the energies.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import RandomEnergySystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> system = RandomEnergySystem(12, temperature=2.0).prepare().system_
        >>> trajectory = simulate(system, until=1.0)
        >>> bool(abs(trajectory.values('ensemble.entropy')[0] - (np.log(2) - 1 / 16)) < 0.05)
        True
    """

    field = 'physics/statistical_mechanics'
    topics = ('disorder', 'glass transition', 'static')
    references = ('Derrida 1981', 'Bertin 2021')

    def __init__(
        self,
        n_spins: int = 10,
        *,
        temperature: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_spins = n_spins
        self.temperature = temperature
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the random energy system.

        It has `2^N` quenched Gaussian energies of variance `N / 2` at a temperature.
        """
        size = 2**self.n_spins
        energies = np.random.default_rng(self.random_state).normal(0.0, np.sqrt(self.n_spins / 2), size)
        visual = VisualSpec('histograms', {'color_by': 'energy'})
        levels = GroupElement(
            'levels',
            element=EnergyLevelElement,
            size=size,
            values={'energy': energies},
            visual=visual,
        )
        sums = BoltzmannSumInteraction(
            'sums',
            target='ensemble',
            levels='levels',
            temperature=self.temperature,
            n_spins=float(self.n_spins),
        )
        return System(
            self.label,
            elements=[levels, BoltzmannAverageElement('ensemble')],
            interactions=[sums],
            time=StaticTime(),
        )
