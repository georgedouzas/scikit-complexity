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
from ....core.components.system import System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system


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
    """Boltzmann sums over a group of levels, which give the log partition function, mean energy, and entropy per spin.

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
        """Return the equations of the log partition function, the mean energy, and the entropy per spin.

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


@register_system(
    field='physics/statistical_mechanics',
    tags=('disorder', 'glass transition', 'static'),
    references=('Derrida 1981', 'Bertin 2021'),
)
def random_energy_system(
    n_spins: int = 10,
    *,
    temperature: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the random energy system, `2^N` quenched Gaussian energies of variance `N / 2` at a temperature.

    The entropy per spin follows `ln 2 - 1 / (4 T^2)` above the glass temperature `T_g = 1 / (2 sqrt(ln 2))` and
    freezes below it, where the Boltzmann weight falls on the few lowest levels (Bertin 2021 §1.5.3). The energies are
    drawn from `random_state`, so a disorder average sweeps it through `GridDesign`.

    Args:
        n_spins:
            The number of spins `N`.

        temperature:
            The temperature.

        random_state:
            The seed or generator of the energies.

    Returns:
        system:
            The system, with the states `ensemble.log_partition`, `ensemble.energy`, and `ensemble.entropy`.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.physics.statistical_mechanics import random_energy_system
        >>> system = random_energy_system(12, temperature=2.0)
        >>> trajectory = Simulation(system).prepare().run(1.0).trajectory_
        >>> bool(abs(trajectory.values('ensemble.entropy')[0] - (np.log(2) - 1 / 16)) < 0.05)
        True
    """
    size = 2**n_spins
    energies = np.random.default_rng(random_state).normal(0.0, np.sqrt(n_spins / 2), size)
    visual = VisualSpec('histograms', {'color_by': 'energy'})
    levels = GroupElement('levels', element=EnergyLevelElement, size=size, values={'energy': energies}, visual=visual)
    sums = BoltzmannSumInteraction(
        'sums',
        target='ensemble',
        levels='levels',
        temperature=temperature,
        n_spins=float(n_spins),
    )
    return System(elements=[levels, BoltzmannAverageElement('ensemble')], interactions=[sums], time=StaticTime())
