"""The random energy system, an exact Boltzmann sum over quenched random energies."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import StaticTime
from ....core.components.system import System
from ....core.specs import VisualSpec
from ..._catalog import register_system
from ._boltzmann_average import BoltzmannAverageElement
from ._boltzmann_sum import BoltzmannSumInteraction
from ._energy_level import EnergyLevelElement


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
