"""The Boltzmann sums over a group of levels at a temperature."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components.base import BasePart
from ....core.components.interactions import Interaction
from ....core.components.laws import EquationLaw
from ....core.components.laws.utils import log_sum_exp, total
from ....core.specs import ParameterSpec, RoleSpec


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
