"""Population dynamics of signed catalytic interactions, with the least populated species replaced at every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import RoleSpec
from ....core.types import Array
from ._competing_species_view import CompetingSpeciesView


class CompetitiveSelectionInteraction(Interaction):
    """Population dynamics of signed catalytic interactions, with the least populated species replaced at every step.

    Between replacements the relative populations follow Eq. (5.37) of Thurner et al. (2018), `dx_i/dt = f_i` with
    `f_i = sum_j M_ij x_j - x_i phi`, where `phi = sum_jk M_jk x_k` keeps the total at one, except that a species at
    zero stays there while `f_i < 0`. The rule integrates them for the time `duration` by Euler steps of `step`, then
    replaces the least populated species, ties broken at random, by a new one whose links to and from every other
    species are drawn from `[-1, 1]` with the probability `p`, whose degradation is drawn from `[-1, 0]`, and whose
    population starts at `1 / N`.

    Args:
        label:
            The label of the interaction.

        species:
            The path of the group of species.

        p:
            The probability of every link of a new species.

        duration:
            The time of the population dynamics between replacements.

        step:
            The Euler step of the population dynamics.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CompetitiveSelectionInteraction
        >>> CompetitiveSelectionInteraction('selection', species='species', p=0.1).p
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        species: Annotated[str, RoleSpec()],
        p: float = 0.05,
        duration: float = 20.0,
        step: float = 0.05,
    ) -> None:
        super().__init__(label)
        self.species = species
        self.p = p
        self.duration = duration
        self.step = step

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the selection rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.select)]

    def select(self, system: SystemView) -> None:
        """Run the population dynamics, then replace the least populated species.

        Args:
            system:
                The system.
        """
        species = system.agents(self.species, CompetingSpeciesView)
        matrix = np.array([np.asarray(item.links, dtype=float) for item in species])
        populations = self._relax(matrix, np.array([item.x for item in species]))
        size = len(species)
        weakest = np.flatnonzero(populations <= populations.min() + 1e-12)
        k = int(system.rng.choice(weakest))
        drawn = system.rng.uniform(-1, 1, (2, size)) * (system.rng.random((2, size)) < self.p)
        matrix[k, :], matrix[:, k] = drawn[0], drawn[1]
        matrix[k, k] = -system.rng.random()
        populations[k] = 1 / size
        populations /= populations.sum()
        for index, item in enumerate(species):
            item.x = float(populations[index])
            item.links = matrix[index]

    def _relax(self, matrix: Array, populations: Array) -> Array:
        """Return the populations after the dynamics of Eq.

        (5.37) for the time `duration`.
        """
        x = populations.copy()
        for _ in range(round(self.duration / self.step)):
            drive = matrix @ x
            f = drive - x * drive.sum()
            f[(x <= 0) & (f < 0)] = 0.0
            x = np.clip(x + self.step * f, 0.0, None)
            total = x.sum()
            x = x / total if total > 0 else np.full_like(x, 1 / x.size)
        return x
