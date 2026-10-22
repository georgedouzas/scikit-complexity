"""Ecology models: a neutral community of constant size whose species arise by speciation and vanish by drift."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.solvers import ModelView
from ....core.system import Observable, System

type Array = npt.NDArray[np.float64]


def _richness(species: Array) -> float:
    """Return the number of distinct species among the individuals."""
    return float(len(np.unique(species)))


class Organism(Element):
    """Individual of a neutral community, labelled by its species.

    Args:
        label:
            The label of the agent or group.

        species:
            The label of the species.

    Examples:
        >>> from skcomplex.models.biology.ecology import Organism
        >>> Organism.group('community', 10, species=0.0).size
        10
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'species'})
    species: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return no laws, since the replacement moves every individual.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class NeutralReplacement(Interaction):
    """Deaths of random individuals at unit rate, each replaced by the offspring of another, a new species at rate `nu`.

    One unit of time holds `N` deaths. The offspring belongs to the species of a parent drawn among the other `N - 1`
    individuals, or with the probability `nu` to a new species, labelled one above the largest label present.

    Args:
        label:
            The label of the interaction.

        community:
            The path of the group of individuals.

        nu:
            The probability that an offspring founds a new species.

    Examples:
        >>> from skcomplex.models.biology.ecology import NeutralReplacement
        >>> NeutralReplacement('replacement', community='community', nu=0.01).nu
        0.01
    """

    community: str = role()
    nu: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the replacement rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.replace)]

    def replace(self, model: ModelView) -> None:
        """Run the deaths and replacements of one unit of time.

        Args:
            model:
                The model.
        """
        individuals = model.agents(self.community)
        species = np.array([individual.species for individual in individuals], dtype=float)
        size = len(species)
        rng = model.rng
        dead = rng.integers(size, size=size)
        parents = (dead + rng.integers(1, size, size=size)) % size
        mutated = rng.random(size) < model.parameter('nu')
        newest = species.max()
        for victim, parent, new in zip(dead, parents, mutated, strict=True):
            newest += new
            species[victim] = newest if new else species[parent]
        for individual, label in zip(individuals, species, strict=True):
            individual.species = float(label)


@register_model(
    field='biology/ecology',
    tags=('neutral theory', 'biodiversity', 'speciation', 'agents', 'species abundance'),
    references=('Bertin 2021', 'Hubbell 2001', 'Vallade and Houchmandzadeh 2003'),
    defaults={'until': 500},
    observables=lambda system: [Observable('richness', group='community', variable='species', reduce=_richness)],
)
def neutral_biodiversity(
    individuals: int = 200,
    *,
    nu: float = 0.01,
    species: int = 1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a neutral community of constant size, where species arise by speciation and vanish by drift.

    Every species is transient, yet the number of species settles around `S_N = sum_n theta / (theta + n)` for `n`
    from 0 to `N - 1`, with `theta = (N - 1) nu / (1 - nu)`, and the mean number of species of `k` individuals around
    `phi_k = theta / k prod_j (N - j + 1) / (N - j + theta)` for `j` from 1 to `k`.

    Args:
        individuals:
            The number of individuals `N`.

        nu:
            The probability that an offspring founds a new species.

        species:
            The initial number of species, to which the individuals belong at random.

        random_state:
            The seed or generator of the initial species.

    Returns:
        system:
            The model, with the group `community` and the number `richness` of species.

    Examples:
        >>> from skcomplex.models.biology.ecology import neutral_biodiversity
        >>> neutral_biodiversity(20).elements[0].size
        20
    """
    labels = np.random.default_rng(random_state).integers(species, size=individuals).astype(float)
    return System(
        components=[Organism.group('community', individuals, species=labels)],
        interactions=[NeutralReplacement('replacement', community='community', nu=nu)],
    )
