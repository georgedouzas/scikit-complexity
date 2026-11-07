"""Ecology systems: a neutral community of constant size whose species arise by speciation and vanish by drift."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import EvolutionInteraction
from ....core.components.laws import BaseLaw
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import AnyArray


class OrganismView(AgentView):
    """View of an organism, with its states as attributes.

    `OrganismElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    species: float


class OrganismElement(Element):
    """Individual of a neutral community, labelled by its species.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        species:
            The label of the species.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import OrganismElement
        >>> GroupElement('community', element=OrganismElement, size=10, values={'species': 0.0}).size
        10
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'species'}),
        *,
        species: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.species = species

    @classmethod
    def _get_view(cls) -> type[OrganismView]:
        """Return the view of an organism, `OrganismView`."""
        return OrganismView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the replacement moves every individual.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class NeutralReplacementInteraction(EvolutionInteraction):
    """Deaths of random individuals at unit rate, each replaced by the offspring of another, a new species at rate `nu`.

    One unit of time holds `N` deaths. The offspring belongs to the species of a parent drawn among the individuals, or
    with the probability `mutation` to a new species, labelled one above the largest label present. It is the neutral
    Moran replacement of `EvolutionInteraction` with mutation to new labels.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of individuals.

        mutation:
            The probability `nu` that an offspring founds a new species.

    Examples:
        >>> from skcomplex.systems.biology.ecology import NeutralReplacementInteraction
        >>> NeutralReplacementInteraction('replacement', target='community', mutation=0.01).change
        'new'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        mutation: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.01,
    ) -> None:
        super().__init__(
            label,
            target=target,
            strategy='species',
            fitness=None,
            selection='proportional',
            replacement='moran',
            mutation=mutation,
            alphabet=None,
            change='new',
        )


def _richness(species: AnyArray) -> float:
    """Return the number of distinct species among the individuals."""
    return float(len(np.unique(species)))


class NeutralBiodiversitySystem(BaseSystem):
    """A neutral community of constant size, where species arise by speciation and vanish by drift.

    Every species is transient, yet the number of species settles around `S_N = sum_n theta / (theta + n)` for `n` from
    0 to `N - 1`, with `theta = (N - 1) nu / (1 - nu)`, and the mean number of species of `k` individuals around `phi_k
    = theta / k prod_j (N - j + 1) / (N - j + theta)` for `j` from 1 to `k`. The system has the group `community` and
    the number `richness` of species.

    Args:
        individuals:
            The number of individuals `N`.

        nu:
            The probability that an offspring founds a new species.

        species:
            The initial number of species, to which the individuals belong at random.

        random_state:
            The seed or generator of the initial species.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import NeutralBiodiversitySystem
        >>> NeutralBiodiversitySystem(20).prepare().system_.elements[0].prepare().size_
        20
    """

    field = 'biology/ecology'
    topics = ('neutral theory', 'biodiversity', 'speciation', 'agents', 'species abundance')
    references = ('Bertin 2021', 'Hubbell 2001', 'Vallade and Houchmandzadeh 2003')
    until = 500
    observables = (ObservableSpec('richness', group='community', variable='species', reduce=_richness),)

    def __init__(
        self,
        individuals: int = 200,
        *,
        nu: float = 0.01,
        species: int = 1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.individuals = individuals
        self.nu = nu
        self.species = species
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system."""
        labels = np.random.default_rng(self.random_state).integers(self.species, size=self.individuals).astype(float)
        return System(
            self.label,
            elements=[
                GroupElement('community', element=OrganismElement, size=self.individuals, values={'species': labels}),
            ],
            interactions=[NeutralReplacementInteraction('replacement', target='community', mutation=self.nu)],
        )
