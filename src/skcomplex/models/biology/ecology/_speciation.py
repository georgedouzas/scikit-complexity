"""Ecology models: a neutral community of constant size whose species arise by speciation and vanish by drift."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import EvolutionInteraction
from ....core.laws import BaseLaw
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import AnyArray
from ..._catalog import register_model
from ._organism_view import OrganismView


def _richness(species: AnyArray) -> float:
    """Return the number of distinct species among the individuals."""
    return float(len(np.unique(species)))


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
        >>> from skcomplex.models.biology.ecology import OrganismElement
        >>> GroupElement('community', element=OrganismElement, size=10, values={'species': 0.0}).size
        10
    """

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
        >>> from skcomplex.models.biology.ecology import NeutralReplacementInteraction
        >>> NeutralReplacementInteraction('replacement', target='community', mutation=0.01).check().change
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


@register_model(
    field='biology/ecology',
    tags=('neutral theory', 'biodiversity', 'speciation', 'agents', 'species abundance'),
    references=('Bertin 2021', 'Hubbell 2001', 'Vallade and Houchmandzadeh 2003'),
    defaults={'until': 500},
    observables=lambda system: [
        ObservableSpec('richness', group='community', variable='species', reduce=_richness),
    ],
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
        >>> neutral_biodiversity(20).elements[0].check().prepare().size_
        20
    """
    labels = np.random.default_rng(random_state).integers(species, size=individuals).astype(float)
    return System(
        elements=[GroupElement('community', element=OrganismElement, size=individuals, values={'species': labels})],
        interactions=[NeutralReplacementInteraction('replacement', target='community', mutation=nu)],
    )
