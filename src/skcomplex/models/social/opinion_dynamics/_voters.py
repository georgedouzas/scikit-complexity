"""Opinion dynamics models: voting and majority rules."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement, LatticeElement
from ....core.laws import BaseLaw, LocalRuleLaw
from ....core.neighborhoods import MooreNeighborhood
from ....core.spaces import LatticeSpace, NetworkSpace
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, TotalisticSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ._variant_voter_view import VariantVoterView
from ._variant_voting import VariantVotingInteraction
from ._voter_imitation import VoterImitationInteraction
from ._voter_view import VoterView

MAJORITY = 5


class VoterElement(Element):
    """Agent of the voter model, whose opinion the imitation sets to that of a random neighbour on a network.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import VoterElement
        >>> group = GroupElement('voters', element=VoterElement, size=20, values={'opinion': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('voters', 'VoterElement', 20)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)}),
        *,
        opinion: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion

    @classmethod
    def _get_view(cls) -> type[VoterView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return VoterView


class MajorityVoterElement(Element):
    """Cell that takes the majority opinion of its Moore neighbourhood, itself included.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import MajorityVoterElement
        >>> MajorityVoterElement('cell', opinion=1).opinion
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        opinion: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the majority rule over the nine cells.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            LocalRuleLaw(s.opinion, MooreNeighborhood(center=True), TotalisticSpec(lambda total: total >= MAJORITY)),
        ]


class VariantVoterElement(Element):
    """Agent of the voter model variants: an imitator, a threshold voter, or a contrarian.

    With a threshold of 0 the voting copies a random neighbour into the agent, as in the voter model. With a positive
    threshold it gives the agent the opinion opposite to its own when the share of neighbours that hold it exceeds the
    threshold, so a threshold of 0.5 gives the majority rule. A contrarian adopts the opposite of what it would
    otherwise adopt.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

        contrarian:
            Whether the agent is a contrarian, 1, or not, 0.

        threshold:
            The share of disagreeing neighbours above which the agent changes its opinion, or 0 to copy a random
            neighbour.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import VariantVoterElement
        >>> group = GroupElement('voters', element=VariantVoterElement, size=10, values={'threshold': 0.5})
        >>> group.label, group.element.__name__, group.size
        ('voters', 'VariantVoterElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)}),
        *,
        opinion: Annotated[float, StateSpec(domain=NN)] = 0.0,
        contrarian: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.contrarian = contrarian
        self.threshold = threshold

    @classmethod
    def _get_view(cls) -> type[VariantVoterView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return VariantVoterView


def _random_cells(shape: Sequence[int], density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)


@register_model(
    field='social/opinion_dynamics',
    tags=('lattice', 'opinion', 'voting'),
    references=('Charbonneau 2017',),
    defaults={'until': 50, 'dt': 1},
)
def majority_rule(
    shape: Sequence[int] = (100, 100),
    *,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return voters that take the majority opinion of their neighbourhood, which coarsens into domains.

    Args:
        shape:
            The number of rows and columns.

        density:
            The probability that a voter starts with opinion 1.

        random_state:
            The seed or generator of the initial opinions.

    Returns:
        system:
            The model, with the opinions `voters.opinion`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import majority_rule
        >>> majority_rule((10, 10)).elements[0].check().prepare().size_
        100
    """
    values = _random_cells(shape, density, random_state)
    return System(
        elements=[
            LatticeElement(
                'voters',
                element=MajorityVoterElement,
                space=LatticeSpace(tuple(shape)),
                values={'opinion': values},
            ),
        ],
    )


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'network'),
    references=('Holley and Liggett 1975',),
    defaults={'until': 100},
)
def voter_model(
    graph: object = None,
    *,
    share: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a voter model with one voter per node of a network.

    Args:
        graph:
            The NetworkX graph, or `None` for a ring of 50 nodes each linked to its four nearest neighbours.

        share:
            The share of voters with opinion 1.

        random_state:
            The seed or generator of the opinions.

    Returns:
        system:
            The model, with the group `voters`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.social.opinion_dynamics import voter_model
        >>> voter_model(nx.cycle_graph(8), random_state=0).elements[0].check().prepare().size_
        8
    """
    rng = np.random.default_rng(random_state)
    graph = nx.watts_strogatz_graph(50, 4, 0.0) if graph is None else graph
    size = nx.number_of_nodes(graph)
    opinions = (rng.random(size) < share).astype(float)
    return System(
        elements=[GroupElement('voters', element=VoterElement, size=size, values={'opinion': opinions})],
        interactions=[VoterImitationInteraction('imitation', target='voters')],
        space=NetworkSpace(graph),
    )


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'network', 'voting', 'contrarians'),
    references=('Thurner, Hanel, and Klimek 2018', 'Galam 2004'),
    defaults={'until': 100},
)
def voter_variants(
    graph: object = None,
    *,
    threshold: float = 0.5,
    contrarians: float = 0.1,
    share: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return voter model variants on a network, with threshold voters and a share of contrarians.

    Args:
        graph:
            The NetworkX graph, or `None` for a ring of 50 nodes each linked to its four nearest neighbours.

        threshold:
            The share of disagreeing neighbours above which a voter changes its opinion, 0.5 for the majority rule, or
            0 to copy a random neighbour.

        contrarians:
            The share of voters that are contrarians.

        share:
            The share of voters with opinion 1.

        random_state:
            The seed or generator of the opinions and of the contrarians.

    Returns:
        system:
            The model, with the group `voters`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.social.opinion_dynamics import voter_variants
        >>> voter_variants(nx.cycle_graph(8), random_state=0).elements[0].check().prepare().size_
        8
    """
    rng = np.random.default_rng(random_state)
    graph = nx.watts_strogatz_graph(50, 4, 0.0) if graph is None else graph
    size = nx.number_of_nodes(graph)
    voters = GroupElement(
        'voters',
        element=VariantVoterElement,
        size=size,
        values={
            'opinion': (rng.random(size) < share).astype(float),
            'contrarian': (rng.random(size) < contrarians).astype(float),
            'threshold': threshold,
        },
    )
    return System(
        elements=[voters],
        interactions=[VariantVotingInteraction('voting', target='voters')],
        space=NetworkSpace(graph),
    )
