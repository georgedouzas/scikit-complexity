"""Define the culture around a metabolic network: biomass, substrates, and the exchanges with them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....core.components.elements import DeclaredElement, Element
from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, LinearProgramLaw, RateLaw, UpdateLaw
from .....core.components.laws.utils import either, where
from .....core.specs import DeclaredVariableSpec, LinearObjectiveSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ._metabolic_network import MetabolicNetworkElement

POSITIVE = RealSet.open(0, oo)


class BiomassElement(Element):
    """Biomass of a culture.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The biomass in gDW per L.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import BiomassElement
        >>> BiomassElement('culture', x=0.01).x
        0.01
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class SubstrateElement(Element):
    """Concentration of a metabolite of a medium.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        c:
            The concentration in mM.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import SubstrateElement
        >>> SubstrateElement('glucose', c=10.0).c
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[float, StateSpec()] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c


class GrowthInteraction(Interaction):
    """Growth of biomass at the rate that a reaction of a metabolic network gives, `dx/dt = v x`.

    Args:
        label:
            The label of the interaction.

        cell:
            The path of the metabolic network.

        biomass:
            The path of the biomass.

        reaction:
            The name of the reaction of growth.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import GrowthInteraction
        >>> GrowthInteraction('growth', cell='cell', biomass='culture', reaction='grow').reaction
        'grow'
    """

    def __init__(
        self,
        label: str,
        *,
        cell: Annotated[str, RoleSpec()] = 'cell',
        biomass: Annotated[str, RoleSpec()] = 'biomass',
        reaction: str = 'Biomass_Ecoli_core',
    ) -> None:
        super().__init__(label)
        self.cell = cell
        self.biomass = biomass
        self.reaction = reaction

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseLaw]:
        """Return the rate of the biomass.

        Args:
            namespace:
                The namespace of the roles.

        Returns:
            laws:
                The rate law of the biomass.
        """
        flux = getattr(namespace.cell, MetabolicNetworkElement.flux_name(self.reaction))
        return [RateLaw(namespace.biomass.x, flux * namespace.biomass.x)]


class ExchangeInteraction(Interaction):
    """Exchange of a metabolite between a metabolic network and its medium, bounded by Michaelis-Menten uptake.

    The law bounds the uptake, the negative flux of the exchange reaction, by `vmax c / (km + c)` and, with an interval,
    by what the medium holds per biomass over the interval (Raman 2021, Section 9.6). The medium changes at the rate
    `dc/dt = v x`, and a depleted medium loses nothing more.

    Args:
        label:
            The label of the interaction.

        cell:
            The path of the metabolic network.

        biomass:
            The path of the biomass.

        substrate:
            The path of the metabolite.

        reaction:
            The name of the exchange reaction.

        vmax:
            The largest uptake.

        km:
            The half-saturation concentration.

        interval:
            The time over which the medium must last, or `None` for no such bound.

        threshold:
            The biomass above which the program solves, or `None` for always.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import ExchangeInteraction
        >>> ExchangeInteraction('uptake', reaction='EX_glc__D_e').vmax
        10.0
    """

    def __init__(
        self,
        label: str,
        *,
        cell: Annotated[str, RoleSpec()] = 'cell',
        biomass: Annotated[str, RoleSpec()] = 'biomass',
        substrate: Annotated[str, RoleSpec()] = 'substrate',
        reaction: str = 'EX_glc__D_e',
        vmax: Annotated[float, ParameterSpec(domain=POSITIVE)] = 10.0,
        km: Annotated[float, ParameterSpec(domain=POSITIVE)] = 0.015,
        interval: float | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(label)
        self.cell = cell
        self.biomass = biomass
        self.substrate = substrate
        self.reaction = reaction
        self.vmax = vmax
        self.km = km
        self.interval = interval
        self.threshold = threshold

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseLaw]:
        """Return the bound of the uptake and the rate of the medium.

        Args:
            namespace:
                The namespace of the roles and parameters.

        Returns:
            laws:
                The linear program law of the bound and the rate law of the medium.
        """
        flux = getattr(namespace.cell, MetabolicNetworkElement.flux_name(self.reaction))
        amount, biomass = namespace.substrate.c, namespace.biomass.x
        held = where(amount > 0, amount, 0)
        constraints = [flux >= -namespace.vmax * held / (namespace.km + held)]
        if self.interval is not None:
            constraints.append(flux >= -held / (biomass * float(self.interval)))
        active = None if self.threshold is None else biomass > float(self.threshold)
        return [
            LinearProgramLaw([flux], constraints, active=active, infeasible='zero'),
            RateLaw(amount, flux * biomass * where(either(amount > 0, flux > 0), 1, 0)),
        ]


class CommunityMediumElement(DeclaredElement):
    """Medium that a community shares, with one algebraic state per shared exchange.

    The element declares the state `u_<reaction>` of the net exchange of every reaction.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        exchanges:
            The names of the reactions.

        variables:
            The declared states that replace the states of the net exchanges, by name, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import CommunityMediumElement
        >>> CommunityMediumElement.state_name('EX_glc__D_e')
        'u_EX_glc__D_e'
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        exchanges: Sequence[str] = (),
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
    ) -> None:
        super().__init__(label, visual=visual, variables=variables)
        self.exchanges = exchanges

    def _declare_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the state of the net exchange of every reaction, with the variables that replace them."""
        flux = StateSpec(algebraic=True)
        return {
            **{self.state_name(reaction): DeclaredVariableSpec(0.0, flux) for reaction in self.exchanges},
            **(self.variables or {}),
        }

    @staticmethod
    def state_name(reaction: str) -> str:
        """Return the name of the state of the net exchange of a reaction.

        Args:
            reaction:
                The name of the reaction.

        Returns:
            name:
                The name of the state.
        """
        return f'u_{MetabolicNetworkElement._identifier(reaction)}'


class CommunityInteraction(Interaction):
    """Joint program of a community: the medium balances the exchanges, and its law holds the objective of growth.

    Args:
        label:
            The label of the interaction.

        members:
            The paths of the metabolic networks.

        medium:
            The path of the shared medium.

        abundances:
            The abundance of every member, which weighs its exchanges and its growth.

        exchanges:
            The lower bound of the net exchange of the community, by reaction.

        growth:
            The name of the reaction of growth of every member.

        common_rate:
            Whether every member grows at one specific rate, the balanced growth of community FBA (Raman 2021, Section
            11.1.3.2), rather than each at its own, as in joint FBA.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import CommunityInteraction
        >>> CommunityInteraction('joint', members=['a', 'b'], medium='medium', abundances=(0.5, 0.5)).abundances
        (0.5, 0.5)
    """

    def __init__(
        self,
        label: str,
        *,
        members: Annotated[list[str], RoleSpec(variadic=True)],
        medium: Annotated[str, RoleSpec()],
        abundances: Sequence[float] = (),
        exchanges: Mapping[str, float] | None = None,
        growth: str = 'Biomass_Ecoli_core',
        common_rate: bool = False,
    ) -> None:
        super().__init__(label)
        self.members = members
        self.medium = medium
        self.abundances = abundances
        self.exchanges = exchanges
        self.growth = growth
        self.common_rate = common_rate

    def _weighted(self, namespace: SimpleNamespace, reaction: str) -> Expression:
        """Return the sum over the members of the flux of a reaction weighted by the abundance."""
        name = MetabolicNetworkElement.flux_name(reaction)
        terms = [
            float(share) * getattr(member, name)
            for share, member in zip(self.abundances, namespace.members, strict=True)
        ]
        return sum(terms[1:], terms[0])

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseLaw]:
        """Return the program that joins the members through the medium.

        Args:
            namespace:
                The namespace of the roles.

        Returns:
            laws:
                The linear program law.
        """
        variables = []
        relations: list[Expression] = []
        for reaction, floor in (self.exchanges or {}).items():
            state = getattr(namespace.medium, CommunityMediumElement.state_name(reaction))
            variables.append(state)
            relations += [state == self._weighted(namespace, reaction), state >= float(floor)]
        name = MetabolicNetworkElement.flux_name(self.growth)
        growth = [getattr(member, name) for member in namespace.members]
        if self.common_rate:
            relations += [item == growth[0] for item in growth[1:]]
        objective = LinearObjectiveSpec(self._weighted(namespace, self.growth))
        return [LinearProgramLaw([*variables, *growth], relations, [objective])]


def community_medium_type(exchanges: Sequence[str]) -> Callable[..., CommunityMediumElement]:
    """Return a function that creates a `CommunityMediumElement` with the state of the net exchange of some reactions.

    Args:
        exchanges:
            The names of the reactions.

    Returns:
        create_medium:
            The function, which takes the label, the visual, and the initial values of the states by name.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import community_medium_type
        >>> community_medium_type(['EX_a'])('medium').get_value('u_EX_a')
        0.0
    """

    def create_medium(label: str, visual: VisualSpec | None = None, **values: float) -> CommunityMediumElement:
        medium = CommunityMediumElement(label, visual, exchanges=tuple(exchanges))
        return medium.set_params(**values)

    return create_medium


class ObjectiveFloorInteraction(Interaction):
    """Lower bound of the objective of a metabolic network as a whole, `c' v >= f` (Raman 2021, Eq. 8.11d).

    Args:
        label:
            The label of the interaction.

        cell:
            The path of the metabolic network.

        weights:
            The coefficient of every reaction in the objective.

        floor:
            The value the objective keeps.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import ObjectiveFloorInteraction
        >>> ObjectiveFloorInteraction('kept', weights={'grow': 1.0}, floor=0.5).floor
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        cell: Annotated[str, RoleSpec()] = 'cell',
        weights: Mapping[str, float] | None = None,
        floor: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.cell = cell
        self.weights = weights
        self.floor = floor

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseLaw]:
        """Return the bound of the objective.

        Args:
            namespace:
                The namespace of the roles.

        Returns:
            laws:
                The linear program law of the bound.
        """
        fluxes = [getattr(namespace.cell, MetabolicNetworkElement.flux_name(name)) for name in self.weights or {}]
        terms = [float(weight) * flux for weight, flux in zip((self.weights or {}).values(), fluxes, strict=True)]
        return [LinearProgramLaw(fluxes, [sum(terms[1:], terms[0]) >= float(self.floor)])]


class SwitchInteraction(Interaction):
    """Update of the switches of genes from a Boolean node that represses them.

    Args:
        label:
            The label of the interaction.

        node:
            The path of the Boolean node, whose state `x` is 1 for on.

        cell:
            The path of the metabolic network.

        genes:
            The names of the genes the node represses.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import SwitchInteraction
        >>> SwitchInteraction('repression', node='node', cell='cell', genes=('b2296',)).genes
        ('b2296',)
    """

    def __init__(
        self,
        label: str,
        *,
        node: Annotated[str, RoleSpec()] = 'node',
        cell: Annotated[str, RoleSpec()] = 'cell',
        genes: Sequence[str] = (),
    ) -> None:
        super().__init__(label)
        self.node = node
        self.cell = cell
        self.genes = genes

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseLaw]:
        """Return the update of every switch.

        Args:
            namespace:
                The namespace of the roles.

        Returns:
            laws:
                One update law per gene.
        """
        return [
            UpdateLaw(getattr(namespace.cell, MetabolicNetworkElement.gene_name(gene)), 1 - namespace.node.x)
            for gene in self.genes
        ]
