"""Define the culture around a metabolic network.

The culture is the biomass, the substrates, and the exchanges with them.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, ConstraintLaw, ObjectiveLaw, RateLaw, UpdateLaw
from .....core.components.laws.utils import either, where
from .....core.components.spaces import BaseSpace
from .....core.types import Role
from ._flux_balance import FluxBalanceInteraction

POSITIVE = RealSet.open(0, oo)


class BiomassElement(Element):
    """Biomass of a culture, in gDW per L.

    Args:
        label:
            The label of the element.

        x:
            The biomass in gDW per L.

        values:
            Further values by name.

        size:
            The number of members, or `None` for one.

        space:
            The space of the members, such as the plate of a culture with one biomass per cell, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import BiomassElement
        >>> BiomassElement('culture', x=0.01).x
        0.01
    """

    def __init__(
        self,
        label: str,
        *,
        x: float | npt.ArrayLike = 0.01,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.x = x


class SubstrateElement(Element):
    """Concentration of a metabolite of a medium, in mM.

    Args:
        label:
            The label of the element.

        c:
            The concentration in mM.

        values:
            Further values by name.

        size:
            The number of members, or `None` for one.

        space:
            The space of the members, such as the plate of a culture with one concentration per cell, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import SubstrateElement
        >>> SubstrateElement('glucose', c=10.0).c
        10.0
    """

    def __init__(
        self,
        label: str,
        *,
        c: float | npt.ArrayLike = 10.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.c = c


class GrowthInteraction(Interaction):
    """Growth of biomass at the rate a reaction of a metabolic network gives.

    The biomass follows `dx/dt = v x`.

    Args:
        label:
            The label of the interaction.

        cell:
            The path of the cell whose values hold the fluxes of the network.

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
        cell: Role = 'cell',
        biomass: Role = 'biomass',
        reaction: str = 'Biomass_Ecoli_core',
    ) -> None:
        super().__init__(label)
        self.cell = cell
        self.biomass = biomass
        self.reaction = reaction

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the biomass.

        Args:
            context:
                The namespace of the cell and the biomass.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            flux = getattr(context.cell, FluxBalanceInteraction.get_flux_name(self.reaction))
            self.laws_: list[BaseLaw] = [RateLaw(context.biomass.x, flux * context.biomass.x)]


class ExchangeInteraction(Interaction):
    """Exchange of a metabolite between a metabolic network and its medium.

    The exchange is bounded by Michaelis-Menten uptake.

    The law bounds the uptake, the negative flux of the exchange reaction, by `vmax c / (km + c)` and, with an interval,
    by what the medium holds per biomass over the interval (Raman 2021, Section 9.6). The medium changes at the rate
    `dc/dt = v x`, and a depleted medium loses nothing more.

    Args:
        label:
            The label of the interaction.

        cell:
            The path of the cell whose values hold the fluxes of the network.

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

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import ExchangeInteraction
        >>> ExchangeInteraction('uptake', reaction='EX_glc__D_e').vmax
        10.0
    """

    def __init__(
        self,
        label: str,
        *,
        cell: Role = 'cell',
        biomass: Role = 'biomass',
        substrate: Role = 'substrate',
        reaction: str = 'EX_glc__D_e',
        vmax: Annotated[float, POSITIVE] = 10.0,
        km: Annotated[float, POSITIVE] = 0.015,
        interval: float | None = None,
    ) -> None:
        super().__init__(label)
        self.cell = cell
        self.biomass = biomass
        self.substrate = substrate
        self.reaction = reaction
        self.vmax = vmax
        self.km = km
        self.interval = interval

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the bounds of the uptake and the rate of the medium.

        Args:
            context:
                The namespace of the roles and parameters.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        flux = getattr(context.cell, FluxBalanceInteraction.get_flux_name(self.reaction))
        amount, biomass = context.substrate.c, context.biomass.x
        held = where(amount > 0, amount, 0)
        laws: list[BaseLaw] = [ConstraintLaw(flux >= -context.vmax * held / (context.km + held), target=flux)]
        if self.interval is not None:
            laws.append(ConstraintLaw(flux >= -held / (biomass * float(self.interval)), target=flux))
        laws.append(RateLaw(amount, flux * biomass * where(either(amount > 0, flux > 0), 1, 0)))
        self.laws_ = laws


class CommunityInteraction(Interaction):
    """Joint flux balance of a community, whose medium balances the exchanges of its members.

    The members share one program, which holds the objective of their growth, weighted by their abundances, and the
    medium holds the net exchange `u_<reaction>` of the community for every shared reaction, which `get_values` gives.

    Args:
        label:
            The label of the interaction.

        members:
            The paths of the cells whose values hold the fluxes of the members.

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
        >>> joint = CommunityInteraction('joint', members=['a', 'b'], medium='medium', exchanges={'EX_a': -10.0})
        >>> joint.get_values()
        {'u_EX_a': 0.0}
    """

    def __init__(
        self,
        label: str,
        *,
        members: tuple[Role, ...],
        medium: Role,
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

    @staticmethod
    def get_state_name(reaction: str) -> str:
        """Return the name of the value of the medium that holds the net exchange of a reaction.

        Args:
            reaction:
                The name of the reaction.

        Returns:
            name:
                The name of the value, such as `u_EX_glc__D_e`.
        """
        return f'u_{FluxBalanceInteraction._get_identifier(reaction)}'

    def get_values(self) -> dict[str, float]:
        """Return the values the medium starts with, the net exchange of every reaction at zero.

        Returns:
            values:
                The values by name.
        """
        return dict.fromkeys(map(self.get_state_name, self.exchanges or {}), 0.0)

    def _get_weighted(self, members: Sequence[SimpleNamespace], reaction: str) -> Expression:
        """Return the sum over the members of the flux of a reaction.

        The flux of every member is weighted by its abundance.
        """
        name = FluxBalanceInteraction.get_flux_name(reaction)
        terms = [float(share) * getattr(member, name) for share, member in zip(self.abundances, members, strict=True)]
        return sum(terms[1:], terms[0])

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the objective that joins the members through the medium, with its balances.

        Args:
            context:
                The namespace of the members and the medium.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        targets = []
        laws: list[BaseLaw] = []
        for reaction, floor in (self.exchanges or {}).items():
            state = getattr(context.medium, self.get_state_name(reaction))
            targets.append(state)
            laws += [
                ConstraintLaw(state == self._get_weighted(context.members, reaction)),
                ConstraintLaw(state >= float(floor)),
            ]
        name = FluxBalanceInteraction.get_flux_name(self.growth)
        growth = [getattr(member, name) for member in context.members]
        if self.common_rate:
            laws += [ConstraintLaw(item == growth[0]) for item in growth[1:]]
        objective = -self._get_weighted(context.members, self.growth)
        self.laws_ = [*laws, ObjectiveLaw([objective], [*targets, *growth])]


class ObjectiveFloorInteraction(Interaction):
    """Lower bound of the objective of a metabolic network as a whole.

    The bound is `c' v >= f` (Raman 2021, Eq. 8.11d).

    Args:
        label:
            The label of the interaction.

        cell:
            The path of the cell whose values hold the fluxes of the network.

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
        cell: Role = 'cell',
        weights: Mapping[str, float] | None = None,
        floor: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.cell = cell
        self.weights = weights
        self.floor = floor

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the bound of the objective.

        Args:
            context:
                The namespace of the cell.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        weights = self.weights or {}
        fluxes = [getattr(context.cell, FluxBalanceInteraction.get_flux_name(name)) for name in weights]
        terms = [float(weight) * flux for weight, flux in zip(weights.values(), fluxes, strict=True)]
        self.laws_: list[BaseLaw] = [ConstraintLaw(sum(terms[1:], terms[0]) >= float(self.floor), target=fluxes[0])]


class SwitchInteraction(Interaction):
    """Update of the switches of genes from a Boolean node that represses them.

    Args:
        label:
            The label of the interaction.

        node:
            The path of the Boolean node, whose state `x` is 1 for on.

        cell:
            The path of the cell whose values hold the switches of the network.

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
        node: Role = 'node',
        cell: Role = 'cell',
        genes: Sequence[str] = (),
    ) -> None:
        super().__init__(label)
        self.node = node
        self.cell = cell
        self.genes = genes

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the update of every switch.

        Args:
            context:
                The namespace of the node and the cell.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [
                UpdateLaw(getattr(context.cell, FluxBalanceInteraction.get_gene_name(gene)), 1 - context.node.x)
                for gene in self.genes
            ]
