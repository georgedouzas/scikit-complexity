"""Natural resources models: harvesting renewable stocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, role, state
from ....core.model import AgentType, Aggregate, Element, Interaction, Law, Rate, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

Stock = Element.define(
    'Stock',
    states={'n': state(default=1000.0, domain=Domain.NONNEGATIVE)},
    parameters={
        'r': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'K': parameter(default=1000.0, domain=Domain.POSITIVE),
    },
    laws=lambda s: [Rate(s.n, s.r * s.n * (1 - s.n / s.K))],
)


class Farmer(AgentType):
    """Farmer who adds the catch of every step to its wealth, reading the resource from the other partition.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        resource:
            The dotted name of the resource stock.

        effort:
            The dotted name of the harvest effort per farmer.

        wealth:
            The initial wealth.

    Examples:
        >>> from skcomplex.models.economics.resources import Farmer
        >>> Farmer.group('farmers', 10).size
        10
    """

    resource: str = 'land/stock.n'
    effort: str = 'harvest.effort'
    wealth: float = state(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the catch rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.catch)]

    def catch(self, agent: AgentView, model: ModelView) -> None:
        """Add the effort times the resource stock at the last synchronisation point to the wealth.

        Args:
            agent:
                The farmer.

            model:
                The model.
        """
        agent.wealth += model.parameters[self.effort] * float(model.inputs.get(self.resource, 0.0))


class Harvest(Interaction):
    """Harvest of a resource proportional to its stock and to the number of harvesters, `-effort * N * n`.

    Args:
        label:
            The label of the interaction.

        resource:
            The path of the resource stock.

        harvesters:
            The path of the aggregate that counts the harvesters.

        effort:
            The harvest rate per harvester and unit of stock.

    Examples:
        >>> from skcomplex.models.economics.resources import Harvest
        >>> Harvest('harvest', resource='land/stock', harvesters='village/size', effort=0.05).effort
        0.05
    """

    resource: str = role()
    harvesters: str = role()
    effort: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the harvest rate.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.resource.n, -s.effort * s.harvesters.value * s.resource.n)]


@register_model(
    field='economics/resources',
    tags=('socio-ecological', 'agents', 'ode', 'co-simulation'),
    references=('Hamill and Gilbert 2016 ch. 10',),
    defaults={'until': 50},
)
def harvesting(farmers: int = 10, *, effort: float = 0.05, r: float = 1.0, K: float = 1000.0) -> System:
    """Return a socio-ecological model of farmer agents harvesting a logistically regrowing resource.

    The resource settles at `K * (1 - effort * farmers / r)` when `effort * farmers < r`, and collapses otherwise.

    Args:
        farmers:
            The number of farmers.

        effort:
            The harvest rate per farmer and unit of stock.

        r:
            The growth rate of the resource.

        K:
            The carrying capacity of the resource.

    Returns:
        system:
            The model, with the ODE subsystem `land` and the agent subsystem `village`.

    Examples:
        >>> from skcomplex.models.economics.resources import harvesting
        >>> [subsystem.label for subsystem in harvesting().systems]
        ['land', 'village']
    """
    land = System('land', components=[Stock('stock', n=K, r=r, K=K)])
    village = System('village', components=[Farmer.group('farmers', farmers), Aggregate('size', group='farmers')])
    harvest = Harvest('harvest', resource='land/stock', harvesters='village/size', effort=effort)
    return System(components=[land, village], interactions=[harvest])
