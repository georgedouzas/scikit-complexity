"""Econophysics models: wealth exchange between agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, state
from ....core.components import Element
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.system import System


class Trader(Element):
    """Agent of the Boltzmann wealth model, who gives one unit of wealth to a random agent when it has any.

    Args:
        label:
            The label of the agent or group.

        wealth:
            The initial wealth.

    Examples:
        >>> from skcomplex.models.economics.econophysics import Trader
        >>> Trader.group('traders', 100)
        Group('traders', element=Trader, size=100)
    """

    wealth: float = state(default=1.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the exchange rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.give)]

    def give(self, agent: AgentView, model: ModelView) -> None:
        """Give one unit of wealth to a random agent of the same group.

        Args:
            agent:
                The trader.

            model:
                The model.
        """
        if agent.wealth <= 0:
            return
        others = model.agents(agent.group)
        receiver = others[int(model.rng.integers(len(others)))]
        receiver.wealth += 1
        agent.wealth -= 1


@register_model(
    field='economics/econophysics',
    tags=('wealth', 'agents', 'exchange'),
    references=('Dragulescu and Yakovenko 2000',),
    defaults={'until': 100},
)
def boltzmann_wealth(n_agents: int = 100) -> System:
    """Return a Boltzmann wealth exchange model where every agent starts with one unit.

    Args:
        n_agents:
            The number of agents.

    Returns:
        system:
            The model, with the group `traders`.

    Examples:
        >>> from skcomplex.models.economics.econophysics import boltzmann_wealth
        >>> boltzmann_wealth(10).elements[0].size
        10
    """
    return System(components=[Trader.group('traders', n_agents)])
