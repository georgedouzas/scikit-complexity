"""Restaurants that each serve one meal every evening, to one of the diners who come chosen at random."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import ClassVar

from ....core.components.interactions import GroupingInteraction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ._diner_view import DinerView


class RestaurantsInteraction(GroupingInteraction):
    """Restaurants that each serve one meal every evening, to one of the diners who come chosen at random.

    The diners are grouped by their restaurant. Every diner learns the crowd at its restaurant, and the first of the
    group in a random order is served. The interaction reports the fraction of restaurants that serve a meal.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of diners.

        n_restaurants:
            The number of restaurants.

        by:
            The state of the diners that holds their restaurant.

        phase:
            The phase the restaurants serve in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import RestaurantsInteraction
        >>> RestaurantsInteraction('restaurants', target='diners', n_restaurants=5).groups
        'state'
    """

    reports: ClassVar[tuple[str, ...]] = ('utilization',)

    def __init__(
        self,
        label: str,
        *,
        target: str,
        n_restaurants: int = 100,
        by: str = 'restaurant',
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, groups='state', by=by, phase=phase)
        self.n_restaurants = n_restaurants

    def act_on(self, members: list[AgentViewProtocol], system: SystemViewProtocol) -> None:
        """Tell every diner of a restaurant its crowd and serve the first diner.

        Args:
            members:
                The diners at one restaurant, in random order.

            system:
                The system view of the restaurants.
        """
        diners = [member for member in members if isinstance(member, DinerView)]
        for position, member in enumerate(diners):
            member.crowd = float(len(diners))
            member.served = float(position == 0)

    def report(self, system: SystemViewProtocol) -> None:
        """Report the fraction of restaurants with at least one diner.

        Args:
            system:
                The system view of the restaurants.
        """
        visited = {float(getattr(agent, str(self.by))) for agent in system.agents('target')}
        system.report('utilization', len(visited) / self.n_restaurants)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that serves the diners and the rule that reports the utilization.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The serving and the reporting rules.
        """
        return [*super()._get_laws(namespace), RuleLaw(self.report, phase=self.phase)]
