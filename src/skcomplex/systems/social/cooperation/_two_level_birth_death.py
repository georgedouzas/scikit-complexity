"""Two-level birth-death systems: competition within and between groups.

The competition is Moran within groups and pairwise conflicts between them.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class PunisherGroupView(AgentView):
    """View of a group of the two-level birth-death process.

    Its states are attributes.
    """

    punishers: float


PAIR = 2


class PunisherGroupElement(Element):
    """Group of defectors and altruistic punishers of a fixed size.

    It is held by its number of punishers.

    Args:
        label:
            The label of the group or the population of groups.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        punishers:
            The number of altruistic punishers of the group.

    Examples:
        >>> from skcomplex.systems.social.cooperation import PunisherGroupElement
        >>> PunisherGroupElement('group', punishers=3.0).punishers
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        punishers: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.punishers = punishers

    @classmethod
    def _get_view(cls) -> type[PunisherGroupView]:
        """Return the class of the views of the agents, whose attributes are the states."""
        return PunisherGroupView


class TwoLevelBirthDeathInteraction(Interaction):
    """Two-level birth-death process of Luo and coauthors.

    Groups have pairwise conflicts between them. Within a group of `n` members with `i` punishers, a punisher reproduces
    at the rate `1 + w pi_P(i / n)` and a defector at `1 + w pi_D(i / n)`, and the offspring replaces a random member of
    the group, the parent included. Every group fights a randomly drawn other group at the rate `Lambda`, wins with the
    probability `rho = 1/2 + (z - u) / 2` of the fractions of punishers, and the winner's copy replaces the loser. The
    events of every step of one unit of time are drawn exactly, by the Gillespie algorithm.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population of groups.

        members:
            The number `n` of members of every group.

        selection:
            The intensity `w` of individual-level selection.

        conflict_rate:
            The rate `Lambda` at which a group fights another group.

        benefit:
            The benefit `b` of the donation game.

        cost:
            The cost `c` of cooperating.

        punishment:
            The punishment `p` of a defector by every punisher it meets.

        interaction_cost:
            The cost `k` a punisher pays per defector it meets.

        fixed_cost:
            The fixed cost `q` of being able to punish.

    Examples:
        >>> from skcomplex.systems.social.cooperation import TwoLevelBirthDeathInteraction
        >>> TwoLevelBirthDeathInteraction('competition', target='groups').members
        20
    """

    draws = True

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        members: Annotated[int, ParameterSpec(domain=NN)] = 20,
        selection: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        conflict_rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        benefit: Annotated[float, ParameterSpec()] = 2.0,
        cost: Annotated[float, ParameterSpec()] = 1.0,
        punishment: Annotated[float, ParameterSpec()] = 0.5,
        interaction_cost: Annotated[float, ParameterSpec()] = 0.5,
        fixed_cost: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.members = members
        self.selection = selection
        self.conflict_rate = conflict_rate
        self.benefit = benefit
        self.cost = cost
        self.punishment = punishment
        self.interaction_cost = interaction_cost
        self.fixed_cost = fixed_cost

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the birth-death rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.compete)]

    def _get_birth_rates(self, punishers: Array) -> tuple[Array, Array]:
        """Return the rates at which the punishers and the defectors reproduce.

        They are given for every group.
        """
        members = float(self.members)
        z = punishers / members
        payoff_punisher = (self.benefit + self.interaction_cost) * z - (
            self.cost + self.fixed_cost + self.interaction_cost
        )
        payoff_defector = (self.benefit - self.punishment) * z
        punisher_births = punishers * np.maximum(1 + self.selection * payoff_punisher, 0.0)
        defector_births = (members - punishers) * np.maximum(1 + self.selection * payoff_defector, 0.0)
        return punisher_births, defector_births

    def compete(self, system: SystemView) -> None:
        """Draw the births, deaths, and conflicts of one unit of time.

        Args:
            system:
                The system.
        """
        groups = system.get_agents(self.target, PunisherGroupView)
        size = len(groups)
        if size < PAIR:
            return
        rng = system.rng
        members = float(self.members)
        punishers = np.array([group.punishers for group in groups])
        time = 0.0
        while True:
            punisher_births, defector_births = self._get_birth_rates(punishers)
            rates = np.concatenate([punisher_births, defector_births, np.full(size, float(self.conflict_rate))])
            total = rates.sum()
            time += rng.exponential(1 / total)
            if time > 1.0:
                break
            event = int(np.searchsorted(np.cumsum(rates), rng.random() * total, side='right'))
            group, kind = event % size, event // size
            if kind == 0:
                punishers[group] += rng.random() >= punishers[group] / members
            elif kind == 1:
                punishers[group] -= rng.random() < punishers[group] / members
            else:
                other = int(rng.integers(size - 1))
                other += other >= group
                victory = 0.5 + (punishers[group] - punishers[other]) / (2 * members)
                winner, loser = (group, other) if rng.random() < victory else (other, group)
                punishers[loser] = punishers[winner]
        for group_view, value in zip(groups, punishers.tolist(), strict=True):
            group_view.punishers = value


class TwoLevelBirthDeathSystem(BaseSystem):
    """The finite two-level birth-death process of defectors and altruistic punishers.

    The punishers and defectors live in competing groups. The population has `m` groups of `n` members. Within groups,
    punishers and defectors reproduce at the rates `1 + w pi_P(z)` and `1 + w pi_D(z)` of the fraction `z` of punishers,
    with `pi_P = (b + k) z - (c + q + k)` and `pi_D = (b - p) z`, and the offspring replaces a random member of the
    group. Every group fights a random other group at the rate `Lambda` and wins with the probability
    `1/2 + (z - u) / 2`, and the winner's copy replaces the loser (Cooney 2024, arXiv 2405.18419 section 3.1 and
    appendix A.1). With many large groups, the density of the fractions of punishers follows `TwoLevelReplicatorSystem`
    with `lambda = Lambda / w`, so between-group conflict sustains the punishers above `lambda* = c + q - p` and the
    defectors take over without it. The groups start with uniformly drawn numbers of punishers.

    The system has the group `groups`.

    Args:
        groups:
            The number `m` of groups.

        members:
            The number `n` of members of every group.

        selection:
            The intensity `w` of individual-level selection.

        conflict_rate:
            The rate `Lambda` at which a group fights another group.

        benefit:
            The benefit `b` of the donation game.

        cost:
            The cost `c` of cooperating.

        punishment:
            The punishment `p` of a defector by every punisher it meets.

        interaction_cost:
            The cost `k` a punisher pays per defector it meets.

        fixed_cost:
            The fixed cost `q` of being able to punish.

        random_state:
            The seed or generator of the initial groups.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.cooperation import TwoLevelBirthDeathSystem
        >>> TwoLevelBirthDeathSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'social/cooperation'
    topics = ('agents', 'multilevel selection', 'punishment', 'birth-death', 'gillespie')
    references = ('Cooney 2024', 'Luo 2014')
    until = 100

    def __init__(
        self,
        groups: int = 50,
        *,
        members: int = 20,
        selection: float = 0.1,
        conflict_rate: float = 0.2,
        benefit: float = 2.0,
        cost: float = 1.0,
        punishment: float = 0.5,
        interaction_cost: float = 0.5,
        fixed_cost: float = 0.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.groups = groups
        self.members = members
        self.selection = selection
        self.conflict_rate = conflict_rate
        self.benefit = benefit
        self.cost = cost
        self.punishment = punishment
        self.interaction_cost = interaction_cost
        self.fixed_cost = fixed_cost
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        population = GroupElement(
            'groups',
            element=PunisherGroupElement,
            size=self.groups,
            values={'punishers': rng.integers(self.members + 1, size=self.groups).astype(float)},
        )
        competition = TwoLevelBirthDeathInteraction(
            'competition',
            target='groups',
            members=self.members,
            selection=self.selection,
            conflict_rate=self.conflict_rate,
            benefit=self.benefit,
            cost=self.cost,
            punishment=self.punishment,
            interaction_cost=self.interaction_cost,
            fixed_cost=self.fixed_cost,
        )
        return CompositeSystem(self.label, elements=[population], interactions=[competition])
