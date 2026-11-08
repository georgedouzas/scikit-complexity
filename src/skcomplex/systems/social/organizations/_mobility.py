"""Mobility systems: individual trajectories.

They follow exploration and preferential return, or move between nested containers.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class ExplorerView(AgentView):
    """View of an individual of the mobility systems, with its states as attributes."""

    x: float
    y: float
    distinct: float
    level: float


class ExplorerElement(Element):
    """Individual of the mobility systems.

    It holds its position, its number of distinct locations, and its last level.

    Args:
        label:
            The label of the individual or the group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        x:
            The first coordinate of the current location.

        y:
            The second coordinate of the current location.

        distinct:
            The number of distinct locations visited.

        level:
            The level-distance of the last displacement in a hierarchy of containers, 0 without one.

    Examples:
        >>> from skcomplex.systems.social.organizations import ExplorerElement
        >>> ExplorerElement('individual').distinct
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        distinct: Annotated[float, StateSpec(domain=NN)] = 1.0,
        level: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.distinct = distinct
        self.level = level

    @classmethod
    def _get_view(cls) -> type[ExplorerView]:
        """Return the class of the views of the agents, whose attributes are the states."""
        return ExplorerView


class ExplorationInteraction(Interaction):
    """Exploration and preferential return of Song and coauthors.

    There is one displacement per individual and step. An individual who has visited `S` distinct locations explores a
    new one with the probability `rho S^-gamma`, at a distance drawn from the power law `P(r) ~ r^-beta` above
    `least_jump` in a uniform direction, and otherwise returns to a visited location with the probability proportional
    to its number of visits.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of individuals.

        rho:
            The factor `rho` of the probability of exploring.

        gamma:
            The exponent `gamma` of the decay of the probability of exploring.

        beta:
            The exponent `beta` of the power law of the jump lengths, above 1.

        least_jump:
            The least jump length.

    Examples:
        >>> from skcomplex.systems.social.organizations import ExplorationInteraction
        >>> ExplorationInteraction('displacement', target='individuals').rho
        0.6
    """

    draws = True

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        rho: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.6,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.21,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(1, oo))] = 1.55,
        least_jump: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.rho = rho
        self.gamma = gamma
        self.beta = beta
        self.least_jump = least_jump

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the displacement rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.move)]

    def move(self, system: SystemView) -> None:
        """Displace every individual once, to a new location or back to a visited one.

        Args:
            system:
                The system.
        """
        rng = system.rng
        memory = system.storage
        for individual in system.agents(self.target, ExplorerView):
            places = memory.setdefault(f'places {individual.id}', [(individual.x, individual.y)])
            visits = memory.setdefault(f'visits {individual.id}', [1.0])
            if not isinstance(places, list) or not isinstance(visits, list):
                continue
            if rng.random() < self.rho * len(places) ** -self.gamma:
                length = self.least_jump * (1 - rng.random()) ** (-1 / (self.beta - 1))
                angle = 2 * np.pi * rng.random()
                place = (individual.x + length * np.cos(angle), individual.y + length * np.sin(angle))
                places.append(place)
                visits.append(1.0)
            else:
                weights = np.asarray(visits)
                index = int(rng.choice(len(places), p=weights / weights.sum()))
                visits[index] += 1.0
                place = places[index]
            individual.x, individual.y = float(place[0]), float(place[1])
            individual.distinct = float(len(places))


class ContainerMobilityInteraction(Interaction):
    """Container model of Alessandretti and coauthors.

    It displaces individuals between locations nested in containers. Space is a nested grid of `levels` levels, every
    container of a level split into `side` by `side` containers of the level below, and the containers of the lowest
    level are the locations. The children of every container have attractivenesses drawn from a power law of the
    exponent `exponent` and normalized, and home is the location of the largest attractiveness at every level. Every
    step every individual draws the level-distance `d` of its displacement from `travel`, chooses a container of level
    `d` other than its own among the children of its container of level `d + 1`, by attractiveness, and then descends to
    a location by the attractivenesses of the children at every level below. Every individual has its own hierarchy,
    drawn at its first step, and starts at home.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of individuals.

        levels:
            The number `L` of levels.

        side:
            The number of children of a container along every axis.

        travel:
            The probabilities of displacements at the level-distances 1 to `L`, the same from any distance from home.

        exponent:
            The exponent of the power law of the attractivenesses.

    Raises:
        ValidationError:
            The probabilities of travel do not match the levels, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.social.organizations import ContainerMobilityInteraction
        >>> ContainerMobilityInteraction('displacement', target='individuals').levels
        3
    """

    draws = True

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        levels: Annotated[int, ParameterSpec(domain=NN)] = 3,
        side: Annotated[int, ParameterSpec(domain=NN)] = 4,
        travel: Sequence[float] = (0.6, 0.3, 0.1),
        exponent: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.levels = levels
        self.side = side
        self.travel = travel
        self.exponent = exponent

    def _check(self) -> None:
        """Raise an error for probabilities of travel that do not match the levels.

        Raises:
            ValidationError:
                The probabilities of travel do not match the levels.
        """
        super()._check()
        if len(self.travel) != self.levels or not np.isclose(sum(self.travel), 1.0):
            error_msg = f'Give {self.levels} probabilities of travel that sum to 1, not {tuple(self.travel)}.'
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the displacement rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.move)]

    def _get_attractiveness(self, rng: np.random.Generator) -> list[Array]:
        """Return the attractiveness of every container of every level.

        Child 0 of every container is the largest.
        """
        children = int(self.side) ** 2
        result = []
        for level in range(int(self.levels)):
            parents = children ** (int(self.levels) - level - 1)
            values = np.sort((1 - rng.random((parents, children))) ** (-1 / self.exponent), axis=1)[:, ::-1]
            result.append((values / values.sum(axis=1, keepdims=True)).ravel())
        return result

    def _get_position(self, location: int) -> tuple[float, float]:
        """Return the centre of a location on the nested grid.

        The location is a unit cell of the lowest level.
        """
        side, children = int(self.side), int(self.side) ** 2
        x = y = 0.0
        for level in range(int(self.levels)):
            child = (location // children**level) % children
            x += (child % side) * side**level
            y += (child // side) * side**level
        return x + 0.5, y + 0.5

    def _descend(self, container: int, level: int, attractiveness: list[Array], rng: np.random.Generator) -> int:
        """Return a location inside a container of a level.

        A child is chosen by attractiveness at every level below.
        """
        children = int(self.side) ** 2
        for below in range(level - 1, 0, -1):
            weights = attractiveness[below - 1][container * children : (container + 1) * children]
            container = container * children + int(rng.choice(children, p=weights))
        return container

    def move(self, system: SystemView) -> None:
        """Displace every individual once.

        The displacement is at a drawn level-distance, to a location chosen by attractiveness.

        Args:
            system:
                The system.
        """
        rng = system.rng
        memory = system.storage
        children, levels = int(self.side) ** 2, int(self.levels)
        travel = np.asarray(self.travel, dtype=float)
        for individual in system.agents(self.target, ExplorerView):
            if f'location {individual.id}' not in memory:
                memory[f'attractiveness {individual.id}'] = self._get_attractiveness(rng)
                memory[f'location {individual.id}'], memory[f'visited {individual.id}'] = 0, {0}
                individual.x, individual.y = self._get_position(0)
            attractiveness = memory[f'attractiveness {individual.id}']
            location = memory[f'location {individual.id}']
            visited = memory[f'visited {individual.id}']
            if not isinstance(attractiveness, list) or not isinstance(location, int) or not isinstance(visited, set):
                continue
            level = int(rng.choice(levels, p=travel)) + 1
            # The location is its digits of base `children`, the lowest level first, and the container of a level its
            # leading digits, so a container of level `level` is the location divided by `children^(level - 1)`.
            own = location // children ** (level - 1)
            siblings = (own // children) * children + np.arange(children)
            weights = attractiveness[level - 1][siblings].copy()
            weights[own % children] = 0.0
            chosen = int(siblings[rng.choice(children, p=weights / weights.sum())])
            location = self._descend(chosen, level, attractiveness, rng)
            visited.add(location)
            memory[f'location {individual.id}'] = location
            individual.x, individual.y = self._get_position(location)
            individual.level = float(level)
            individual.distinct = float(len(visited))


class ExplorationPreferentialReturnSystem(BaseSystem):
    """The exploration and preferential return model of human mobility.

    The model is that of Song and coauthors. At every displacement an individual who has visited `S` distinct locations
    explores a new place with the probability `rho S^-gamma`, at a distance drawn from the power law `P(r) ~ r^-beta`,
    and otherwise returns to a visited location chosen in proportion to its number of visits (arXiv 2109.07381 Methods,
    the baseline of Fig. 2). The number of distinct locations grows as `n^(1 / (1 + gamma))` after `n` displacements,
    and the visits of the locations follow Zipf's law. The defaults `rho = 0.6` and `gamma = 0.21` are the values of
    Song and coauthors, and every individual starts at the origin.

    The system has the group `individuals`.

    Args:
        n_individuals:
            The number of individuals.

        rho:
            The factor `rho` of the probability of exploring.

        gamma:
            The exponent `gamma` of the decay of the probability of exploring.

        beta:
            The exponent `beta` of the power law of the jump lengths, above 1.

        least_jump:
            The least jump length.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import ExplorationPreferentialReturnSystem
        >>> ExplorationPreferentialReturnSystem(3).prepare().system_.elements[0].prepare().size_
        3
    """

    field = 'social/organizations'
    topics = ('agents', 'human mobility', 'exploration', 'preferential return')
    references = ('Song, Koren, Wang, and Barabási 2010', 'Alessandretti, Aslak, and Lehmann 2020')
    until = 500

    def __init__(
        self,
        n_individuals: int = 10,
        *,
        rho: float = 0.6,
        gamma: float = 0.21,
        beta: float = 1.55,
        least_jump: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_individuals = n_individuals
        self.rho = rho
        self.gamma = gamma
        self.beta = beta
        self.least_jump = least_jump

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        individuals = GroupElement('individuals', element=ExplorerElement, size=self.n_individuals)
        displacement = ExplorationInteraction(
            'displacement',
            target='individuals',
            rho=self.rho,
            gamma=self.gamma,
            beta=self.beta,
            least_jump=self.least_jump,
        )
        return System(self.label, elements=[individuals], interactions=[displacement])


class ContainerMobilitySystem(BaseSystem):
    """The container model of human mobility of Alessandretti, Aslak, and Lehmann.

    Its trajectories cross scales. Space is a hierarchy of `L` levels of containers, a nested grid where every container
    holds `side` by `side` containers of the level below, down to the locations, and every container has an
    attractiveness among its siblings, drawn from a power law. An individual travels at the level-distance `d`, the
    highest level at which its origin and destination differ, with the probability `p_d`, chooses the destination
    container of level `d` among the other children of its container of level `d + 1` by attractiveness, and descends by
    attractiveness to a location (arXiv 2109.07381 Methods, Generating traces). The nested grid, the power law of the
    attractivenesses, and the probabilities of travel that do not depend on the distance from home follow Supplementary
    Note 3. Displacements are a mixture of the scales of the containers, from one cell to `side^(L - 1)` cells.

    The system has the group `individuals`.

    Args:
        n_individuals:
            The number of individuals.

        levels:
            The number `L` of levels.

        side:
            The number of children of a container along every axis.

        travel:
            The probabilities `p_d` of displacements at the level-distances 1 to `L`.

        exponent:
            The exponent of the power law of the attractivenesses.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import ContainerMobilitySystem
        >>> ContainerMobilitySystem(3).prepare().system_.elements[0].prepare().size_
        3
    """

    field = 'social/organizations'
    topics = ('agents', 'human mobility', 'hierarchy', 'containers', 'scales')
    references = ('Alessandretti, Aslak, and Lehmann 2020',)
    until = 500

    def __init__(
        self,
        n_individuals: int = 10,
        *,
        levels: int = 3,
        side: int = 4,
        travel: Sequence[float] = (0.6, 0.3, 0.1),
        exponent: float = 1.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_individuals = n_individuals
        self.levels = levels
        self.side = side
        self.travel = travel
        self.exponent = exponent

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        individuals = GroupElement('individuals', element=ExplorerElement, size=self.n_individuals)
        displacement = ContainerMobilityInteraction(
            'displacement',
            target='individuals',
            levels=self.levels,
            side=self.side,
            travel=self.travel,
            exponent=self.exponent,
        )
        return System(self.label, elements=[individuals], interactions=[displacement])
