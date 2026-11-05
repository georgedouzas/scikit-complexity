"""Mortal person who eats, ages, raises children, moves, and, in the monetary economy, works and founds firms."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._plot_view import PlotView
from ._settler_view import SettlerView

PROPENSITIES = {'fertility': 0.3, 'exploration': 1.0, 'enterprise': 0.3, 'restlessness': 0.1}


ALIVE, STARVED, AGED = 0.0, 1.0, 2.0


OLD_AGE_DEATH = 1 / 7


class SettlerElement(Element):
    """Mortal person who eats, ages, raises children, moves, and, in the monetary economy, works and founds firms.

    Every step a person eats `need` food and starves when its food runs out, and past `lifespan` it dies of old age
    with probability 1/7. A child takes `need` food from its parent and follows it. A fed adult has a child with the
    probability `fertility`, at most once every `spacing` steps, and gives it `endowment` food. The child inherits the
    four propensities, each moved by a normal mutation and kept in [0, 1]. A person that harvested too little and holds
    less than `reserve` steps of food moves, far with the probability `exploration`. In the monetary economy an
    unemployed adult on a patch with food but no firm founds one with the probability `enterprise`, and a worker quits
    with the probability `restlessness`.

    Args:
        label:
            The label of the person or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        food:
            The food the person holds.

        money:
            The money the person holds.

        loans:
            The loans the person holds, negative for a debt taken over from a firm.

        age:
            The age in steps.

        parent:
            The id of the parent, or -1.

        employer:
            The id of the firm that employs the person, or -1.

        harvested:
            The food the person harvested or earned in the step.

        born:
            The step of the person's last child.

        fate:
            0 while alive, 1 after starving, and 2 after dying of old age, until the census removes the person.

        fertility:
            The probability of a child in a step.

        exploration:
            The probability of moving far.

        enterprise:
            The probability of founding a firm.

        restlessness:
            The probability of quitting a job.

        labour:
            The labour of an adult in a step.

        need:
            The food a person eats in a step.

        maturity:
            The age of adulthood.

        lifespan:
            The age after which death by old age is possible.

        spacing:
            The fewest steps between two children.

        endowment:
            The food a parent gives a newborn child.

        mutation:
            The standard deviation of the mutation of a propensity.

        sight:
            The distance within which a person sees firms and explores.

        reserve:
            The steps of food a person buys up to, and below which a person that harvests too little moves.

        monetary:
            One in the monetary economy, zero among hunter-gatherers.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.macroeconomics import SettlerElement
        >>> GroupElement('people', element=SettlerElement, size=3, variable=True).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        food: Annotated[float, StateSpec()] = 10.0,
        money: Annotated[float, StateSpec(quantity='money')] = 0.0,
        loans: Annotated[float, StateSpec(quantity='credit')] = 0.0,
        age: Annotated[float, StateSpec(domain=NN)] = 20.0,
        parent: Annotated[float, StateSpec(reference='people')] = -1.0,
        employer: Annotated[float, StateSpec(reference='businesses')] = -1.0,
        harvested: Annotated[float, StateSpec()] = 0.0,
        born: Annotated[float, StateSpec()] = -100.0,
        fate: Annotated[float, StateSpec()] = ALIVE,
        fertility: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        exploration: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        enterprise: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        restlessness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        labour: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        need: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        maturity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        lifespan: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 85.0,
        spacing: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        endowment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        mutation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        sight: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        reserve: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        monetary: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.food = food
        self.money = money
        self.loans = loans
        self.age = age
        self.parent = parent
        self.employer = employer
        self.harvested = harvested
        self.born = born
        self.fate = fate
        self.fertility = fertility
        self.exploration = exploration
        self.enterprise = enterprise
        self.restlessness = restlessness
        self.labour = labour
        self.need = need
        self.maturity = maturity
        self.lifespan = lifespan
        self.spacing = spacing
        self.endowment = endowment
        self.mutation = mutation
        self.sight = sight
        self.reserve = reserve
        self.monetary = monetary

    @classmethod
    def _get_view(cls) -> type[SettlerView]:
        """Return the view of a person, with its states and parameters."""
        return SettlerView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of living in the phase of living.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.live, phase='living')]

    def _feed_child(self, agent: SettlerView, system: SystemView) -> None:
        """Take food from the living parent and follow it."""
        parent = agent.referenced('parent')
        if not isinstance(parent, SettlerView):
            return
        given = min(agent.need, max(parent.food - parent.need, 0.0))
        parent.food -= given
        agent.food += given
        agent.move_to(parent.location)

    def _breed(self, agent: SettlerView, system: SystemView) -> None:
        """Have a child with mutated propensities at the probability of fertility."""
        fed = agent.food >= agent.endowment + agent.need
        spaced = system.time - agent.born >= agent.spacing
        if not (fed and spaced and system.rng.random() < agent.fertility):
            return
        traits = {
            name: float(np.clip(getattr(agent, name) + system.rng.normal(0.0, agent.mutation), 0.0, 1.0))
            for name in PROPENSITIES
        }
        agent.food -= agent.endowment
        agent.born = system.time
        agent.create(None, None, food=agent.endowment, age=0.0, parent=float(agent.id), born=system.time, **traits)

    def _work_or_found(self, agent: SettlerView, system: SystemView) -> bool:
        """Quit a job at the restlessness, or found a firm on a free patch with food, and return whether employed."""
        if agent.employer >= 0:
            if system.rng.random() < agent.restlessness:
                agent.employer = -1.0
            return agent.employer >= 0
        if agent.agents_here('businesses') or SettlerElement._patch_here(agent).food <= 0:
            return False
        if system.rng.random() >= agent.enterprise:
            return False
        firm = agent.create('businesses', owner=float(agent.id))
        agent.employer = float(firm.id)
        return True

    def live(self, agent: SettlerView, system: SystemView) -> None:
        """Age, eat, die, have children, and move or work.

        Args:
            agent:
                The person.

            system:
                The system.
        """
        agent.age += 1
        adult = agent.age >= agent.maturity
        if not adult:
            self._feed_child(agent, system)
        agent.food -= agent.need
        if agent.food < 0:
            agent.fate = STARVED
            return
        if agent.age > agent.lifespan and system.rng.random() < OLD_AGE_DEATH:
            agent.fate = AGED
            return
        if not adult:
            return
        self._breed(agent, system)
        employed = agent.monetary > 0 and self._work_or_found(agent, system)
        if not employed and agent.harvested < agent.need and agent.food < agent.reserve * agent.need:
            SettlerElement._wander(agent, system, agent.sight)
        agent.harvested = 0.0

    @staticmethod
    def _patch_here(agent: AgentView) -> PlotView:
        """Return the patch of the cell of an agent."""
        return next(patch for patch in agent.agents_here('patches') if isinstance(patch, PlotView))

    @staticmethod
    def _cell(agent: AgentView) -> tuple[int, ...]:
        """Return the cell of an agent of the lattice."""
        location = agent.location
        if not isinstance(location, tuple):
            error_msg = f'The agent {agent.id} is not on a lattice cell.'
            raise TypeError(error_msg)
        return tuple(int(item) for item in location)

    @staticmethod
    def _wander(agent: SettlerView, system: SystemView, sight: float) -> None:
        """Move to a random cell within sight with the propensity to explore, and otherwise to a neighbouring cell."""
        if system.rng.random() < agent.exploration:
            width, height = system.dimensions or (1, 1)
            reach = int(sight)
            dx, dy = system.rng.integers(-reach, reach + 1, size=2)
            x, y = SettlerElement._cell(agent)
            agent.move_to(((x + int(dx)) % int(width), (y + int(dy)) % int(height)))
            return
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(system.rng.integers(len(cells)))])
