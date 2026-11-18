"""Ecology systems: agent-based predation of prey by predators on a grid."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Mapping, Sequence
from functools import partial
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array
from ...general.agents import BirthDeathInteraction, MovementInteraction


class PredatorView(AgentView):
    """View of a predator, with its state and parameters as attributes.

    `PredatorElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    meals: float
    conversion: float
    death: float


class PredatorElement(Element):
    """Predator that walks at random, eats prey, breeds after a meal, and dies.

    In its own activation a predator walks to a random neighbouring cell, eats and removes one random prey of the new
    cell, counted in `meals`, breeds with the probability `conversion` after a meal, dies with the probability
    `death`, and digests its meals, by member interactions.

    Args:
        label:
            The label of the agent or group.

        meals:
            The prey the predator ate in the current step.

        conversion:
            The probability of breeding after eating a prey.

        death:
            The probability of dying at every step.

        values:
            Further values by name, such as the cells `x` and `y` of the predators, or `None`.

        size:
            The number of predators, or `None` for one.

        space:
            The grid the predators walk on, whose axes `x` and `y` place them, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import PredatorElement
        >>> PredatorElement('foxes', size=50).size
        50
    """

    visual = VisualSpec('agents', {'radius': 0.4})

    def __init__(
        self,
        label: str,
        *,
        meals: Annotated[float | Array, NN] = 0.0,
        conversion: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.5,
        death: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.1,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.meals = meals
        self.conversion = conversion
        self.death = death

    @classmethod
    def _get_view(cls) -> type[PredatorView]:
        """Return the view of a predator, `PredatorView`."""
        return PredatorView


class PreyView(AgentView):
    """View of a prey, with its parameters as attributes.

    `PreyElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    birth: float


class PreyElement(Element):
    """Prey that walks at random and gives birth when alone in its cell.

    In its own activation a prey walks to a random neighbouring cell and then gives birth with the probability
    `birth` when no other prey shares the new cell, by member interactions.

    Args:
        label:
            The label of the agent or group.

        birth:
            The probability of giving birth at every step.

        values:
            Further values by name, such as the cells `x` and `y` of the prey, or `None`.

        size:
            The number of prey, or `None` for one.

        space:
            The grid the prey walk on, whose axes `x` and `y` place them, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import PreyElement
        >>> PreyElement('rabbits', size=200).size
        200
    """

    visual = VisualSpec('agents', {'radius': 0.3})

    def __init__(
        self,
        label: str,
        *,
        birth: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.2,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.birth = birth

    @classmethod
    def _get_view(cls) -> type[PreyView]:
        """Return the view of a prey, `PreyView`."""
        return PreyView


class AgentBasedLotkaVolterraSystem(BaseSystem):
    """Prey and predators that walk on a grid, whose numbers cycle.

    Their numbers cycle like the Lotka-Volterra equations.

    Prey give birth only in cells they do not share with other prey, which limits their density, and predators eat prey
    they meet and breed after a meal. Demographic noise makes the cycles irregular and can end them by extinction. The
    system has the groups `prey` and `predators`.

    Args:
        width:
            The number of columns of the torus.

        height:
            The number of rows of the torus.

        prey:
            The initial number of prey.

        predators:
            The initial number of predators.

        birth:
            The probability that a prey gives birth at every step.

        conversion:
            The probability that a predator breeds after eating a prey.

        death:
            The probability that a predator dies at every step.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.ecology import AgentBasedLotkaVolterraSystem
        >>> system = AgentBasedLotkaVolterraSystem(10, 10, prey=50, predators=5).prepare().system_
        >>> [group.size for group in system.elements]
        [50, 5]
    """

    field = 'biology/ecology'
    topics = ('predator-prey', 'agents', 'grid', 'oscillation')
    references = ('Argun et al. 2021', 'Lotka 1925', 'Volterra 1926')
    until = 200

    def __init__(
        self,
        width: int = 30,
        height: int = 30,
        *,
        prey: int = 300,
        predators: int = 60,
        birth: float = 0.2,
        conversion: float = 0.5,
        death: float = 0.1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.width = width
        self.height = height
        self.prey = prey
        self.predators = predators
        self.birth = birth
        self.conversion = conversion
        self.death = death
        self.random_state = random_state

    @staticmethod
    def _get_birth(prey: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the probability `birth` that a prey gives birth.

        It is zero when other prey share its cell.
        """
        del system
        if not isinstance(prey, PreyView) or prey.get_agents_here(prey.group):
            return 0.0
        return float(prey.birth)

    @staticmethod
    def _get_conversion(predator: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the probability `conversion` that a predator breeds in this step.

        It is the probability of breeding after a meal, and zero without one.
        """
        del system
        return float(predator.conversion) if isinstance(predator, PredatorView) and predator.meals > 0 else 0.0

    @staticmethod
    def _get_death(predator: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the probability `death` that a predator dies in a step."""
        del system
        return float(predator.death) if isinstance(predator, PredatorView) else 0.0

    @staticmethod
    def _digest(predator: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Clear the meals of a predator, which starts its next step hungry."""
        del system
        if isinstance(predator, PredatorView):
            predator.meals = 0.0

    @staticmethod
    def _eat(predator: AgentViewProtocol, system: SystemViewProtocol, *, prey: str) -> None:
        """Let a predator eat one prey of its cell, drawn at random among several, which dies."""
        groups = set(system.get_groups(prey))
        found = [other for other in predator.get_agents_here() if other.group in groups]
        if len(found) > 1:
            found = [found[int(index)] for index in system.rng.permutation(len(found))[:1]]
        for item in found:
            if predator.alive and item.alive:
                if isinstance(predator, PredatorView):
                    predator.meals += 1.0
                item.remove()

    def _get_prey(self, group: str) -> list[MovementInteraction | BirthDeathInteraction]:
        """Return the member interactions of a prey.

        A prey walks, then gives birth when alone in the new cell.
        """
        return [
            MovementInteraction(f'{group}_walk', target=group, schedule='member'),
            BirthDeathInteraction(f'{group}_birth', target=group, birth=self._get_birth, schedule='member'),
        ]

    def _get_predators(
        self,
        group: str,
        prey: str,
    ) -> list[MovementInteraction | Interaction | BirthDeathInteraction]:
        """Return the member interactions of a predator.

        A predator walks, eats a prey of its cell, breeds after it, dies, and digests.
        """
        eat = partial(self._eat, prey=prey)
        return [
            MovementInteraction(f'{group}_walk', target=group, schedule='member'),
            Interaction(f'{group}_eating', roles={'target': group}, laws=lambda s: [UpdateLaw(s.target, eat)]),
            BirthDeathInteraction(f'{group}_breeding', target=group, birth=self._get_conversion, schedule='member'),
            BirthDeathInteraction(f'{group}_death', target=group, death=self._get_death, schedule='member'),
            Interaction(
                f'{group}_digestion',
                roles={'target': group},
                laws=lambda s: [UpdateLaw(s.target, self._digest)],
            ),
        ]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of prey and predators that walk on a grid.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        cells = rng.integers(0, [self.width, self.height], size=(self.prey + self.predators, 2))
        space = LatticeSpace((self.width, self.height), boundary='periodic')
        prey, predators = cells[: self.prey], cells[self.prey :]
        self.system_ = System(
            self.label,
            elements=[
                PreyElement(
                    'prey',
                    birth=self.birth,
                    values={'x': prey[:, 0], 'y': prey[:, 1]},
                    size=self.prey,
                    space=space,
                ),
                PredatorElement(
                    'predators',
                    conversion=self.conversion,
                    death=self.death,
                    values={'x': predators[:, 0], 'y': predators[:, 1]},
                    size=self.predators,
                    space=space,
                ),
            ],
            interactions=[*self._get_prey('prey'), *self._get_predators('predators', 'prey')],
            space=space,
        )
