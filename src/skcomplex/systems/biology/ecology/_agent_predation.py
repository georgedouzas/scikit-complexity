"""Ecology systems: agent-based predation of prey by predators on a grid."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import (
    LifeCycleInteraction,
    MapInteraction,
    MovementInteraction,
    TransferInteraction,
)
from ....core.components.laws import Source
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class PredatorView(AgentView):
    """View of a predator, with its state and parameters as attributes.

    `PredatorElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    meals: float
    conversion: float
    death: float


class PredatorElement(Element):
    """Predator that walks at random, eats a prey of its cell, breeds after a meal, and dies.

    In its own activation a predator walks to a random neighbouring cell, eats and removes one random prey of the new
    cell, counted in `meals`, breeds with the probability `conversion` after a meal, dies with the probability
    `death`, and digests its meals, by member interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        meals:
            The prey the predator ate in the current step.

        conversion:
            The probability of breeding after eating a prey.

        death:
            The probability of dying at every step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import PredatorElement
        >>> GroupElement('foxes', element=PredatorElement, size=50).size
        50
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.4}),
        *,
        meals: Annotated[float, StateSpec(domain=NN)] = 0.0,
        conversion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        death: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
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
    """Prey that walks at random and gives birth in cells it does not share with other prey.

    In its own activation a prey walks to a random neighbouring cell and then gives birth with the probability
    `birth` when no other prey shares the new cell, by member interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        birth:
            The probability of giving birth at every step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import PreyElement
        >>> GroupElement('rabbits', element=PreyElement, size=200).size
        200
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.3}),
        *,
        birth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.birth = birth

    @classmethod
    def _get_view(cls) -> type[PreyView]:
        """Return the view of a prey, `PreyView`."""
        return PreyView


def _get_birth(prey: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return the probability `birth` that a prey gives birth, zero when other prey share its cell."""
    del system
    if not isinstance(prey, PreyView) or prey.agents_here(prey.group):
        return 0.0
    return float(prey.birth)


def _get_conversion(predator: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return the probability `conversion` that a predator breeds after a meal in this step, zero without one."""
    del system
    return float(predator.conversion) if isinstance(predator, PredatorView) and predator.meals > 0 else 0.0


def _get_death(predator: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return the probability `death` that a predator dies in a step."""
    del system
    return float(predator.death) if isinstance(predator, PredatorView) else 0.0


def _get_digested(predator: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return zero, the meals a predator starts its next step with."""
    del predator, system
    return 0.0


def _get_prey(group: str) -> list[MovementInteraction | LifeCycleInteraction]:
    """Return the member interactions of a prey: walk, then give birth when alone in the new cell."""
    return [
        MovementInteraction(f'{group}_walk', target=group, schedule='member'),
        LifeCycleInteraction(f'{group}_birth', target=group, birth=_get_birth, schedule='member'),
    ]


def _get_predators(
    group: str,
    prey: str,
) -> list[MovementInteraction | TransferInteraction | LifeCycleInteraction | MapInteraction]:
    """Return the member interactions of a predator: walk, eat a prey of its cell, breed after it, die, digest."""
    return [
        MovementInteraction(f'{group}_walk', target=group, schedule='member'),
        TransferInteraction(
            f'{group}_eating',
            source=prey,
            target=group,
            state=Source(),
            into='meals',
            amount=1.0,
            pairing='cell',
            form='agent',
            schedule='member',
            actor='target',
            partners=1,
            remove='source',
        ),
        LifeCycleInteraction(f'{group}_breeding', target=group, birth=_get_conversion, schedule='member'),
        LifeCycleInteraction(f'{group}_death', target=group, death=_get_death, schedule='member'),
        MapInteraction(
            f'{group}_digestion',
            target=group,
            state='meals',
            function=_get_digested,
            form='agent',
            schedule='member',
        ),
    ]


@register_system(
    field='biology/ecology',
    tags=('predator-prey', 'agents', 'grid', 'oscillation'),
    references=('Argun et al. 2021', 'Lotka 1925', 'Volterra 1926'),
    defaults={'until': 200},
)
def agent_based_lotka_volterra(
    width: int = 30,
    height: int = 30,
    *,
    prey: int = 300,
    predators: int = 60,
    birth: float = 0.2,
    conversion: float = 0.5,
    death: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return prey and predators that walk on a grid, whose numbers cycle like the Lotka-Volterra equations.

    Prey give birth only in cells they do not share with other prey, which limits their density, and predators eat
    prey they meet and breed after a meal. Demographic noise makes the cycles irregular and can end them by extinction.

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

    Returns:
        system:
            The system, with the groups `prey` and `predators`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import agent_based_lotka_volterra
        >>> [group.size for group in agent_based_lotka_volterra(10, 10, prey=50, predators=5).elements]
        [50, 5]
    """
    rng = np.random.default_rng(random_state)
    cells = rng.integers(0, [width, height], size=(prey + predators, 2))
    locations = [tuple(int(item) for item in cell) for cell in cells]
    return System(
        elements=[
            GroupElement(
                'prey',
                element=PreyElement,
                size=prey,
                values={'birth': birth},
                variable=True,
                location=locations[:prey],
            ),
            GroupElement(
                'predators',
                element=PredatorElement,
                size=predators,
                values={'conversion': conversion, 'death': death},
                variable=True,
                location=locations[prey:],
            ),
        ],
        interactions=[*_get_prey('prey'), *_get_predators('predators', 'prey')],
        space=LatticeSpace((width, height), boundary='periodic'),
    )
