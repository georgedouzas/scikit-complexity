"""Ecology systems: predators and prey that share emotions through marks they leave on the ground."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Callable
from functools import partial

import numpy as np

from ....core.components.elements import FieldElement, GroupElement
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
from ..._catalog import register_system
from ..common.stigmergy import EvaporationInteraction
from ._emotion_marks import EmotionMarksElement
from ._forager import PLANTS, ForagerElement
from ._forager_view import ForagerView
from ._plant import PlantElement
from ._plant_view import PlantView
from ._stigmergic_predator import PREY, StigmergicPredatorElement
from ._stigmergic_prey import StigmergicPreyElement

MARKS = 'ground/marks'


PREDATORS = 'predators'


EMOTIONS = ('prey_happiness', 'prey_fear', 'predator_happiness', 'predator_anger')


PREY_TRAITS: dict[str, float] = {'maturity': 3.0, 'reproduction': 0.5, 'starvation': 10.0}


PREDATOR_TRAITS: dict[str, float] = {'maturity': 20.0, 'reproduction': 0.1, 'starvation': 8.0}


def _kind(agent: AgentViewProtocol) -> str:
    """Return the label of an agent's group, without the path of its system."""
    return agent.group.rpartition('/')[2]


def _occupants(agent: AgentViewProtocol) -> dict[object, AgentViewProtocol]:
    """Return the agents around an agent by their cell."""
    return {other.location: other for other in agent.neighbors()}


def _empty_cells(agent: AgentViewProtocol) -> list[object]:
    """Return the neighbouring cells of an agent that no agent occupies."""
    occupied = _occupants(agent)
    return [cell for cell in agent.neighbor_cells() if cell not in occupied]


def _get_empty_cell(agent: AgentViewProtocol | None, system: SystemViewProtocol) -> object:
    """Return a random empty neighbouring cell of a parent, or `None` when there is none."""
    cells = [] if agent is None else _empty_cells(agent)
    return cells[int(system.rng.integers(len(cells)))] if cells else None


def _get_one(agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return one, the step by which the age and the hunger of an organism grow."""
    del agent, system
    return 1.0


def _spreads(plant: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether a mature plant has an empty neighbouring cell to spread into."""
    del system
    return isinstance(plant, PlantView) and plant.age >= plant.maturity and bool(_empty_cells(plant))


def _starves(animal: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether the hunger of an animal passes `starvation`, so it dies."""
    del system
    return isinstance(animal, ForagerView) and animal.hunger > animal.starvation


def _get_fear(prey: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return the number of predators around a prey, its experience of fear."""
    del system
    return float(sum(_kind(other) == PREDATORS for other in _occupants(prey).values()))


def _get_anger(predator: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return the hunger of a predator as a share of its starvation, its experience of anger."""
    del system
    return float(predator.hunger / predator.starvation) if isinstance(predator, ForagerView) else 0.0


def _get_preference(
    animal: ForagerView,
    system: SystemViewProtocol,
    cell: object,
    occupant: AgentViewProtocol | None,
    marks: tuple[str, str, str],
) -> float:
    """Return the preference of an animal for a neighbouring cell, given its food and the marks it reads."""
    food, happiness_mark, distress_mark = marks
    meal = 1.0 if occupant is not None and _kind(occupant) == food else 0.0
    if not animal.emotions:
        return meal
    happiness = system.field(f'{MARKS}.{happiness_mark}', cell) - animal.happiness
    distress = system.field(f'{MARKS}.{distress_mark}', cell) - animal.distress
    return meal + happiness - distress


def _get_preferred(animal: AgentViewProtocol, system: SystemViewProtocol, marks: tuple[str, str, str]) -> object:
    """Return the empty or food cell around an animal of highest preference, with a little noise, or `None`."""
    if not isinstance(animal, ForagerView):
        return None
    occupants = _occupants(animal)
    options = [
        (cell, occupants.get(cell))
        for cell in animal.neighbor_cells()
        if cell not in occupants or _kind(occupants[cell]) == marks[0]
    ]
    if not options:
        return None
    noise = system.rng.random(len(options)) * 1e-3
    scores = [_get_preference(animal, system, cell, occupant, marks) for cell, occupant in options]
    return options[int(np.argmax(np.asarray(scores) + noise))][0]


def _get_fed(food: AgentViewProtocol, animal: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return minus the hunger of an animal, which a meal clears."""
    del food, system
    return -float(animal.hunger) if isinstance(animal, ForagerView) else 0.0


def _get_marked(animal: ForagerView, system: SystemViewProtocol, mark: str) -> float:
    """Return the mean of a mark over the cells around an animal."""
    return float(np.mean([system.field(f'{MARKS}.{mark}', cell) for cell in animal.neighbor_cells()]))


def _get_happiness(animal: AgentViewProtocol, system: SystemViewProtocol, mark: str) -> float:
    """Return the next happiness of an animal, from a meal in the step and the happiness marked around it."""
    if not isinstance(animal, ForagerView):
        return 0.0
    if not animal.emotions:
        return float(animal.happiness)
    ate = float(animal.hunger == 0)
    keep, share = animal.memory, animal.communication
    return (1 - keep) * (ate + share * _get_marked(animal, system, mark)) + keep * animal.happiness


def _get_distress(animal: AgentViewProtocol, system: SystemViewProtocol, mark: str) -> float:
    """Return the next distress of an animal, from its experience of the step and the distress marked around it."""
    if not isinstance(animal, ForagerView):
        return 0.0
    if not animal.emotions:
        return float(animal.distress)
    keep, share = animal.memory, animal.communication
    return (1 - keep) * (animal.experience + share * _get_marked(animal, system, mark)) + keep * animal.distress


def _feels(animal: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> bool:
    """Return whether emotions guide an animal, so it marks its cell."""
    del cell, system
    return isinstance(animal, ForagerView) and bool(animal.emotions)


def _get_marked_happiness(animal: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> float:
    """Return the happiness an animal marks its cell with."""
    del cell, system
    return float(animal.happiness) if isinstance(animal, ForagerView) else 0.0


def _get_marked_distress(animal: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> float:
    """Return the distress an animal marks its cell with."""
    del cell, system
    return float(animal.distress) if isinstance(animal, ForagerView) else 0.0


def _get_breeding(animal: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return the probability that a mature animal with an empty neighbouring cell breeds, raised by happiness."""
    del system
    if not isinstance(animal, ForagerView) or animal.age < animal.maturity or not _empty_cells(animal):
        return 0.0
    return float(animal.reproduction * (1 + animal.emotions * (animal.happiness - animal.distress)))


def _get_growth(system: str, label: str, state: str) -> MapInteraction:
    """Return the map that raises a state of every member of a group by one, in the activation of the member."""
    return MapInteraction(
        f'{label}_{state}',
        target=f'{system}/{label}',
        state=state,
        function=_get_one,
        mode='increment',
        form='agent',
        schedule='member',
    )


def _get_plants(system: str, label: str) -> list[MapInteraction | LifeCycleInteraction]:
    """Return the member interactions of a plant: age, and spread into an empty neighbouring cell at maturity."""
    return [
        _get_growth(system, label, 'age'),
        LifeCycleInteraction(
            f'{label}_spreading',
            target=f'{system}/{label}',
            birth=_spreads,
            placement=_get_empty_cell,
            reset={'age': 0.0},
            schedule='member',
        ),
    ]


def _get_animals(
    system: str,
    label: str,
    element: type['ForagerElement'],
    sense: Callable[[AgentViewProtocol, SystemViewProtocol], float],
) -> list[MapInteraction | LifeCycleInteraction | MovementInteraction | TransferInteraction]:
    """Return the member interactions of an animal: age, starve, sense, move and eat, feel, mark, and breed."""
    group = f'{system}/{label}'
    marks = (element.food, element.happiness_mark, element.distress_mark)
    return [
        _get_growth(system, label, 'age'),
        _get_growth(system, label, 'hunger'),
        LifeCycleInteraction(f'{label}_starvation', target=group, death=_starves, schedule='member'),
        MapInteraction(
            f'{label}_experience',
            target=group,
            state='experience',
            function=sense,
            form='agent',
            schedule='member',
        ),
        MovementInteraction(
            f'{label}_foraging',
            target=group,
            step=partial(_get_preferred, marks=marks),
            schedule='member',
        ),
        TransferInteraction(
            f'{label}_meal',
            source=f'{system}/{element.food}',
            target=group,
            state=Source(),
            into='hunger',
            amount=_get_fed,
            pairing='cell',
            actor='target',
            remove='source',
            form='agent',
            schedule='member',
        ),
        MapInteraction(
            f'{label}_happiness',
            target=group,
            state='happiness',
            function=partial(_get_happiness, mark=element.happiness_mark),
            form='agent',
            schedule='member',
        ),
        MapInteraction(
            f'{label}_distress',
            target=group,
            state='distress',
            function=partial(_get_distress, mark=element.distress_mark),
            form='agent',
            schedule='member',
        ),
        TransferInteraction(
            f'{label}_marking',
            source=group,
            target=MARKS,
            state=Source(),
            into=element.happiness_mark,
            amount=_get_marked_happiness,
            legs=((Source(), element.distress_mark, _get_marked_distress),),
            pairing='site',
            guard=_feels,
            form='agent',
            schedule='member',
        ),
        LifeCycleInteraction(
            f'{label}_breeding',
            target=group,
            birth=_get_breeding,
            placement=_get_empty_cell,
            inherit=('hunger',),
            reset={'age': 0.0},
            schedule='member',
        ),
    ]


def _random_cells(rng: np.random.Generator, width: int, height: int, counts: list[int]) -> list[list[tuple[int, int]]]:
    """Return distinct random cells for several groups."""
    picks = rng.choice(width * height, size=sum(counts), replace=False)
    cells = [(int(pick % width), int(pick // width)) for pick in picks]
    bounds = np.cumsum([0, *counts])
    return [cells[start:end] for start, end in itertools.pairwise(bounds)]


@register_system(
    field='biology/ecology',
    tags=('predator-prey', 'agents', 'stigmergy', 'emotions', 'co-simulation'),
    references=('Carmichael and Hadžikadić 2019', 'Olsen 2019'),
    defaults={'until': 100, 'dt': 1},
)
def stigmergic_predator_prey(
    width: int = 30,
    height: int = 30,
    *,
    plants: int = 300,
    prey: int = 120,
    predators: int = 20,
    emotions: bool = True,
    decay: float = 0.2,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return plants, prey, and predators on a grid, whose animals share emotions through decaying marks.

    Every cell holds at most one organism. Prey eat plants and fear predators, and predators eat prey and grow angry
    with hunger. Happiness draws animals of the same kind towards the cells where it was marked and distress drives
    them away, a trail stigmergy of emotions that the cellular automaton of Olsen in Carmichael and Hadžikadić (2019)
    uses to study cooperation within each species.

    Args:
        width:
            The number of columns of the torus.

        height:
            The number of rows of the torus.

        plants:
            The initial number of plants.

        prey:
            The initial number of prey.

        predators:
            The initial number of predators.

        emotions:
            Whether emotions guide the animals.

        decay:
            The rate at which the marks decay.

        random_state:
            The seed or generator of the initial cells.

    Returns:
        system:
            The system, with the groups `life/plants`, `life/prey`, and `life/predators`, and the marks
            `ground/marks`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import stigmergic_predator_prey
        >>> [child.label for child in stigmergic_predator_prey(10, 10, plants=20, prey=10, predators=5).systems]
        ['life', 'ground']
    """
    rng = np.random.default_rng(random_state)
    plant_cells, prey_cells, predator_cells = _random_cells(rng, width, height, [plants, prey, predators])
    flag = float(emotions)
    life = System(
        'life',
        elements=[
            GroupElement(
                PLANTS,
                element=PlantElement,
                size=plants,
                values={'age': rng.integers(0, 4, plants).astype(float)},
                variable=True,
                location=plant_cells,
            ),
            GroupElement(
                PREY,
                element=StigmergicPreyElement,
                size=prey,
                values={'emotions': flag, **PREY_TRAITS},
                variable=True,
                location=prey_cells,
            ),
            GroupElement(
                PREDATORS,
                element=StigmergicPredatorElement,
                size=predators,
                values={'emotions': flag, **PREDATOR_TRAITS},
                variable=True,
                location=predator_cells,
            ),
        ],
        space=LatticeSpace((width, height), boundary='periodic'),
    )
    marks = FieldElement(
        'marks',
        element=EmotionMarksElement,
        space=LatticeSpace((width, height)),
        values={'decay': decay},
    )
    evaporations = [
        EvaporationInteraction(f'evaporation_{mark}', target='marks', state=mark, evaporation='decay')
        for mark in EMOTIONS
    ]
    return System(
        systems=[life, System('ground', elements=[marks], interactions=evaporations)],
        interactions=[
            *_get_plants('life', PLANTS),
            *_get_animals('life', PREY, StigmergicPreyElement, _get_fear),
            *_get_animals('life', PREDATORS, StigmergicPredatorElement, _get_anger),
        ],
    )
