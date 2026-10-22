"""Ecology models: predators and prey that share emotions through marks they leave on the ground."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from typing import Any, ClassVar

import numpy as np

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import AgentType, Element, Field, LatticeSpace, Law, Rate, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

MARKS = 'ground/marks'
PLANTS = 'plants'
PREY = 'prey'
PREDATORS = 'predators'
PREY_TRAITS: dict[str, Any] = {'maturity': 3.0, 'reproduction': 0.5, 'starvation': 10.0}
PREDATOR_TRAITS: dict[str, Any] = {'maturity': 20.0, 'reproduction': 0.1, 'starvation': 8.0}

EmotionMarks = Element.define(
    'EmotionMarks',
    states={
        'prey_happiness': state(default=0.0, domain=Domain.NONNEGATIVE),
        'prey_fear': state(default=0.0, domain=Domain.NONNEGATIVE),
        'predator_happiness': state(default=0.0, domain=Domain.NONNEGATIVE),
        'predator_anger': state(default=0.0, domain=Domain.NONNEGATIVE),
    },
    parameters={'decay': parameter(default=0.2, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [
        Rate(s.prey_happiness, -s.decay * s.prey_happiness),
        Rate(s.prey_fear, -s.decay * s.prey_fear),
        Rate(s.predator_happiness, -s.decay * s.predator_happiness),
        Rate(s.predator_anger, -s.decay * s.predator_anger),
    ],
)


def _kind(agent: AgentView) -> str:
    """Return the label of an agent's group, without the path of its system."""
    return agent.group.rpartition('/')[2]


def _occupants(agent: AgentView) -> dict[object, AgentView]:
    """Return the agents around an agent by their cell."""
    return {other.location: other for other in agent.neighbors()}


def _empty_cells(agent: AgentView) -> list[object]:
    """Return the neighbouring cells of an agent that no agent occupies."""
    occupied = _occupants(agent)
    return [cell for cell in agent.neighbor_cells() if cell not in occupied]


def _spawn(agent: AgentView, model: ModelView, **values: float) -> None:
    """Place an offspring of an agent in a random empty neighbouring cell, if there is one, with given states."""
    cells = _empty_cells(agent)
    if cells:
        agent.create(None, cells[int(model.rng.integers(len(cells)))], **values)
        agent.age = 0.0


class Plant(AgentType):
    """Plant that spreads into an empty neighbouring cell whenever it reaches maturity.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

        age:
            The steps since the plant last spread.

        maturity:
            The age at which the plant spreads.

    Examples:
        >>> from skcomplex.models.biology.ecology import Plant
        >>> Plant.group('plants', 100).size
        100
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'radius': 0.25})
    age: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    maturity: float = parameter(default=4.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the spreading rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.spread)]

    def spread(self, agent: AgentView, model: ModelView) -> None:
        """Age, and spread into an empty neighbouring cell at maturity.

        Args:
            agent:
                The plant.

            model:
                The model.
        """
        agent.age += 1
        if agent.age >= agent.maturity:
            _spawn(agent, model)


class Forager(AgentType):
    """Animal that hunts its food guided by its emotions and those its kind left on the ground, and breeds.

    Every step the animal ages and grows hungry, and starves when its hunger passes `starvation`. It moves to the
    neighbouring cell, empty or holding food, of highest preference: 1 for food, plus, with emotions, the happiness
    marked there minus its own and its own distress minus the distress marked there. It eats food it moves onto.
    Its emotions follow `E' = (1 - memory) (X + communication C) + memory E`, with the experience `X` of the step and
    the mean emotion `C` marked around it, and it marks its cell with them. A mature animal breeds into an empty
    neighbouring cell with a probability that happiness raises and distress lowers, and its offspring shares its
    hunger, so a lineage starves together when its food is gone. Subclasses name their food, the
    marks they read, and their experience of distress.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

        age:
            The steps since the animal last bred.

        hunger:
            The steps since the animal last ate.

        happiness:
            The happiness, which eating raises.

        distress:
            The fear or anger.

        maturity:
            The age at which the animal can breed.

        reproduction:
            The probability of breeding at maturity without emotions.

        starvation:
            The hunger at which the animal dies.

        memory:
            The share of its previous emotion that an animal keeps.

        communication:
            The weight of the emotions marked around the animal.

        emotions:
            Whether emotions guide the animal, 1 or 0.

    Examples:
        >>> from skcomplex.models.biology.ecology import StigmergicPrey
        >>> StigmergicPrey.group('prey', 10).size
        10
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'happiness', 'radius': 0.35})
    food: ClassVar[str] = PLANTS
    happiness_mark: ClassVar[str] = 'prey_happiness'
    distress_mark: ClassVar[str] = 'prey_fear'
    age: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    hunger: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    happiness: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    distress: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    maturity: float = parameter(default=5.0, domain=Domain.POSITIVE)
    reproduction: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    starvation: float = parameter(default=10.0, domain=Domain.POSITIVE)
    memory: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    communication: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    emotions: float = parameter(default=1.0, domain=Domain.BOOLEAN)

    def _preference(self, agent: AgentView, model: ModelView, cell: object, occupant: AgentView | None) -> float:
        """Return the preference of an animal for a neighbouring cell."""
        food = 1.0 if occupant is not None and _kind(occupant) == self.food else 0.0
        if not agent.emotions:
            return food
        happiness = model.field(f'{MARKS}.{self.happiness_mark}', cell) - agent.happiness
        distress = model.field(f'{MARKS}.{self.distress_mark}', cell) - agent.distress
        return food + happiness - distress

    def _move(self, agent: AgentView, model: ModelView, occupants: dict[object, AgentView]) -> float:
        """Move an animal to its preferred cell, eat the food there, and return 1 if it ate."""
        options = [
            (cell, occupants.get(cell))
            for cell in agent.neighbor_cells()
            if cell not in occupants or _kind(occupants[cell]) == self.food
        ]
        if not options:
            return 0.0
        noise = model.rng.random(len(options)) * 1e-3
        scores = [self._preference(agent, model, cell, occupant) for cell, occupant in options]
        cell, occupant = options[int(np.argmax(np.asarray(scores) + noise))]
        if occupant is None:
            agent.move_to(cell)
            return 0.0
        occupant.remove()
        agent.hunger = 0.0
        agent.move_to(cell)
        return 1.0

    def _feel(self, agent: AgentView, model: ModelView, happiness: float, distress: float) -> None:
        """Update the emotions of an animal from its experiences and the marks around it, and mark its cell."""
        cells = agent.neighbor_cells()
        marked_happiness = np.mean([model.field(f'{MARKS}.{self.happiness_mark}', cell) for cell in cells])
        marked_distress = np.mean([model.field(f'{MARKS}.{self.distress_mark}', cell) for cell in cells])
        keep, share = agent.memory, agent.communication
        agent.happiness = (1 - keep) * (happiness + share * marked_happiness) + keep * agent.happiness
        agent.distress = (1 - keep) * (distress + share * marked_distress) + keep * agent.distress
        model.deposit(f'{MARKS}.{self.happiness_mark}', agent.location, agent.happiness)
        model.deposit(f'{MARKS}.{self.distress_mark}', agent.location, agent.distress)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.live)]

    def sense(self, agent: AgentView, occupants: dict[object, AgentView]) -> float:
        """Return the distress the animal experiences from its neighbours.

        Args:
            agent:
                The animal.

            occupants:
                The agents around the animal by their cell.

        Returns:
            distress:
                The experience of distress.
        """
        del occupants
        return float(agent.hunger / agent.starvation)

    def live(self, agent: AgentView, model: ModelView) -> None:
        """Age, starve, sense, move and eat, feel, and breed.

        Args:
            agent:
                The animal.

            model:
                The model.
        """
        agent.age += 1
        agent.hunger += 1
        if agent.hunger > agent.starvation:
            agent.remove()
            return
        distress = self.sense(agent, _occupants(agent))
        ate = self._move(agent, model, _occupants(agent))
        if agent.emotions:
            self._feel(agent, model, ate, distress)
        if agent.age < agent.maturity:
            return
        chance = agent.reproduction * (1 + agent.emotions * (agent.happiness - agent.distress))
        if model.rng.random() < chance:
            _spawn(agent, model, hunger=agent.hunger)


class StigmergicPrey(Forager):
    """Prey that eats plants and fears the predators around it.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

        age:
            The steps since the animal last bred.

        hunger:
            The steps since the animal last ate.

        happiness:
            The happiness, which eating raises.

        distress:
            The fear, which predators around it raise.

        maturity:
            The age at which the animal can breed.

        reproduction:
            The probability of breeding at maturity without emotions.

        starvation:
            The hunger at which the animal dies.

        memory:
            The share of its previous emotion that an animal keeps.

        communication:
            The weight of the emotions marked around the animal.

        emotions:
            Whether emotions guide the animal, 1 or 0.

    Examples:
        >>> from skcomplex.models.biology.ecology import StigmergicPrey
        >>> StigmergicPrey.food
        'plants'
    """

    def sense(self, agent: AgentView, occupants: dict[object, AgentView]) -> float:
        """Return the number of predators around the prey, its experience of fear.

        Args:
            agent:
                The animal.

            occupants:
                The agents around the animal by their cell.

        Returns:
            distress:
                The experience of distress.
        """
        del agent
        return float(sum(_kind(other) == PREDATORS for other in occupants.values()))


class StigmergicPredator(Forager):
    """Predator that eats prey and grows angry with hunger.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

        age:
            The steps since the animal last bred.

        hunger:
            The steps since the animal last ate.

        happiness:
            The happiness, which eating raises.

        distress:
            The anger, which hunger raises.

        maturity:
            The age at which the animal can breed.

        reproduction:
            The probability of breeding at maturity without emotions.

        starvation:
            The hunger at which the animal dies.

        memory:
            The share of its previous emotion that an animal keeps.

        communication:
            The weight of the emotions marked around the animal.

        emotions:
            Whether emotions guide the animal, 1 or 0.

    Examples:
        >>> from skcomplex.models.biology.ecology import StigmergicPredator
        >>> StigmergicPredator.food
        'prey'
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'distress', 'radius': 0.45})
    food: ClassVar[str] = PREY
    happiness_mark: ClassVar[str] = 'predator_happiness'
    distress_mark: ClassVar[str] = 'predator_anger'


def _random_cells(rng: np.random.Generator, width: int, height: int, counts: list[int]) -> list[list[tuple[int, int]]]:
    """Return distinct random cells for several groups."""
    picks = rng.choice(width * height, size=sum(counts), replace=False)
    cells = [(int(pick % width), int(pick // width)) for pick in picks]
    bounds = np.cumsum([0, *counts])
    return [cells[start:end] for start, end in itertools.pairwise(bounds)]


@register_model(
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
            The model, with the groups `life/plants`, `life/prey`, and `life/predators`, and the marks
            `ground/marks`.

    Examples:
        >>> from skcomplex.models.biology.ecology import stigmergic_predator_prey
        >>> [child.label for child in stigmergic_predator_prey(10, 10, plants=20, prey=10, predators=5).systems]
        ['life', 'ground']
    """
    rng = np.random.default_rng(random_state)
    plant_cells, prey_cells, predator_cells = _random_cells(rng, width, height, [plants, prey, predators])
    flag = float(emotions)
    life = System(
        'life',
        components=[
            Plant.group(PLANTS, plants, location=plant_cells, age=rng.integers(0, 4, plants).astype(float)),
            StigmergicPrey.group(PREY, prey, location=prey_cells, emotions=flag, **PREY_TRAITS),
            StigmergicPredator.group(PREDATORS, predators, location=predator_cells, emotions=flag, **PREDATOR_TRAITS),
        ],
        space=LatticeSpace((width, height), boundary='periodic'),
    )
    marks = Field('marks', EmotionMarks, LatticeSpace((width, height)), values={'decay': decay})
    return System(components=[life, System('ground', components=[marks])])
