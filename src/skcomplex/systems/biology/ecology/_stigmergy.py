"""Predators and prey that share emotions through marks on the ground."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Callable
from functools import partial
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement, GroupElement
from ....core.components.interactions import (
    BirthDeathInteraction,
    MapInteraction,
    MovementInteraction,
    TransferInteraction,
)
from ....core.components.laws import Source
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.stigmergy import EvaporationInteraction

PLANTS = 'plants'

PREY = 'prey'

MARKS = 'ground/marks'

PREDATORS = 'predators'

EMOTIONS = ('prey_happiness', 'prey_fear', 'predator_happiness', 'predator_anger')

PREY_TRAITS: dict[str, float] = {'maturity': 3.0, 'reproduction': 0.5, 'starvation': 10.0}

PREDATOR_TRAITS: dict[str, float] = {'maturity': 20.0, 'reproduction': 0.1, 'starvation': 8.0}


class EmotionMarksElement(Element):
    """Marks of one cell, the emotions of prey and predators.

    They are the happiness and fear of prey and the happiness and anger of predators.

    Animals with emotions leave their feelings in the cell they stand on, and the marks evaporate at the rate `decay`,
    which the system adds as an evaporation of every mark.

    Args:
        label:
            The label of the element or the lattice field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        prey_happiness:
            The initial happiness left by prey.

        prey_fear:
            The initial fear left by prey.

        predator_happiness:
            The initial happiness left by predators.

        predator_anger:
            The initial anger left by predators.

        decay:
            The rate at which every mark evaporates.

    Examples:
        >>> from skcomplex.systems.biology.ecology import EmotionMarksElement
        >>> EmotionMarksElement('marks').decay
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        prey_happiness: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        prey_fear: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        predator_happiness: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        predator_anger: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        decay: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.prey_happiness = prey_happiness
        self.prey_fear = prey_fear
        self.predator_happiness = predator_happiness
        self.predator_anger = predator_anger
        self.decay = decay


class ForagerView(AgentView):
    """View of a forager or a predator, with its states and parameters as attributes.

    `ForagerElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    age: float
    hunger: float
    happiness: float
    distress: float
    experience: float
    maturity: float
    reproduction: float
    starvation: float
    memory: float
    communication: float
    emotions: float


class ForagerElement(Element):
    """Animal that hunts its food guided by emotions, and breeds.

    The emotions are its own and those its kind left on the ground.

    Every step the animal ages and grows hungry, and starves when its hunger passes `starvation`. It moves to the
    neighbouring cell, empty or holding food, of highest preference: 1 for food, plus, with emotions, the happiness
    marked there minus its own and its own distress minus the distress marked there. It eats food it moves onto.
    Its emotions follow `E' = (1 - memory) (X + communication C) + memory E`, with the experience `X` of the step and
    the mean emotion `C` marked around it, and it marks its cell with them. A mature animal breeds into an empty
    neighbouring cell with a probability that happiness raises and distress lowers, and its offspring shares its
    hunger, so a lineage starves together when its food is gone. Subclasses name their food and the marks they read,
    and the system adds the ageing, the hunger, the starvation, the experience of distress, the move, the meal, the
    emotions, the marks, and the breeding as member interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The steps since the animal last bred.

        hunger:
            The steps since the animal last ate.

        happiness:
            The happiness, which eating raises.

        distress:
            The fear or anger.

        experience:
            The distress the animal experienced in the step, before it moved.

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import StigmergicPreyElement
        >>> GroupElement('prey', element=StigmergicPreyElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'happiness', 'radius': 0.35}),
        *,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        hunger: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        happiness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        distress: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        experience: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        maturity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        reproduction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        starvation: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        memory: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        communication: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        emotions: Annotated[float, ParameterSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.age = age
        self.hunger = hunger
        self.happiness = happiness
        self.distress = distress
        self.experience = experience
        self.maturity = maturity
        self.reproduction = reproduction
        self.starvation = starvation
        self.memory = memory
        self.communication = communication
        self.emotions = emotions

    food: ClassVar[str] = PLANTS
    happiness_mark: ClassVar[str] = 'prey_happiness'
    distress_mark: ClassVar[str] = 'prey_fear'

    @classmethod
    def _get_view(cls) -> type[ForagerView]:
        """Return the view of a forager or a predator, `ForagerView`."""
        return ForagerView


class StigmergicPredatorElement(ForagerElement):
    """Predator that eats prey and grows angry with hunger.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The steps since the animal last bred.

        hunger:
            The steps since the animal last ate.

        happiness:
            The happiness, which eating raises.

        distress:
            The anger, which hunger raises.

        experience:
            The distress the animal experienced in the step, before it moved.

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
        >>> from skcomplex.systems.biology.ecology import StigmergicPredatorElement
        >>> StigmergicPredatorElement.food
        'prey'
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'distress', 'radius': 0.45}),
        *,
        age: float = 0.0,
        hunger: float = 0.0,
        happiness: float = 0.0,
        distress: float = 0.0,
        experience: float = 0.0,
        maturity: float = 5.0,
        reproduction: float = 0.3,
        starvation: float = 10.0,
        memory: float = 0.3,
        communication: float = 0.1,
        emotions: float = 1.0,
    ) -> None:
        super().__init__(
            label,
            visual=visual,
            age=age,
            hunger=hunger,
            happiness=happiness,
            distress=distress,
            experience=experience,
            maturity=maturity,
            reproduction=reproduction,
            starvation=starvation,
            memory=memory,
            communication=communication,
            emotions=emotions,
        )

    food: ClassVar[str] = PREY
    happiness_mark: ClassVar[str] = 'predator_happiness'
    distress_mark: ClassVar[str] = 'predator_anger'


class StigmergicPreyElement(ForagerElement):
    """Prey that eats plants and fears the predators around it.

    Args:
        label:
            The label of the agent or group.

        age:
            The steps since the animal last bred.

        hunger:
            The steps since the animal last ate.

        happiness:
            The happiness, which eating raises.

        distress:
            The fear, which predators around it raise.

        experience:
            The distress the animal experienced in the step, before it moved.

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
        >>> from skcomplex.systems.biology.ecology import StigmergicPreyElement
        >>> StigmergicPreyElement.food
        'plants'
    """


class PlantView(AgentView):
    """View of a plant, with its states and parameters as attributes.

    `PlantElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    age: float
    maturity: float


class PlantElement(Element):
    """Plant that spreads into an empty neighbouring cell whenever it reaches maturity.

    The system adds the ageing and the spreading as member interactions, and a plant that spreads starts its age anew.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The steps since the plant last spread.

        maturity:
            The age at which the plant spreads.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import PlantElement
        >>> GroupElement('plants', element=PlantElement, size=100).size
        100
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.25}),
        *,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        maturity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.age = age
        self.maturity = maturity

    @classmethod
    def _get_view(cls) -> type[PlantView]:
        """Return the view of a plant, `PlantView`."""
        return PlantView


class StigmergicPredatorPreySystem(BaseSystem):
    """Plants, prey, and predators on a grid.

    The animals share emotions through decaying marks.

    Every cell holds at most one organism. Prey eat plants and fear predators, and predators eat prey and grow angry
    with hunger. Happiness draws animals of the same kind towards the cells where it was marked and distress drives them
    away, a trail stigmergy of emotions that the cellular automaton of Olsen in Carmichael and Hadžikadić (2019) uses to
    study cooperation within each species. The system has the groups `life/plants`, `life/prey`, and `life/predators`,
    and the marks `ground/marks`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import StigmergicPredatorPreySystem
        >>> system = StigmergicPredatorPreySystem(10, 10, plants=20, prey=10, predators=5).prepare().system_
        >>> [child.label for child in system.systems]
        ['life', 'ground']
    """

    field = 'biology/ecology'
    topics = ('predator-prey', 'agents', 'stigmergy', 'emotions', 'co-simulation')
    references = ('Carmichael and Hadžikadić 2019', 'Olsen 2019')
    until = 100
    dt = 1

    def __init__(
        self,
        width: int = 30,
        height: int = 30,
        *,
        plants: int = 300,
        prey: int = 120,
        predators: int = 20,
        emotions: bool = True,
        decay: float = 0.2,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.plants = plants
        self.prey = prey
        self.predators = predators
        self.emotions = emotions
        self.decay = decay
        self.random_state = random_state

    @staticmethod
    def _kind(agent: AgentViewProtocol) -> str:
        """Return the label of an agent's group, without the path of its system."""
        return agent.group.rpartition('/')[2]

    @staticmethod
    def _occupants(agent: AgentViewProtocol) -> dict[object, AgentViewProtocol]:
        """Return the agents around an agent by their cell."""
        return {other.location: other for other in agent.neighbors()}

    def _empty_cells(self, agent: AgentViewProtocol) -> list[object]:
        """Return the neighbouring cells of an agent that no agent occupies."""
        occupied = self._occupants(agent)
        return [cell for cell in agent.neighbor_cells() if cell not in occupied]

    def _get_empty_cell(self, agent: AgentViewProtocol | None, system: SystemViewProtocol) -> object:
        """Return a random empty neighbouring cell of a parent, or `None`.

        It is `None` when there is none.
        """
        cells = [] if agent is None else self._empty_cells(agent)
        return cells[int(system.rng.integers(len(cells)))] if cells else None

    @staticmethod
    def _get_one(agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return one, the step of the age and the hunger of an organism.

        The age and the hunger grow by it.
        """
        del agent, system
        return 1.0

    def _spreads(self, plant: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether a mature plant has an empty neighbouring cell to spread into."""
        del system
        return isinstance(plant, PlantView) and plant.age >= plant.maturity and bool(self._empty_cells(plant))

    @staticmethod
    def _starves(animal: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether the hunger of an animal passes `starvation`.

        The animal then dies.
        """
        del system
        return isinstance(animal, ForagerView) and animal.hunger > animal.starvation

    def _get_fear(self, prey: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the number of predators around a prey, its experience of fear."""
        del system
        return float(sum(self._kind(other) == PREDATORS for other in self._occupants(prey).values()))

    @staticmethod
    def _get_anger(predator: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the anger of a predator, its hunger as a share of its starvation."""
        del system
        return float(predator.hunger / predator.starvation) if isinstance(predator, ForagerView) else 0.0

    def _get_preference(
        self,
        animal: ForagerView,
        system: SystemViewProtocol,
        cell: object,
        occupant: AgentViewProtocol | None,
        marks: tuple[str, str, str],
    ) -> float:
        """Return the preference of an animal for a neighbouring cell.

        It depends on its food and the marks it reads.
        """
        food, happiness_mark, distress_mark = marks
        meal = 1.0 if occupant is not None and self._kind(occupant) == food else 0.0
        if not animal.emotions:
            return meal
        happiness = system.field(f'{MARKS}.{happiness_mark}', cell) - animal.happiness
        distress = system.field(f'{MARKS}.{distress_mark}', cell) - animal.distress
        return meal + happiness - distress

    def _get_preferred(
        self,
        animal: AgentViewProtocol,
        system: SystemViewProtocol,
        marks: tuple[str, str, str],
    ) -> object:
        """Return the empty or food cell around an animal of highest preference.

        The choice has a little noise, and it is `None` when there is no such cell.
        """
        if not isinstance(animal, ForagerView):
            return None
        occupants = self._occupants(animal)
        options = [
            (cell, occupants.get(cell))
            for cell in animal.neighbor_cells()
            if cell not in occupants or self._kind(occupants[cell]) == marks[0]
        ]
        if not options:
            return None
        noise = system.rng.random(len(options)) * 1e-3
        scores = [self._get_preference(animal, system, cell, occupant, marks) for cell, occupant in options]
        return options[int(np.argmax(np.asarray(scores) + noise))][0]

    @staticmethod
    def _get_fed(food: AgentViewProtocol, animal: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return minus the hunger of an animal.

        A meal clears the hunger.
        """
        del food, system
        return -float(animal.hunger) if isinstance(animal, ForagerView) else 0.0

    @staticmethod
    def _get_marked(animal: ForagerView, system: SystemViewProtocol, mark: str) -> float:
        """Return the mean of a mark over the cells around an animal."""
        return float(np.mean([system.field(f'{MARKS}.{mark}', cell) for cell in animal.neighbor_cells()]))

    def _get_happiness(self, animal: AgentViewProtocol, system: SystemViewProtocol, mark: str) -> float:
        """Return the next happiness of an animal.

        It comes from a meal in the step and the happiness marked around it.
        """
        if not isinstance(animal, ForagerView):
            return 0.0
        if not animal.emotions:
            return float(animal.happiness)
        ate = float(animal.hunger == 0)
        keep, share = animal.memory, animal.communication
        return (1 - keep) * (ate + share * self._get_marked(animal, system, mark)) + keep * animal.happiness

    def _get_distress(self, animal: AgentViewProtocol, system: SystemViewProtocol, mark: str) -> float:
        """Return the next distress of an animal.

        It comes from its experience of the step and the distress marked around it.
        """
        if not isinstance(animal, ForagerView):
            return 0.0
        if not animal.emotions:
            return float(animal.distress)
        keep, share = animal.memory, animal.communication
        return (1 - keep) * (
            animal.experience + share * self._get_marked(animal, system, mark)
        ) + keep * animal.distress

    @staticmethod
    def _feels(animal: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> bool:
        """Return whether emotions guide an animal.

        An animal guided by emotions marks its cell.
        """
        del cell, system
        return isinstance(animal, ForagerView) and bool(animal.emotions)

    @staticmethod
    def _get_marked_happiness(animal: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> float:
        """Return the happiness an animal marks its cell with."""
        del cell, system
        return float(animal.happiness) if isinstance(animal, ForagerView) else 0.0

    @staticmethod
    def _get_marked_distress(animal: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> float:
        """Return the distress an animal marks its cell with."""
        del cell, system
        return float(animal.distress) if isinstance(animal, ForagerView) else 0.0

    def _get_breeding(self, animal: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the probability that a mature animal breeds.

        The animal needs an empty neighbouring cell, and happiness raises the probability.
        """
        del system
        if not isinstance(animal, ForagerView) or animal.age < animal.maturity or not self._empty_cells(animal):
            return 0.0
        return float(animal.reproduction * (1 + animal.emotions * (animal.happiness - animal.distress)))

    def _get_growth(self, system: str, label: str, state: str) -> MapInteraction:
        """Return the map that raises a state of every member of a group by one.

        It acts in the activation of the member.
        """
        return MapInteraction(
            f'{label}_{state}',
            target=f'{system}/{label}',
            state=state,
            function=self._get_one,
            mode='increment',
            form='agent',
            schedule='member',
        )

    def _get_plants(self, system: str, label: str) -> list[MapInteraction | BirthDeathInteraction]:
        """Return the member interactions of a plant.

        A plant ages and spreads into an empty neighbouring cell at maturity.
        """
        return [
            self._get_growth(system, label, 'age'),
            BirthDeathInteraction(
                f'{label}_spreading',
                target=f'{system}/{label}',
                birth=self._spreads,
                placement=self._get_empty_cell,
                reset={'age': 0.0},
                schedule='member',
            ),
        ]

    def _get_animals(
        self,
        system: str,
        label: str,
        element: type['ForagerElement'],
        sense: Callable[[AgentViewProtocol, SystemViewProtocol], float],
    ) -> list[MapInteraction | BirthDeathInteraction | MovementInteraction | TransferInteraction]:
        """Return the member interactions of an animal.

        An animal ages, starves, senses, moves and eats, feels, marks, and breeds.
        """
        group = f'{system}/{label}'
        marks = (element.food, element.happiness_mark, element.distress_mark)
        return [
            self._get_growth(system, label, 'age'),
            self._get_growth(system, label, 'hunger'),
            BirthDeathInteraction(f'{label}_starvation', target=group, death=self._starves, schedule='member'),
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
                step=partial(self._get_preferred, marks=marks),
                schedule='member',
            ),
            TransferInteraction(
                f'{label}_meal',
                source=f'{system}/{element.food}',
                target=group,
                state=Source(),
                into='hunger',
                amount=self._get_fed,
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
                function=partial(self._get_happiness, mark=element.happiness_mark),
                form='agent',
                schedule='member',
            ),
            MapInteraction(
                f'{label}_distress',
                target=group,
                state='distress',
                function=partial(self._get_distress, mark=element.distress_mark),
                form='agent',
                schedule='member',
            ),
            TransferInteraction(
                f'{label}_marking',
                source=group,
                target=MARKS,
                state=Source(),
                into=element.happiness_mark,
                amount=self._get_marked_happiness,
                legs=((Source(), element.distress_mark, self._get_marked_distress),),
                pairing='site',
                guard=self._feels,
                form='agent',
                schedule='member',
            ),
            BirthDeathInteraction(
                f'{label}_breeding',
                target=group,
                birth=self._get_breeding,
                placement=self._get_empty_cell,
                inherit=('hunger',),
                reset={'age': 0.0},
                schedule='member',
            ),
        ]

    @staticmethod
    def _random_cells(
        rng: np.random.Generator,
        width: int,
        height: int,
        counts: list[int],
    ) -> list[list[tuple[int, int]]]:
        """Return distinct random cells for several groups."""
        picks = rng.choice(width * height, size=sum(counts), replace=False)
        cells = [(int(pick % width), int(pick // width)) for pick in picks]
        bounds = np.cumsum([0, *counts])
        return [cells[start:end] for start, end in itertools.pairwise(bounds)]

    def _build_system(self) -> System:
        """Return the parts of plants, prey, and predators on a grid."""
        rng = np.random.default_rng(self.random_state)
        plant_cells, prey_cells, predator_cells = self._random_cells(
            rng,
            self.width,
            self.height,
            [self.plants, self.prey, self.predators],
        )
        flag = float(self.emotions)
        life = System(
            'life',
            elements=[
                GroupElement(
                    PLANTS,
                    element=PlantElement,
                    size=self.plants,
                    values={'age': rng.integers(0, 4, self.plants).astype(float)},
                    variable=True,
                    location=plant_cells,
                ),
                GroupElement(
                    PREY,
                    element=StigmergicPreyElement,
                    size=self.prey,
                    values={'emotions': flag, **PREY_TRAITS},
                    variable=True,
                    location=prey_cells,
                ),
                GroupElement(
                    PREDATORS,
                    element=StigmergicPredatorElement,
                    size=self.predators,
                    values={'emotions': flag, **PREDATOR_TRAITS},
                    variable=True,
                    location=predator_cells,
                ),
            ],
            space=LatticeSpace((self.width, self.height), boundary='periodic'),
        )
        marks = FieldElement(
            'marks',
            element=EmotionMarksElement,
            space=LatticeSpace((self.width, self.height)),
            values={'decay': self.decay},
        )
        evaporations = [
            EvaporationInteraction(f'evaporation_{mark}', target='marks', state=mark, evaporation='decay')
            for mark in EMOTIONS
        ]
        return System(
            self.label,
            systems=[life, System('ground', elements=[marks], interactions=evaporations)],
            interactions=[
                *self._get_plants('life', PLANTS),
                *self._get_animals('life', PREY, StigmergicPreyElement, self._get_fear),
                *self._get_animals('life', PREDATORS, StigmergicPredatorElement, self._get_anger),
            ],
        )
