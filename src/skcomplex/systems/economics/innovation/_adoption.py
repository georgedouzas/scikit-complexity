"""Innovation systems: households that adopt a networked product when they can afford it and know a user."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import deque
from collections.abc import Sequence
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array


class HouseholdView(AgentView):
    """Household as its rule sees it, with its income, phone, and class.

    It reads its `income`, its `phone`, its `social_class`, the `growth` of its income, and the `threshold` of affording
    a phone.
    """

    income: float
    phone: float
    social_class: float
    growth: float
    threshold: float


UPPER = 0
MIDDLE = 1
LOWER = 2
INCOME_SIGMA = 0.6
MOORE = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)


class HouseholdElement(Element):
    """Household whose income grows every year and that gets a phone when it can afford one and a neighbour has one.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        income:
            The initial income, relative to the mean income at the start.

        phone:
            1 for a household with a phone and 0 otherwise.

        social_class:
            0 for the upper, 1 for the middle, and 2 for the lower class.

        growth:
            The yearly growth rate of the income.

        threshold:
            The income from which a household can afford a phone.

    Examples:
        >>> from skcomplex.systems.economics.innovation import HouseholdElement
        >>> GroupElement('households', element=HouseholdElement, size=4, values={'threshold': 1.5}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'phone', 'color_range': (0.0, 1.0)}),
        *,
        income: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        phone: Annotated[float, StateSpec(domain=NN)] = 0,
        social_class: Annotated[float, StateSpec(domain=NN)] = MIDDLE,
        growth: Annotated[float, ParameterSpec()] = 0.02,
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.income = income
        self.phone = phone
        self.social_class = social_class
        self.growth = growth
        self.threshold = threshold

    @classmethod
    def _get_view(cls) -> type[HouseholdView]:
        """Return the view of the agents of the element."""
        return HouseholdView

    @staticmethod
    def get_income(agent: HouseholdView, system: SystemView) -> float:
        """Return the income grown by a year.

        Args:
            agent:
                The household.

            system:
                The system view of the growth.

        Returns:
            income:
                The income of the next year.
        """
        return agent.income * (1 + agent.growth)

    @staticmethod
    def get_phone(agent: HouseholdView, system: SystemView) -> float:
        """Return 1 for a household that has a phone or gets one, affordable with a neighbour that has one, else 0.

        Args:
            agent:
                The household.

            system:
                The system view of the adoption.

        Returns:
            phone:
                1 for a household with a phone and 0 otherwise.
        """
        if agent.phone or agent.income < agent.threshold:
            return float(agent.phone)
        return float(any(isinstance(neighbor, HouseholdView) and neighbor.phone for neighbor in agent.neighbors()))


class PhoneAdoptionSystem(BaseSystem):
    """Hamill and Gilbert's adoption of fixed-line phones by British households from 1951.

    One household lives in every cell of a torus. The upper class lives in the top left quadrant, the lower class in
    the bottom left, and the middle class elsewhere, so neighbours tend to be alike. Incomes are lognormal, scaled by
    class and normalised to a mean of 1, and grow every year. A household gets a phone once its income reaches the
    threshold and one of its eight neighbours has a phone. Households decide one after the other in a new random order
    every year, so phones spread along chains of affording neighbours within a year. The first owners are a cluster of
    affording households grown from the richest upper-class household.

    The system has the group `households` and the share of households with a phone `share`.

    Args:
        shape:
            The number of columns and rows.

        growth:
            The yearly growth rate of incomes.

        threshold:
            The income from which a household can afford a phone.

        initial_share:
            The share of households with a phone at the start.

        class_incomes:
            The income scale of the upper, middle, and lower classes.

        random_state:
            The seed or generator of the incomes.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.innovation import PhoneAdoptionSystem
        >>> PhoneAdoptionSystem((10, 10)).prepare().system_.elements[0].prepare().size_
        100
    """

    field = 'economics/innovation'
    topics = ('diffusion', 'adoption', 'network effect', 'agents')
    references = ('Hamill and Gilbert 2016',)
    until = 50
    observables = (ObservableSpec('share', group='households', variable='phone'),)

    def __init__(
        self,
        shape: Sequence[int] = (30, 30),
        *,
        growth: float = 0.02,
        threshold: float = 1.2,
        initial_share: float = 0.1,
        class_incomes: Sequence[float] = (1.6, 1.0, 0.6),
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.growth = growth
        self.threshold = threshold
        self.initial_share = initial_share
        self.class_incomes = class_incomes
        self.random_state = random_state

    @staticmethod
    def _social_classes(width: int, height: int) -> Array:
        """Return the class of every cell: upper in the top left quadrant, lower in the bottom left, else middle."""
        columns, rows = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
        left, top = columns < width // 2, rows >= height // 2
        return np.where(left & top, UPPER, np.where(left & ~top, LOWER, MIDDLE)).ravel().astype(float)

    @staticmethod
    def _seed_network(incomes: Array, classes: Array, shape: tuple[int, int], threshold: float, share: float) -> Array:
        """Return the initial owners, grown from the richest upper-class household over affording neighbours."""
        width, height = shape
        owners = np.zeros(len(incomes))
        seed = int(np.argmax(np.where(classes == UPPER, incomes, -np.inf)))
        queue, target = deque([seed]), round(share * len(incomes))
        while queue and owners.sum() < target:
            cell = queue.popleft()
            if owners[cell] or incomes[cell] < threshold:
                continue
            owners[cell] = 1.0
            column, row = divmod(cell, height)
            queue.extend(((column + dx) % width) * height + (row + dy) % height for dx, dy in MOORE)
        return owners

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        growth = self.growth
        width, height = (int(side) for side in self.shape)
        rng = np.random.default_rng(self.random_state)
        classes = self._social_classes(width, height)
        scales = np.asarray(self.class_incomes, dtype=float)[classes.astype(int)]
        incomes = scales * rng.lognormal(0.0, INCOME_SIGMA, len(classes))
        incomes /= incomes.mean()
        owners = self._seed_network(incomes, classes, (width, height), self.threshold, self.initial_share)
        locations = [(column, row) for column in range(width) for row in range(height)]
        households = GroupElement(
            'households',
            element=HouseholdElement,
            size=len(locations),
            values={
                'income': incomes,
                'phone': owners,
                'social_class': classes,
                'growth': growth,
                'threshold': self.threshold,
            },
            location=locations,
        )
        growth = MapInteraction(
            'growth',
            target='households',
            state='income',
            function=HouseholdElement.get_income,
            form='agent',
        )
        adoption = MapInteraction(
            'adoption',
            target='households',
            state='phone',
            function=HouseholdElement.get_phone,
            schedule='random-sequential',
            form='agent',
        )
        return System(
            self.label,
            elements=[households],
            interactions=[growth, adoption],
            space=LatticeSpace((width, height), boundary='periodic'),
            time=DiscreteTime(),
        )
