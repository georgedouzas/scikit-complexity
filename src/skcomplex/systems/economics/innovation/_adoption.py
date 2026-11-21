"""Innovation systems: households that adopt a networked product.

They adopt when they can afford it and know a user.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import deque
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast, runtime_checkable

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

UPPER = 0
MIDDLE = 1
LOWER = 2
INCOME_SIGMA = 0.6
MOORE = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)


@runtime_checkable
class HouseholdView(AgentViewProtocol, Protocol):
    """Household as its rule sees it, with its income, phone, and class.

    It reads its `income`, its `phone`, its `social_class`, the `growth` of its income, and the `threshold` of affording
    a phone.
    """

    income: float
    phone: float
    social_class: float
    growth: float
    threshold: float


class HouseholdElement(Element):
    """Household whose income grows every year.

    It gets a phone when it can afford one and a neighbour has one.

    Args:
        label:
            The label of the agent or group.

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

        size:
            The number of households, or `None` for one.

        space:
            The space the households live on, or `None` for none.

        values:
            More values by name, such as the cells `x` and `y` of the households on a lattice, or `None` for none.

    Examples:
        >>> from skcomplex.systems.economics.innovation import HouseholdElement
        >>> HouseholdElement('households', threshold=1.5, size=4).size
        4
    """

    visual = VisualSpec('agents', {'color_by': 'phone', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        income: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        phone: Annotated[object, NN] = 0.0,
        social_class: Annotated[object, NN] = float(MIDDLE),
        growth: object = 0.02,
        threshold: Annotated[object, RealSet.closed_open(0, oo)] = 1.2,
        size: int | None = None,
        space: BaseSpace | None = None,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.income = income
        self.phone = phone
        self.social_class = social_class
        self.growth = growth
        self.threshold = threshold


class AdoptionInteraction(Interaction):
    """Growth of the incomes of households and their adoption of a phone.

    Every year the income of every household grows by its rate, `income' = income (1 + growth)`. Then the households
    decide one after the other in a new random order, and a household gets a phone when its income reaches its
    threshold and one of its neighbours has a phone, so phones spread along chains of affording neighbours within a
    year.

    Args:
        label:
            The label of the interaction.

        target:
            The households, on the cells of a lattice.

    Examples:
        >>> from skcomplex.systems.economics.innovation import AdoptionInteraction
        >>> AdoptionInteraction('adoption').target
        'households'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'households') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the growth of the incomes and the adoption of phones as the rules of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context, self._grow), UpdateLaw(context, self._adopt)]

    def _grow(self, system: SystemView) -> None:
        """Grow the income of every household by a year."""
        for household in cast(list[HouseholdView], system.get_agents('target')):
            household.income *= 1 + household.growth

    def _adopt(self, system: SystemView) -> None:
        """Give a phone to every household that can afford it and has a neighbour with one, in a random order."""
        households = cast(list[HouseholdView], system.get_agents('target'))
        for index in system.rng.permutation(len(households)):
            household = households[int(index)]
            if household.alive and not household.phone and household.income >= household.threshold:
                household.phone = float(
                    any(
                        isinstance(neighbor, HouseholdView) and neighbor.phone for neighbor in household.get_neighbors()
                    ),
                )


class PhoneAdoptionSystem(BaseSystem):
    """Hamill and Gilbert's adoption of fixed-line phones by British households.

    The adoption starts from 1951. One household lives in every cell of a torus. The upper class lives in the top left
    quadrant, the lower class in the bottom left, and the middle class elsewhere, so neighbours tend to be alike.
    Incomes are lognormal, scaled by class and normalised to a mean of 1, and grow every year. A household gets a phone
    once its income reaches the threshold and one of its eight neighbours has a phone. Households decide one after the
    other in a new random order every year, so phones spread along chains of affording neighbours within a year. The
    first owners are a cluster of affording households grown from the richest upper-class household.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.innovation import PhoneAdoptionSystem
        >>> PhoneAdoptionSystem((10, 10)).prepare().system_.elements[0].size
        100
    """

    field = 'economics/innovation'
    topics = ('diffusion', 'adoption', 'network effect', 'agents')
    references = ('Hamill and Gilbert 2016',)
    until = 50
    observables = (Observable('share', group='households', variable='phone'),)

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.growth = growth
        self.threshold = threshold
        self.initial_share = initial_share
        self.class_incomes = class_incomes
        self.random_state = random_state

    @staticmethod
    def _get_social_classes(width: int, height: int) -> Array:
        """Return the class of every cell.

        It is upper in the top left quadrant, lower in the bottom left, else middle.
        """
        columns, rows = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
        left, top = columns < width // 2, rows >= height // 2
        return np.where(left & top, UPPER, np.where(left & ~top, LOWER, MIDDLE)).ravel().astype(float)

    @staticmethod
    def _seed_network(incomes: Array, classes: Array, shape: tuple[int, int], threshold: float, share: float) -> Array:
        """Return the initial owners.

        They are grown from the richest upper-class household over affording neighbours.
        """
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the households on a torus and their adoption."""
        super()._prepare(context)
        width, height = (int(side) for side in self.shape)
        rng = np.random.default_rng(self.random_state)
        classes = self._get_social_classes(width, height)
        scales = np.asarray(self.class_incomes, dtype=float)[classes.astype(int)]
        incomes = scales * rng.lognormal(0.0, INCOME_SIGMA, len(classes))
        incomes /= incomes.mean()
        owners = self._seed_network(incomes, classes, (width, height), self.threshold, self.initial_share)
        columns, rows = np.divmod(np.arange(width * height), height)
        households = HouseholdElement(
            'households',
            income=incomes,
            phone=owners,
            social_class=classes,
            growth=self.growth,
            threshold=self.threshold,
            size=width * height,
            space=LatticeSpace((width, height), boundary='periodic'),
            values={'x': columns, 'y': rows},
        )
        self.system_ = System(
            self.label,
            elements=[households],
            interactions=[AdoptionInteraction('adoption', target='households')],
        )
