"""Innovation models: households that adopt a networked product when they can afford it and know a user."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import deque
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import Element, LatticeSpace, Law, Observable, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

type Array = npt.NDArray[np.float64]

UPPER = 0
MIDDLE = 1
LOWER = 2
INCOME_SIGMA = 0.6
MOORE = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)


def _social_classes(width: int, height: int) -> Array:
    """Return the class of every cell, upper in the top left quadrant, lower in the bottom left, middle elsewhere."""
    columns, rows = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    left, top = columns < width // 2, rows >= height // 2
    return np.where(left & top, UPPER, np.where(left & ~top, LOWER, MIDDLE)).ravel().astype(float)


def _seed_network(incomes: Array, classes: Array, shape: tuple[int, int], threshold: float, share: float) -> Array:
    """Return the initial owners, a cluster grown from the richest upper-class household over affording neighbours."""
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


class Household(Element):
    """Household whose income grows every year and that gets a phone when it can afford one and a neighbour has one.

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

    Examples:
        >>> from skcomplex.models.economics.innovation import Household
        >>> Household.group('households', 4, threshold=1.5).size
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'phone', 'color_range': (0.0, 1.0)})
    income: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    phone: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    social_class: float = state(default=MIDDLE, domain=Domain.NONNEGATIVE_INTEGER)
    growth: float = parameter(default=0.02, domain=Domain.REAL)
    threshold: float = parameter(default=1.2, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the yearly rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.adopt)]

    def adopt(self, agent: AgentView, model: ModelView) -> None:
        """Grow the income and get a phone if it is affordable and a neighbour has one.

        Args:
            agent:
                The household.

            model:
                The model.
        """
        agent.income = agent.income * (1 + agent.growth)
        if agent.phone or agent.income < agent.threshold:
            return
        if any(neighbor.phone for neighbor in agent.neighbors()):
            agent.phone = 1


@register_model(
    field='economics/innovation',
    tags=('diffusion', 'adoption', 'network effect', 'agents'),
    references=('Hamill and Gilbert 2016',),
    defaults={'until': 50},
    observables=lambda system: [Observable('share', group='households', variable='phone')],
)
def phone_adoption(
    shape: Sequence[int] = (30, 30),
    *,
    growth: float = 0.02,
    threshold: float = 1.2,
    initial_share: float = 0.1,
    class_incomes: Sequence[float] = (1.6, 1.0, 0.6),
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Hamill and Gilbert's adoption of fixed-line phones by British households from 1951.

    One household lives in every cell of a torus. The upper class lives in the top left quadrant, the lower class in
    the bottom left, and the middle class elsewhere, so neighbours tend to be alike. Incomes are lognormal, scaled by
    class and normalised to a mean of 1, and grow every year. A household gets a phone once its income reaches the
    threshold and one of its eight neighbours has a phone. The first owners are a cluster of affording households
    grown from the richest upper-class household.

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

    Returns:
        system:
            The model, with the group `households` and the share of households with a phone `share`.

    Examples:
        >>> from skcomplex.models.economics.innovation import phone_adoption
        >>> phone_adoption((10, 10)).elements[0].size
        100
    """
    width, height = (int(side) for side in shape)
    rng = np.random.default_rng(random_state)
    classes = _social_classes(width, height)
    scales = np.asarray(class_incomes, dtype=float)[classes.astype(int)]
    incomes = scales * rng.lognormal(0.0, INCOME_SIGMA, len(classes))
    incomes /= incomes.mean()
    owners = _seed_network(incomes, classes, (width, height), threshold, initial_share)
    locations = [(column, row) for column in range(width) for row in range(height)]
    households = Household.group(
        'households',
        len(locations),
        location=locations,
        income=incomes,
        phone=owners,
        social_class=classes,
        growth=growth,
        threshold=threshold,
    )
    return System(
        components=[households],
        space=LatticeSpace((width, height), boundary='periodic'),
    )
