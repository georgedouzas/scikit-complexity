"""Model rule of Charbonneau's traffic model: a random driver brakes now and then, and every car moves."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, RoleSpec
from ._motorist_view import MotoristView


class CarelessBrakingInteraction(Interaction):
    """Model rule of Charbonneau's traffic model: a random driver brakes now and then, and every car moves.

    Every step, with probability `probability`, one random driver brakes by the braking of its group whatever the
    distance to the car ahead. Then every car moves by its speed, but no closer than `gap` to the car ahead.

    Args:
        label:
            The label of the interaction.

        cars:
            The path of the group of cars.

        probability:
            The probability that a random driver brakes in a step.

        gap:
            The smallest distance a car keeps to the car ahead.

    Examples:
        >>> from skcomplex.models.engineering.traffic import CarelessBrakingInteraction
        >>> CarelessBrakingInteraction('bozo', cars='cars').probability
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        cars: Annotated[str, RoleSpec()],
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        gap: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.cars = cars
        self.probability = probability
        self.gap = gap

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of random braking and motion.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.drive)]

    def drive(self, model: ModelView) -> None:
        """Make a random driver brake with the interaction's probability, then move every car.

        Args:
            model:
                The model.
        """
        cars = sorted(model.agents('cars', MotoristView), key=lambda car: car.rank)
        if not cars:
            return
        if model.rng.random() < model.parameter('probability'):
            careless = cars[int(model.rng.integers(len(cars)))]
            careless.v = max(careless.v - careless.braking, 0.0)
        gap = model.parameter('gap')
        positions = [car.x for car in cars]
        for index, car in enumerate(cars):
            moved = car.x + car.v
            car.x = min(moved, positions[index + 1] - gap) if index + 1 < len(cars) else moved
