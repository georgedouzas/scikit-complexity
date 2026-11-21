"""Define the interaction that moves an amount from one stock to another."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import ClassVar

from sage.all__sagemath_symbolics import SR

from ....core.components.base import BaseComponent
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.errors import ValidationError
from ....core.types import Role


class FlowInteraction(Interaction):
    """An interaction that moves an amount from one stock of an element to another.

    The flow `f` takes `f` from the stock it leaves and gives `f` to the stock it enters, as the one rate law
    `RateLaw({leaves: -1, enters: +1}, f, label=label)`, so the total of the stocks is kept and the trajectory records
    `f` as `<element>.<label>`. A flow from outside the system enters a stock and leaves none, and a
    flow out of the system leaves a stock and enters none. The amount is a function of the namespace whose `target`
    holds the values of the element and whose `t` holds the time.

    Unlike `EquationInteraction`, which gives one state its rate, it changes two stocks by one amount, so what leaves
    one enters the other. Choose it when an amount moves between stocks, as in system dynamics.

    It appears in many fields. Water drains from a bathtub, people adopt a product in a diffusion of innovations, and
    capital depreciates in an economy.

    Args:
        label:
            The label of the interaction and the name of its flow, such as `'births'`.

        target:
            The path of the element that holds the stocks.

        leaves:
            The stock the flow leaves, or `None` for a flow from outside the system.

        enters:
            The stock the flow enters, or `None` for a flow out of the system.

        amount:
            The amount per unit of time as a function of the namespace, or a number.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.general.dynamics import FlowInteraction
        >>> drain = FlowInteraction('drain', target='tub', leaves='water', amount=lambda s: s.target.k * s.target.water)
        >>> system = System(elements=[Element('tub', values={'water': 10.0, 'k': 0.5})], interactions=[drain])
        >>> trajectory = system.prepare().simulate(until=2.0, dt=2.0)
        >>> [round(float(value), 4) for value in trajectory.get_values('tub.drain')]
        [5.0, 1.8394]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        leaves: str | None = None,
        enters: str | None = None,
        amount: Callable[[SimpleNamespace], object] | float,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.leaves = leaves
        self.enters = enters
        self.amount = amount

    def _check_arguments(self) -> None:
        """Raise an error for an argument the interaction cannot take.

        Raises:
            ValidationError:
                The flow neither leaves nor enters a stock.
        """
        super()._check_arguments()
        if self.leaves is None and self.enters is None:
            error_msg = f'The flow {self.label!r} neither leaves nor enters a stock. Give `leaves`, `enters`, or both.'
            raise ValidationError(error_msg)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate law of the flow, which maps its stocks to signs.

        Args:
            context:
                The namespace of the element.
        """
        super()._prepare(context)
        amount = SR(self.amount(context) if callable(self.amount) else self.amount)
        signs = {self.leaves: -1, self.enters: 1}
        stocks = {getattr(context.target, name): sign for name, sign in signs.items() if name is not None}
        self.laws_: list[BaseComponent] = [RateLaw(stocks, amount, label=self.label)]
