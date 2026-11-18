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
from ....core.components.spaces import BaseSpace
from ....core.errors import ValidationError
from ....core.specs import Role


class FlowInteraction(Interaction):
    """An interaction that moves an amount per unit of time from one stock of an element to another.

    The flow `f` takes `f` from the stock it leaves and gives `f` to the stock it enters, as the two rate terms
    `RateLaw(leaves, -f, label=label)` and `RateLaw(enters, f, label=label)`, so the total of the stocks is kept and the
    trajectory records `f` as `<element>.<label>`. A flow from outside the system enters a stock and leaves none, and a
    flow out of the system leaves a stock and enters none. The amount is a function of the namespace whose `target`
    holds the values of the element and whose `t` holds the time.

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
            The amount per unit of time as a function of the namespace.

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
        amount: Callable[[SimpleNamespace], object],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.leaves = leaves
        self.enters = enters
        self.amount = amount

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate term of the stock the flow leaves and of the stock it enters.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the element.

        Raises:
            ValidationError:
                The flow neither leaves nor enters a stock.
        """
        super()._prepare(context)
        if self.leaves is None and self.enters is None:
            error_msg = f'The flow {self.label!r} neither leaves nor enters a stock. Give `leaves`, `enters`, or both.'
            raise ValidationError(error_msg)
        if not isinstance(context, SimpleNamespace):
            return
        amount = SR(self.amount(context))
        laws: list[BaseComponent] = []
        if self.leaves is not None:
            laws.append(RateLaw(getattr(context.target, self.leaves), -amount, label=self.label))
        if self.enters is not None:
            laws.append(RateLaw(getattr(context.target, self.enters), amount, label=self.label))
        self.laws_: list[BaseComponent] = laws
