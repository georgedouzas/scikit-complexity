"""Define the element whose values come with their allowed values and quantities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import Annotated

from ....core.components.elements import Element, Quantity


class AnnotatedElement(Element):
    """An element of values by name, with the allowed values and quantities of some.

    It holds what a class would write in `Annotated`, for values known only as data. The values
    `{'n': 40.0}` with `allowed={'n': RealSet.closed_open(0, oo)}` and `quantities={'n': 'people'}` are read as
    `n: Annotated[float, RealSet.closed_open(0, oo), Quantity('people')] = 40.0`, so a negative `n` is rejected and the
    conservation monitor follows the people. The generic systems that build one element from a mapping of states, such
    as `DynamicalSystem` and `StockFlowSystem`, build this element.

    Unlike `Element`, whose subclass writes the allowed values and quantities in `Annotated` on its arguments, it takes
    them as data. Choose it when the names of the values are known only when a system is built.

    It appears in many fields. The stocks of a bathtub, the people of a diffusion of innovations, and the oil of an
    economy are each its values.

    Args:
        label:
            The name of the element, unique among the components of its system, such as `'population'`.

        values:
            The values by name, such as `{'n': 40.0, 'r': 0.5}`, or `None` for none.

        allowed:
            The allowed values of some values by name, a `RealSet` such as `RealSet.closed_open(0, oo)` or a set such
            as `{'S', 'I', 'R'}`, or `None` for none.

        quantities:
            The quantity some values hold by name, such as `{'n': 'people'}`, or `None` for none.

    Examples:
        >>> from sage.all__sagemath_symbolics import oo
        >>> from sage.sets.real_set import RealSet
        >>> from skcomplex.systems.general.dynamics import AnnotatedElement
        >>> from skcomplex.core.errors import ValidationError
        >>> from skcomplex.core.systems import System
        >>> tank = AnnotatedElement('tank', values={'level': -1.0}, allowed={'level': RealSet.closed_open(0, oo)})
        >>> try:
        ...     System(elements=[tank]).prepare()
        ... except ValidationError as error:
        ...     print(error)
        The system is not valid, with 1 error(s):
        - tank: tank.level = -1.0 is not in its domain [0, +oo). Correct the component as the message describes. [value]
    """

    def __init__(
        self,
        label: str,
        *,
        values: Mapping[str, object] | None = None,
        allowed: Mapping[str, object] | None = None,
        quantities: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.allowed = allowed
        self.quantities = quantities

    def _get_hints(self) -> dict[str, object]:
        """Return the hint of every value with allowed values or a quantity.

        Each hint is what `Annotated` would write. The arguments `allowed` and `quantities` describe the values, so they
        are no values themselves.
        """
        allowed, quantities = self.allowed or {}, self.quantities or {}
        hints: dict[str, object] = {}
        for name in [*allowed, *(name for name in quantities if name not in allowed)]:
            items = [allowed[name]] if name in allowed else []
            items += [Quantity(quantities[name])] if name in quantities else []
            hints[name] = Annotated[object, *items]
        return hints
