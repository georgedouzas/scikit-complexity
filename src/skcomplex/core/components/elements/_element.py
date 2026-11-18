"""Define the element whose states and parameters its subclass declares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import inspect
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

from ...errors import ValidationError
from ...specs import ConnectorSpec, ParameterSpec, StateSpec
from .._tags import Tags
from ..base import BaseComponent
from ..spaces import BaseSpace, NetworkSpace
from ._base import BaseElement

RESERVED = ('label', 'size', 'space', 'values')
LINKS = ('edges', 'hyperedges')
RESERVED_MEANINGS = {
    'size': 'number of members',
    'space': 'space of the members',
    'values': 'mapping of values by name',
}


class Element(BaseElement):
    """An element whose states and parameters are declared by its own subclass.

    A subclass takes each state and parameter as an argument of its `__init__`, annotated with `StateSpec` or
    `ParameterSpec` inside `Annotated`. It may return laws of its own from `_get_laws`, such as the growth of a
    population, and it lists their classes in `law_classes`, so that `get_tags` can name their kinds before a run.

    A subclass that connects to other elements, such as a resistor in a circuit, declares its ports in `ports`, a
    mapping from the name of each port to its `ConnectorSpec`. It then takes one algebraic state per variable of a port,
    named `<port>_<variable>`. An equation, not a rate, fixes an algebraic state at every time. A flow is positive into
    the element.

    An element whose laws are agent rules, Python functions that change one member at each step, is an agent. Under
    discrete time each member runs its rules with its states as attributes, and these states may hold any value, not
    only numbers. A member is removed only when its states tagged with a quantity reach zero. Its states and parameters
    cannot take the names that the agent view or Mesa reads, such as `location`, `alive`, `position`, or `system`.

    Unlike `BaseElement`, it returns no laws by default, and it can connect through ports or act as an agent. Choose it
    when a kind of entity deserves a class of its own, such as a population with a growth law or a capacitor with two
    pins, and choose `DeclaredElement` when the states are known only as data.

    It appears in many fields. A prey population in ecology, a capacitor in an electric circuit, a firm in an economy,
    and a pendulum in mechanics are each a subclass of it.

    Args:
        label:
            The name of the element, unique among the components of its system, such as `'rabbits'`.

        values:
            Values by name, such as `{'x': 10.0, 'm': 1.0}`, in place of a subclass that takes them as arguments. A
            derivative or a past value that a law reads starts from the value named with primes or a past index, such
            as `{'x': 10.0, "x'": 3.0}` for a ball thrown upwards or `{'y': 1.0, 'y[-1]': 0.5}` for an economy. An
            element with many members takes one starting value per member as an array, such as the velocities of a
            gas.

        size:
            The number of members, each with its own values, such as `1000` molecules of a gas, or `None` for one.

        space:
            The space of the element, with one member per site, such as a lattice of cells, or `None` for none.

        on:
            The part of a network space that holds one member each: `'edges'` for every link, such as the signed
            links of a network, `'hyperedges'` for every hyperedge, or `None` for every site.

    Attributes:
        default_visual_:
            How the element is drawn by default, the `visual` of its class.

    Examples:
        A population and a bank deposit both grow at a rate proportional to their size, so one class serves both.

        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.specs import ParameterSpec, StateSpec
        >>> class GrowthElement(Element):
        ...     law_classes = (RateLaw,)
        ...
        ...     def __init__(
        ...         self, label, x: Annotated[float, StateSpec()] = 1.0, r: Annotated[float, ParameterSpec()] = 0.5
        ...     ):
        ...         super().__init__(label)
        ...         self.x = x
        ...         self.r = r
        ...
        ...     def _get_laws(self, namespace):
        ...         return [RateLaw(namespace.x, namespace.r * namespace.x)]
        >>> rabbits = GrowthElement('rabbits', x=40.0, r=0.5)
        >>> deposit = GrowthElement('deposit', x=250.0, r=0.05)
        >>> trajectory = simulate(CompositeSystem(elements=[rabbits, deposit]), until=1, dt=1.0)
        >>> trajectory.get_values('rabbits.x').round(1).tolist(), trajectory.get_values('deposit.x').round(2).tolist()
        ([40.0, 65.9], [250.0, 262.82])
    """

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {}
    law_classes: ClassVar[tuple[type, ...]] = ()

    def __init__(
        self,
        label: str,
        *,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
        on: str | None = None,
    ) -> None:
        super().__init__(label)
        self.values = values
        self.size = size
        self.space = space
        self.on: object = on

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, checking the reserved names.

        Args:
            context:
                The space the element inherits from its system, which an element does not read.

        Raises:
            ValidationError:
                `size` is not a number of members, `space` is not a space, or `values` is not a mapping, as when a
                subclass stores a quantity of its own under a reserved name.
        """
        super()._prepare(context)
        size = self.size
        checks = {
            'size': size is None or (isinstance(size, numbers.Integral) and not isinstance(size, bool) and size >= 1),
            'space': self.space is None or isinstance(self.space, BaseSpace),
            'values': self.values is None or isinstance(self.values, Mapping),
        }
        for name, valid in checks.items():
            if not valid:
                error_msg = (
                    f'Element {self.label!r} has {name} = {getattr(self, name)!r}, but {name} is the '
                    f'{RESERVED_MEANINGS[name]}. The names {", ".join(RESERVED)} are reserved, so rename the value, '
                    f'such as half_side for the half side of a table.'
                )
                raise ValidationError(error_msg)
        self._check_on()

    def _get_tags(self) -> Tags:
        """Return the tags of the element, with the tags of its space when it has one.

        A field on a phase space thus asks for a solver that supports phase spaces, as a group on it does.
        """
        tags = super()._get_tags()
        if not isinstance(self.space, BaseSpace):
            return tags
        return replace(tags, space=self.space.get_tags().space)

    def _check_on(self) -> None:
        """Raise an error for an `on` that names no part of a network space.

        A value named `on` that the element declares as its own state or parameter names no place.
        """
        if isinstance(type(self)._get_specs().get('on'), StateSpec | ParameterSpec):
            return
        on = getattr(self, 'on', None)
        if on is None or (on in LINKS and isinstance(self.space, NetworkSpace)):
            return
        error_msg = (
            f'Element {self.label!r} has on = {on!r}, but on is the part of a network space that holds one member '
            "each, 'edges' or 'hyperedges', such as Element('links', values={'sign': 1}, space=network, on='edges')."
        )
        raise ValidationError(error_msg)

    def _get_hints(self) -> dict[str, object]:
        """Return the type hint of every value argument by name, in their order.

        The hints are the annotations of the arguments of the class, such as `Annotated[float, Quantity('energy')]`,
        from which the kind, the allowed values, the quantity, and the period of each value are read. A generic
        subclass that takes its values through `values` gives their hints here instead, and the arguments it leaves
        out, such as the allowed values themselves, are no values.
        """
        parameters = inspect.signature(type(self)).parameters
        return {name: parameter.annotation for name, parameter in parameters.items() if name not in (*RESERVED, 'on')}

    @classmethod
    def _get_view(cls) -> type | None:
        """Return the class of the views of the element's agents.

        The result is `None` for the engine's plain view. An agent element names a subclass of the engine's `AgentView`
        that annotates its states and parameters, such as `class WordView(AgentView): count: float`, so its rules read
        them with checked types.
        """
        return None

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the laws of the element for a namespace of symbols, none by default.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return []
