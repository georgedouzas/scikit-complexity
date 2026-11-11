"""Define the element whose states and parameters its subclass declares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import ClassVar

from ...specs import ConnectorSpec
from ..base import BaseComponent
from ._base import BaseElement


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

        visual:
            How the element is drawn, such as `VisualSpec('line')`, or `None` for `default_visual_`.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

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
