"""Define elements, the entities of a system that carry states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseElement


class Element(BaseElement):
    """Base class of entities that carry state, such as particles, populations, agents, or firms.

    A subclass takes its states, its parameters, and `visual` as arguments of its own `__init__`, annotating each state
    with a `StateSpec` and each parameter with a `ParameterSpec` inside `Annotated`, and returns its intrinsic dynamics
    from `_get_laws`, which returns no laws by default. A subclass drawn in its own way gives `visual` a default in its
    signature.

    An element whose laws have the agent trait, such as rules, alone or in a group, is an agent: under discrete time it
    runs on the agent stepper, whose rules receive every member as an agent with its states as attributes. Its states
    then hold any value, not only numbers. A member is removed only when its states tagged with a quantity are settled
    to zero, and its states and parameters cannot take the names of the agent view, such as `location` or `alive`, or
    of the attributes Mesa reads, such as `position` or `model`.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.specs import StateSpec
        >>> from skcomplex.specs import VisualSpec
        >>> from sage.all__sagemath_symbolics import oo
        >>> from sage.sets.real_set import RealSet
        >>> class PopulationElement(Element):
        ...     def __init__(
        ...         self,
        ...         label,
        ...         n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        ...         visual=VisualSpec('line'),
        ...     ):
        ...         super().__init__(label, visual=visual)
        ...         self.n = n
        >>> PopulationElement('rabbits', n=40)
        PopulationElement(label='rabbits', n=40)
    """

    def _get_laws(self, namespace: object) -> list[object]:
        """Return the laws of the element for a namespace of symbols, none by default.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return []
