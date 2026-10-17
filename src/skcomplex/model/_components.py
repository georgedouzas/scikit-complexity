"""Define the elements and interactions that systems are built from."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from typing import Any, ClassVar, dataclass_transform

from sklearn.base import BaseEstimator

from ..core import VisualSpec, parameter, role, state
from ._laws import Law
from ._symbols import Symbols


@dataclass_transform(kw_only_default=True, field_specifiers=(dataclasses.field, state, parameter, role))
@dataclasses.dataclass(eq=False, repr=False)
class Component(BaseEstimator):
    """Base class of elements and interactions.

    Every subclass becomes a dataclass whose fields after `label` are keyword-only, so the constructor stores every
    argument unmodified under its own name.

    Args:
        label:
            The label of the component, unique among the children of its system.
    """

    label: str

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Turn the subclass into a keyword-only dataclass."""
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(kw_only=True, eq=False, repr=False)(cls)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the laws of the component.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class Element(Component):
    """Base class of entities that carry state, such as particles, populations, agents, or firms.

    Subclasses declare their states with `state` and their parameters with `parameter`, and can override `laws` to
    return their intrinsic dynamics. The class variable `visual` names how the state is drawn.

    Args:
        label:
            The label of the element, unique among the children of its system.

    Examples:
        >>> from skcomplex.core import Domain, state
        >>> from skcomplex.model import Element
        >>> class Population(Element):
        ...     n: float = state(1.0, domain=Domain.NONNEGATIVE)
        >>> Population('rabbits', n=40)
        Population(label='rabbits', n=40)
    """

    visual: ClassVar[VisualSpec] = VisualSpec()


class Interaction(Component):
    """Base class of laws that couple elements to each other or to the environment.

    Subclasses declare their roles with `role` and their parameters with `parameter`, and override `laws`. Inside
    `laws`, `s.<role>` is the namespace of the element bound to the role and `s.<parameter>` is a symbol.

    Args:
        label:
            The label of the interaction, unique among the children of its system.

    Examples:
        >>> from skcomplex.core import Domain, parameter, role
        >>> from skcomplex.model import Interaction, Rate
        >>> class Growth(Interaction):
        ...     target: str = role()
        ...     r: float = parameter(1.0, domain=Domain.POSITIVE)
        ...
        ...     def laws(self, s):
        ...         return [Rate(s.target.n, s.r * s.target.n)]
        >>> Growth('growth', target='rabbits', r=0.5)
        Growth(label='growth', r=0.5, target='rabbits')
    """
