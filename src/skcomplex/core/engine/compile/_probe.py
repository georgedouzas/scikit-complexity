"""Probe the laws of the parts of a system before compiling, for the passes that need to know their kinds early."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import var

from ...components.elements import BaseElement, GroupElement
from ...components.interactions import BaseInteraction
from ...errors import SkcomplexError
from ...specs import RoleSpec


def _probe_namespace(part: BaseElement | BaseInteraction) -> SimpleNamespace:
    """Return the namespace of free symbols of the declared names of a part, which probing its laws reads.

    An element gives its states and its parameters, with `other`, the neighbour of a member, holding the same
    names. An interaction gives its parameters and an empty namespace per role, or an empty tuple for a variadic role,
    since the names of the bound elements are not known before compiling. Both give the time `t`.
    """
    specs = part._get_part_specs()
    names = {name: var(name) for name, spec in specs.items() if not isinstance(spec, RoleSpec)}
    roles = {
        name: () if spec.variadic else SimpleNamespace() for name, spec in specs.items() if isinstance(spec, RoleSpec)
    }
    other = {} if roles else {'other': SimpleNamespace(**names)}
    return SimpleNamespace(**names, **roles, **other, t=var('t'))


def probe_laws(component: object) -> list[object] | None:
    """Return the laws of a component for a namespace of its declared names, before its states are declared.

    A group returns the laws of a representative member. The namespace holds a free symbol for every declared state
    and parameter, the time `t`, `other` for the neighbour of a member, and an empty namespace per role. Laws that
    read other names, such as the states of the elements a role binds, or that index a vector state, are not listed:
    the errors that reading them raises leave the laws unlisted.

    Args:
        component:
            The element, group, or interaction.

    Returns:
        laws:
            The laws, or `None` when they cannot be listed.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.engine.compile._probe import probe_laws
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.specs import StateSpec
        >>> class DecayElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.x = x
        ...
        ...     def _get_laws(self, namespace):
        ...         return [RateLaw(namespace.x, -namespace.x)]
        >>> [type(law).__name__ for law in probe_laws(DecayElement('decay'))]
        ['RateLaw']
    """
    if not isinstance(component, BaseElement | BaseInteraction):
        return None
    try:
        member = component._get_member() if isinstance(component, GroupElement) else component
        return list(member.prepare().run(_probe_namespace(member)).laws_)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ArithmeticError, RuntimeError, SkcomplexError):
        return None
