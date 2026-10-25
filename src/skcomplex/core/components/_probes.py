"""Define the probing of the laws of a component with free symbols, before its states are declared."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

from ..base import VariableKind, declarations_of, symbol
from ..laws import BaseLaw, Symbols
from ._component import BaseComponent
from ._components import GroupComponent


def probe_laws(component: object) -> list[BaseLaw] | None:
    """Return the laws of a component for a namespace of free symbols, before its states are declared.

    A group returns the laws of a representative member, and an interaction reads one namespace of free symbols per
    role. Laws that cannot be written with free symbols, such as laws that index a vector state, are not listed.

    Args:
        component:
            The element, group, or interaction.

    Returns:
        laws:
            The laws, or `None` when they cannot be listed.

    Examples:
        >>> from skcomplex.core.components import ElementComponent, probe_laws
        >>> from skcomplex.core.laws import RateLaw
        >>> DecayComponent = ElementComponent.define('DecayComponent', laws=lambda s: [RateLaw(s.x, -s.x)])
        >>> [type(law).__name__ for law in probe_laws(DecayComponent('decay'))]
        ['RateLaw']
    """
    if isinstance(component, GroupComponent):
        component_type: type[BaseComponent] = component.element
    elif isinstance(component, BaseComponent):
        component_type = type(component)
    else:
        return None
    if component_type.laws is BaseComponent.laws:
        return []
    roles = declarations_of(component_type, VariableKind.ROLE)
    try:
        member = (
            component.element(component.label, **(component.values or {}))
            if isinstance(component, GroupComponent)
            else component
        )
        return list(member.laws(_FreeSymbols(**{name: _FreeSymbols() for name in roles})))
    except Exception:  # noqa: BLE001
        return None


class _FreeSymbols(Symbols):
    """Namespace that returns a free symbol for every name, so laws can be listed before the states are declared."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Return a free symbol of the name, or a namespace of free symbols for `other`, the neighbour of a member."""
        if name.startswith('__'):
            raise AttributeError(name)
        if name == 'other':
            return _FreeSymbols()
        return symbol(name)
