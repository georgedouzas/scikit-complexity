"""Declare the state variables, parameters, and roles of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._domains import Domain

METADATA_KEY = 'skcomplex'


class VariableKind(Enum):
    """Role a declared field plays in a component."""

    STATE = 'state'
    PARAMETER = 'parameter'
    ROLE = 'role'


@dataclass(frozen=True)
class Variable:
    """Declaration of a field of a component.

    Args:
        kind:
            Whether the field is a state variable, a parameter, or a role.

        domain:
            The set of values the field takes.

        vector:
            Whether the field has one component per dimension of the enclosing space.
    """

    kind: VariableKind
    domain: Domain = Domain.REAL
    vector: bool = False


def _declare_field(default: Any, declaration: Variable) -> Any:
    metadata = {METADATA_KEY: declaration}
    if default is dataclasses.MISSING:
        return dataclasses.field(metadata=metadata)
    if isinstance(default, list | tuple | dict | set):
        return dataclasses.field(default_factory=lambda: copy.deepcopy(default), metadata=metadata)
    return dataclasses.field(default=default, metadata=metadata)


def state(default: Any = 0.0, *, domain: Domain = Domain.REAL, vector: bool = False) -> Any:
    """Declare a state variable of an element, with its initial value as default.

    Args:
        default:
            The default initial value. A scalar default of a vector state applies to every component.

        domain:
            The set of values the state takes.

        vector:
            Whether the state has one component per dimension of the enclosing space.

    Returns:
        field:
            A dataclass field that carries the declaration.

    Examples:
        >>> from skcomplex.core import Domain, state
        >>> from skcomplex.model import Element
        >>> class Population(Element):
        ...     n: float = state(1.0, domain=Domain.NONNEGATIVE)
        >>> Population('rabbits', n=40).n
        40
    """
    return _declare_field(default, Variable(VariableKind.STATE, domain, vector))


def parameter(
    default: Any = dataclasses.MISSING,
    *,
    domain: Domain = Domain.REAL,
    vector: bool = False,
) -> Any:
    """Declare a parameter of an element or an interaction.

    Args:
        default:
            The default value. A parameter without a default must be given to the constructor.

        domain:
            The set of values the parameter takes.

        vector:
            Whether the parameter has one component per dimension of the enclosing space.

    Returns:
        field:
            A dataclass field that carries the declaration.

    Examples:
        >>> from skcomplex.core import Domain, parameter
        >>> from skcomplex.model import Interaction
        >>> class Growth(Interaction):
        ...     r: float = parameter(1.0, domain=Domain.POSITIVE)
        >>> Growth('growth', r=0.5).r
        0.5
    """
    return _declare_field(default, Variable(VariableKind.PARAMETER, domain, vector))


def role() -> Any:
    """Declare a role of an interaction, bound to an element path in the constructor.

    Returns:
        field:
            A required dataclass field that carries the declaration.

    Examples:
        >>> from skcomplex.core import role
        >>> from skcomplex.model import Interaction
        >>> class Mortality(Interaction):
        ...     target: str = role()
        >>> Mortality('death', target='foxes').target
        'foxes'
    """
    return _declare_field(dataclasses.MISSING, Variable(VariableKind.ROLE))


def declarations_of(component: Any, kind: VariableKind) -> dict[str, Variable]:
    """Return the declared fields of one kind of a component, in declaration order.

    Args:
        component:
            A component class or instance.

        kind:
            The kind of field to return.

    Returns:
        declarations:
            A mapping from field name to declaration.
    """
    return {
        field.name: field.metadata[METADATA_KEY]
        for field in dataclasses.fields(component)
        if METADATA_KEY in field.metadata and field.metadata[METADATA_KEY].kind is kind
    }
