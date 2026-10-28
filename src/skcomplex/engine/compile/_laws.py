"""Name the law classes the compiler treats alike, and read the values the arguments of a law hold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from collections.abc import Collection, Mapping

from sage.modules.free_module_element import FreeModuleElement

from ...core.laws import (
    BaseLaw,
    BoundaryLaw,
    CrossingLaw,
    DestructionLaw,
    DriveLaw,
    EncounterLaw,
    ExtremalLaw,
    FluxLaw,
    LocalRuleLaw,
    ProliferationLaw,
    RelaxationLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
)
from ...core.laws.utils import Call

FORMLESS_LAWS = (BoundaryLaw, CrossingLaw, EncounterLaw, ProliferationLaw, DestructionLaw, FluxLaw)
"""Laws that leave the form of a system as the other laws decide it."""

KERNEL_FREE_LAWS = (
    TransferLaw,
    EncounterLaw,
    ProliferationLaw,
    DestructionLaw,
    LocalRuleLaw,
    DriveLaw,
    RelaxationLaw,
    ExtremalLaw,
    SampleLaw,
    FluxLaw,
)
"""Laws that their own passes lower, which no numeric kernel is written for."""

LATTICE_LAWS = (LocalRuleLaw, DriveLaw, RelaxationLaw, ExtremalLaw, SampleLaw)
"""Laws that update the sites of lattices and networks."""

AGENT_LAWS = (RuleLaw,)
"""Laws that make an element an agent, run on the agent stepper."""


def values_of(law: BaseLaw, *, include: Collection[str] | None = None, exclude: Collection[str] = ()) -> list[object]:
    """Return the values of the arguments of a law besides its label, flattened in the order of its constructor.

    A call gives its arguments, a mapping its keys and values in pairs, and a vector, a list, or a tuple its items, at
    any depth. Every other value is returned as it is.

    Args:
        law:
            The law.

        include:
            The names of the arguments to read, or `None` for every argument.

        exclude:
            The names of the arguments to leave out.

    Returns:
        values:
            The flattened values.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import JumpLaw
        >>> from skcomplex.engine.compile._laws import values_of
        >>> s, i = var('s'), var('i')
        >>> values_of(JumpLaw(0.1 * s * i, {s: -1, i: 1}), include=('rate',))
        [0.1*i*s]
    """
    names = [
        name
        for name in inspect.signature(type(law)).parameters
        if name != 'label' and (include is None or name in include) and name not in exclude
    ]
    stack: list[object] = [getattr(law, name) for name in reversed(names)]
    found: list[object] = []
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            stack += reversed([item for pair in value.items() for item in pair])
        elif isinstance(value, FreeModuleElement | list | tuple):
            stack += reversed(list(value))
        elif isinstance(value, Call):
            stack += reversed(value.arguments)
        else:
            found.append(value)
    return found


def replaced[L: BaseLaw](law: L, **changes: object) -> L:
    """Return a new law of the class of a law, with some of its arguments changed.

    Args:
        law:
            The law.

        **changes:
            The new values of arguments, by name.

    Returns:
        law:
            The new law, which the compiler reads without checking it again.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import RateLaw
        >>> from skcomplex.engine.compile._laws import replaced
        >>> x = var('x')
        >>> replaced(RateLaw(x, -x), value=-2 * x).value
        -2*x
    """
    return type(law)(**{**law.get_params(deep=False), **changes})
