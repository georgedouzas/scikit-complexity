"""Keep the laws of the owners a compilation compiles, masked to the rows present of capacity groups."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from dataclasses import replace
from typing import cast

from sage.all__sagemath_symbolics import SR

from ....core.base import Expression, SymbolicVector, vector
from ....core.laws import BaseLaw, Invariant, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from .._census import kind_of, law_owner
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._pipeline import BasePass
from .._specs import KernelSpec, mask_of, masked_value
from .._tree import Node


def _kept_update(law: UpdateLaw, mask: Expression) -> UpdateLaw:
    """Return an update law that keeps the value of its target on the free rows of a capacity layout."""
    value, kept = masked_value(law.value, mask), masked_value(law.target, 1 - mask)
    if isinstance(value, SymbolicVector):
        return UpdateLaw(law.target, vector(a + b for a, b in zip(value, cast('SymbolicVector', kept), strict=True)))
    return UpdateLaw(law.target, SR(value) + SR(kept))


def _masked(spec: KernelSpec, laws: list[BaseLaw], declarations: Declarations) -> list[BaseLaw]:
    """Return laws that leave the free rows of a capacity layout unchanged and fire no event for them."""
    mask = mask_of(spec, declarations)
    if mask is None:
        return laws
    masked: list[BaseLaw] = []
    for law in laws:
        if isinstance(law, RateLaw | NoiseLaw):
            masked.append(replace(law, value=masked_value(law.value, mask)))
        elif isinstance(law, JumpLaw) and law.rate is not None:
            masked.append(replace(law, rate=masked_value(law.rate, mask)))
        elif isinstance(law, UpdateLaw) and isinstance(law.value, Expression | SymbolicVector | int | float):
            masked.append(_kept_update(law, mask))
        else:
            masked.append(law)
    return masked


def _owned(specs: Laws, owners: frozenset[str] | None, agents: set[str]) -> Laws:
    """Return the laws of the components in `owners`, or every law."""
    return [
        (spec, laws if owners is None else [law for law in laws if law_owner(spec, law, agents) in owners])
        for spec, laws in specs
    ]


def _law_kinds(specs: list[tuple[KernelSpec, list[BaseLaw]]], agents: set[str]) -> dict[str, set[type]]:
    """Return the kinds of the laws of every component, without the invariants of members."""
    kinds: dict[str, set[type]] = defaultdict(set)
    for spec, laws in specs:
        for law in laws:
            if not (isinstance(law, Invariant) and law.is_member):
                kinds[law_owner(spec, law, agents)].add(kind_of(law))
    return kinds


class OwnersPass(BasePass):
    """Pass that keeps the laws of the compiled owners, masks them to the rows present, and finds their kinds."""

    reads = ('shifted', 'apart', 'nodes', 'layout')
    writes = ('drawn', 'kinds', 'drawn_apart')

    def run(self, compilation: Compilation) -> None:
        """Keep the laws of the owners of the compilation and find their kinds.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        agents = {node.path for node in cast('list[Node]', compilation.nodes) if node.is_agent}
        owned = _owned(cast('Laws', compilation.shifted), compilation.owners, agents)
        compilation.drawn = [(spec, _masked(spec, laws, declarations)) for spec, laws in owned]
        compilation.drawn_apart = _owned(cast('Laws', compilation.apart), compilation.owners, agents)
        compilation.kinds = _law_kinds([*compilation.drawn, *compilation.drawn_apart], agents)
