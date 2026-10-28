"""Keep the laws of the owners a compilation compiles, masked to the rows present of capacity groups."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from typing import cast

from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.laws import BaseLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ...problems import vector
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._pipeline import BasePass
from .._specs import KernelSpec
from .._tree import Node
from ._census import CensusPass
from ._kernels import KernelsPass
from ._topology import MaskRequestsPass


class OwnersPass(BasePass):
    """Pass that keeps the laws of the compiled owners, masks them to the rows present, and finds their kinds."""

    reads = ('shifted', 'apart', 'nodes', 'layout')
    writes = ('drawn', 'kinds', 'drawn_apart')

    def _run(self, compilation: Compilation) -> None:
        """Keep the laws of the owners of the compilation and find their kinds.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        agents = {node.path for node in cast('list[Node]', compilation.nodes) if node.is_agent}
        owned = self._owned(cast('Laws', compilation.shifted), compilation.owners, agents)
        compilation.drawn = [(spec, self._masked(spec, laws, declarations)) for spec, laws in owned]
        compilation.drawn_apart = self._owned(cast('Laws', compilation.apart), compilation.owners, agents)
        compilation.kinds = self._law_kinds([*compilation.drawn, *compilation.drawn_apart], agents)

    @staticmethod
    def _kept_update(law: UpdateLaw, mask: Expression) -> UpdateLaw:
        """Return an update law that keeps the value of its target on the free rows of a capacity layout."""
        value, kept = (
            MaskRequestsPass._masked_value(law.value, mask),
            MaskRequestsPass._masked_value(law._get_targets()[0], 1 - mask),
        )
        if isinstance(value, FreeModuleElement):
            return UpdateLaw(
                law._get_targets()[0],
                vector(a + b for a, b in zip(value, cast('FreeModuleElement', kept), strict=True)),
            )
        return UpdateLaw(law._get_targets()[0], SR(value) + SR(kept))

    @staticmethod
    def _masked(spec: KernelSpec, laws: list[BaseLaw], declarations: Declarations) -> list[BaseLaw]:
        """Return laws that leave the free rows of a capacity layout unchanged and fire no event for them."""
        mask = MaskRequestsPass._mask_of(spec, declarations)
        if mask is None:
            return laws
        masked: list[BaseLaw] = []
        for law in laws:
            if isinstance(law, RateLaw | NoiseLaw):
                masked.append(BasePass._replaced(law, value=MaskRequestsPass._masked_value(law.value, mask)))
            elif isinstance(law, JumpLaw) and law.rate is not None:
                masked.append(BasePass._replaced(law, rate=MaskRequestsPass._masked_value(law.rate, mask)))
            elif isinstance(law, UpdateLaw) and isinstance(law.value, Expression | FreeModuleElement | int | float):
                masked.append(OwnersPass._kept_update(law, mask))
            else:
                masked.append(law)
        return masked

    @staticmethod
    def _owned(specs: Laws, owners: frozenset[str] | None, agents: set[str]) -> Laws:
        """Return the laws of the components in `owners`, or every law."""
        return [
            (
                spec,
                laws if owners is None else [law for law in laws if CensusPass._law_owner(spec, law, agents) in owners],
            )
            for spec, laws in specs
        ]

    @staticmethod
    def _law_kinds(specs: list[tuple[KernelSpec, list[BaseLaw]]], agents: set[str]) -> dict[str, set[type]]:
        """Return the kinds of the laws of every component."""
        kinds: dict[str, set[type]] = defaultdict(set)
        for spec, laws in specs:
            for law in laws:
                kinds[CensusPass._law_owner(spec, law, agents)].add(KernelsPass._kind_of(law))
        return kinds
