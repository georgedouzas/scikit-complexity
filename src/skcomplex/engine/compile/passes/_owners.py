"""Keep the laws of the owners a compilation compiles, masked to the rows present of capacity groups."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict

from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.laws import BaseLaw, JumpLaw, LinearProgramLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.spaces import DiscreteTime
from ..._utils import vector
from ...problems import DeclarationsSpec, KernelSpec
from .._base import BasePass
from .._compilation import Compilation, Laws
from ._census import CensusPass
from ._fractional_form import FractionalFormPass
from ._kernels import KernelsPass
from ._mask_requests import MaskRequestsPass


class OwnersPass(BasePass):
    """Pass that keeps the laws of the compiled owners, masks them to the rows present, and finds their kinds."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('shifted', 'apart', 'programs', 'nodes', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('drawn', 'kinds', 'drawn_apart')

    def _run(self, inputs: object = None) -> None:
        """Keep the laws of the owners of the compilation and find their kinds."""
        compilation = self.compilation_
        declarations = Compilation._required(compilation.layout, 'layout')
        agents = {node.path for node in Compilation._required(compilation.nodes, 'nodes') if node.is_agent}
        owned = self._owned(Compilation._required(compilation.shifted, 'shifted'), compilation.owners, agents)
        compilation.drawn = [(spec, self._masked(spec, laws, declarations)) for spec, laws in owned]
        apart = Compilation._required(compilation.apart, 'apart')
        compilation.drawn_apart = self._owned(apart, compilation.owners, agents)
        programs = self._owned(Compilation._required(compilation.programs, 'programs'), compilation.owners, agents)
        compilation.kinds = self._law_kinds([*compilation.drawn, *compilation.drawn_apart, *programs], agents)
        self._set_program_steps(compilation.kinds, FractionalFormPass._declared_times(compilation.system))

    @staticmethod
    def _kept_update(law: UpdateLaw, mask: Expression) -> UpdateLaw:
        """Return an update law that keeps the value of its target on the free rows of a capacity layout."""
        value, kept = (
            MaskRequestsPass._masked_value(law.value, mask),
            MaskRequestsPass._masked_value(law._get_targets()[0], 1 - mask),
        )
        if isinstance(value, FreeModuleElement) and isinstance(kept, FreeModuleElement):
            return BasePass._replaced(law, value=vector(a + b for a, b in zip(value, kept, strict=True)))
        return BasePass._replaced(law, value=SR(value) + SR(kept))

    @staticmethod
    def _masked(spec: KernelSpec, laws: list[BaseLaw], declarations: DeclarationsSpec) -> list[BaseLaw]:
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
    def _set_program_steps(kinds: dict[str, set[type]], times: dict[str, type | None]) -> None:
        """Count a linear program under discrete time as an update, since each step of a map solves it."""
        for owner, classes in kinds.items():
            time = times.get(owner.rpartition('/')[0])
            if LinearProgramLaw in classes and time is not None and issubclass(time, DiscreteTime):
                classes.add(UpdateLaw)

    @staticmethod
    def _law_kinds(specs: list[tuple[KernelSpec, list[BaseLaw]]], agents: set[str]) -> dict[str, set[type]]:
        """Return the kinds of the laws of every component."""
        kinds: dict[str, set[type]] = defaultdict(set)
        for spec, laws in specs:
            for law in laws:
                kinds[CensusPass._law_owner(spec, law, agents)].add(KernelsPass._kind_of(law))
        return kinds
