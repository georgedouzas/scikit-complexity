"""Define the transfer state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ...specs._compiled_delay_spec import _CompiledDelaySpec
from ._born_spec import _BornSpec
from ._kernel_spec import KernelSpec
from ._record_spec import _RecordSpec

type Term = tuple[KernelSpec, str, str, Expression, str | None, str | None]


@dataclass(frozen=True)
class TransferStateSpec(BaseSpec):
    """State of the lowering of flows between the pass that expands flows and the pass that lowers them.

    Args:
        quantities:
            The quantity of every tagged state by entry name.

        born:
            The expanded flow of every law a flow became, by the identity of the law.

        flows:
            The expanded flows, in the order they were expanded.

        records:
            The recorded expressions of the flows.

        delays:
            The delays and smoothings lowered to hidden states.

        terms:
            The flows and rates of every spec, with their sources and sinks, for the loop terms.
    """

    quantities: Mapping[str, str]
    born: Mapping[int, _BornSpec]
    flows: tuple[_BornSpec, ...]
    records: tuple[_RecordSpec, ...]
    delays: tuple[_CompiledDelaySpec, ...]
    terms: tuple[Term, ...]

    def replaced(self, specs: Mapping[int, KernelSpec]) -> TransferStateSpec:
        """Return the state with replaced kernel specs, such as specs whose rows are settled.

        Args:
            specs:
                The new spec of every replaced spec, by the identity of the replaced spec.

        Returns:
            state:
                The state that holds the new specs.
        """
        born = {id(item): replace(item, spec=specs.get(id(item.spec), item.spec)) for item in self.flows}
        return replace(
            self,
            born={key: born[id(item)] for key, item in self.born.items()},
            flows=tuple(born[id(item)] for item in self.flows),
            records=tuple(replace(record, spec=specs.get(id(record.spec), record.spec)) for record in self.records),
            terms=tuple((specs.get(id(spec), spec), *rest) for spec, *rest in self.terms),
        )
