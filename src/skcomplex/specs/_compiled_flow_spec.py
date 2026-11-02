"""Define the compiled flow spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _CompiledFlowSpec(BaseSpec):
    """Transfer of a compiled system between two holders."""

    path: str
    source: str | None
    sink: str | None
    source_label: str
    sink_label: str
    quantity: str | None
