"""Define the weight spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec
from ._topology_request_spec import TopologyRequestSpec


@dataclass(frozen=True, eq=False)
class _WeightSpec(BaseSpec):
    """Target of the weight of the candidate nodes of a topology request, lowered with the reductions of laws."""

    request: TopologyRequestSpec
