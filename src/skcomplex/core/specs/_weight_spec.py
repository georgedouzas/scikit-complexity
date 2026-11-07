"""Define the weight spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec
from ._topology_request_spec import TopologyRequestSpec


@dataclass(frozen=True, eq=False)
class _WeightSpec(BaseSpec):
    """Target of the weight or the score of the candidate nodes of a topology request, lowered like laws."""

    request: TopologyRequestSpec
    score: bool = False
