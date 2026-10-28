"""Target of the weight of the candidate nodes of a topology request, lowered with the reductions of laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....specs import TopologyRequestSpec


@dataclass(frozen=True, eq=False)
class _Weight:
    """Target of the weight of the candidate nodes of a topology request, lowered with the reductions of laws."""

    request: TopologyRequestSpec
