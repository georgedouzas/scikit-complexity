"""Define the weight spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...components.laws import TopologyRequestSpec
from ...specs._base_spec import BaseSpec


@dataclass(frozen=True, eq=False)
class _WeightSpec(BaseSpec):
    """Target of the weight or the score of the candidate nodes of a topology request.

    It is lowered like laws.
    """

    request: TopologyRequestSpec
    score: bool = False
