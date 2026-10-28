"""Topology request of a law with the rows it runs for and the kernels that decide and direct it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...specs import BaseSpec, TopologyRequestSpec
from ._law_kernel import LawKernel
from ._network_rows import NetworkRows


@dataclass(frozen=True)
class CompiledRequest(BaseSpec):
    """Topology request of a law with the rows it runs for and the kernels that decide and direct it.

    Args:
        request:
            The request.

        owner:
            The path of the component whose law requests it.

        network:
            The variable groups of the network it changes.

        rows:
            `'node'` for a law of the nodes, `'edge'` for a law of the edges of a layer, or `'single'` for a law that
            runs once.

        condition:
            The kernel of the probability that an update law fires the request for every row, or `None` for a jump.

        toward:
            The kernel of the weight of every node row as a new end, or `None` for equal weights.
    """

    request: TopologyRequestSpec
    owner: str
    network: NetworkRows
    rows: str
    condition: LawKernel | None = None
    toward: LawKernel | None = None
