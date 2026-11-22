"""Define the compiled request spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from ...base import BaseSpec
from ...components.laws import TopologyRequestSpec
from ._law_kernel import LawKernel
from ._network_rows_spec import _NetworkRowsSpec


@dataclass(frozen=True)
class CompiledRequestSpec(BaseSpec):
    """A spec of a request to change a network, with the kernels that decide it.

    A law may ask to add or remove a node or an edge, such as a new node that joins old nodes in proportion to their
    degree. The spec holds that request, the rows of the network it runs for, and the compiled kernels of its
    probability, of the weight of every candidate end, and of the score of every candidate edge. It grows the web of
    citations, the friendships of a school, and the trade links of countries.

    Read more in the user guide, under Extend the library.

    Args:
        request:
            The request.

        owner:
            The path of the component whose law requests it.

        network:
            The variable groups of the network it changes.

        rows:
            `'node'` for a law of the nodes, `'edge'` for a law of the edges of a layer, `'hyperedge'` for a law of the
            hyperedges, or `'single'` for a law that runs once.

        condition:
            The kernel of the probability that an update law fires the request for every row, or `None` for a jump.

        toward:
            The kernel of the weight of every node row as a new end, or `None` for equal weights.

        score:
            The kernel of the score of every node row as the end of a candidate edge, or `None` for none.

    See Also:
        TopologyKernel: Applies the requests between steps.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import BarabasiAlbertSystem
        >>> request = BarabasiAlbertSystem().prepare().simulate(until=0).problem.topology.requests[0]
        >>> request.owner, request.rows
        ('growth', 'single')
    """

    request: TopologyRequestSpec
    owner: str
    network: _NetworkRowsSpec
    rows: str
    condition: LawKernel | None = None
    toward: LawKernel | None = None
    score: LawKernel | None = None
