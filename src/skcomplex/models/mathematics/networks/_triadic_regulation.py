"""Regulation of every link by a third node, which lets the link be active with probability `p` in its phase."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.sets.real_set import RealSet

from ....core.interactions import HyperedgeInteraction
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import bernoulli, where
from ....specs import ParameterSpec, RoleSpec


class TriadicRegulationInteraction(HyperedgeInteraction):
    """Regulation of every link by a third node, which lets the link be active with probability `p` in its phase.

    Every hyperedge `(i, j, r)` holds the two ends of the link `(i, j)` and its regulating node `r`. In every step the
    link becomes active with probability `p` when a positive regulator is active or a negative regulator is inactive,
    and inactive otherwise.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group on the nodes, whose activity the regulators read.

        links:
            The path of the group on the edges, whose activity the regulation sets.

        p:
            The probability that a link the regulation allows is active.

        phase:
            The phase of the step the regulation runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.mathematics.networks import TriadicRegulationInteraction
        >>> TriadicRegulationInteraction('regulation', nodes='nodes', links='links').size
        3
    """

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        p: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.5,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, size=3)
        self.nodes = nodes
        self.links = links
        self.p = p
        self.phase = phase

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the activity of the link from its regulating node, the third member of the hyperedge."""
        regulator = s.others[1].active
        allowed = where(s.edge.sign > 0, regulator, 1 - regulator)
        return [UpdateLaw(s.edge.active, bernoulli('retained', s.p) * allowed, phase=self.phase)]
