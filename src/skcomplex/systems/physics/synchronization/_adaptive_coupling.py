"""Define adaptive Kuramoto-Sakaguchi coupling, whose weights relax to a phase-lagged sine of the phase difference."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import sin

from ....core.components.interactions import NeighborInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, RoleSpec


class AdaptiveCouplingInteraction(NeighborInteraction):
    """Adaptive Kuramoto-Sakaguchi coupling, whose weights relax towards a phase-lagged sine of the phase difference.

    On every directed link from `j` to `i`, the phase of `i` turns by `-K kappa_ij sin(theta_i - theta_j + alpha)` and
    the weight relaxes as `kappa_ij' = -epsilon (kappa_ij + sin(theta_i - theta_j + beta))`, so it grows between
    oscillators whose phases differ by about `-beta - pi / 2`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group of phase oscillators on the nodes.

        links:
            The path of the group of adaptive links on the edges.

        K:
            The coupling strength, `1 / N` for the mean coupling of `N` oscillators.

        alpha:
            The phase lag of the coupling.

        beta:
            The phase lag of the adaptation.

        epsilon:
            The rate of the adaptation.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import AdaptiveCouplingInteraction
        >>> AdaptiveCouplingInteraction('adaptation', nodes='nodes', links='links').epsilon
        0.01
    """

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec()] = 1.0,
        alpha: Annotated[float, ParameterSpec()] = 0.0,
        beta: Annotated[float, ParameterSpec()] = 0.0,
        epsilon: Annotated[float, ParameterSpec()] = 0.01,
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.links = links
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the coupling of the target of every link and the adaptation of its weight.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the ends `source` and `target` and the `edge`.

        Returns:
            laws:
                The rate law of the phase of the target and the rate law of the weight of the link.
        """
        s = namespace
        difference = s.target.theta - s.source.theta
        return [
            RateLaw(s.target.theta, -s.K * s.edge.kappa * sin(difference + s.alpha)),
            RateLaw(s.edge.kappa, -s.epsilon * (s.edge.kappa + sin(difference + s.beta))),
        ]
