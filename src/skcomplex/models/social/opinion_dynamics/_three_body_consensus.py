"""Nonlinear three-body consensus of Neuhäuser, Mellor, and Lambiotte, on the triangles active at the time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.interactions import HyperedgeInteraction
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import where
from ....specs import ParameterSpec, RoleSpec

TRIANGLE = 3


class ThreeBodyConsensusInteraction(HyperedgeInteraction):
    """Nonlinear three-body consensus of Neuhäuser, Mellor, and Lambiotte, on the triangles active at the time.

    The focal node `i` of an active triangle `ijk` moves towards the two others at the rate `s(|x_j - x_k|) (x_j + x_k -
    2 x_i)`, with `s(d) = exp(reinforcement * d)`, so a pair that agrees pulls harder for a negative reinforcement. A
    triangle is active when `sin(2 pi t / period + phase) > 0`, with the phase of the triangle member of the group on
    hyperedges.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of nodes.

        triangles:
            The path of the group of the schedules of the triangles.

        reinforcement:
            The exponent of the modulating function `s`.

        period:
            The period of the schedules.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import ThreeBodyConsensusInteraction
        >>> ThreeBodyConsensusInteraction('consensus', target='nodes', triangles='triangles').size
        3
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        triangles: Annotated[str, RoleSpec()],
        reinforcement: Annotated[float, ParameterSpec()] = 0.0,
        period: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, size=TRIANGLE)
        self.target = target
        self.triangles = triangles
        self.reinforcement = reinforcement
        self.period = period

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the pull of the two other members of an active triangle on the focal node."""
        x = s.member.x
        first, second = (other.x for other in s.others)
        active = where(sin(2 * pi * s.t / s.period + s.hyperedge.phase) > 0, 1, 0)
        pull = exp(s.reinforcement * abs(first - second)) * (first + second - 2 * x)
        return [RateLaw(x, active * pull)]
