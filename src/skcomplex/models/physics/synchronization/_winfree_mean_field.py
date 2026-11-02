"""Winfree coupling `theta_i' += K mean_j P(theta_j) R(theta_i)` over every member `j` of the group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, sin

from ....core.interactions import CouplingInteraction
from ....specs import ParameterSpec, RoleSpec


class WinfreeMeanFieldInteraction(CouplingInteraction):
    """Winfree coupling `theta_i' += K mean_j P(theta_j) R(theta_i)` over every member `j` of the group.

    The influence `P(theta) = 1 + cos(theta)` that an oscillator sends and the sensitivity `R(theta) = -sin(theta)` with
    which it responds make every oscillator respond to the mean influence of the group.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of oscillators.

        K:
            The coupling strength.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.models.physics.synchronization import WinfreeMeanFieldInteraction
        >>> coupling = WinfreeMeanFieldInteraction('coupling', target='oscillators')
        >>> member, other = SimpleNamespace(theta=var('a')), SimpleNamespace(theta=var('b'))
        >>> coupling.function(SimpleNamespace(K=var('K'), member=member, other=other))
        -K*(cos(b) + 1)*sin(a)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            read='theta',
            write='theta',
            function=self._response,
            reduce='mean',
            mode='group',
            include_self=True,
        )
        self.K = K

    def _response(self, namespace: SimpleNamespace) -> Expression:
        """Return the response `K P(theta_j) R(theta_i)` of a member to one other member."""
        return namespace.K * (1 + cos(namespace.other.theta)) * -sin(namespace.member.theta)
