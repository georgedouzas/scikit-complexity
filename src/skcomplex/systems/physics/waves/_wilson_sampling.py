"""Monte Carlo sampling of angle-valued link variables by the local Wilson action of a link."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import SamplingInteraction
from ....core.components.laws.utils import exterior_derivative, neighbors
from ....core.components.spaces import CofaceNeighborhood
from ....core.specs import ParameterSpec, RoleSpec


class WilsonSamplingInteraction(SamplingInteraction):
    """Monte Carlo sampling of link angles with the local energy `-beta sum_p cos(theta_p)` of the plaquettes of a link.

    Here `theta_p` is the oriented sum of the link angles around the plaquette `p`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group on the edges of a cell complex.

        beta:
            The inverse coupling of the Wilson action.

        state:
            The name of the state that holds the link angles.

        move:
            `'rotate'` for U(1), or `'random-value'` for Z_N.

        values:
            The angles a random-value move draws from.

        width:
            The largest change of a rotation.

        rule:
            The acceptance, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        schedule:
            `'colored'` or `'random-sequential'`.

    Examples:
        >>> from skcomplex.systems.physics.waves import WilsonSamplingInteraction
        >>> WilsonSamplingInteraction('wilson', target='links', beta=2.0).beta
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        state: str = 'theta',
        move: str = 'rotate',
        values: Sequence[float] = (),
        width: float = 1.5,
        rule: str = 'metropolis',
        schedule: str = 'colored',
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            objective=self._get_action,
            move=move,
            rule=rule,
            schedule=schedule,
            values=values,
            width=width,
        )
        self.beta = beta

    def _get_action(self, namespace: SimpleNamespace) -> Expression:
        """Return the terms of the Wilson action that hold the angle of a link."""
        theta = getattr(namespace.target, self.state)
        return -namespace.beta * neighbors(cos(exterior_derivative(theta)), over=CofaceNeighborhood())
