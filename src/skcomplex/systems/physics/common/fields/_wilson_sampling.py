"""Monte Carlo sampling of angle or matrix link variables.

The sampling uses the local Wilson action of a link.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import SamplingInteraction
from .....core.components.laws.utils import exterior_derivative, neighbors
from .....core.components.spaces import CofaceNeighborhood
from .....core.specs import HolonomyScoreSpec, MatrixGroupSpec, ParameterSpec, RoleSpec
from .....core.types import Array, ComplexArray


class WilsonSamplingInteraction(SamplingInteraction):
    """Monte Carlo sampling of link variables.

    The sampling uses the local Wilson action of the plaquettes of a link.

    For link angles the action is `-beta sum_p cos(theta_p)`, where `theta_p` is the oriented sum of the link angles
    around the plaquette `p`. For the matrices of a group `SU(N)` it is `-(beta / N) sum_p Re tr U_p`, where `U_p` is
    the path-ordered product of the link matrices around `p`, and the move `'rotate'` multiplies a link by a random
    matrix of the group near the identity of spread `width`.

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
            `'rotate'` for U(1) or a matrix group, or `'random-value'` for Z_N.

        values:
            The angles a random-value move draws from.

        width:
            The largest change of a rotation.

        rule:
            The acceptance, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        schedule:
            `'colored'` or `'random-sequential'`.

        group:
            The matrix group of the links, or `None` for link angles.

        hits:
            The number of moves proposed per link and sweep of matrix links.

    Examples:
        >>> from skcomplex.systems.physics.common.fields import WilsonSamplingInteraction
        >>> WilsonSamplingInteraction('wilson', target='links', beta=2.0).beta
        2.0
        >>> from skcomplex.core.specs import MatrixGroupSpec
        >>> color = WilsonSamplingInteraction('wilson', target='links', state='U', group=MatrixGroupSpec('SU', 3))
        >>> color.objective.group.degree
        3
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        state: str = "theta",
        move: str = "rotate",
        values: Sequence[float] = (),
        width: float = 1.5,
        rule: str = "metropolis",
        schedule: str = "colored",
        group: MatrixGroupSpec | None = None,
        hits: int = 1,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            objective=self._get_action if group is None else HolonomyScoreSpec(self._get_trace_action, group, hits),
            move=move,
            rule=rule,
            schedule=schedule,
            values=values,
            width=width,
        )
        self.beta = beta
        self.group = group
        self.hits = hits

    def _get_action(self, namespace: SimpleNamespace) -> Expression:
        """Return the terms of the Wilson action that hold the angle of a link."""
        theta = getattr(namespace.target, self.state)
        return -namespace.beta * neighbors(cos(exterior_derivative(theta)), over=CofaceNeighborhood())

    def _get_trace_action(self, holonomies: ComplexArray, beta: float) -> Array:
        """Return the Wilson action `-(beta / N) Re tr U_p` of the plaquette holonomies."""
        degree = holonomies.shape[-1]
        return -beta / degree * np.trace(holonomies, axis1=-2, axis2=-1).real
