"""Define Kuramoto coupling, the sine coupling of phases with a phase lag."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import Expression, cos, sin

from .....core.components.interactions import CouplingInteraction
from .....core.components.laws.utils import count, total
from .....core.components.spaces import BaseNeighborhood
from .....core.components.types import Target
from .....core.specs import ParameterSpec, RoleSpec


class KuramotoCouplingInteraction(CouplingInteraction):
    """Kuramoto coupling `theta_i' += K sum_j sin(theta_j - theta_i - alpha)`.

    The coupling acts on the phases of a group.

    The sum, or the mean, runs over the members `j` that the mode selects: the neighbours along the edges of a lattice
    or network, or within a neighbourhood such as `RadiusNeighborhood(r)`, every other member of the group, every member
    of a separate source group, or the next member. With `include_self` it also runs over `j = i`, whose term
    `-K sin(alpha)` slows every phase. Over the whole group without weights or delay, the coupling reads the group sums
    of `sin theta` and `cos theta` once, so it costs time linear in the size of the group.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of oscillators.

        source:
            The path of the group of oscillators the target couples to in the mode `'group'`, or `None` for the target.

        K:
            The coupling strength of one pair.

        alpha:
            The phase lag.

        state:
            The name of the state that holds the phases.

        reduce:
            `'sum'` or `'mean'`.

        mode:
            `'edges'`, `'group'`, or `'next'`.

        weights:
            The name of the state of the edge group in the mode `'edges'`, a matrix in the mode `'group'`, or `None`.

        include_self:
            Whether the sum also runs over the member itself.

        delay:
            The delay of the phases of the other members, 0 for none.

        neighborhood:
            The neighbourhood of the mode `'edges'`, or `None` for the edges of the lattice or network.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the graph of the network.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.physics.common.oscillators import (
        ...     KuramotoCouplingInteraction,
        ...     PhaseOscillatorElement,
        ... )
        >>> from skcomplex.core.engine.runtime import simulate
        >>> phases = GroupElement('phases', element=PhaseOscillatorElement, size=4, values={'theta': [0, 1, 2, 3]})
        >>> coupling = KuramotoCouplingInteraction('coupling', target='phases', K=2.0, mode='group', reduce='mean')
        >>> system = CompositeSystem(elements=[phases], interactions=[coupling])
        >>> theta = simulate(system, until=20.0, dt=1.0).get_values('phases.theta')
        >>> round(float(np.ptp(theta[-1])), 6)
        0.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str | None, RoleSpec(optional=True)] = None,
        K: Annotated[float, ParameterSpec()] = 1.0,
        alpha: Annotated[float, ParameterSpec()] = 0.0,
        state: str = 'theta',
        reduce: str = 'sum',
        mode: str = 'edges',
        weights: str | npt.ArrayLike | scipy.sparse.sparray | None = None,
        include_self: bool = False,
        delay: float = 0.0,
        neighborhood: BaseNeighborhood | None = None,
        layer: str | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            source=source,
            read=state,
            write=state,
            function=self._get_sine,
            reduce=reduce,
            mode=mode,
            weights=weights,
            include_self=include_self,
            delay=delay,
            neighborhood=neighborhood,
            layer=layer,
        )
        self.K = K
        self.alpha = alpha
        self.state = state

    def _get_sine(self, namespace: SimpleNamespace) -> Expression:
        """Return the term `K sin(theta_j - theta_i - alpha)` of one other member."""
        member, other = getattr(namespace.member, self.state), getattr(namespace.other, self.state)
        return namespace.K * sin(other - member - namespace.alpha)

    def _get_group_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over the group.

        The reduction uses the group sums of `sin theta` and `cos theta` when it can.
        """
        if self.weights is not None or self.delay != 0:
            return super()._get_group_reduction(namespace)
        source, target = namespace.source, namespace.target
        shifted = getattr(target, self.state) + namespace.alpha
        phases = getattr(source, self.state)
        sines, cosines = total(sin(phases), over=source), total(cos(phases), over=source)
        reduction = namespace.K * (sines * cos(shifted) - cosines * sin(shifted))
        if self._excludes_self():
            reduction = reduction + namespace.K * sin(namespace.alpha)
        if self.reduce == 'mean':
            reduction = reduction / (count(over=source) - (1 if self._excludes_self() else 0))
        return reduction
