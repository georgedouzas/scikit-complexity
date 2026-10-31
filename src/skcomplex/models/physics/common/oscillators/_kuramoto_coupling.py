"""Define Kuramoto coupling, the sine coupling of phases with a phase lag."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import Expression, cos, sin

from .....core.interactions import CouplingInteraction
from .....core.laws.utils import count, total
from .....core.spaces import BaseNeighborhood
from .....core.types import Target
from .....specs import ParameterSpec, RoleSpec


class KuramotoCouplingInteraction(CouplingInteraction):
    """Kuramoto coupling `theta_i' += K sum_j sin(theta_j - theta_i - alpha)` of the phases of a group.

    The sum, or the mean, runs over the members `j` that the mode selects: the neighbours along the edges of a lattice
    or network, or within a neighbourhood such as `RadiusNeighborhood(r)`, every other member of the group, or the next
    member. With `include_self` it also runs over `j = i`, whose term `-K sin(alpha)` slows every phase. Over the whole
    group without weights or delay, the coupling reads the group sums of `sin theta` and `cos theta` once, so it costs
    time linear in the size of the group.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of oscillators.

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
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.physics.common.oscillators import (
        ...     KuramotoCouplingInteraction,
        ...     PhaseOscillatorElement,
        ... )
        >>> phases = GroupElement('phases', element=PhaseOscillatorElement, size=4, values={'theta': [0, 1, 2, 3]})
        >>> coupling = KuramotoCouplingInteraction('coupling', target='phases', K=2.0, mode='group', reduce='mean')
        >>> system = System(elements=[phases], interactions=[coupling])
        >>> theta = Simulation(system, dt=1.0).check().prepare().run(until=20.0).trajectory_.values('phases.theta')
        >>> round(float(np.ptp(theta[-1])), 6)
        0.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
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
            read=state,
            write=state,
            function=self._sine,
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

    def _sine(self, namespace: SimpleNamespace) -> Expression:
        """Return the term `K sin(theta_j - theta_i - alpha)` of one other member."""
        member, other = getattr(namespace.member, self.state), getattr(namespace.other, self.state)
        return namespace.K * sin(other - member - namespace.alpha)

    def _get_group_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over the group, from the group sums of `sin theta` and `cos theta` when it can."""
        if self.weights is not None or self.delay != 0:
            return super()._get_group_reduction(namespace)
        source, target = namespace.source, namespace.target
        shifted = getattr(target, self.state) + namespace.alpha
        phases = getattr(source, self.state)
        sines, cosines = total(sin(phases), over=source), total(cos(phases), over=source)
        reduction = namespace.K * (sines * cos(shifted) - cosines * sin(shifted))
        if not self.include_self:
            reduction = reduction + namespace.K * sin(namespace.alpha)
        if self.reduce == 'mean':
            reduction = reduction / (count(over=source) - (0 if self.include_self else 1))
        return reduction
