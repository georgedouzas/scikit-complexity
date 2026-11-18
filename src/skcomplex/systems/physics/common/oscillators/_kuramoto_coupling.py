"""Define Kuramoto coupling, the sine coupling of phases with a phase lag."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import SR, Expression, cos, sin

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.components.laws.utils import count, neighbors, total
from .....core.components.spaces import BaseNeighborhood, BaseSpace
from .....core.errors import ValidationError
from .....core.specs import Role

REDUCTIONS = ('sum', 'mean')


class KuramotoCouplingInteraction(Interaction):
    """Kuramoto coupling `theta_i' += K sum_j sin(theta_j - theta_i - alpha)` of the phases of a group.

    The sum, or the mean, runs over every other member of the group, or over the members within a neighbourhood such
    as `RadiusNeighborhood(r)`. With `include_self` it also runs over `j = i`, whose term `-K sin(alpha)` slows every
    phase. Over the whole group, the coupling reads the group sums of `sin theta` and `cos theta` once, so it costs time
    linear in the size of the group, as for fireflies that all see each other or the generators of one grid.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of oscillators.

        K:
            The coupling strength of one pair.

        alpha:
            The phase lag.

        reduce:
            `'sum'` or `'mean'`.

        include_self:
            Whether the sum also runs over the member itself.

        neighborhood:
            The neighbourhood of every member, or `None` for the whole group.

    Raises:
        ValidationError:
            The reduction is neither `'sum'` nor `'mean'`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.oscillators import (
        ...     KuramotoCouplingInteraction,
        ...     PhaseOscillatorElement,
        ... )
        >>> phases = PhaseOscillatorElement('phases', theta=[0.0, 1.0, 2.0, 3.0], size=4)
        >>> coupling = KuramotoCouplingInteraction('coupling', target='phases', K=2.0, reduce='mean')
        >>> system = System(elements=[phases], interactions=[coupling]).prepare()
        >>> theta = system.simulate(until=20.0, dt=1.0).get_values('phases.theta')
        >>> round(float(np.ptp(theta[-1])), 6)
        0.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        K: float = 1.0,
        alpha: float = 0.0,
        reduce: str = 'sum',
        include_self: bool = False,
        neighborhood: BaseNeighborhood | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.K = K
        self.alpha = alpha
        self.reduce = reduce
        self.include_self = include_self
        self.neighborhood = neighborhood

    def _get_group_sum(self, context: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the sum of the sines over the whole group and the number of terms, from the group sums."""
        theta = context.target.theta
        shifted = theta + context.alpha
        sines, cosines = total(sin(theta)), total(cos(theta))
        summed = sines * cos(shifted) - cosines * sin(shifted)
        if self.include_self:
            return summed, count()
        return summed + sin(context.alpha), count() - 1

    def _get_neighbour_sum(self, context: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the sum of the sines over the neighbourhood and the number of terms."""
        theta, other = context.target.theta, context.target.other.theta
        summed = neighbors(sin(other - theta - context.alpha), over=self.neighborhood)
        terms = neighbors(SR(1), over=self.neighborhood)
        if self.include_self:
            return summed - sin(context.alpha), terms + 1
        return summed, terms

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the phases from the lagged sines of the other phases.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.

        Raises:
            ValidationError:
                The reduction is neither `'sum'` nor `'mean'`.
        """
        super()._prepare(context)
        if self.reduce not in REDUCTIONS:
            error_msg = f'Unknown reduction {self.reduce!r}. Use one of {", ".join(REDUCTIONS)}.'
            raise ValidationError(error_msg)
        if isinstance(context, SimpleNamespace):
            get_sum = self._get_group_sum if self.neighborhood is None else self._get_neighbour_sum
            summed, terms = get_sum(context)
            rate = context.K * summed if self.reduce == 'sum' else context.K * summed / terms
            self.laws_ = [RateLaw(context.target.theta.diff(context.t), rate)]
