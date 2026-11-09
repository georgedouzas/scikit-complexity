"""Define sample laws.

They are Monte Carlo sweeps that accept moves from the change of a local objective.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ...specs import MoveSpec, ObjectiveSpec, RequirementSpec, SamplerSpec
from ..spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ._base import BaseLaw


class SampleLaw(BaseLaw):
    """Monte Carlo sweep that samples states by moves.

    The moves are accepted from the change of an objective, one sweep per unit of time.

    Args:
        objective:
            The local objective of a site and its neighbourhood.

        move:
            The move proposed at a site.

        sampler:
            The acceptance of moves, with its intensity and schedule, or `None` for Metropolis acceptance at the
            parameter `intensity` of the element.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import SampleLaw
        >>> from skcomplex.core.components.spaces import BallNeighborhood
        >>> from skcomplex.core.specs import ObjectiveSpec, MoveSpec, SamplerSpec
        >>> spin = var('spin')
        >>> cross = BallNeighborhood(metric='manhattan')
        >>> law = SampleLaw(ObjectiveSpec(-spin, cross), MoveSpec(spin), SamplerSpec('glauber', 2.0))
        >>> law.prepare().targets_
        (spin,)
    """

    law_tag: ClassVar[str] = 'sampling'

    def __init__(
        self,
        objective: ObjectiveSpec,
        move: MoveSpec,
        sampler: SamplerSpec | None = None,
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.objective = objective
        self.move = move
        self.sampler = sampler

    def _get_targets(self) -> tuple[object, ...]:
        """Return the lattice state the moves change."""
        return (self.move.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the sweep requires: discrete time on a lattice or a network."""
        return RequirementSpec(time=(DiscreteTime,), space=(LatticeSpace, NetworkSpace))

    def _prepare(self, context: str | None = None) -> None:
        """Check the objective, the move, and the sampler.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        for spec in (self.objective, self.move, self.sampler):
            if spec is not None:
                spec._check()
