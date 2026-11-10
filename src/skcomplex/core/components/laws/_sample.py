"""Define the law of a Monte Carlo sweep that samples states by moves."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ...specs import MoveSpec, ObjectiveSpec, RequirementSpec, SamplerSpec
from ..spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ._base import BaseLaw


class SampleLaw(BaseLaw):
    """A law of a Monte Carlo sweep that samples states by proposed moves.

    In each unit of time the law makes one sweep. It proposes moves and accepts them by the change of `objective`, so
    the states follow the distribution of a Markov chain Monte Carlo method.

    Unlike `UpdateLaw`, which sets the next value by a rule, it proposes values and accepts or rejects them. Choose it
    when the states follow a Boltzmann distribution of an energy.

    It appears in many fields. Spins of a magnet in the Ising model, cells in the cellular Potts model, and residents in
    a model of segregation are sampled this way.

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
