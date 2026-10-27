"""Define sample laws, Monte Carlo sweeps that accept moves from the change of a local energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

from ..spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ..specs import EnergySpec, MoveSpec, RequirementSpec, SamplerSpec
from ._base import BaseLaw


class SampleLaw(BaseLaw):
    """Monte Carlo sweep that samples states by moves accepted from the change of an energy, one per unit of time.

    Args:
        energy:
            The local energy of a site and its neighbourhood.

        move:
            The move proposed at a site.

        sampler:
            The acceptance of moves, with its temperature and schedule, or `None` for Metropolis acceptance at the
            parameter `temperature` of the element.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import SampleLaw
        >>> from skcomplex.core.spaces import VonNeumannNeighborhood
        >>> from skcomplex.core.specs import EnergySpec, MoveSpec, SamplerSpec
        >>> spin = var('spin')
        >>> law = SampleLaw(EnergySpec(-spin, VonNeumannNeighborhood()), MoveSpec(spin), SamplerSpec('glauber', 2.0))
        >>> law.check().prepare().targets_
        (spin,)
    """

    def __init__(
        self,
        energy: EnergySpec,
        move: MoveSpec,
        sampler: SamplerSpec | None = None,
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.energy = energy
        self.move = move
        self.sampler = sampler

    def _get_targets(self) -> tuple[object, ...]:
        """Return the lattice state the moves change."""
        return (self.move.target,)

    def _get_requirements(self) -> RequirementSpec:
        """Return what the sweep requires of a system: discrete time on a lattice or a network."""
        return RequirementSpec(time=(DiscreteTime,), space=(LatticeSpace, NetworkSpace))

    def _check(self) -> Self:
        """Check the energy, the move, and the sampler.

        Returns:
            law:
                The law itself.
        """
        for spec in (self.energy, self.move, self.sampler):
            if spec is not None:
                spec.check()
        return self
