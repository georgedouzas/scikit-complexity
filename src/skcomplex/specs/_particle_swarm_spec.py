"""Define the settings of particle swarm optimisation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ParticleSwarmSpec(BaseSpec):
    """Settings of particle swarm optimisation.

    Args:
        inertia:
            The weight of the previous velocity.

        cognitive:
            The attraction to the best position of a particle.

        social:
            The attraction to the best position of the swarm.

    Examples:
        >>> from skcomplex.specs import ParticleSwarmSpec
        >>> ParticleSwarmSpec().inertia
        0.9
    """

    inertia: float = 0.9
    cognitive: float = 2.0
    social: float = 2.0
