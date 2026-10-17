"""Collection of physics systems."""

from ._classical_mechanics import (
    ClassicalMechanicsSystem,
    ExternalForce,
    ExternalPotential,
    InternalForce,
    InternalPotential,
    PointParticle,
)

__all__: list[str] = [
    'ClassicalMechanicsSystem',
    'ExternalForce',
    'ExternalPotential',
    'InternalForce',
    'InternalPotential',
    'PointParticle',
]
