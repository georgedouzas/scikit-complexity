"""Particles: point particles, potentials, pair potentials, and the LAMMPS backend that runs them."""

from .._formalism import Formalism, register
from ._backends import NATIVE_FIXES, NATIVE_PAIRS, LammpsFix, LammpsSolver, register_lammps_fix, register_lammps_pair
from ._components import ParticleElement
from ._interactions import PairPotentialInteraction, PotentialInteraction

__all__: list[str] = [
    'NATIVE_FIXES',
    'NATIVE_PAIRS',
    'LammpsFix',
    'LammpsSolver',
    'PairPotentialInteraction',
    'ParticleElement',
    'PotentialInteraction',
    'register_lammps_fix',
    'register_lammps_pair',
]

register(
    Formalism(
        'particles',
        components=(ParticleElement, PotentialInteraction, PairPotentialInteraction),
        solvers={'lammps': LammpsSolver},
    ),
)
