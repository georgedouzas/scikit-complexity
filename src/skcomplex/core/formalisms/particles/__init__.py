"""Particles: point particles, potentials, pair potentials, and the LAMMPS backend that runs them."""

from .._formalism import Formalism, register
from ._backends import NATIVE_FIXES, NATIVE_PAIRS, LammpsBackend, LammpsFix, register_lammps_fix, register_lammps_pair
from ._components import Particle
from ._interactions import PairPotential, Potential

__all__: list[str] = [
    'NATIVE_FIXES',
    'NATIVE_PAIRS',
    'LammpsBackend',
    'LammpsFix',
    'PairPotential',
    'Particle',
    'Potential',
    'register_lammps_fix',
    'register_lammps_pair',
]

register(
    Formalism(
        'particles',
        components=(Particle, Potential, PairPotential),
        solvers={'lammps': LammpsBackend},
    ),
)
