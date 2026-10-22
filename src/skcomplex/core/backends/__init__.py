"""Backends that run compiled problems: NumPy, Numba, Mesa for agents, and LAMMPS for particles.

Importing the package registers every backend and the solvers that run in them.
"""

from ._base import ExternalBackend
from ._lammps import NATIVE_FIXES, NATIVE_PAIRS, LammpsBackend, LammpsFix, register_lammps_fix, register_lammps_pair
from ._mesa import AgentView, MesaStepper, ModelView
from ._numba import NumbaBackend
from ._numpy import NumpyBackend

__all__: list[str] = [
    'NATIVE_FIXES',
    'NATIVE_PAIRS',
    'AgentView',
    'ExternalBackend',
    'LammpsBackend',
    'LammpsFix',
    'MesaStepper',
    'ModelView',
    'NumbaBackend',
    'NumpyBackend',
    'register_lammps_fix',
    'register_lammps_pair',
]
