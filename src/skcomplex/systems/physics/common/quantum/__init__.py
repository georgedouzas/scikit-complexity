"""Quantum realizations shared by the physics packages.

The package holds the Hilbert space of a subsystem, qubits and bosonic modes, the generator and dissipator laws, the
compiler and the solvers of their dynamics, and it measures operator systems: their spectra, ground states, purity,
and entanglement.
"""

from ._compiler import QuantumCompiler
from ._dissipator import DissipatorLaw
from ._generator import GeneratorLaw
from ._hilbert_spec import HilbertSpec
from ._jump_trajectories import JumpTrajectoriesSolver
from ._lindblad import LindbladSolver
from ._measures import OperatorMeasuresResult, operator_measures
from ._mode import ModeElement, mode_space
from ._operator import OperatorSolver
from ._qubit import QUBIT, QubitElement
from ._sector_spec import SectorSpec
from ._unitary import UnitarySolver

__all__: list[str] = [
    'QUBIT',
    'DissipatorLaw',
    'GeneratorLaw',
    'HilbertSpec',
    'JumpTrajectoriesSolver',
    'LindbladSolver',
    'ModeElement',
    'OperatorMeasuresResult',
    'OperatorSolver',
    'QuantumCompiler',
    'QubitElement',
    'SectorSpec',
    'UnitarySolver',
    'mode_space',
    'operator_measures',
]
