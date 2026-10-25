"""Analyse quantum systems: spectra, ground states, purity, and entanglement entropy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp

from ...core.base import AnalysisError, ParameterError, Vector
from ...core.laws import Symbols
from ...core.problems import BaseProblem
from ...core.system import System
from ...engine.compile import compile_system
from ...engine.runtime import Trajectory
from ._passes import DENSE_LIMIT, QuantumProblem, hamiltonian_at

DIAGONAL_TOLERANCE = 1e-12
SECTOR_TOLERANCE = 1e-9

type Sector = tuple[Callable[[Symbols], Any], float]


@dataclass(frozen=True)
class Spectrum:
    """Lowest eigenvalues and eigenstates of the Hamiltonian of a quantum system.

    Args:
        energies:
            The eigenvalues in increasing order.

        states:
            The eigenstates as columns over the whole space.
    """

    energies: npt.NDArray[np.float64]
    states: npt.NDArray[np.complex128]


def _quantum_problem(system: System | BaseProblem) -> QuantumProblem:
    """Return the quantum problem of a system, compiling it when needed."""
    problem = system if isinstance(system, BaseProblem) else compile_system(system)
    if not isinstance(problem, QuantumProblem):
        error_msg = f'The system compiles to the {problem.form.value} form, but the analysis needs quantum elements.'
        raise AnalysisError(error_msg)
    return problem


def _parameters_of(problem: QuantumProblem, values: Mapping[str, Any] | None) -> Vector:
    """Return the initial parameter vector with some values replaced by dotted name."""
    p = problem.initial_parameters.copy()
    for name, value in (values or {}).items():
        try:
            entry, indices = problem.parameters.locate(name)
        except KeyError as error:
            error_msg = f'No parameter named {name!r}. Available names are: {", ".join(problem.parameters.names)}.'
            raise ParameterError(error_msg) from error
        p[indices] = entry.encode(value)
    return p


def _sector_basis(
    problem: QuantumProblem,
    sector: Sector,
    p: Vector,
) -> npt.NDArray[np.intp]:
    """Return the basis states whose eigenvalue of a conserved diagonal operator equals a value."""
    expression, value = sector
    terms = problem.lowering.terms(expression(problem.namespaces['']), 'the sector')
    matrix = sum((term.value(0.0, p) * term.matrix for term in terms), start=0 * terms[0].matrix)
    diagonal = matrix.diagonal()
    if abs(matrix).sum() - np.abs(diagonal).sum() > DIAGONAL_TOLERANCE:
        error_msg = 'The operator of a sector must be diagonal in the product basis, such as a sum of `z` operators.'
        raise AnalysisError(error_msg)
    basis = np.flatnonzero(np.abs(diagonal - value) < SECTOR_TOLERANCE)
    if not basis.size:
        error_msg = f'No basis state has the eigenvalue {value} of the sector operator.'
        raise AnalysisError(error_msg)
    return basis


def _states_of(trajectory: Trajectory) -> tuple[QuantumProblem, npt.NDArray[np.complex128]]:
    """Return the quantum problem of a trajectory and its states, one per frame."""
    problem = trajectory.problem
    if not isinstance(problem, QuantumProblem):
        error_msg = 'The trajectory is not the trajectory of a quantum system.'
        raise AnalysisError(error_msg)
    return problem, np.asarray(trajectory.values(problem.states.names[0]))


def _lowest_states(
    problem: QuantumProblem,
    p: Vector,
    k: int,
    basis: npt.NDArray[np.intp] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Return the lowest eigenvalues and eigenstates of the Hamiltonian at time 0 through QuTiP, within a basis."""
    dimension = math.prod(problem.dimensions)
    matrix = hamiltonian_at(problem, 0.0, p)
    indices = np.arange(dimension) if basis is None else np.asarray(basis, dtype=np.intp)
    matrix = sp.csr_array(matrix[indices][:, indices])
    count = min(k, indices.size)
    sparse = indices.size > DENSE_LIMIT and count < indices.size - 1
    energies, kets = qutip.Qobj(matrix).eigenstates(sparse=sparse, eigvals=count)
    states = np.zeros((dimension, count), dtype=complex)
    for column, ket in enumerate(kets):
        states[indices, column] = ket.full().ravel()
    return np.real(np.asarray(energies)), states


def spectrum(
    system: System | BaseProblem,
    k: int = 6,
    *,
    parameters: Mapping[str, Any] | None = None,
    sector: Sector | None = None,
) -> Spectrum:
    """Return the lowest eigenvalues and eigenstates of the Hamiltonian of a quantum system, by exact diagonalisation.

    QuTiP diagonalises small spaces densely and larger ones with a sparse eigensolver. A sector restricts the basis to
    the states with a given eigenvalue of a conserved operator that is diagonal in the product basis, such as the total
    magnetisation.

    Args:
        system:
            The quantum system, or its compiled problem.

        k:
            The number of eigenvalues.

        parameters:
            Parameter values by dotted name that replace the initial ones.

        sector:
            A function of the system namespace that returns the conserved operator, and its eigenvalue, or `None`.

    Returns:
        spectrum:
            The eigenvalues and eigenstates.

    Raises:
        AnalysisError:
            The system is not quantum, or the sector operator is not diagonal or has no such eigenvalue.

        ParameterError:
            A parameter name is not part of the system.

    Examples:
        >>> from skcomplex.formalisms.quantum import spectrum
        >>> from skcomplex.models import create
        >>> spectrum(create('quantum_heisenberg', n_sites=4), k=2).energies.round(6).tolist()
        [-2.0, -1.0]
    """
    problem = _quantum_problem(system)
    p = _parameters_of(problem, parameters)
    basis = None if sector is None else _sector_basis(problem, sector, p)
    energies, states = _lowest_states(problem, p, k, basis)
    return Spectrum(energies, states)


def ground_state(
    system: System | BaseProblem,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> tuple[float, npt.NDArray[np.complex128]]:
    """Return the lowest eigenvalue and eigenstate of the Hamiltonian of a quantum system.

    Args:
        system:
            The quantum system, or its compiled problem.

        parameters:
            Parameter values by dotted name that replace the initial ones.

    Returns:
        ground:
            The ground state energy and the ground state vector.

    Examples:
        >>> from skcomplex.formalisms.quantum import ground_state
        >>> from skcomplex.models import create
        >>> energy, state = ground_state(create('quantum_heisenberg', n_sites=4))
        >>> round(energy, 6), state.shape
        (-2.0, (16,))
    """
    found = spectrum(system, 1, parameters=parameters)
    return float(found.energies[0]), found.states[:, 0]


def purity(trajectory: Trajectory) -> npt.NDArray[np.float64]:
    """Return the purity `Tr(rho^2)` of the state of a quantum trajectory at every frame.

    Args:
        trajectory:
            The trajectory of a quantum system.

    Returns:
        purity:
            One value per frame, 1 for pure states.

    Raises:
        AnalysisError:
            The trajectory is not quantum.

    Examples:
        >>> from skcomplex.formalisms.quantum import purity
        >>> from skcomplex.models import run
        >>> trajectory = run('quantum_heisenberg', n_sites=4, gamma=0.5, settings={'until': 1.0}).trajectory_
        >>> bool(purity(trajectory)[-1] < 1)
        True
    """
    problem, states = _states_of(trajectory)
    if problem.pure:
        return np.sum(np.abs(states) ** 2, axis=1) ** 2
    return np.sum(np.abs(states) ** 2, axis=(1, 2))


def entanglement_entropy(trajectory: Trajectory, subsystems: Sequence[str]) -> npt.NDArray[np.float64]:
    """Return the von Neumann entropy of the reduced state of some subsystems at every frame, in nats.

    Args:
        trajectory:
            The trajectory of a quantum system.

        subsystems:
            The keys of the subsystems to keep, such as `chain[0]`, or the path of a whole element or group.

    Returns:
        entropy:
            One value per frame.

    Raises:
        AnalysisError:
            The trajectory is not quantum, or a subsystem is not part of it.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.formalisms.quantum import entanglement_entropy
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.physics.realizations.quantum import SpinComponent
        >>> singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
        >>> system = System(components=[SpinComponent.group('pair', 2)])
        >>> trajectory = Simulation(system, dt=1.0, initial_state={'system.state': singlet}).run(until=1.0).trajectory_
        >>> bool(np.allclose(entanglement_entropy(trajectory, ['pair[0]']), np.log(2)))
        True
    """
    problem, states = _states_of(trajectory)
    keep = []
    for name in subsystems:
        found = [index for index, key in enumerate(problem.subsystems) if key == name or key.startswith(f'{name}[')]
        if not found:
            error_msg = (
                f'{name} is not a subsystem of the trajectory. The subsystems are {", ".join(problem.subsystems)}.'
            )
            raise AnalysisError(error_msg)
        keep += found
    dimensions = list(problem.dimensions)
    values = []
    for state in states:
        if problem.pure:
            quantum = qutip.Qobj(state.reshape(-1, 1), dims=[dimensions, [1] * len(dimensions)])
        else:
            quantum = qutip.Qobj(state, dims=[dimensions, dimensions])
        values.append(qutip.entropy_vn(quantum.ptrace(sorted(set(keep)))))
    return np.asarray(values, dtype=float)


def hamiltonian_of(system: System | BaseProblem) -> Any:  # noqa: ANN401
    """Return the Hamiltonian of a quantum system as one operator, printed in operator notation.

    Args:
        system:
            The quantum system, or its compiled problem.

    Returns:
        hamiltonian:
            The sum of the Hamiltonian laws of the system.

    Raises:
        AnalysisError:
            The system is not quantum.

    Examples:
        >>> from skcomplex.formalisms.quantum import hamiltonian_of
        >>> from skcomplex.formalisms.quantum import GeneratorLaw
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.physics.realizations.quantum import QubitComponent
        >>> DrivenComponent = QubitComponent.define(
        ...     'DrivenComponent', parameters={'omega': 1.0}, laws=lambda s: [GeneratorLaw(s.omega / 2 * s.z)]
        ... )
        >>> hamiltonian_of(System(components=[DrivenComponent('q')])).text
        '1/2*omega_q*q.z'
    """
    return _quantum_problem(system).hamiltonian_operator
