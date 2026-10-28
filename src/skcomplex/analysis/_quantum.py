"""Analyse a quantum system: its spectrum, its ground state, and the purity and entanglement of its trajectories."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp

from ..core.system import System
from ..engine.compile import QuantumCompiler, compile_system
from ..engine.compile.passes._quantum_compiler import DENSE_LIMIT
from ..engine.runtime import Trajectory
from ..errors import AnalysisError, ParameterError
from ..specs import BaseProblemSpec, QuantumProblemSpec, RunStateSpec
from ..types import Array, DataKind, Sector
from ._base import BaseAnalysis, Source

KINDS = ('spectrum', 'ground-state', 'purity', 'entanglement')
SETTINGS = {
    'spectrum': {'n_states', 'parameters', 'sector'},
    'ground-state': {'parameters'},
    'purity': set(),
    'entanglement': {'subsystems'},
}
"""Settings every kind reads, besides `kind`."""
DIAGONAL_TOLERANCE = 1e-12
SECTOR_TOLERANCE = 1e-9


class QuantumAnalysis(BaseAnalysis):
    """Spectrum and ground state of a quantum system, or the purity and entanglement of the states of its trajectory.

    The kinds read a system or a trajectory:

    - `'spectrum'`: the lowest eigenvalues and eigenstates of the Hamiltonian, by exact diagonalisation. QuTiP
      diagonalises small spaces densely and larger ones with a sparse eigensolver. A sector restricts the basis to the
      states with a given eigenvalue of a conserved operator that is diagonal in the product basis, such as the total
      magnetisation.
    - `'ground-state'`: the lowest eigenvalue and eigenstate of the Hamiltonian.
    - `'purity'`: the purity `Tr(rho^2)` of the state of a trajectory at every frame, 1 for pure states.
    - `'entanglement'`: the von Neumann entropy of the reduced state of some subsystems at every frame of a
      trajectory, in nats.

    Args:
        kind:
            The quantity to compute, one of the kinds above.

        n_states:
            The number of eigenvalues of the spectrum.

        parameters:
            Parameter values by dotted name that replace the initial ones.

        sector:
            A function of the system namespace that returns the conserved operator, and its eigenvalue, or `None`.

        subsystems:
            The keys of the subsystems to keep, such as `chain[0]`, or the path of a whole element or group.

    Attributes:
        problem_:
            The compiled quantum problem.

        trajectory_:
            The trajectory of `'purity'` and `'entanglement'`.

        energies_:
            The eigenvalues of the spectrum in increasing order.

        states_:
            The eigenstates of the spectrum as columns over the whole space.

        energy_:
            The ground state energy.

        state_:
            The ground state vector.

        purity_:
            The purity at every frame.

        entropy_:
            The entanglement entropy at every frame.

    Raises:
        AnalysisError:
            The system or the trajectory is not quantum, the sector operator is not diagonal or has no such eigenvalue,
            or a subsystem is not part of the trajectory.

        ParameterError:
            A parameter name is not part of the system.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from sage.all__sagemath_symbolics import CC, SR
        >>> from skcomplex.analysis import QuantumAnalysis
        >>> from skcomplex.core.elements import Element, GroupElement
        >>> from skcomplex.core.laws import GeneratorLaw
        >>> from skcomplex.core.laws.utils import local_operator
        >>> from skcomplex.specs import HilbertSpec, StateSpec
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.compile import QuantumCompiler
        >>> from skcomplex.engine.runtime import Simulation
        >>> operators = {'z': (np.diag([1, -1]), True), 'minus': (np.array([[0, 0], [1, 0]]), False)}
        >>> QUBIT = HilbertSpec(2, operators, basis={'up': 0, 'down': 1})
        >>> class QubitElement(Element):
        ...     def __init__(
        ...         self, label, psi: Annotated[object, StateSpec(domain=CC**2)] = 'up', hilbert=QUBIT
        ...     ):
        ...         super().__init__(label)
        ...         self.psi = psi
        ...         self.hilbert = hilbert
        ...
        ...     def _get_laws(self, s):
        ...         return [GeneratorLaw(local_operator(s.psi, SR.symbol('z')))]
        >>> system = System(elements=[QubitElement('q')])
        >>> QuantumAnalysis(n_states=2).check().prepare(system).run().energies_.round(6).tolist()
        [-1.0, 1.0]
        >>> singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
        >>> pair = System(elements=[GroupElement('pair', element=QubitElement, size=2)])
        >>> trajectory = Simulation(
        ...     pair, compiler=QuantumCompiler(), dt=1.0, initial_state={'system.state': singlet},
        ... ).check().run(until=1.0).trajectory_
        >>> analysis = QuantumAnalysis('entanglement', subsystems=['pair[0]']).check().prepare(trajectory).run()
        >>> bool(np.allclose(analysis.entropy_, np.log(2)))
        True
    """

    def __init__(
        self,
        kind: str = 'spectrum',
        *,
        n_states: int = 6,
        parameters: Mapping[str, Any] | None = None,
        sector: Sector | None = None,
        subsystems: Sequence[str] | None = None,
    ) -> None:
        self.kind = kind
        self.n_states = n_states
        self.parameters = parameters
        self.sector = sector
        self.subsystems = subsystems

    def _check(self) -> Self:
        """Validate the kind, that the kind reads every changed setting, and the subsystems of the entanglement."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind])
        if self.kind == 'entanglement' and not self.subsystems:
            error_msg = f'{type(self).__name__} of kind entanglement needs the subsystems to keep. Give subsystems.'
            raise AnalysisError(error_msg)
        if self.n_states < 1:
            error_msg = f'{type(self).__name__} has n_states={self.n_states!r}. Use 1 or more.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the system kind."""
        return DataKind.SYSTEM

    def _prepare(self, source: Source) -> Self:
        """Compile the quantum system, or read the quantum problem of a trajectory."""
        if self.kind in {'purity', 'entanglement'}:
            if not isinstance(source, Trajectory):
                error_msg = (
                    f'{type(self).__name__} of kind {self.kind!r} reads a trajectory, not {type(source).__name__}.'
                )
                raise AnalysisError(error_msg)
            self.problem_, _ = self._states_of(source)
            self.trajectory_ = source
            return self
        self.problem_ = self._quantum_problem(self._read_system(source))
        return self

    def _run(self, inputs: object = None) -> Self:
        """Diagonalise the Hamiltonian, or measure the states of the trajectory."""
        if self.kind == 'purity':
            problem, states = self._states_of(self.trajectory_)
            if problem.pure:
                self.purity_ = np.sum(np.abs(states) ** 2, axis=1) ** 2
            else:
                self.purity_ = np.sum(np.abs(states) ** 2, axis=(1, 2))
            return self
        if self.kind == 'entanglement':
            problem, states = self._states_of(self.trajectory_)
            self.entropy_ = self._entanglement(problem, states, self.subsystems)  # type: ignore[arg-type]
            return self
        p = self._parameters_of(self.problem_, self.parameters)
        if self.kind == 'ground-state':
            energies, states = self._lowest_states(self.problem_, p, 1, None)
            self.energy_ = float(energies[0])
            self.state_ = states[:, 0]
            return self
        basis = None if self.sector is None else self._sector_basis(self.problem_, self.sector, p)
        self.energies_, self.states_ = self._lowest_states(self.problem_, p, self.n_states, basis)
        return self

    @staticmethod
    def _quantum_problem(system: System | BaseProblemSpec) -> QuantumProblemSpec:
        """Return the quantum problem of a system, compiling it when needed."""
        problem = system if isinstance(system, BaseProblemSpec) else compile_system(system, compiler=QuantumCompiler())
        if not isinstance(problem, QuantumProblemSpec):
            error_msg = (
                f'The system compiles to the {problem.form.value} form, but the analysis needs quantum elements.'
            )
            raise AnalysisError(error_msg)
        return problem

    @staticmethod
    def _parameters_of(problem: QuantumProblemSpec, values: Mapping[str, Any] | None) -> Array:
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

    @staticmethod
    def _sector_basis(
        problem: QuantumProblemSpec,
        sector: Sector,
        p: Array,
    ) -> npt.NDArray[np.intp]:
        """Return the basis states whose eigenvalue of a conserved diagonal operator equals a value."""
        expression, value = sector
        terms = problem.lowering.run((expression(problem.namespaces['']), 'the sector'))
        matrix = sum(
            (term.run(RunStateSpec(0.0, np.zeros(0), p)) * term.matrix for term in terms), start=0 * terms[0].matrix,
        )
        diagonal = matrix.diagonal()
        if abs(matrix).sum() - np.abs(diagonal).sum() > DIAGONAL_TOLERANCE:
            error_msg = (
                'The operator of a sector must be diagonal in the product basis, such as a sum of `z` operators.'
            )
            raise AnalysisError(error_msg)
        basis = np.flatnonzero(np.abs(diagonal - value) < SECTOR_TOLERANCE)
        if not basis.size:
            error_msg = f'No basis state has the eigenvalue {value} of the sector operator.'
            raise AnalysisError(error_msg)
        return basis

    @staticmethod
    def _states_of(trajectory: Trajectory) -> tuple[QuantumProblemSpec, npt.NDArray[np.complex128]]:
        """Return the quantum problem of a trajectory and its states, one per frame."""
        problem = trajectory.problem
        if not isinstance(problem, QuantumProblemSpec):
            error_msg = 'The trajectory is not the trajectory of a quantum system.'
            raise AnalysisError(error_msg)
        return problem, np.asarray(trajectory.values(problem.states.names[0]))

    @staticmethod
    def _lowest_states(
        problem: QuantumProblemSpec,
        p: Array,
        k: int,
        basis: npt.NDArray[np.intp] | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
        """Return the lowest eigenvalues and eigenstates of the Hamiltonian at time 0 through QuTiP, within a basis."""
        dimension = math.prod(problem.dimensions)
        matrix = problem._hamiltonian_at(0.0, p)
        indices = np.arange(dimension) if basis is None else np.asarray(basis, dtype=np.intp)
        matrix = sp.csr_array(matrix[indices][:, indices])
        count = min(k, indices.size)
        sparse = indices.size > DENSE_LIMIT and count < indices.size - 1
        energies, kets = qutip.Qobj(matrix).eigenstates(sparse=sparse, eigvals=count)
        states = np.zeros((dimension, count), dtype=complex)
        for column, ket in enumerate(kets):
            states[indices, column] = ket.full().ravel()
        return np.real(np.asarray(energies)), states

    @staticmethod
    def _entanglement(
        problem: QuantumProblemSpec, states: npt.NDArray[np.complex128], subsystems: Sequence[str],
    ) -> Array:
        """Return the von Neumann entropy of the reduced state of some subsystems at every frame, in nats."""
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
