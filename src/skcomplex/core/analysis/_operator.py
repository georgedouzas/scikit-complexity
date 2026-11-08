"""Analyse an operator system.

The analyses are its spectrum, its ground state, and the purity and entanglement of its trajectories.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp

from ..components.system import System
from ..engine.compile.passes._operator_compiler import DENSE_LIMIT, OperatorCompiler
from ..engine.problems import BaseProblemSpec
from ..engine.problems._operator_problem_spec import OperatorProblemSpec
from ..engine.runtime import Trajectory
from ..errors import AnalysisError, ParameterError
from ..specs import AnalysisTags, RunStateSpec, SectorSpec
from ..types import Array, DataKind
from ._base import _check_choice, _check_settings, _composite, _read_system
from .types import Source

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

TAGS = AnalysisTags(DataKind.SYSTEM)


@dataclass(frozen=True)
class OperatorMeasuresResult:
    """Spectrum or ground state of an operator system, or purity or entanglement.

    The purity or the entanglement is that of the states of its trajectory.

    Args:
        problem:
            The compiled operator problem.

        trajectory:
            The trajectory of `'purity'` and `'entanglement'`, or `None` for the other kinds.

        energies:
            The eigenvalues of the spectrum in increasing order, set for `'spectrum'`.

        states:
            The eigenstates of the spectrum as columns over the whole space, set for `'spectrum'`.

        energy:
            The ground state energy, set for `'ground-state'`.

        state:
            The ground state vector, set for `'ground-state'`.

        purity:
            The purity at every frame, set for `'purity'`.

        entropy:
            The entanglement entropy at every frame, set for `'entanglement'`.
    """

    problem: OperatorProblemSpec
    trajectory: Trajectory | None = None
    energies: Array | None = None
    states: npt.NDArray[np.complex128] | None = None
    energy: float | None = None
    state: npt.NDArray[np.complex128] | None = None
    purity: Array | None = None
    entropy: Array | None = None


def operator_measures(
    source: Source,
    *,
    kind: str = 'spectrum',
    n_states: int = 6,
    parameters: Mapping[str, object] | None = None,
    sector: SectorSpec | None = None,
    subsystems: Sequence[str] | None = None,
) -> OperatorMeasuresResult:
    """Return a spectrum, ground state, purity, or entanglement of an operator system.

    The purity and the entanglement are those of its trajectory.

    The kinds read a system or a trajectory:

    - `'spectrum'`: the lowest eigenvalues and eigenstates of the generator, by exact diagonalisation. QuTiP
      diagonalises small spaces densely and larger ones with a sparse eigensolver. A sector restricts the basis to the
      states with a given eigenvalue of a conserved operator that is diagonal in the product basis, such as the total
      magnetisation.
    - `'ground-state'`: the lowest eigenvalue and eigenstate of the generator.
    - `'purity'`: the purity `Tr(rho^2)` of the state of a trajectory at every frame, 1 for pure states.
    - `'entanglement'`: the von Neumann entropy of the reduced state of some subsystems at every frame of a
      trajectory, in nats.

    Args:
        source:
            The operator system or its compiled problem for `'spectrum'` and `'ground-state'`, or a trajectory of it.

        kind:
            The quantity to compute, one of the kinds above.

        n_states:
            The number of eigenvalues of the spectrum.

        parameters:
            Parameter values by dotted name that replace the initial ones.

        sector:
            The sector of a conserved diagonal operator whose basis states the spectrum keeps, or `None` for every
            basis state.

        subsystems:
            The keys of the subsystems to keep, such as `chain[0]`, or the path of a whole element or group.

    Returns:
        result:
            The spectrum, the ground state, the purity, or the entanglement entropy.

    Raises:
        AnalysisError:
            The system or the trajectory holds no operators, the sector operator is not diagonal or lacks the
            eigenvalue, or a subsystem is not part of the trajectory.

        ParameterError:
            A parameter name is not part of the system.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from sage.all__sagemath_symbolics import CC, SR
        >>> from skcomplex.core.analysis import operator_measures
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.core.components.laws import GeneratorLaw
        >>> from skcomplex.core.components.laws.utils import local_operator
        >>> from skcomplex.core.specs import HilbertSpec, StateSpec
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import simulate
        >>> operators = {'z': (np.diag([1, -1]), True), 'minus': (np.array([[0, 0], [1, 0]]), False)}
        >>> QUBIT = HilbertSpec(2, operators, basis={'up': 0, 'down': 1})
        >>> class TwoLevelElement(Element):
        ...     def __init__(
        ...         self, label, psi: Annotated[object, StateSpec(domain=CC**2)] = 'up', hilbert=QUBIT
        ...     ):
        ...         super().__init__(label)
        ...         self.psi = psi
        ...         self.hilbert = hilbert
        ...
        ...     def _get_laws(self, s):
        ...         return [GeneratorLaw(local_operator(s.psi, SR.symbol('z')))]
        >>> system = System(elements=[TwoLevelElement('q')])
        >>> operator_measures(system, n_states=2).energies.round(6).tolist()
        [-1.0, 1.0]
        >>> singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
        >>> pair = System(elements=[GroupElement('pair', element=TwoLevelElement, size=2)])
        >>> trajectory = simulate(pair, until=1.0, dt=1.0, initial_state={'system.state': singlet})
        >>> result = operator_measures(trajectory, kind='entanglement', subsystems=['pair[0]'])
        >>> bool(np.allclose(result.entropy, np.log(2)))
        True
    """
    source = _composite(source)
    _check_choice('operator_measures', 'kind', kind, KINDS)
    _check_settings(
        operator_measures,
        'kind',
        SETTINGS[kind],
        kind=kind,
        n_states=n_states,
        parameters=parameters,
        sector=sector,
        subsystems=subsystems,
    )
    if kind == 'entanglement' and not subsystems:
        error_msg = 'operator_measures of kind entanglement needs the subsystems to keep. Give subsystems.'
        raise AnalysisError(error_msg)
    if n_states < 1:
        error_msg = f'operator_measures has n_states={n_states!r}. Use 1 or more.'
        raise AnalysisError(error_msg)
    if kind in {'purity', 'entanglement'}:
        if not isinstance(source, Trajectory):
            error_msg = f'operator_measures of kind {kind!r} reads a trajectory, not {type(source).__name__}.'
            raise AnalysisError(error_msg)
        problem, states = _states_of(source)
        if kind == 'entanglement':
            return OperatorMeasuresResult(
                problem,
                trajectory=source,
                entropy=_entanglement(problem, states, subsystems or ()),
            )
        purity = np.sum(np.abs(states) ** 2, axis=1) ** 2 if problem.pure else np.sum(np.abs(states) ** 2, axis=(1, 2))
        return OperatorMeasuresResult(problem, trajectory=source, purity=purity)
    problem = _operator_problem(_read_system(source))
    p = _parameters_of(problem, parameters)
    if kind == 'ground-state':
        energies, states = _lowest_states(problem, p, 1, None)
        return OperatorMeasuresResult(problem, energy=float(energies[0]), state=states[:, 0])
    basis = None if sector is None else _sector_basis(problem, sector, p)
    energies, states = _lowest_states(problem, p, n_states, basis)
    return OperatorMeasuresResult(problem, energies=energies, states=states)


def _operator_problem(system: System | BaseProblemSpec) -> OperatorProblemSpec:
    """Return the operator problem of a system, compiling it when needed."""
    problem = system if isinstance(system, BaseProblemSpec) else OperatorCompiler().prepare(system).run().problem_
    if not isinstance(problem, OperatorProblemSpec):
        error_msg = f'The system compiles to the {problem.form} form, but the analysis needs operator elements.'
        raise AnalysisError(error_msg)
    return problem


def _parameters_of(problem: OperatorProblemSpec, values: Mapping[str, object] | None) -> Array:
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
    problem: OperatorProblemSpec,
    sector: SectorSpec,
    p: Array,
) -> npt.NDArray[np.intp]:
    """Return the basis states with a value as eigenvalue of a conserved operator.

    The operator is diagonal.
    """
    if problem.lowering is None:
        error_msg = 'The operator problem has no lowering of operators, so its sectors cannot be read.'
        raise AnalysisError(error_msg)
    terms = problem.lowering.run((sector.operator(problem.namespaces['']), 'the sector')).kernels_
    if not terms:
        error_msg = 'The operator of a sector lowers to no term.'
        raise AnalysisError(error_msg)
    state = RunStateSpec(0.0, np.zeros(0), p)
    matrix = terms[0].matrix * terms[0].run(state).values_
    for term in terms[1:]:
        matrix = matrix + term.matrix * term.run(state).values_
    diagonal = matrix.diagonal()
    if abs(matrix).sum() - np.abs(diagonal).sum() > DIAGONAL_TOLERANCE:
        error_msg = 'The operator of a sector must be diagonal in the product basis, such as a sum of `z` operators.'
        raise AnalysisError(error_msg)
    basis = np.flatnonzero(np.abs(diagonal - sector.value) < SECTOR_TOLERANCE)
    if not basis.size:
        error_msg = f'No basis state has the eigenvalue {sector.value} of the sector operator.'
        raise AnalysisError(error_msg)
    return basis


def _states_of(trajectory: Trajectory) -> tuple[OperatorProblemSpec, npt.NDArray[np.complex128]]:
    """Return the operator problem of a trajectory and its states, one per frame."""
    problem = trajectory.problem
    if not isinstance(problem, OperatorProblemSpec):
        error_msg = 'The trajectory is not the trajectory of an operator system.'
        raise AnalysisError(error_msg)
    return problem, np.asarray(trajectory.values(problem.states.names[0]))


def _lowest_states(
    problem: OperatorProblemSpec,
    p: Array,
    k: int,
    basis: npt.NDArray[np.intp] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Return the lowest eigenvalues and eigenstates of the generator at time 0.

    They are found through QuTiP, within a basis.
    """
    dimension = math.prod(problem.dimensions)
    matrix = problem._generator_at(0.0, p)
    indices = np.arange(dimension) if basis is None else np.asarray(basis, dtype=np.intp)
    matrix = sp.csr_array(matrix[indices][:, indices])
    count = min(k, indices.size)
    sparse = indices.size > DENSE_LIMIT and count < indices.size - 1
    energies, kets = qutip.Qobj(matrix).eigenstates(sparse=sparse, eigvals=count)
    states = np.zeros((dimension, count), dtype=complex)
    for column, ket in enumerate(kets):
        states[indices, column] = ket.full().ravel()
    return np.real(np.asarray(energies)), states


def _entanglement(
    problem: OperatorProblemSpec,
    states: npt.NDArray[np.complex128],
    subsystems: Sequence[str],
) -> Array:
    """Return the von Neumann entropy of the reduced state of some subsystems.

    It is given at every frame, in nats.
    """
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
            density = qutip.Qobj(state.reshape(-1, 1), dims=[dimensions, [1] * len(dimensions)])
        else:
            density = qutip.Qobj(state, dims=[dimensions, dimensions])
        values.append(qutip.entropy_vn(density.ptrace(sorted(set(keep)))))
    return np.asarray(values, dtype=float)
