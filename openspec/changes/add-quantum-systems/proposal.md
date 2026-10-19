# Proposal

## Why

Mihailović et al. (2023 §7.3) treat the quantum Heisenberg model, a chain of spins coupled by the exchange of their
spin operators. Its state is a vector in a Hilbert space whose dimension grows as `2^N`, its dynamics is a Hamiltonian
operator, and what a reader measures are expectation values of operators. The framework has only real states that laws
change one symbol at a time, so the inventory of `organize-model-library` records the model as out of scope. The user
directive to include every model turns this into a missing general capability: quantum states and operators, usable by
any field that has a quantum subsystem, such as spin chains, qubits, cavities, and open quantum systems.

## What Changes

- Add quantum subsystems as elements: spins of any spin quantum number, qubits as spin one half, bosonic modes truncated
  at a declared number of quanta, and a single particle on the sites of a space. A group of quantum subsystems on a
  lattice or a network is the tensor product of their Hilbert spaces, ordered by member.
- Add operator expressions built from the local operators of subsystems, such as `s.x`, `s.z`, `s.plus`, `a.number`,
  and `a.annihilate`, with sums, products, adjoints, and coefficients. The coefficients are Sage expressions of
  parameters and time, so a drive, a ramp, or a schedule enters a Hamiltonian the way it enters a rate law today.
- Add quantum laws: `Hamiltonian` terms, on an element or as an interaction between neighbours of a space, and
  `Dissipator` terms with a collapse operator and a rate, which make the system open.
- Add a quantum form of compiled systems. Its state is a pure state vector or a density matrix, stored as complex
  values in the flat state vector, and initial states are product states, named basis states, or explicit vectors.
- Add solvers for the quantum form: unitary Schrödinger evolution, the Lindblad master equation, and quantum
  trajectories by Monte Carlo wave functions, as steppers that hand each frame to QuTiP.
- Add expectation values as observables, recorded in trajectories and streamed to Rerun, and the reduced quantities
  that come from the state: the energy, the purity, and the entanglement entropy of a subsystem.
- Add exact diagonalisation as an analysis of a quantum system: the lowest eigenvalues and eigenstates of its
  Hamiltonian, and the ground state as an initial state for a run.
- Add the quantum Heisenberg model under `physics/statistical_mechanics`: the spin one half XXZ chain with exchange,
  anisotropy, a field, and optional dephasing, quenched from a Néel or a single spin flip state.
- Add an operator formulation to the existing `schrodinger` model, a particle on the same periodic lattice as a
  single-particle Hilbert space with hopping and a potential, which agrees with the two-field formulation and accepts
  position dephasing.
- Add QuTiP as a runtime dependency.

## Capabilities

### New Capabilities

- `quantum-systems`: quantum subsystems and their Hilbert spaces, operator expressions with symbolic coefficients,
  Hamiltonian and dissipator laws, the quantum form and its states, unitary, Lindblad, and trajectory solvers,
  expectation observables, exact diagonalisation, and the streaming of quantum runs to Rerun.

### Modified Capabilities

- `model-library`: the quantum Heisenberg model, and the operator formulation of the Schrödinger model.

## Impact

- `skcomplex.core.base` gains `Form.QUANTUM` and the named error `HilbertSpaceError`.
- `skcomplex.core.model` gains `Spin`, `Qubit`, `Mode`, `Site`, `Operator`, `Hamiltonian`, `Dissipator`, `Exchange`, and
  `expectation`, and `System` accepts quantum elements and laws.
- `skcomplex.core.compile` lowers a quantum system to a `QuantumProblem` that holds the operator terms, the lowered
  coefficient functions, and the Hilbert space dimensions.
- `skcomplex.core.backends` gains the QuTiP adapter and registers the solvers `unitary`, `lindblad`, and
  `quantum_trajectories`.
- `skcomplex.core.analysis` gains `spectrum`, `ground_state`, `purity`, and `entanglement_entropy`.
- `skcomplex.models.physics.statistical_mechanics` gains `quantum_heisenberg`, and
  `skcomplex.models.physics.waves.schrodinger` gains the `formulation` parameter.
- New runtime dependency: `qutip` 5.3 or later, BSD-3-Clause, with wheels for Linux and macOS on Python 3.12 and
  3.13. It needs only NumPy, SciPy, and `packaging`, which the project already installs.
- Depends on `add-complex-states` for the paired storage of complex values in state vectors and trajectories, and on
  `add-model-gallery` for the examples and thumbnails.
- No breaking change: classical systems keep their forms, layouts, and solvers.
