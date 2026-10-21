# quantum-systems Specification

## Purpose
Lets a user model quantum systems of spins, qubits, and bosonic modes with symbolic Hamiltonians and dissipators, run
them with unitary, Lindblad, or quantum trajectory dynamics through the same interface as classical systems, and read
expectation values, spectra, and entanglement from the result.

## Requirements

### Requirement: Quantum subsystems have Hilbert spaces
The modelling layer SHALL provide quantum subsystems as elements with a declared Hilbert space: spins of spin `j`
with dimension `2j + 1`, qubits, bosonic modes truncated at a declared number of quanta, and a particle on the sites
of a space. A group of subsystems on a space SHALL be the tensor product of its members' spaces in member order.
A system whose total dimension exceeds its declared limit SHALL be rejected.

#### Scenario: Dimension of a spin chain
- **WHEN** a user builds a group of 10 spins one half on a periodic lattice and a mode truncated at 4 quanta
- **THEN** the compiled system has the Hilbert space dimensions `[2] * 10 + [5]` and the total dimension 5120

#### Scenario: Too large a space
- **WHEN** a user compiles a group of 40 spins one half with a dimension limit of `2**24`
- **THEN** compiling raises a named error that names the group, its dimension, and the limit

### Requirement: Operators are built from local operators with symbolic coefficients
Every quantum subsystem SHALL expose its local operators: spin components, raising, and lowering operators for
spins, Pauli matrices for qubits, and annihilation, creation, and number operators for modes, with `ħ = 1`. Operators
SHALL combine by sums, products, scalar multiplication, and the adjoint. Their coefficients SHALL be symbolic
expressions of parameters and time, validated like the coefficients of classical laws.

#### Scenario: Exchange term
- **WHEN** a user writes `J * (a.x * b.x + a.y * b.y + delta * a.z * b.z)` for two spins `a` and `b`
- **THEN** the operator has three terms whose coefficients are `J`, `J`, and `J * delta`, each acting on both spins

#### Scenario: Driven qubit
- **WHEN** a user writes `omega / 2 * q.z + A * cos(nu * t) * q.x` for a qubit `q`
- **THEN** the Hamiltonian at time `t` equals the matrix of that operator with the parameter values of the run

#### Scenario: Classical symbol in an operator
- **WHEN** an operator coefficient uses a state of a classical element
- **THEN** compiling raises a named error that names the operator and the state, since a coefficient may read
  parameters, time, and observables only

### Requirement: Hamiltonians and dissipators are laws
Systems SHALL accept Hamiltonian laws on a quantum element, and Hamiltonian interactions between the neighbours of a
space, which add one term per pair of neighbours. Systems SHALL accept dissipator laws with a collapse operator and a
non-negative rate, which make the system open. The Hamiltonian SHALL be checked to be Hermitian at compile time.

#### Scenario: Interaction over a lattice
- **WHEN** a nearest-neighbour exchange interaction acts on a group of 8 spins on a periodic lattice
- **THEN** the compiled Hamiltonian holds 8 exchange terms, one per bond, and equals the matrix of their sum

#### Scenario: Non-Hermitian Hamiltonian
- **WHEN** a user declares the Hamiltonian `q.plus` for a qubit `q`
- **THEN** compiling raises a named error that names the law and says to add its adjoint

### Requirement: Quantum systems compile to a quantum form
A system with quantum elements SHALL compile to the quantum form. Its state SHALL be a normalised pure state when the
system has no dissipator and the run uses no mixed initial state, and a density matrix otherwise. Initial states SHALL
be given per subsystem as named basis states, product states, or explicit vectors, or for the whole system as a
vector, a density matrix, or the ground state. Classical and quantum elements SHALL NOT share a partition.

#### Scenario: Néel state
- **WHEN** a chain of 6 spins one half starts from the product state up, down, up, down, up, down
- **THEN** the initial state vector has norm 1 and a single non-zero amplitude at the index of that basis state

#### Scenario: Unnormalised vector
- **WHEN** a user gives an initial vector of norm 2
- **THEN** creating the simulation raises a named error that names the system and the norm

### Requirement: Quantum runs use the simulation interface
A quantum system SHALL run through `Simulation`, `create`, and `run` like any other system, with frames at
instants of physical time separated by `dt`. Parameters SHALL change between frames through `update_parameters` and
schedules. The trajectory SHALL record the expectation observables, the energy, and the complex state at every frame.

#### Scenario: Run by name
- **WHEN** a user calls `run('quantum_heisenberg', settings={'until': 2.0, 'dt': 0.1})`
- **THEN** the trajectory has 21 frames, its state at each frame is a complex vector of the system's dimension, and its
  observables hold the expectation of `S^z` of every site at every frame

#### Scenario: Quench by parameter update
- **WHEN** a user changes the anisotropy of a running spin chain through `update_parameters` at time 1
- **THEN** the frames after time 1 follow the new Hamiltonian, and the energy observable jumps at that frame

### Requirement: Unitary evolution follows the Schrödinger equation
The unitary solver SHALL evolve a pure state by `i d psi / dt = H(t) psi` with `ħ = 1`, keep its norm, and agree with
the exact propagator for a time-independent Hamiltonian.

#### Scenario: Rabi oscillation
- **WHEN** a qubit with Hamiltonian `omega / 2 * q.x` starts in the up state and runs to time `pi / omega`
- **THEN** the expectation of `q.z` goes from 1 to -1 as `cos(omega t)` within 1e-6, and the norm stays 1 within 1e-8

#### Scenario: Conserved magnetisation
- **WHEN** an isotropic Heisenberg chain of 10 spins runs for 10 time units from a Néel state
- **THEN** the total magnetisation and the energy stay constant within 1e-8

### Requirement: Open systems follow the Lindblad master equation
The Lindblad solver SHALL evolve a density matrix by the master equation with the declared dissipators, keep its trace
1 and its Hermiticity, and reduce to unitary evolution when every rate is zero.

#### Scenario: Spontaneous decay
- **WHEN** a qubit starts in the excited state with the dissipator `q.minus` of rate `gamma`
- **THEN** the excited population decays as `exp(-gamma t)` within 1e-6 and the trace stays 1 within 1e-10

#### Scenario: Dephasing lowers purity
- **WHEN** a spin chain with dephasing on every site runs from a pure state
- **THEN** the purity starts at 1 and never increases from frame to frame

### Requirement: Quantum trajectories sample the master equation
The quantum trajectory solver SHALL evolve pure states with random quantum jumps drawn from the simulation's random
stream, so a seeded run is reproducible, and the average over trajectories SHALL converge to the Lindblad solution.

#### Scenario: Reproducible jumps
- **WHEN** a decaying qubit runs twice with the trajectory solver and the same seed
- **THEN** the two trajectories are identical, including the jump times

#### Scenario: Ensemble average
- **WHEN** 500 seeded trajectories of a decaying qubit are averaged
- **THEN** the mean excited population is within three standard errors of the Lindblad solution at every frame

### Requirement: Expectation values are observables
The modelling layer SHALL provide expectation values of operator expressions as observables, and reductions of local
expectations over the members of a quantum group, such as the magnetisation of every site or their sum. The analysis
layer SHALL provide the energy, the purity, and the von Neumann entanglement entropy of a set of subsystems, at every
frame of a trajectory.

#### Scenario: Local magnetisation
- **WHEN** a trajectory of a chain of 8 spins is asked for the observable `sz` of its spin group
- **THEN** it returns an array of 8 real values per frame, each the expectation of the local `S^z` of one site

#### Scenario: Entangled pair
- **WHEN** the entanglement entropy of one spin of the singlet of two spins one half is computed
- **THEN** it equals `ln 2` within 1e-12

### Requirement: Exact diagonalisation gives spectra and ground states
The analysis layer SHALL compute the lowest eigenvalues and eigenstates of a quantum system's Hamiltonian at given
parameter values, by dense diagonalisation for small spaces and a sparse eigensolver otherwise, optionally within the
sector of a conserved quantity. The ground state SHALL be usable as the initial state of a run.

#### Scenario: Heisenberg ring of four spins
- **WHEN** the spectrum of the isotropic antiferromagnetic Heisenberg ring of 4 spins one half with `J = 1` is computed
- **THEN** its lowest eigenvalue is -2 within 1e-12

#### Scenario: Ground state is stationary
- **WHEN** a run starts from the ground state of a time-independent Hamiltonian
- **THEN** every expectation observable stays constant within 1e-8

### Requirement: Quantum runs stream to Rerun
A quantum run SHALL stream to Rerun through the existing observer. Local expectation values of a group on a
one-dimensional lattice SHALL be logged as a lattice image with its space-time image. System observables, the energy,
and the purity of open systems SHALL be logged as scalar series. The default blueprint SHALL make the space-time image
the primary view of a spin chain, and the series the primary view of a system with one subsystem.

#### Scenario: Light cone
- **WHEN** a chain of 12 spins starting with one flipped spin in the middle is streamed to Rerun
- **THEN** the recording holds a space-time image of the local magnetisation in which the flip spreads in a cone from
  the middle site
