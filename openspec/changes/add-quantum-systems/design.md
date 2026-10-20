# Design

## Context

Every state in the framework is a real scalar that a law changes through a Sage expression, and the compiler packs
the states into one flat float vector that a stepper advances frame by frame (see proposal.md, Why). A quantum system
breaks two of these assumptions. Its state is one vector of `2^N` complex amplitudes that no law reads one component
at a time, and its dynamics is a linear operator built from non-commuting local operators. The rest of the pipeline
still fits:
- the system's parameters, time, schedules, and `update_parameters` stay what they are;
- `Simulation` already advances a problem by frames of physical time through a `Stepper`;
- `ExternalBackend` already hands a whole compiled problem to a third-party library, as `LammpsBackend` does;
- `add-complex-states` stores complex values as contiguous float pairs in state vectors and trajectories;
- `ImageVisual` already logs a space-time image of a one-dimensional lattice.

## Goals / Non-Goals

**Goals:**

- Write a quantum model as subsystems on a space, a Hamiltonian, and dissipators, with symbolic coefficients, and run
  it with `create` and `run` like a classical model.
- Keep the core domain-agnostic: spins, qubits, and modes are general subsystems, and nothing in the core knows about
  magnetism.
- Take every numerical method from a mature library, so the core holds only the lowering of operators and the adapter
  to the solvers.

**Non-Goals:**

- Tensor network and DMRG methods for long chains. Exact states limit chains to about 24 spins one half.
- Variational quantum Monte Carlo with neural states, which NetKet covers with its own model language.
- Fermionic second quantisation and quantum circuits. A later change can add them on OpenFermion or a circuit
  library, with the same operator terms.
- Coupling a quantum partition to classical partitions in one co-simulation. The quantum form is a partition of its
  own, and mixed quantum and classical dynamics is a later change.

## Decisions

### Library survey

The survey ran on 2026-10-19 against PyPI and GitHub.

| Library | Version and date | Licence | Wheels for Python 3.12 and 3.13 | Maintenance and use | Fit |
|---|---|---|---|---|---|
| QuTiP | 5.3.1, 2026-08-04 | BSD-3-Clause | Compiled wheels for manylinux x86_64 and aarch64, and macOS x86_64 and arm64, for CPython 3.11 to 3.14 | 2087 stars, commits on 2026-10-19, about 217,000 downloads a month | Chosen. Operators, sparse CSR and dense data, Schrödinger, Lindblad, and Monte Carlo solvers with a stepwise `start` and `step` API, time-dependent `QobjEvo` with Python coefficients, `ptrace`, entropies, and eigensolvers. Needs only NumPy, SciPy, and `packaging` |
| QuSpin | 1.0.1, 2026-04-08 | BSD-3-Clause | Pure wheel plus `quspin-extensions` 0.1.8 compiled for manylinux x86_64 and macOS, no Linux aarch64 | 338 stars, about 4,900 downloads a month | Rejected as the base: fast symmetry-reduced exact diagonalisation of spin chains, but no Monte Carlo wave functions, a string-based operator language, and fewer platforms. Kept as a possible later backend for symmetry sectors |
| dynamiqs | 0.3.7, 2026-10-15 | Apache-2.0 | Pure wheel, needs JAX, diffrax, and QuTiP | 328 stars, pre-1.0 | Rejected: GPU and gradient solvers on JAX, a heavy stack the project avoids, and it depends on QuTiP anyway |
| NetKet | 3.23.0, 2026-10-19 | Apache-2.0 | Pure wheel, needs JAX and Flax | 703 stars | Rejected: variational Monte Carlo of ground states, not exact dynamics or open systems |
| OpenFermion | 1.8.1, 2026-07-17 | Apache-2.0 | Pure wheel, needs Cirq and SymPy | 1744 stars | Rejected: fermionic operators for chemistry, with no solvers of its own |

QuTiP becomes a runtime dependency rather than an extra. It is permissive, it ships compiled wheels for every
supported platform, and it adds no package the project does not already install. A prototype on QuTiP 5.3.1 built the
12-site XX chain as sparse CSR in 26 ms, and `SESolver.step` reproduced the single magnon probabilities `J_d(t)²` at
time 2 within 2e-6, with QuTiP's default tolerances. `MCSolver.start` with a fixed seed gave identical jump records on
two runs.

### Operators are terms with Sage coefficients

The Sage symbolic ring is commutative, and the project imports Sage only through `sage.all__sagemath_symbolics`, which
has no free algebra. So an `Operator` is a small term structure in `skcomplex.core.model`: a mapping from an ordered
tuple of `(subsystem key, LocalOperator)` factors to a Sage coefficient, where a `LocalOperator` is a named matrix of
one subsystem. Products concatenate factors and keep their order, sums add the coefficients of equal factor tuples, a
number added to an operator is a multiple of the identity, and the adjoint conjugates the coefficient and reverses and
adjoins the factors. The compiler sorts the factors of different subsystems into tensor product order, since they
commute, and keeps the order of factors on one subsystem. The coefficients are ordinary Sage expressions of parameters
and time, so the existing NumPy printer lowers them unchanged. An operator never becomes a Sage expression, and
classical laws never see one.

`Spin(label, j=1/2)`, `Qubit(label)`, `Mode(label, quanta=n)`, and `Site(label, space=...)` are subclasses of the
`QuantumElement` base, whose namespace exposes their local operators, such as `s.x` and `a.annihilate`, built from
QuTiP's `jmat`, `sigmax`, `destroy`, and `num`. Inside a group, every member is a subsystem keyed `<path>[<member>]`.
`Exchange` is an interaction over the neighbours of the space of its group, written once for a bond with `s.first` and
`s.second` and expanded to one term per bond. `Hamiltonian(operator)` and `Dissipator(operator, rate)` are the two
quantum laws. They are not members of the classical `Law` union, so the classical compiler keeps its types.

### The quantum form and its problem

`Form.QUANTUM` joins the forms. `compile_system` dispatches a system tree that holds quantum elements to
`compile_quantum` in the new module `skcomplex.core.compile._quantum`, the way it dispatches Mesa and rewriting
systems, so the classical compiler is untouched. A system that mixes quantum and classical elements raises
`IncompatibleLawsError`. The result is a `QuantumProblem`, a `Problem` subclass whose extra fields hold:
- the Hilbert space dimensions and the subsystem keys, in tensor product order;
- the Hamiltonian as `QuantumTerm`s, each a sparse CSR matrix built with `scipy.sparse.kron` and a coefficient lowered
  by the NumPy printer to a function of time and the parameter vector;
- the terms of every collapse operator, with the square root of its rate folded into the coefficients;
- the namespaces of the systems and of the members, which observables and sectors receive.

Terms with numeric coefficients fold into one constant matrix and terms with equal coefficients are summed, so the
isotropic Heisenberg chain is one matrix and one coefficient. The compiler checks that the Hamiltonian is Hermitian
numerically, at sampled parameter values, and names the laws that are not. It checks the total dimension against
`max_dimension`, raising `HilbertSpaceError`. A coefficient that reads a symbol other than parameters and time raises
`ValidationError`.

The compiler stores the state under `<system label>.state` as one complex `Entry` of the layout, so it uses the paired
storage of `add-complex-states`: the real and imaginary part of every amplitude are adjacent floats, and the
trajectory decodes them into a complex array. The `_re` and `_im` symbols and the rewriting of complex laws in
`compile/_complex.py` do not apply, because no Sage expression ever reads the quantum state. The problem stores a pure
state of shape `(dimension,)` for a closed system with a pure initial state, and a density matrix of shape
`(dimension, dimension)` otherwise. Initial states come from the elements, as basis labels such as `'up'`, indices, or
vectors, per member for groups, or from a `QuantumSystem`, a `System` subclass with `initial` and `max_dimension`,
as a vector, a matrix, or `'ground'`, the lowest eigenstate at the initial parameters.

### Solvers are QuTiP steppers in the solvers layer

`skcomplex.core.solvers` gains the base `QuantumSolver` and `UnitaryEvolution`, `LindbladEvolution`, and
`QuantumTrajectories`, steppers of `Form.QUANTUM` registered as `unitary`, `lindblad`, and `quantum_trajectories`.
They live in the solvers layer rather than the backends layer, because the default solver of a form has to be
registered whenever the runtime is imported, and `skcomplex.core.backends` registers its names only when it is
imported. `lindblad` is the default: it uses QuTiP's `SESolver` for a closed pure problem, which is the master equation
without dissipators, and `MESolver` otherwise. `unitary` refuses open problems and density matrices.

A solver wraps the CSR matrices as `Qobj` with the problem's dimensions, folds the coefficients that do not depend on
time into one constant `Qobj` at the current parameters, and keeps the time-dependent ones as `QobjEvo` pairs with
functions of time. It creates the QuTiP solver with `atol=1e-12` and `rtol=1e-10`, and starts it from the current state
and time at the first frame and again whenever the parameter vector changes, so a parameter update or a schedule
applies from the next frame on. This replaces passing the parameters through QuTiP's `args`, which would call one
Python function per coefficient at every right-hand side evaluation. `QuantumTrajectories` seeds QuTiP's `MCSolver`
from the simulation's generator at every start, so one seeded run is one reproducible trajectory, and stores the
projector of the pure state when the problem holds a density matrix.

Frames are instants of physical time, so the trajectory reads the same way as a classical one: `times` holds the
frames, `trajectory['<system>.state']` the complex states, and `trajectory.observables` the expectations.

### Exact diagonalisation is an analysis, not a run

`skcomplex.core.analysis.spectrum(system, k=6, *, parameters=None, sector=None)` compiles the system, sums the terms at
time 0 and the given parameter values, and returns a `Spectrum` of the lowest `k` eigenvalues and eigenstates through
QuTiP's `Qobj.eigenstates`, dense up to a dimension of 1024 and sparse above. A sector is a function of the system
namespace that returns a conserved operator diagonal in the product basis, such as the total magnetisation, with its
eigenvalue, and restricts the basis to the states with that eigenvalue. `ground_state` returns the lowest one.
`purity` and `entanglement_entropy` read a trajectory frame by frame, the latter through QuTiP's `ptrace` and
`entropy_vn`, and `hamiltonian_of` returns the Hamiltonian as one operator printed in operator notation, while
`equations_of` raises `AnalysisError` for a quantum system, which has no state equations.

### Observables

`expectation(label, expression, *, group=None, reduce=None)` makes an `Observable` whose value is `Tr(ρ O)` or
`<psi|O|psi>`, a float for a Hermitian operator. For a group it records one expectation per member unless `reduce`
names a reduction. The `ObservableEvaluator` delegates every observable of a quantum problem to the quantum module,
which adds `energy` to every quantum run, and `purity` to open ones, unless the system declares an observable of that
name. Expectations are computed with SciPy sparse products on the frame's state.

### Rerun

Quantum elements default to the new `ExpectationVisual`, a subclass of `ImageVisual` registered as `expectations`. A
group logs the expectation of the displayed local operator of every member, `z` for spins and qubits and `number` for
modes, as a lattice image with a space-time image in one dimension, which is the light cone of a quench and the
thumbnail of the gallery example. A particle on sites logs the probability of every site the same way, and a single
spin, qubit, or mode logs its Hermitian local expectations as series. Observables, the energy, and the purity go through
the observer's scalar series. The raw state is not logged, since its size grows as `2^N`.

### The Schrödinger model gains an operator formulation

The existing `schrodinger` model is a particle on a periodic lattice written as two real fields. Its operator
formulation, `formulation='operators'`, builds the same particle as one `Site` subsystem over the `LatticeSpace`, with
the Hamiltonian `-D laplacian + V`, where the compiler builds the Laplacian of the site from the same dealiased
spectral multiplier the fields use. Both formulations evolve the same discrete equation and agree within 1e-8 on 128
sites. The operator formulation accepts position dephasing at a rate `gamma`, one projector dissipator per site, which
the field formulation cannot express. The registered defaults keep `ETDRK4` for the fields, so the operator formulation
runs with `settings={'solver': 'auto'}` or through `Simulation`. The model keeps its name, so the inventory gains no
new entry.

### The quantum Heisenberg model

`quantum_heisenberg(n_sites=12, *, J=1.0, delta=1.0, h=0.0, gamma=0.0, boundary='periodic', initial='flip')` returns a
`QuantumSystem` with a lattice of spins one half, the `XXZExchange` interaction, a `ZeemanField`, a `Dephasing` per
spin when `gamma > 0`, and the observables `sz` per site and `magnetization`. Its defaults run to time 8 with
`dt = 0.05`. Its example shows the space-time image of `sz` as its thumbnail, and an analysis figure of the magnon
against `J_d(t)²` and of the ground-state energy per site from `spectrum` in the sector of zero magnetisation for 4 to
14 sites against `1/4 - ln 2`.

## Risks / Trade-offs

- [Exponential cost in the number of subsystems] → `max_dimension` rejects systems that would not fit, with a message
  that names every element and its dimension, and sectors cut the work of spectra.
- [Trajectories of density matrices are large] → A density matrix of dimension 4096 takes 268 MB per frame. The state
  is the simulation's working vector, which observables read, so a run too large to record uses `record=False` with
  the Rerun observer. The gallery examples stay at 12 sites for pure states and at most 48 sites or 6 spins for
  density matrices.
- [The spectral Laplacian is dense] → The Lindblad superoperator of a dense Hamiltonian grows as `n^3`, so dephased
  packets stay below about 64 sites.
- [Operator terms are not Sage expressions] → `hamiltonian_of` prints the Hamiltonian in operator notation.
- [QuTiP API changes] → The adapter uses only `Qobj`, `QobjEvo`, the three solver classes, `ptrace`, `entropy_vn`,
  `eigenstates`, and `groundstate`, all stable since QuTiP 5.0, and the manifest pins `qutip>=5.3,<6`.
- [Coupling to classical partitions] → A quantum system cannot share a co-simulation with classical partitions, so the
  catalog test of observables read by classical laws skips quantum models.
