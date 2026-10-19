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
has no free algebra. So an `Operator` is a small term structure: a mapping from an ordered tuple of `(subsystem path,
local operator name)` factors to a Sage coefficient. Products concatenate factors, sort factors of different subsystems
by subsystem order, since they commute, and keep the order of factors on one subsystem. Sums add coefficients of equal
factor tuples, and the adjoint conjugates the coefficient and reverses and adjoins the factors. The coefficients are
ordinary Sage expressions of parameters, time, observables, and the functions `pulse`, `ramp`, and `step`, so the
existing validation, domains, and printers apply to them unchanged. An operator never becomes a Sage expression, and
classical laws never see one.

`Spin(label, j=1/2)`, `Qubit(label)`, `Mode(label, quanta=n)`, and `Site(label, space)` are elements whose namespace
exposes their local operators, such as `s.x` and `a.annihilate`. Inside a `Group` on a `LatticeSpace` or `NetworkSpace`,
a member's local operators carry its index, and `Exchange` is an interaction over the neighbours of the space, written
once for a pair `(a, b)` and expanded to one term per bond, the way pair interactions expand today.
`Hamiltonian(operator)` and `Dissipator(operator, rate)` are the two quantum laws.

### The quantum form and its problem

`Form.QUANTUM` joins the forms. `compile_system` lowers a system with quantum elements to a `QuantumProblem`, a
`Problem` whose extra fields hold:
- the Hilbert space dimensions, ordered by element and member;
- one sparse CSR matrix per distinct factor tuple, built with `scipy.sparse.kron` from the local matrices;
- the coefficient of each matrix lowered by the existing NumPy printer to a function of time and the parameter vector;
- the collapse operators and their rates, lowered the same way;
- the expectation observables as operator terms.

Constant coefficients are folded at compile time, so a time-independent Hamiltonian is one matrix. The compiler checks
that the Hamiltonian equals its adjoint symbolically on the coefficients and term by term, and checks the total
dimension against `max_dimension`, a parameter of the system with a default of `2**20`, raising `HilbertSpaceError`.

The state vector of the problem holds the pure state or the density matrix in the paired complex layout of
`add-complex-states`, under the variable `<system path>.state`. The problem holds a density matrix when the system has
a dissipator or the initial state is mixed. Initial states come from the elements, as basis labels such as `'up'` or a
number of quanta, or from the system as a vector, a matrix, or `'ground'`, which takes the lowest eigenstate at the
initial parameters.

### Solvers are QuTiP steppers in the backends layer

`skcomplex.core.backends` gains `UnitaryEvolution`, `LindbladEvolution`, and `QuantumTrajectories`, `ExternalBackend`
subclasses for `Form.QUANTUM` registered as `unitary`, `lindblad`, and `quantum_trajectories`. The default solver of the
form is `lindblad` when the problem has a dissipator and `unitary` otherwise. `build` wraps the CSR matrices as `Qobj`
with the problem's dimensions and assembles one `QobjEvo` of `[matrix, coefficient]` pairs, where each coefficient
calls the lowered function with QuTiP's `args` holding the parameter vector. It then creates `SESolver`, `MESolver`, or
`MCSolver` with tolerances `atol=1e-10` and `rtol=1e-8`. The first `advance` calls `start` at the initial state and
time, and every `advance(t, x, p, until)` calls `step(until, args={'p': p})`, so a parameter update or a schedule
reaches the next frame without rebuilding the solver. `QuantumTrajectories` draws its seed from the simulation's
generator, so one run is one reproducible trajectory, and ensembles of trajectories are ensembles of seeded runs from
`skcomplex.core.experiments`.

Frames are instants of physical time, so a quantum run has a time axis as faithful as a classical one, and the
trajectory reads the same way: `times` holds the frames, `trajectory['<system>.state']` the complex states, and
`trajectory.observables` the expectations.

### Exact diagonalisation is an analysis, not a run

Diagonalising a Hamiltonian is a static computation with no time axis, so it does not go through `Simulation`.
`skcomplex.core.analysis.spectrum(system, k=6, parameters=None, sector=None)` compiles the system, sums the matrices at
the parameter values, and returns the lowest `k` eigenvalues and eigenstates through QuTiP's `Qobj.eigenstates`, dense
below a dimension of 4096 and sparse with SciPy's `eigsh` above. A sector restricts the basis to the states with a given
eigenvalue of a conserved diagonal operator, such as the total magnetisation. `ground_state` returns the lowest one,
and the initial state `'ground'` uses it. Purity and the von Neumann entanglement entropy of a set of subsystems read a
trajectory frame by frame through QuTiP's `ptrace` and `entropy_vn`. This keeps one honest meaning for every frame of
a trajectory, and lets the gallery show a spectrum next to a run.

### Observables

`expectation(operator)` makes an `Observable` whose value is `Tr(ρ O)` or `<psi|O|psi>`, recorded as a real number for
a Hermitian operator. A reduction over a quantum group, such as `Observable('sz', group='spins', expression=lambda s:
s.z)`, records one expectation per member. The energy of a system is recorded under `<system>/energy` by default. The
existing `ObservableEvaluator` gains the quantum case, which uses the QuTiP `expect` function on the frame's state.

### Rerun

Local expectations of a quantum group on a lattice are logged through `ImageVisual` like a classical lattice state, so
a chain gets its space-time image, which is the light cone of a quench and the thumbnail of its gallery example.
System observables, the energy, and the purity go through `ScalarsVisual`. The raw state is not logged, since its size
grows as `2^N` and no view can show it.

### The Schrödinger model gains an operator formulation

The existing `schrodinger` model is a particle on a periodic lattice written as two real fields. Its operator
formulation builds the same particle as one `Site` subsystem of dimension `n_sites` over the `LatticeSpace`, with the
kinetic operator from the lattice's spectral Laplacian and the potential as a diagonal operator. Both formulations then
evolve the same discrete equation, which gives a direct cross-check of the quantum solvers against `ETDRK4`, and the
operator formulation can add position dephasing, which the field formulation cannot express. The model keeps its name
and its default formulation, so the inventory gains no new entry.

`Site` is the general subsystem of a single particle on the sites of a space, with the position projectors and the
hopping operators of the space as its local operators. It serves tight-binding models in any field.

### The quantum Heisenberg model

`quantum_heisenberg(n_sites=12, *, J=1.0, delta=1.0, h=0.0, gamma=0.0, boundary='periodic', initial='flip')` returns a
group of spins one half on a `LatticeSpace`, an `Exchange` interaction with the XXZ operator, a field term, a
dephasing `Dissipator(s.z, gamma)` per spin when `gamma > 0`, and the observables `sz` and the total magnetisation.
Its defaults run to time 8 with `dt = 0.05`, 161 frames within the gallery budget. Its example shows the space-time
image of `sz` as its thumbnail, and an analysis figure of the ground-state energy per site from `spectrum` for 4 to 16
sites against `1/4 - ln 2`.

## Risks / Trade-offs

- [Exponential cost in the number of subsystems] → `max_dimension` rejects systems that would not fit, with a message
  that names the dimension, and sectors halve the work of spectra. Chains of up to 16 spins run in seconds.
- [Trajectories of density matrices are large] → A density matrix of dimension 4096 takes 268 MB per frame, so the
  trajectory records the state only when it has at most `2**20` components, a setting of the system, and records the
  observables always. The gallery examples stay at 12 sites for pure states and 6 sites for density matrices, whose
  Lindblad runs also stay fast.
- [Operator terms are not Sage expressions] → `equations_of` prints a quantum system as its Hamiltonian and dissipators
  in operator notation, so documentation still shows the equations.
- [QuTiP API changes] → The adapter uses only `Qobj`, `QobjEvo`, the three solver classes, `expect`, `ptrace`,
  `entropy_vn`, and `eigenstates`, all stable since QuTiP 5.0, and the manifest pins `qutip>=5.3,<6`.
- [Dependence on `add-complex-states`] → The quantum states use its paired layout and complex trajectories, so this
  change is applied after it.
