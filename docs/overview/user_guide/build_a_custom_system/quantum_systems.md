# Quantum systems

This page builds a quantum system, such as a chain of spins, and follows its state in time or finds its spectrum. A
quantum system is a set of quantum subsystems whose state is one vector of their joint Hilbert space, the complex
vector space whose unit vectors are the possible states.

## A spectrum from the catalog

This section finds the two lowest energies of a chain of four spins, the Heisenberg chain of the catalog. The analysis
`operator_measures` diagonalises the generator of the system exactly, and `n_states` sets how many of the lowest
eigenvalues it returns. It measures quantum systems only, so it lives in the quantum package of physics,
`skcomplex.systems.physics.common.quantum`, beside the qubit and the bosonic mode, rather than in
`skcomplex.core.analysis`.

```python
from skcomplex.systems.physics.common.quantum import operator_measures
from skcomplex.systems.physics.statistical_mechanics import QuantumHeisenbergSystem

spectrum = operator_measures(QuantumHeisenbergSystem(n_sites=4), n_states=2)
spectrum.energies.round(6).tolist()
```

The two lowest energies are -2.0 and -1.0, in units of the coupling.

The core names the machinery of operators for its mathematics, linear operator dynamics on tensor products, while the
quantum package of physics keeps the words of physics. It holds the Hilbert space of a subsystem, the laws that build
the Hamiltonian and the dissipators, and the solvers that run them.

Every subsystem holds its state `psi` in the Hilbert space of a `HilbertSpec`, which names its operators. A subsystem
of your own declares the operators it needs. Typical subsystems have these operators.

- A spin has the components `x`, `y`, `z` and the ladder operators `plus` and `minus` of spin `j`.
- A qubit has the Pauli matrices.
- A mode has `annihilate`, `create`, and `number`, truncated at `quanta`.
- A particle on the sites of a lattice or a network has its `laplacian` and its `adjacency`.

A group of subsystems, such as an [`Element`](../core_components/elements.md#element) of spins with one member per
site of a space, is the tensor product of the spaces of its members, in member order. The initial state of every subsystem is a basis
label, an index, or a vector. The `HilbertSpec` names the compiler of quantum systems, so `prepare` compiles a system
of such elements to the operator form by itself.

## Operators and laws

This section writes the generator of a system from the operators of its subsystems. A spin one half is an
[`Element`](../core_components/elements.md#element) whose state `psi` lives in the two-dimensional space of a
`HilbertSpec` with the three spin components. Neighbouring spins on a ring exchange their spin with the coupling `J`,
the Heisenberg exchange `J (Sx Sx + Sy Sy + Sz Sz)`. An
[`Interaction`](../core_components/interactions.md#interaction) bound `among` the spins writes the exchange once for a
bond and sets it as a `GeneratorLaw`, as the Heisenberg chain of the catalog does. Here the chain has six sites and starts in
the alternating state up, down, up, down, up, down, and its observable `sz` records the expectation of `Sz` at every
site.

```python
import numpy as np

from skcomplex.systems.physics.statistical_mechanics import QuantumHeisenbergSystem

chain = QuantumHeisenbergSystem(n_sites=6, initial='neel').prepare()
trajectory = chain.simulate(until=2.0, dt=0.5)
trajectory.get_values('sz').shape, round(float(np.ptp(trajectory.get_values('energy'))), 8)
```

The record of `sz` has the shape `(5, 6)`, five output times by six sites. The energy varies by 0.0 over the run, so
the unitary evolution conserves it exactly.

Laws name the operators of a subsystem with `local_operator(s.psi, name)`, from
`skcomplex.core.components.laws.utils`. Operators combine by sums, numbers, and symbolic coefficients of parameters
and time. `operator_product` and `dagger` keep the order of products and adjoints. Two laws of the quantum package
build the dynamics.

- `GeneratorLaw(operator)` adds to the generator, with `ħ = 1`.
- `DissipatorLaw(operator, rate)` adds a Lindblad dissipator, a term that lets the system lose energy or coherence to
  its surroundings, which makes the system open.

An interaction bound `among` a group is written once for a bond, with `s.source` and `s.target`. It acts on every pair
of neighbours of the space of that group, here the
[`LatticeSpace`](../core_components/spaces.md#latticespace) of six sites. Compiling checks that the generator is
Hermitian, and that the dimension stays within the `max_dimension` of the operator solver.

A whole-system initial state, such as an entangled vector or `'ground'`, is given to `simulate` as
`initial_state={'<system label>.state': value}`.

## Solvers, observables, and spectra

This section chooses how the state evolves, records the expectations of operators, and diagonalises the generator
exactly. `prepare` compiles a system of quantum elements to the operator form. Its state is stored as complex pairs
under `<system>.state`, a pure state for a closed system and a density matrix for an open one. The solvers of the
quantum package hand every frame to QuTiP.

- `UnitarySolver` solves the Schrödinger equation.
- `LindbladSolver`, the default, solves the master equation.
- `JumpTrajectoriesSolver` draws quantum jumps from the random stream of the run.

An `Observable` whose expression is an operator records its expectation. One over a group gives one value per
member when its `reduce` keeps the values, as `sz` of the chain does. Every quantum run also records its `energy`, and
its `purity` when it is open. Groups on a lattice stream the space-time image of their local expectations to Rerun.

`operator_measures(system, kind=...)` analyses quantum systems by its kind.

- `'spectrum'` and `'ground-state'` diagonalise the generator exactly, optionally in the sector of a conserved
  diagonal operator.
- `'purity'` and `'entanglement'` read trajectories.

`symbolic(system, kind='hamiltonian')` prints the Hamiltonian in operator notation.
