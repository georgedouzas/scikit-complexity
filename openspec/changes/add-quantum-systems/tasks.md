# Tasks

## 1. Dependency and core vocabulary

- [x] 1.1 Declare `qutip>=5.3,<6` as a runtime dependency and lock it, and verify that `pdm lock` and `pdm sync` pass
- [x] 1.2 Add `Form.QUANTUM` and `HilbertSpaceError` to `skcomplex.core.base`, and verify the too large a space scenario

## 2. Subsystems and operators

- [x] 2.1 Add `QuantumElement`, `Spin`, `Qubit`, `Mode`, and `Site` with their dimensions and local operators, and verify
  the local matrices against QuTiP's `jmat`, `sigmax`, and `destroy`
- [x] 2.2 Add `LocalOperator` and `Operator` terms with sums, products, adjoints, and Sage coefficients, and verify the
  exchange term and classical symbol scenarios
- [x] 2.3 Add `Hamiltonian`, `Dissipator`, `Exchange` over the neighbours of lattice and network spaces, and
  `QuantumSystem`, and verify the interaction over a lattice and non-Hermitian Hamiltonian scenarios

## 3. Compilation

- [x] 3.1 Lower quantum systems to a `QuantumProblem` in a new compile module dispatched by `compile_system`, with sparse
  CSR terms, lowered coefficient functions, collapse operators, and the dimension limit, and verify the dimension of a
  spin chain and too large a space scenarios and that a constant Hamiltonian folds to one matrix
- [x] 3.2 Add initial states from basis labels, product states, vectors, density matrices, and `'ground'`, stored as one
  complex entry in the paired layout, and verify the Néel state and unnormalised vector scenarios
- [x] 3.3 Add `expectation` and quantum reductions to the observable evaluator, with the energy and the purity of open
  systems by default, and verify the local magnetisation scenario

## 4. Solvers

- [x] 4.1 Add `UnitaryEvolution` on QuTiP's `SESolver` in `skcomplex.core.solvers`, registered as `unitary`, and verify
  the Rabi oscillation and conserved magnetisation scenarios
- [x] 4.2 Add `LindbladEvolution` on `MESolver`, registered as `lindblad` and the default of the form, and verify the
  spontaneous decay and dephasing scenarios
- [x] 4.3 Add `QuantumTrajectories` on `MCSolver` seeded from the simulation's generator, and verify the reproducible
  jumps and ensemble average scenarios
- [x] 4.4 Restart the QuTiP solver from the current state when the parameters change, and verify the quench by
  parameter update scenario
- [x] 4.5 Record the complex states and the observables at every frame, and verify the run by name scenario

## 5. Analysis and visualization

- [x] 5.1 Add `spectrum`, `Spectrum`, and `ground_state` with dense and sparse paths and conserved sectors, and verify
  the Heisenberg ring of four spins and ground state is stationary scenarios
- [x] 5.2 Add `purity` and `entanglement_entropy` over trajectory frames, and verify the entangled pair scenario
- [x] 5.3 Add `ExpectationVisual`, which logs local expectations of quantum groups as lattice and space-time images and
  becomes the primary view of the default blueprint, and verify the light cone scenario on a recording
- [x] 5.4 Add `hamiltonian_of`, which prints a quantum system in operator notation, make `equations_of` refuse quantum
  systems, and verify both

## 6. Models

- [x] 6.1 Add `quantum_heisenberg` to `skcomplex.models.physics.statistical_mechanics` as a registered factory, and
  verify the ground state of the ring, single magnon, and dephasing scenarios and the conformance suite
- [x] 6.2 Add the `formulation` and `gamma` parameters to `schrodinger`, and verify the formulations agree and dephased
  packet scenarios
- [x] 6.3 Mark `quantum_heisenberg` implemented in `docs/overview/inventory.toml` and drop its blocked reason, and verify
  the inventory check

## 7. Gallery, documentation, and gate

- [x] 7.1 Add `docs/examples/physics/statistical_mechanics/plot_quantum_heisenberg.py` with the light cone as its
  thumbnail and the magnon and the ground-state energy per site against `1/4 - ln 2` as its analysis figure
- [x] 7.2 Extend `docs/examples/physics/waves/plot_schrodinger.py` with an analysis figure of the agreement of the
  formulations and of the purity of dephased packets, keeping the Rerun screenshot as its thumbnail
- [x] 7.3 Document quantum subsystems, operators, laws, solvers, observables, and spectra in the user guide
- [x] 7.4 Run formatting, linting, type checks, the tests of the change, the catalog conformance of its models, the
  layout and inventory tests, and the doctests of its modules
