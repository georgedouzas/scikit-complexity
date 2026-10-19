# Tasks

## 1. Dependency and core vocabulary

- [ ] 1.1 Declare `qutip>=5.3,<6` as a runtime dependency, update the Project Profile's dependency list, and verify
  that `pdm install`, `deptry`, and `pip-audit` pass on Linux and macOS with Python 3.12 and 3.13
- [ ] 1.2 Add `Form.QUANTUM` and `HilbertSpaceError` to `skcomplex.core.base`, and verify the errors module test and
  the form enumeration test

## 2. Subsystems and operators

- [ ] 2.1 Add `Spin`, `Qubit`, `Mode`, and `Site` with their dimensions and local operators, and verify the dimension
  of a spin chain scenario and the local matrices against QuTiP's `jmat`, `sigmax`, and `destroy`
- [ ] 2.2 Add `Operator` terms with sums, products, adjoints, and Sage coefficients, and verify the exchange term,
  driven qubit, and classical symbol scenarios
- [ ] 2.3 Add `Hamiltonian`, `Dissipator`, and `Exchange` over the neighbours of lattice and network spaces, and verify
  the interaction over a lattice and non-Hermitian Hamiltonian scenarios

## 3. Compilation

- [ ] 3.1 Lower quantum systems to a `QuantumProblem` with sparse CSR term matrices, lowered coefficient functions,
  collapse operators, and the dimension limit, and verify the too large a space scenario and that a constant
  Hamiltonian folds to one matrix
- [ ] 3.2 Add initial states from basis labels, product states, vectors, density matrices, and `'ground'`, stored in
  the paired complex layout, and verify the Néel state and unnormalised vector scenarios
- [ ] 3.3 Add `expectation` and quantum reductions to the observables and their evaluator, and verify the local
  magnetisation scenario on both the NumPy and Numba backends

## 4. Solvers

- [ ] 4.1 Add `UnitaryEvolution` on QuTiP's `SESolver` with `start` and `step`, registered as `unitary`, and verify
  the Rabi oscillation and conserved magnetisation scenarios
- [ ] 4.2 Add `LindbladEvolution` on `MESolver`, registered as `lindblad` and chosen by default for open systems, and
  verify the spontaneous decay and dephasing scenarios
- [ ] 4.3 Add `QuantumTrajectories` on `MCSolver` seeded from the simulation's generator, and verify the reproducible
  jumps and ensemble average scenarios
- [ ] 4.4 Pass the parameter vector to the coefficients through the solver `args`, and verify the quench by parameter
  update scenario and a scheduled drive
- [ ] 4.5 Record the states within the recording limit and the observables always, and verify the run by name
  scenario and a density matrix above the limit that records observables only

## 5. Analysis and visualization

- [ ] 5.1 Add `spectrum` and `ground_state` with dense and sparse paths and conserved sectors, and verify the
  Heisenberg ring of four spins and ground state is stationary scenarios
- [ ] 5.2 Add `purity` and `entanglement_entropy` over trajectory frames, and verify the entangled pair scenario
- [ ] 5.3 Log local expectations of quantum groups as lattice images and system observables as series, set the
  default blueprint, and verify the light cone scenario on a recording
- [ ] 5.4 Print quantum systems in operator notation through `equations_of`, and verify a docstring example of the
  XXZ Hamiltonian

## 6. Models

- [ ] 6.1 Add `quantum_heisenberg` to `skcomplex.models.physics.statistical_mechanics` as a registered factory, and
  verify the ground state of the ring, single magnon, and dephasing scenarios and the conformance suite
- [ ] 6.2 Add the `formulation` and `gamma` parameters to `schrodinger`, and verify the formulations agree and
  dephased packet scenarios
- [ ] 6.3 Mark `quantum_heisenberg` implemented in `docs/overview/inventory.toml` and drop its out-of-scope reason,
  and verify the inventory registered scenario with the inventory check

## 7. Gallery, documentation, and gate

- [ ] 7.1 Add `docs/examples/physics/statistical_mechanics/plot_quantum_heisenberg.py` with the light cone as its
  thumbnail and the ground-state energy per site against `1/4 - ln 2` as its analysis figure, and verify the gallery
  example scenario within the recording budget
- [ ] 7.2 Extend `docs/examples/physics/waves/plot_schrodinger.py` with an analysis figure of the purity of the
  dephased packet, and verify that its thumbnail stays the Rerun screenshot
- [ ] 7.3 Document quantum subsystems, operators, laws, solvers, observables, and spectra in the user guide, add a
  changelog entry, and verify the documentation test passes
- [ ] 7.4 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
