## MODIFIED Requirements

### Requirement: Expectation values are observables
The modelling layer SHALL provide expectation values of operator expressions as observables, and reductions of local
expectations over the members of a quantum group, such as the magnetisation of every site or their sum. The quantum
package of physics, `skcomplex.systems.physics.common.quantum`, SHALL provide the energy, the purity, and the von
Neumann entanglement entropy of a set of subsystems, at every frame of a trajectory.

#### Scenario: Local magnetisation
- **WHEN** a trajectory of a chain of 8 spins is asked for the observable `sz` of its spin group
- **THEN** it returns an array of 8 real values per frame, each the expectation of the local `S^z` of one site

#### Scenario: Entangled pair
- **WHEN** the entanglement entropy of one spin of the singlet of two spins one half is computed with
  `operator_measures` of `skcomplex.systems.physics.common.quantum`
- **THEN** it equals `ln 2` within 1e-12

### Requirement: Exact diagonalisation gives spectra and ground states
The quantum package of physics, `skcomplex.systems.physics.common.quantum`, SHALL compute the lowest eigenvalues and
eigenstates of a quantum system's Hamiltonian at given parameter values, by dense diagonalisation for small spaces and
a sparse eigensolver otherwise, optionally within the sector of a conserved quantity. The ground state SHALL be usable
as the initial state of a run. The general analysis layer, `skcomplex.core.analysis`, SHALL NOT provide these
measures.

#### Scenario: Heisenberg ring of four spins
- **WHEN** the spectrum of the isotropic antiferromagnetic Heisenberg ring of 4 spins one half with `J = 1` is computed
- **THEN** its lowest eigenvalue is -2 within 1e-12

#### Scenario: Ground state is stationary
- **WHEN** a run starts from the ground state of a time-independent Hamiltonian
- **THEN** every expectation observable stays constant within 1e-8

#### Scenario: Measures of the quantum package
- **WHEN** a user imports `operator_measures` and `OperatorMeasuresResult` from
  `skcomplex.systems.physics.common.quantum`
- **THEN** both imports succeed, and `operator_measures(system, n_states=1)` of a Heisenberg ring returns its lowest
  energy
