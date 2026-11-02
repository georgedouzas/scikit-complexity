# Spec Delta

## ADDED Requirements

### Requirement: Quantum spin chain models
The library SHALL provide the quantum Heisenberg model (Mihailović et al. 2023 §7.3) under
`physics/statistical_mechanics`: a chain of spins one half with the XXZ Hamiltonian
`J Σ (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j) - h Σ S^z_i` over neighbouring pairs, optional dephasing at a rate
`gamma`, and an initial Néel state, a single flipped spin, a domain wall, or the ground state.

#### Scenario: Ground state of the ring
- **WHEN** the spectrum of `quantum_heisenberg` with 4 periodic sites, `J = 1`, `Δ = 1`, and `h = 0` is computed
- **THEN** its lowest eigenvalue is -2 within 1e-12

#### Scenario: Single magnon spreads as a Bessel light cone
- **WHEN** `quantum_heisenberg` runs with 12 periodic sites, `J = 1`, `Δ = 0`, and one flipped spin at site 6, to
  time 2
- **THEN** at every frame the probability `1/2 - <S^z_{6+d}>` of finding the flip at distance `d` equals
  `J_d(J t)²` within 1e-5, and the total magnetisation stays constant within 1e-8

#### Scenario: Dephasing destroys coherence
- **WHEN** `quantum_heisenberg` runs with 6 periodic sites from a Néel state with `gamma = 0.1`
- **THEN** the solver is the Lindblad solver, the purity never increases from frame to frame and ends below 1, and
  the total magnetisation stays constant within 1e-8

### Requirement: Operator formulation of the Schrödinger model
The `schrodinger` model SHALL accept a `formulation` of `'fields'`, the default, which keeps its two real fields, or
`'operators'`, which builds a particle on the same lattice as a single-particle Hilbert space with the kinetic operator
of the lattice's spectral Laplacian and the harmonic potential as a diagonal operator, and accepts position dephasing
at a rate `gamma`. Both formulations SHALL register under the one name `schrodinger` in `physics/waves`.

#### Scenario: Formulations agree
- **WHEN** `schrodinger` runs with 128 sites to time 4 once with each formulation
- **THEN** the probability of every site, `(u² + w²) h` of the fields with the spacing `h` and `|psi|²` of the
  operators, agrees within 1e-6 at every frame

#### Scenario: Dephased packet
- **WHEN** `schrodinger` runs with the operator formulation and `gamma = 0.5`
- **THEN** the trace stays 1 within 1e-10 and the purity never increases from frame to frame

### Requirement: Quantum models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as out of scope or blocked by this change SHALL register under the name of
its entry. A model that the library already implements SHALL gain the new formulation under the same name, not be
registered twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `quantum_heisenberg` in `physics/statistical_mechanics` (Mi 7.3)
  - `schrodinger` in `physics/waves` (Sc 4), with the operator formulation

#### Scenario: Gallery example of a quantum model
- **WHEN** the gallery runs the example `docs/examples/physics/statistical_mechanics/plot_quantum_heisenberg.py`
- **THEN** it creates `quantum_heisenberg` by name, streams the run to Rerun within the recording budget, its
  thumbnail is the space-time image of the local magnetisation with a light cone, and its analysis figure compares the
  ground-state energy per site with `1/4 - ln 2`
