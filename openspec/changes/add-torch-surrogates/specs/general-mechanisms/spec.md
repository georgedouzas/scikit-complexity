## ADDED Requirements

### Requirement: Gradients of equivariant network potentials
`EquivariantPotentialInteraction` SHALL move the vector states of every member of a group down the gradient of a
many-body network potential of all members, invariant to rotations, translations, and permutations of members of one
kind, given by a saved or published network and an integer kind per member, divided by an optional divisor.

#### Scenario: Gradient of the network
- **WHEN** the interaction acts on a group of 8 members with a small untrained network
- **THEN** the rate of every member equals minus the finite-difference gradient of the network potential, divided by
  its divisor, within 1e-5 relative error

#### Scenario: Rotation equivariance
- **WHEN** every state of the group is rotated by one rotation in three dimensions
- **THEN** the rates rotate by the same rotation within 1e-6 relative error

#### Scenario: Wrong number of kinds
- **WHEN** the kinds differ in length from the size of the group
- **THEN** preparing the system raises `ValidationError`, whose message names both lengths

### Requirement: Network potentials respect periodic spaces
On a periodic continuous space, the network potential of `EquivariantPotentialInteraction` SHALL read the members by
the minimum image within the cutoff of the network, and its potential SHALL be recorded as an observable of the run.

#### Scenario: Members across the boundary
- **WHEN** two members of a periodic box lie on opposite faces, closer by the minimum image than the cutoff
- **THEN** their rates equal those of the same pair placed side by side in the middle of the box

#### Scenario: Recorded potential
- **WHEN** a group with the interaction runs with velocity Verlet and no thermostat for 1000 steps
- **THEN** the trajectory holds the potential at every frame, and the total of the potential and the kinetic term drifts
  by less than 1e-3 of its initial value
