# Spec Delta

## ADDED Requirements

### Requirement: Coupling of separate elements and hubs
`CouplingInteraction` SHALL couple a set of separate elements to each other, and a single hub element to a group by
any reduction.

#### Scenario: Global inhibition
- **WHEN** a hub element reads the largest state of a lattice group and inhibits every member
- **THEN** it uses `CouplingInteraction`, and the trajectory equals the previous model's

### Requirement: Parametric stoichiometry
`ReactionInteraction` SHALL take stoichiometric coefficients that are parameters.

#### Scenario: Oregonator
- **WHEN** the Oregonator's reactions change `x` and `y` scaled by `1 / epsilon` and `1 / delta`
- **THEN** it uses `ReactionInteraction`, and its limit cycle equals the previous model's

### Requirement: Transfers along edges
`TransferInteraction` SHALL move amounts along the edges of a lattice or network in both directions at parametric
rates, with one draw per edge shared by both ends.

#### Scenario: Parallel exclusion process
- **WHEN** particles hop right on a ring under parallel update with exclusion
- **THEN** the current equals the exact parallel-update current, and no site holds two particles

### Requirement: Visits drawn with replacement
The agent forms SHALL run a phase as N visits by members drawn with replacement, each composing the member
interactions of the visited member.

#### Scenario: Perishables market
- **WHEN** the perishables market runs on the general classes
- **THEN** its magnetization over 20 seeds equals the previous model's at every temperature tested

### Requirement: Agent coupling over radii
The agent form of `CouplingInteraction` SHALL sum terms over several radii with a priority among them and clamp the
resulting turn and speed.

#### Scenario: Couzin zones
- **WHEN** fish repel within one radius, align within a second, and attract within a third
- **THEN** repulsion overrides the other zones, and the polarization over 20 seeds equals the previous model's

### Requirement: Rules read observable history
An agent rule SHALL read the recorded values of an observable at earlier rounds.

#### Scenario: Attendance history
- **WHEN** a choice reads the bar attendance of the previous rounds
- **THEN** no reporter-only interaction records it
