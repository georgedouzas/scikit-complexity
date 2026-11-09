## MODIFIED Requirements

### Requirement: Ready-made systems are classes
Every ready-made system SHALL be a class on `BaseSystem`, or on a general core system such as `ReactionNetworkSystem`,
`DynamicalSystem`, `NetworkDynamicalSystem`, `ReactionDiffusionSystem`, or `StockFlowSystem`, in the module of its
subfield. It SHALL take its model parameters as arguments of its constructor and SHALL give its field path, topics,
references, and tuned run settings as class attributes. `all_systems()` SHALL find every such class.

#### Scenario: Browse the catalog
- **WHEN** a user calls `all_systems(field='biology')` or `all_systems(field='biology/ecology')`
- **THEN** the result holds the name and class of every system of that field or subfield, and each class gives its
  field, topics, and references as class attributes

#### Scenario: Create by name
- **WHEN** a user builds `LotkaVolterraSystem(a=0.2)`, imported from `skcomplex.systems.biology.ecology`
- **THEN** the result is a system that `simulate` runs like the same system built from its components, to the tuned
  end time of its class

## ADDED Requirements

### Requirement: Ready-made systems use the general system of their shape
A ready-made system whose structure is that of a general core system SHALL subclass it, unless its element class is a
part that other systems or users compose, such as an element of a field `common` package. Its own element class SHALL
then leave the catalog, and its equations SHALL move into the hooks of the system.

#### Scenario: Map of one field
- **WHEN** a contributor reads `HenonMapSystem` of `mathematics/dynamical_systems`
- **THEN** it subclasses `DynamicalSystem` in discrete time, and the module defines no element class for it

#### Scenario: Shared element
- **WHEN** an element class such as the Lorenz element of `mathematics/common/flows` serves several systems
- **THEN** the systems that compose it keep it, and none of them rebases onto `DynamicalSystem`

### Requirement: Rebasing keeps a system's interface and results
Rebasing a ready-made system onto a general system SHALL keep its public name, its import path, its constructor
parameters with their defaults, its state and observable names, its tags, and its trajectory. A rebase that changes any
of these SHALL NOT be made, and the system SHALL keep its base.

#### Scenario: Same trajectory
- **WHEN** a rebased system is simulated with its defaults, `random_state=0` where it takes one, and its tuned run
  settings, before and after the rebase
- **THEN** both trajectories hold the same names and equal values

#### Scenario: Same parameters
- **WHEN** a user calls `get_params()` on a rebased system
- **THEN** the result equals the one before the rebase
