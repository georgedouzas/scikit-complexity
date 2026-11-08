## ADDED Requirements

### Requirement: Ready-made systems are imported from their subfield
Every ready-made system SHALL be a public class of its subfield package, imported as
`from skcomplex.systems.<field>.<subfield> import <Name>System`, and SHALL carry its field, topics, references, and
tuned run settings as class attributes. `simulate` SHALL use the tuned run settings for every argument the call leaves
out.

#### Scenario: Direct use
- **WHEN** a user runs `simulate(SIRSystem())` after `from skcomplex.systems.biology.epidemiology import SIRSystem`
- **THEN** the run uses the class's tuned end time, solver hint, and observables

### Requirement: Systems are discovered with all_systems
`all_systems(field=None, topics=(), tags=None)` SHALL return the name and class of every ready-made system, in the role
of `sklearn.utils.all_estimators()`. Selecting by field or topics SHALL read the class definitions statically and import
only the modules of the systems it returns. Filtering by capability tags SHALL import the systems it filters.

#### Scenario: Listing by field without imports
- **WHEN** a user calls `all_systems(field='physics/mechanics')` in a fresh interpreter
- **THEN** the systems of that field are returned, and no module of another subfield is imported

#### Scenario: Filtering by capability
- **WHEN** a user calls `all_systems(tags={'randomness': 'stochastic', 'space': 'network'})`
- **THEN** every listed system has those tags

## REMOVED Requirements

### Requirement: Using a system imports only its module
**Reason**: Systems are classes imported from their subfield, so there is no creation by name.
**Migration**: Import the system class from its subfield and construct it with the model's parameters.

### Requirement: The listing agrees with the registrations
**Reason**: There are no registrations. A system class is found by `all_systems()`.
**Migration**: Declare field, topics, and references as class attributes of the system class.

### Requirement: A catalog simulation carries the tuned settings
**Reason**: `create_simulation` is removed. The tuned settings are class attributes that `simulate` reads.
**Migration**: Call `simulate(SystemClass(**params))`, overriding any tuned setting by argument.
