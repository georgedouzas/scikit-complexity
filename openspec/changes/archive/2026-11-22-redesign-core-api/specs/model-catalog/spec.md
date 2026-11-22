## MODIFIED Requirements

### Requirement: Ready-made systems are imported from their subfield
Every ready-made system SHALL be a public class of its subfield package, imported as
`from skcomplex.systems.<field>.<subfield> import <Name>System`, and SHALL carry its field, topics, references, and
tuned run settings as class attributes. `simulate` SHALL use the tuned run settings for every argument the call leaves
out.

#### Scenario: Direct use
- **WHEN** a user runs `SIRSystem().prepare().simulate()` after
  `from skcomplex.systems.biology.epidemiology import SIRSystem`
- **THEN** the run uses the class's tuned end time, solver, and observables
