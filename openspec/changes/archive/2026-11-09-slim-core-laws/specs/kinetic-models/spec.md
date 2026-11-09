## MODIFIED Requirements

### Requirement: Encounters derive gain and loss equations
Bilinear laws SHALL derive the gain and loss equations of the kinetic theory from encounter rates and transition
probabilities. A conserved bilinear law SHALL take the loss from its first density, so conservative encounters keep the
density constant. Proliferative and destructive encounters SHALL change it as declared. Bilinear laws SHALL be of the
law kind `rate`, in the continuous class, and SHALL declare that they require continuous time and a lattice space.

#### Scenario: Conservative encounters
- **WHEN** a wealth system with only conservative encounters runs for 100 time units
- **THEN** the total density stays constant within 1e-10 relative error

#### Scenario: Invalid transition
- **WHEN** a transition probability sums to 0.9 over the outcomes of one encounter
- **THEN** validation reports an issue whose path is the encounter and whose message names the activities and the sum,
  before compilation

#### Scenario: Wrong substrate
- **WHEN** a bilinear law is placed in a system with discrete time
- **THEN** validation reports that it needs continuous time and suggests declaring `ContinuousTime()`

#### Scenario: Conserved law without outcomes
- **WHEN** a conserved bilinear law is given no outcome table
- **THEN** preparing it raises a `ValidationError` that asks for the outcome probabilities
