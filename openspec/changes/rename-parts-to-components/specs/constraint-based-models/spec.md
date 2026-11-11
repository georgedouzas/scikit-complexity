## MODIFIED Requirements

### Requirement: Switches gate variables through adaptive parameters
A bound SHALL be able to read adaptive parameters with the domain `{0, 1}` through `either` and `both`, so that a
knock-out sets a parameter to 0 at construction, between runs, or by an `UpdateLaw` of another component.

#### Scenario: Isozymes
- **WHEN** a reaction is bounded by `1000 * either(g1, g2)` and only `g1` is switched off
- **THEN** the reaction keeps its bound, and switching off `g2` as well sets it to zero

#### Scenario: Live knock-out
- **WHEN** the switch of an essential reaction is set to 0 between two runs of a program
- **THEN** the objective of the second run is zero within 1e-9

#### Scenario: Regulation gates a reaction
- **WHEN** an `UpdateLaw` sets the switch of a reaction from a Boolean node that represses it, and the node is on
- **THEN** the next solve has that reaction at zero, and after the node turns off the reaction may carry flux
