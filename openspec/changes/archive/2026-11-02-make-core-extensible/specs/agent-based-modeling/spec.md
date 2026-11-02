## ADDED Requirements

### Requirement: Agent states take object values
An agent state SHALL be able to hold a hashable object, such as a tuple, a string, or a frozen set, and the run SHALL
keep it in snapshots, count it in compositions, and record it in observable histories.

#### Scenario: Queue of orders
- **WHEN** an agent state holds a tuple of pending order identifiers that rules append to and pop from
- **THEN** each snapshot keeps the tuples as they were, and the composition counts agents by tuple

#### Scenario: Analysis of an object state
- **WHEN** an analysis that needs numbers reads an object state
- **THEN** it raises an error naming the state and saying it holds objects
