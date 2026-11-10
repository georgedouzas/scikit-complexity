## ADDED Requirements

### Requirement: Basin stability starts count states from integers

Basin stability SHALL draw each start of a state whose domain holds only integers from the integers within the given
bounds, and each start of a real state from the real interval.

#### Scenario: Basin stability of a counting process

- **WHEN** basin stability is run on a system whose state counts events, with bounds from 0 to 5
- **THEN** every start of that state is an integer from 0 to 5 and the runs finish without a domain error

#### Scenario: Real states are unchanged

- **WHEN** basin stability is run on a double well with real states and a fixed seed
- **THEN** the starts and the fractions equal those drawn before this change
