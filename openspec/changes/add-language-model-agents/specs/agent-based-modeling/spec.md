## ADDED Requirements

### Requirement: Rules query oracles
An agent rule SHALL ask an oracle for an answer with `system.query(name, request)` of the system view, and
`SystemViewProtocol` SHALL declare the method. A query SHALL return the answer of the oracle passed to the run under
that name, or the recorded answer in a replay.

#### Scenario: A rule that asks
- **WHEN** a rule stores `system.query('player', f'step {system.time}')` in an object state of its agent, and the oracle
  echoes its request
- **THEN** the state holds `'step 1'`, `'step 2'`, and `'step 3'` in the frames at times 1, 2, and 3

#### Scenario: Query outside an agent rule
- **WHEN** a symbolic law tries to query an oracle
- **THEN** compiling raises `ValidationError`, whose message says that only agent rules query oracles
