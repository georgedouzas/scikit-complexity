## ADDED Requirements

### Requirement: Rules query oracles
A rule of an agent, `rule(agent, system)`, or of the whole system, `rule(system)`, SHALL ask an oracle for an answer
with `system.query(name, request)` of the system view, and `SystemViewProtocol` SHALL declare the method. A query SHALL
return the answer of the oracle passed to the run under that name, or the recorded answer in a replay.

#### Scenario: A rule that asks
- **WHEN** a rule stores `system.query('player', f'step {system.time}')` in an object state of its agent, and the oracle
  echoes its request
- **THEN** the state holds `'step 1'`, `'step 2'`, and `'step 3'` in the frames at times 1, 2, and 3

#### Scenario: Query of a name the run was not given
- **WHEN** a rule of a plain `System` queries the oracle `'judge'` in a run given only the oracle `'player'`
- **THEN** the query raises `ValidationError`, whose message names `'judge'` and the oracles of the run
