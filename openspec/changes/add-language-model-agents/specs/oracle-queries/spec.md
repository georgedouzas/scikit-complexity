## ADDED Requirements

### Requirement: Oracles are specs the caller passes
`skcomplex.core.engine.runtime` SHALL provide `OracleSpec(respond, name, cache)`, a frozen value of a function
`respond(request, *, seed)` that returns a string or a JSON value, and `System.simulate` SHALL take `oracles`, a mapping
from the name a rule uses to an `OracleSpec`. A ready-made system SHALL declare the names its rules query in the class
variable `oracles`, and its run SHALL raise `ValidationError` before the first step naming every declared oracle that
the caller did not pass.

#### Scenario: Missing oracle
- **WHEN** a user calls `LlmNamingGameSystem(player='oracle').prepare().simulate(until=10)` without passing the oracle
  `player`
- **THEN** `simulate` raises `ValidationError` before the first step, and the message names `player`

#### Scenario: Not callable
- **WHEN** a user passes `OracleSpec(respond='gpt')` to `simulate`
- **THEN** `simulate` raises `ValidationError` before the first step, whose message asks for a function of a request and
  a seed

### Requirement: Queries are seeded and cached
A query SHALL draw its call seed from a generator spawned from the random state of the run for queries alone, and SHALL
return the stored answer when the cache holds one for the same oracle name, request, and seed, and call the oracle
otherwise.

#### Scenario: Same seed, same answers
- **WHEN** a system queries a fake oracle that returns its seed as text, twice with the same random state
- **THEN** both trajectories hold the same seeds and answers

#### Scenario: Cache on disk
- **WHEN** a run with an oracle that counts its calls and a cache file is repeated with the same random state
- **THEN** the second run makes no call, and every record of it is marked as cached

#### Scenario: Other draws unchanged
- **WHEN** a system with a named draw runs with and without one query per step at the same random state
- **THEN** the named draws of both runs are equal

### Requirement: Queries are recorded
Every query SHALL be kept in the trajectory as a `QueryRecordSpec` with the oracle name, the time, the agent, the
request, the seed, the answer, and whether the cache served it. `save_queries(trajectory, path)` and
`load_queries(path)` SHALL write and read the records as JSON lines.

#### Scenario: Records of a run
- **WHEN** a system whose one agent queries a fake oracle once per step runs 5 steps
- **THEN** the trajectory holds 5 records in the order of the steps, each with its request, seed, and answer

### Requirement: Queries are replayed
`System.simulate(replay=...)` SHALL accept a trajectory or a query log, SHALL return the recorded answers in order
without
calling any oracle, and SHALL raise `ValidationError` naming the step, the agent, and the request when a rule sends a
request that differs from the record.

#### Scenario: Replay without the oracle
- **WHEN** a run of the language-model naming game with a fake oracle is saved with `save_queries` and replayed from
  `load_queries` with no oracle passed
- **THEN** the replayed states equal the recorded states

#### Scenario: Replay that does not match
- **WHEN** a recorded run is replayed with another pool of names, so that the first prompt differs
- **THEN** `simulate` raises `ValidationError`, whose message names the first step and the agent of the differing
  request

#### Scenario: No credential in the record
- **WHEN** a run with `chat_oracle` and a fake client is saved with `save_queries`
- **THEN** the file holds no value of the environment variable that names the key
