## ADDED Requirements

### Requirement: Parts declare capability tags
Every element, interaction, and law SHALL return its capability tags from a plain method `get_tags()`, as a frozen
`Tags` value with these fields: `time` (continuous, discrete, static), `domains` (real, integer, boolean, categorical,
complex, vector, matrix group, power series), `space` (none, continuous, lattice, network, hypergraph, cell complex,
phase space, curved spacetime), `laws` (rate, noise, jump, update, equation, sampling, topology), `randomness`
(deterministic, stochastic), `structure` (fixed, variable membership, changing topology), and flags such as
`differentiable`. Tags SHALL depend only on the parameters of the part, so they are known before `prepare`.

#### Scenario: Tags of an element
- **WHEN** `get_tags()` is called on an element with a boolean state on a lattice
- **THEN** its tags hold the domain `boolean` and the space `lattice`, before the element is prepared

#### Scenario: Tags that follow a parameter
- **WHEN** an interaction whose `kind` parameter selects a jump law is given `kind='stochastic'`
- **THEN** its tags hold the law kind `jump` and the randomness `stochastic`

### Requirement: Systems derive their tags
A `System` SHALL derive its tags from its parts and SHALL NOT declare them: time and space SHALL agree across its parts,
law kinds and domains SHALL be the union of its parts', and randomness SHALL be stochastic when any part is. Parts
whose time or space disagree SHALL make `get_tags()` raise an error that names the parts and the tag.

#### Scenario: A stochastic system
- **WHEN** a system holds a rate interaction and a noise interaction
- **THEN** its tags hold the law kinds `rate` and `noise` and the randomness `stochastic`

#### Scenario: Parts that disagree
- **WHEN** a system holds an element on a lattice and an element on a network with no space that joins them
- **THEN** `get_tags()` raises an error that names both elements and the tag `space`

### Requirement: Roles accept tags
An interaction SHALL declare, per role, the tags of the elements it accepts, and `prepare` SHALL reject an element
whose tags it does not accept, with a message that names the role, the element, and the tag.

#### Scenario: A rate law on a boolean state
- **WHEN** a rate interaction binds an element whose state has the domain `boolean`
- **THEN** preparing the system raises an error that names the role, the element, and the domain `boolean`

### Requirement: Functions read tags
`simulate`, solvers, backends, analyses, and experiments SHALL declare the tags they support or need, and SHALL read
the tags of the system or of the trajectory before doing any work. `simulate` SHALL choose the default solver from the
tags, and SHALL reject a solver or a backend that does not support them, naming the tag.

#### Scenario: A solver that does not fit
- **WHEN** `simulate` is called with an ODE solver on a system whose tags hold the law kind `jump`
- **THEN** it raises an error naming the solver and the law kind `jump` before compiling

#### Scenario: An analysis that needs a deterministic system
- **WHEN** a Lyapunov analysis is given a trajectory of a system whose randomness is `stochastic`
- **THEN** it raises an error that names the tag `randomness`

### Requirement: Capability tags are separate from catalog topics
Capability tags SHALL be general and SHALL live in the core. Topical tags of the catalog, such as `epidemic` or
`chaos`, SHALL stay catalog metadata of `skcomplex.systems`, and the catalog SHALL show and filter by both.

#### Scenario: Filtering the catalog by capability
- **WHEN** a user filters the catalog for discrete-time stochastic systems on networks
- **THEN** the listing holds every such system and no other, whatever its topical tags
