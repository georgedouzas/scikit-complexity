## ADDED Requirements

### Requirement: Systems derive their tags from their components
A `CompositeSystem` SHALL derive its tags from its components and SHALL NOT declare them: time and space SHALL agree
across its components, law kinds and domains SHALL be the union of those of its components, and randomness SHALL be
stochastic when any component is. Components whose time or space disagree SHALL make `get_tags()` raise an error that
names the components and the tag.

#### Scenario: A stochastic system
- **WHEN** a system holds a rate interaction and a noise interaction
- **THEN** its tags hold the law kinds `rate` and `noise` and the randomness `stochastic`

#### Scenario: Components that disagree
- **WHEN** a system holds an element on a lattice and an element on a network with no space that joins them
- **THEN** `get_tags()` raises an error that names both elements and the tag `space`

## RENAMED Requirements

- FROM: `### Requirement: Parts declare capability tags`
- TO: `### Requirement: Components declare capability tags`

## REMOVED Requirements

### Requirement: Systems derive their tags
**Reason**: Its text and a scenario name called the building blocks of a system parts, and a scenario cannot be renamed in place.
**Migration**: See the requirement "Systems derive their tags from their components", which replaces it.

## MODIFIED Requirements

### Requirement: Components declare capability tags
Every element, interaction, and law SHALL return its capability tags from a plain method `get_tags()`, as a frozen
`Tags` value with these fields: `time` (continuous, discrete, static), `domains` (real, integer, boolean, categorical,
complex, vector, matrix group, power series), `space` (none, continuous, lattice, network, hypergraph, cell complex,
phase space, curved spacetime), `laws` (rate, noise, jump, update, equation, sampling, topology), `randomness`
(deterministic, stochastic), `structure` (fixed, variable membership, changing topology), and flags such as
`differentiable`. Tags SHALL depend only on the parameters of the component, so they are known before `prepare`.

#### Scenario: Tags of an element
- **WHEN** `get_tags()` is called on an element with a boolean state on a lattice
- **THEN** its tags hold the domain `boolean` and the space `lattice`, before the element is prepared

#### Scenario: Tags that follow a parameter
- **WHEN** an interaction whose `kind` parameter selects a jump law is given `kind='stochastic'`
- **THEN** its tags hold the law kind `jump` and the randomness `stochastic`
