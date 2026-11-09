## MODIFIED Requirements

### Requirement: Systems derive their tags
A `CompositeSystem` SHALL derive its tags from its parts and SHALL NOT declare them: time and space SHALL agree across its parts,
law kinds and domains SHALL be the union of its parts', and randomness SHALL be stochastic when any part is. Parts
whose time or space disagree SHALL make `get_tags()` raise an error that names the parts and the tag.

#### Scenario: A stochastic system
- **WHEN** a system holds a rate interaction and a noise interaction
- **THEN** its tags hold the law kinds `rate` and `noise` and the randomness `stochastic`

#### Scenario: Parts that disagree
- **WHEN** a system holds an element on a lattice and an element on a network with no space that joins them
- **THEN** `get_tags()` raises an error that names both elements and the tag `space`
