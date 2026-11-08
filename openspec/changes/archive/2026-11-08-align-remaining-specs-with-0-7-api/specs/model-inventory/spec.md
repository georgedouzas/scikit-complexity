## MODIFIED Requirements

### Requirement: Systems exist to exercise the core
The library's systems SHALL serve first to reveal what the core needs and to exercise its general primitives, and only
second as ready-made examples. Every inventory entry SHALL record the core primitives it exercises, named from a
vocabulary the inventory defines, and the inventory check SHALL reject a primitive outside that vocabulary.

#### Scenario: Primitives of an entry
- **WHEN** a contributor reads the inventory entry of `BakSneppenSystem`
- **THEN** it lists the lattice and extremal-update primitives it exercises

#### Scenario: Unknown primitive
- **WHEN** an entry names a primitive that the vocabulary does not define
- **THEN** the inventory check fails and names the entry and the primitive
