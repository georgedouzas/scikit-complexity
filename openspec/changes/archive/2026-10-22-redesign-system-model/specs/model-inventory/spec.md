# Spec Delta

## ADDED Requirements

### Requirement: Entries map to the system vocabulary
Every inventory entry SHALL declare its time kind, its space kind, its component kinds, its interaction bindings, and
its law classes from the vocabulary of `system-modeling`, implemented and pending entries alike. A check SHALL fail on any
entry that does not map. Coverage SHALL be reported per category of the vocabulary.

#### Scenario: Unmapped entry
- **WHEN** an entry lists a law class that the vocabulary does not define
- **THEN** the inventory check fails, names the entry, and asks for the general primitive to be specified

#### Scenario: Coverage by category
- **WHEN** the inventory is published
- **THEN** its coverage table counts implemented and pending entries for every time kind, space kind, component kind,
  binding, and law class
