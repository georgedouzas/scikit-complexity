## MODIFIED Requirements

### Requirement: Systems form a class hierarchy
`BaseSystem` SHALL be the base of every system: it SHALL hold `prepare`, `run`, `get_tags()` derived from the parts,
nesting, and the passing of model parameters down to the parts. `CompositeSystem(BaseSystem)` SHALL be the generic composite of
elements, interactions, and nested systems. A class SHALL NOT be chosen by mathematical form, which the tags give.

#### Scenario: Generic composite
- **WHEN** a user builds `CompositeSystem(elements=[...], interactions=[...])`
- **THEN** it is a `BaseSystem`, and its tags come from its parts
