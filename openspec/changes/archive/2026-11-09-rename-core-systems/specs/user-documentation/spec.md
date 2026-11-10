## MODIFIED Requirements

### Requirement: The documentation explains the system hierarchy
The documentation SHALL hold a systems page that explains `BaseSystem`, the composite `CompositeSystem`, general systems of the
core, and field realizations, with an executed example of a subclass that changes a ready-made system. The catalog
pages SHALL show, for each system, its class, its import path, its model parameters, and its capability tags.

#### Scenario: A catalog page
- **WHEN** a reader opens the page of `SIRSystem`
- **THEN** it shows `from skcomplex.systems.biology.epidemiology import SIRSystem`, the parameters, and the tags
