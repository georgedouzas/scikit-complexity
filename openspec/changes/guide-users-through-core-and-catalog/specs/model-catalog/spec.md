# Spec Delta

## ADDED Requirements

### Requirement: The documentation holds a browsable catalog
The documentation SHALL hold an index of every registered model by field and tag, with its description, generated from
`catalog()`. Each model page SHALL show how to select the model by name and list the element and interaction classes its system
uses.

#### Scenario: Index follows the catalog
- **WHEN** a model is registered
- **THEN** the next documentation build lists it in the index under its field and its tags

#### Scenario: Model page
- **WHEN** a reader opens the page of `lotka_volterra`
- **THEN** it shows `create('lotka_volterra')`, and lists the element and interaction classes of its system
