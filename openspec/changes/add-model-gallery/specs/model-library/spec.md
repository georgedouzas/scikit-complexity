# Spec Delta

## ADDED Requirements

### Requirement: Every registered model has a gallery example
Every model registered in the library SHALL ship with a gallery example and a reviewed screenshot. A change that adds or
rebuilds a model SHALL add or update its example in the same change, and a model whose screenshot cannot show its
behaviour within the gallery budget SHALL say so in its caption and show the closest faithful view.

#### Scenario: New model without an example
- **WHEN** a change registers a new model and adds no example
- **THEN** the gallery coverage check fails and names the model

#### Scenario: Behaviour beyond the budget
- **WHEN** a model's known behaviour, such as a 100 thousand year glacial cycle, needs a longer run than the budget
- **THEN** its example runs it with coarser frames within the budget, and its caption states what the image shows
