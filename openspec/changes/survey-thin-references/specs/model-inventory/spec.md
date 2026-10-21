# Spec Delta

## ADDED Requirements

### Requirement: Every reference source is surveyed section by section
Every textbook and paper in `references` SHALL be surveyed section by section, and the survey SHALL be recorded with the
sections read, the models found, and the primitives each exercises, so a source with few inventory entries is a source
with few models and not an unread one.

#### Scenario: Thinly cited textbook
- **WHEN** a contributor reads the survey record of a textbook cited by four inventory entries or fewer
- **THEN** it lists every chapter read and, for each, the models recorded or the reason none was

#### Scenario: Paper in the references
- **WHEN** a paper is added to `references/papers`
- **THEN** its models are in the inventory under its arXiv identifier, or its survey record says why it has none
