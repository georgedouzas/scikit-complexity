# Spec Delta

## MODIFIED Requirements

### Requirement: Every analysis works in the default install
Every analysis SHALL work in the default install, those backed by a copyleft library included, because every library
the core uses is a runtime dependency.

#### Scenario: Default install
- **WHEN** a user lists the analyses of a jump path in the default install
- **THEN** the Markov state model is listed under the path kind and estimates a model from the path
