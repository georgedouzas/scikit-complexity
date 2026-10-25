## ADDED Requirements

### Requirement: Rewriting rules are laws of the rewritten component
`Rewriting` and `Rewrite` SHALL derive from `BaseLaw`, and a word, patch, or host graph SHALL return its rewriting
rules from `laws`. No rewriting rule SHALL be an interaction.

#### Scenario: Rule of a word
- **WHEN** a word returns a `Rewrite` of the productions `A -> AB` and `B -> A` from `laws`, and runs for five steps
- **THEN** the word is `ABAABABAABAAB`, and `Rewrite` is not a subclass of `BaseInteraction`
