# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## Unreleased

### Features

- Every numeric law takes a function: crossing guards and resets, jump changes, transition matrices, renewal waiting
  times, reductions through `reduction`, observables through `ObservableSpec(function=...)`, and `FluxLaw(speed=...)`.
- `CallSpec` gains `relative`, `random`, and `vectorized`.
- Subclasses rename roles and parameters through `_get_role` and `_get_parameter` in the symbolic and agent forms.
- `PartnerSpec` selects partners for transfers, matching, and groupings, and a settlement credits both ends of a pair.
- Agent states with `StateSpec(domain=None)` hold hashable objects, and `StateSpec(role=...)` names headings and
  velocities in snapshots.

### Behaviour changes

- Analyses read the headings and velocities of agents from `StateSpec(role=...)` alone, not from the names of the states.
- Errors replace silent wrong results: a function as a Stratonovich amplitude, a phase of update laws with an activation,
  and grid-only variants in continuous space.

## [0.1.0](https://github.com/georgedouzas/sage-physics/releases/tag/0.1.0) - 2023-09-19

<small>[Compare with first commit](https://github.com/georgedouzas/sage-physics/compare/eb1319601a393032e7230466e82e3cebb325f57c...0.1.0)</small>

### Bug Fixes

- Use correct type definition ([23120a1](https://github.com/georgedouzas/sage-physics/commit/23120a151149dbb05f452918ecad1b13cfdfe0c0) by georgedouzas).

### Code Refactoring

- Prepare stable release ([ceadea6](https://github.com/georgedouzas/sage-physics/commit/ceadea6673406bbdae813778f1dd06f242cb5858) by georgedouzas).

### Chore

- Rename to main ([5b9b89d](https://github.com/georgedouzas/sage-physics/commit/5b9b89d918e5afe92a07ea4da337bf3213f79678) by georgedouzas).
- Remove cli files ([9ccc379](https://github.com/georgedouzas/sage-physics/commit/9ccc3799280529a616f233196da72567e2218f85) by georgedouzas).

