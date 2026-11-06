# Proposal

## Why

Three public verbs make a modeller learn a step that carries no decision: 3,799 of the 5,043 `.check()` calls in the
repository are written as `.check().prepare(...)`. scikit-learn validates its parameters inside `fit`, with no public
validation step. Two verbs, `prepare` and `run`, give one rule: build it, prepare it, run it.

## What Changes

- **BREAKING** Remove the public `check()` from `BaseConfig`, `BasePart`, `BaseSpec`, and `ModelEntry`, with no alias.
- `prepare` validates first through a private `_validated`, which runs the `_check` hook, so a bad setting raises the same `ValidationError` from `prepare`.
- **BREAKING** Remove `NotCheckedError`. `run` on an unprepared object still raises `NotPreparedError`.
- A spec, a law, an element, or an interaction held by another object is validated when its owner is prepared, so a bad
  spec fails at `prepare`, not where it was written. This later failure is the accepted cost of the change.
- Inside the library, an owner validates a part it does not prepare by calling the part's `_validated`.
- Migrate `src`, `tests`, the guide, the README, the gallery, doctests, the specs, and the constitution.
- No result of any model or analysis changes.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `base-interfaces`: the verbs, the constructors, the values, the results, and the rule that running needs preparing.
- `analysis-interface`: analyses have two verbs and validate in `prepare`.
- `experiments`: experiments and designs have two verbs and validate in `prepare`.
- `system-modeling`: parts have two verbs; a Hamiltonian interaction reports an unpaired coordinate from `prepare`.
- `package-layout`: engine classes validate in `prepare`, and engine dataclasses have no `check()`.
- `higher-order-interactions`: a wrong grouping setting is reported from `prepare`.
- `learning-dynamics`: an unknown optimizer is reported from `prepare`.

## Impact

- **Public API:** every call to `check()` breaks, and a release note must say so. The hook `_check` that a user
  implements in a subclass stays the same.
- **Code:** the base classes, about 320 non-doctest calls in `src`, 2,596 in `tests`, and 1,447 in `docs`, including
  all 567 gallery files that call it, whose cached gallery runs are invalidated.
- **Documents:** `README.md`, the user guide, `openspec/constitution.md`, and the main specs.
- **Dependencies:** none.
