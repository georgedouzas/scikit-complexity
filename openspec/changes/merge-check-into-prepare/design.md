# Design

## Context

Observed in the code on 2026-11-05:

- `BaseConfig` in `src/skcomplex/base.py` defines `check`, `prepare`, and `run`. `check` calls `_check` and sets
  `_checked`, and `prepare` and `run` raise `NotCheckedError` without it.
- `BaseSpec` defines its own `check` and `prepare` and skips the guards, since a frozen value cannot be marked.
- `BasePart.check` in `src/skcomplex/core/base.py` also validates the label and every declared spec.
- `ModelEntry` is not a `BaseConfig` but has a `check`, which `register_model` calls in `models/_catalog.py`.
- `System._check` calls `check` on every part, and other owners do the same for the specs and laws they hold.
- `report_.check()` in `engine/compile/_base.py` raises when the report of a compiler holds an error.
- There are 398 `_check` hooks, which stay as they are.
- `prepare()` with no argument works on a lone interaction, element, spec, system, analysis, and simulation. A compiler
  without a system raises a `ValidationError` naming the missing system.

## Goals / Non-Goals

**Goals:**

- Exactly two public verbs on every configurable class.
- The same errors with the same messages, raised from `prepare`.
- No object is partly prepared when validation fails.

**Non-Goals:**

- Changing what any `_check` validates.
- Collecting validation failures from `_check` into the compiler's report.
- Changing a result of any model or analysis.
- Validating in `__init__`, which stays a plain store of its arguments.

## Decisions

**`prepare` validates first, through `_validated`.** `BaseConfig` gets a private `_validated`, which calls `self._check()`
and returns the object, and `prepare` calls it before `self._prepare(context)`. `BasePart` overrides it to validate the
label and the declared specs first, since `BasePart.check` did this outside the `_check` hook that subclasses implement.
`_check` stays the hook a class implements, and `_validated` is what a caller runs. `BaseSpec.prepare` calls it too, so a
lone spec has one way to validate. Setting a parameter and preparing again therefore revalidates it.

**Owners validate before preparing anything.** `System._check` keeps validating every part, so a bad interaction raises
before any element is prepared. A part is validated again when it is prepared, which is harmless, and task 1.4 measures
the cost.

**Inside the library, an owner calls `part._validated()`.** An owner often validates something it will not prepare, such
as a spec field, a law, or the report of a compiler, and calling `prepare` on a part may set attributes. So a call to
`.check()` in `src` becomes `._validated()` on the same object. The alternative, a public validation method under another
name, keeps three verbs. `ModelEntry` gets a private `_validated` on the same rule.

**Removing `NotCheckedError`.** Nothing can be unvalidated once `prepare` validates, so the class and its guards go.
`_is_prepared` remains, and so do its exemption for specs and its `NotPreparedError`.

**The migration is mechanical first.** A script in the scratchpad, not the repository, applies these rewrites, multi-line
chains included, and the remaining cases are fixed by hand:

| Before | After |
|---|---|
| `X.check().prepare(y)` | `X.prepare(y)` |
| `X.check().run(...)`, expecting `NotPreparedError` | `X.run(...)` |
| `pytest.raises(...)` around `X.check()` | `X.prepare()` |
| `X.check().attribute` in a doctest | `X.attribute` |
| `y = X.check()` where `y` is used later | `y = X` |
| `part.check()` in `src` | `part._validated()` |

A test that only asserts a valid object passes `check()` becomes `prepare(...)` with what the object needs.

## Risks / Trade-offs

- A bad spec fails later than where it was written → the error names the part and the argument, as it does today, and
  the user accepted this.
- A test that expected a failure from `check` could now pass or fail for another reason, such as a missing context →
  `pytest.raises` keeps its `match`, and `prepare` validates before it reads the context.
- `prepare` now validates in places that call it often, such as a law prepared during compilation → task 1.4 measures
  compile time before and after on the catalog tests.
- A blunt rewrite could hit a `check` of another library or a generated source → task 2.1 reviews every call whose
  receiver is not in the library, and the translator and printer sources are read by hand.

## Migration Plan

This is a breaking change for the next release, with no alias, and the release notes give the one-line rule: delete
`.check()`, and call `.prepare(...)` where `check()` stood alone.

## Open Questions

None.
