# Design

## Context

See proposal.md for the motivation. The current state that shapes the approach:
- `BaseConfig[ContextT, InputT]` in `skcomplex/base.py` defines the verbs `check()`, `prepare(context)`, and
  `run(inputs, final=False) -> object`, and the hooks `_check`, `_prepare`, `_run`, and `_release`, all with no-op
  bodies. Its 22 direct children are listed in `skcomplex.base`'s import graph. `BasePart`, `BasePrinter`,
  `BaseLowering`, and `BaseSolver` pass type parameters on to their children.
- 192 classes implement `_run`:
  - 67 return `None`;
  - 57 return `Self`;
  - 68 return a value: laws return `object`, kernels return arrays, records, or tuples, and printers return code or
    a compiled function.
- 140 classes implement `_prepare`, and most return `self`.
- `BaseSpec` already overrides `check()` and `_is_checked`, because a frozen spec cannot be marked as checked.
- `Simulation`, `BaseAnalysis`, `BaseObserver`, `BaseClock`, and the Rerun visual override the public `run` to name
  their input, and already return `Self`.

## Goals / Non-Goals

**Goals:**
- A class that skips `_check`, `_prepare`, or `_run` cannot be created, except a spec.
- `run()` returns the object everywhere, and every result has one attribute name per family.
- mypy checks the result attributes, so no caller narrows a run's result with `isinstance`.

**Non-Goals:**
- Changing what any class computes, or its `prepare` context and `run` input types.
- Removing the named-input overrides of `run`, which the user chose to keep.
- Typing the `prepare` and `run` inputs without `| None`, which needs type parameter defaults (Python 3.13).

## Decisions

### Hooks return `None`, verbs return `self`
`_check() -> None`, `_prepare(context) -> None`, and `_run(inputs) -> None`. The verbs call the hook and return
`self`. Hooks that return `self` today drop the return.

Alternative considered: hooks return `Self` so they can be chained inside a class. No caller chains hooks, and two
return conventions would be a second way.

### The result attribute is declared on the family's base
Each family base declares its result as a class-level annotation with a docstring entry under `Attributes`, for
example `state_: Array` on `BaseSolver` and `problem_: BaseProblemSpec` on `BaseCompiler`.

Where children produce different types, the base is generic in the result's type, or a child narrows the
annotation:
- `BaseLowering[LoweredT]` declares `lowered_: LoweredT`. Its current type parameter, misnamed `InputT`, already
  names the output.
- `BaseKernel` declares `values_: object`, and each kernel narrows it, such as `values_: Array` on `EncounterKernel`
  and `values_: Sparse` on `JacobianKernel`.
- `BasePrinter` declares `code_: object`, and each printer narrows it. `ExpressionPrinter` sets either the printed
  assignments or the compiled function, depending on its `compiled` argument.

Alternative considered: a third type parameter `OutputT` with a typed return. The user chose the scikit-learn style.

### Laws and parts set attributes on themselves
Laws are classes, not frozen dataclasses, so `run` sets `value_` on the law. Elements, interactions, and systems
set `laws_`. Kernels and the relaxation and renewal solvers call `law.run(state)` once per step and read `value_`
right after it. Nothing reads a law's result later, so keeping only the last result is safe.

### Specs keep default hooks
`BaseSpec` implements `_check` (accepts every field), `_prepare`, and `_run` (do nothing). The 19 specs that validate
fields keep their `_check`. This is the only base with default hooks, by the user's decision.

### Named-input overrides forward to the base verb
`Simulation.run(until, final)`, `BaseObserver.run(frame, final)`, `BaseClock.run(elapsed, final)`, the Rerun
visual's `run(frame, final)`, and `BaseAnalysis.run(inputs, final)` call `super().run(...)` and return `self`. They
add no behaviour of their own.

### The `ABC` stays on `BaseConfig`
Children that are bases of their own family, such as `BaseAnalysis` and `BaseDesign`, keep `ABC` in their bases only
where they add abstract methods of their own. Otherwise the inherited `ABCMeta` suffices, and the extra `ABC` goes.

### Generated source is loaded by a `SourceLoader`
Chosen by the user. `SourceLoader(BaseConfig[str | Path | None, str])` holds the logic of today's
`BaseConfig._load_source` and the constant `CACHE_DIR`, unchanged at `~/.cache/skcomplex/kernels`. Its lowest user
is `ExpressionPrinter`, so the layer contract places it in `skcomplex.engine.problems`. The injected functions that
`_load_source` takes today are an argument of its constructor. `NumbaBackend`, `ExpressionPrinter`,
`LatticeProblemPass`, and `LyapunovAnalysis` hold one where they load source.

## Risks / Trade-offs

- **[Risk] A missed caller reads the return value, now `self`, and fails at run time rather than at import.**
  → mypy types `run()` as returning `Self`, so a caller that treats the result as an array, a problem, or a list is
  a type error. The full test suite and doctests run after each layer.
- **[Risk] Hot loops slow down.** Kernels and laws run once per step. → Setting and reading an attribute costs a
  dictionary write and read. A profile of a long agent and lattice run before and after must show no regression
  beyond noise.
- **[Risk] A kernel or law shared between two callers in one step overwrites its own result before it is read.** →
  Every call site reads the attribute right after the call, with no call in between. The review of each layer
  checks this for kernels shared between partitions of a co-simulation.
- **[Trade-off] Named-input overrides are a second signature of `run`.** The user kept them for readability of the
  most-used calls. They forward to the base and add no behaviour.

## Migration Plan

Once `BaseConfig.run` returns `self`, every caller that reads a return value breaks at once, so no intermediate
state passes the full suite. The change therefore lands on one branch, `restore-abstract-hooks`, and merges into
`development` only when the checks session and the full suite pass.

On the branch, one commit per layer, in the order of the import layers:
1. `skcomplex.base` and `BaseSpec`: the verbs return `self`, and the hooks return `None`. The hooks stay concrete
   until step 6.
2. `core`;
3. `engine`, as problems and kernels, then backends, compile, solvers, runtime, and visuals;
4. `analysis` and `experiments`;
5. `models`;
6. the hooks become abstract, once every class implements all three.

Each layer's commit updates that layer's classes, every caller of their `run()` in any layer, its tests, doctests, and docs.
It passes ruff, black, pydoclint, and `mypy` on the files it changed. The full suite runs from step 5 on. Rollback
is dropping the branch, or reverting the merge.
