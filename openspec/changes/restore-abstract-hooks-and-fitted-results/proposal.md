# Proposal

## Why

`BaseConfig` derives from `ABC` but declares no abstract method. Commit `cd12ad90` replaced the abstract `_check`,
`_prepare`, and `_run` hooks with no-op defaults, so a child can skip a hook and nothing complains. `run()` also
returns `object`, so results reach callers in two ways:
- some callers read the return value and narrow it with `isinstance`, such as the simulation reading the problem of
  a compiler and the state of a solver;
- others read an attribute ending in `_`, such as `analysis.mean_` or `protocol.trades_`.

The project follows the scikit-learn contract, where a fitted object keeps its results as attributes ending in `_`
and the verb returns the object. This change makes every configurable class follow it.

## What Changes

- **Hooks are abstract again.** `BaseConfig` declares `_check`, `_prepare`, and `_run` as abstract methods, so `ABC`
  means something and a class that skips a hook cannot be created. `_release` stays an optional hook with a no-op
  default.
- **Specs keep defaults.** `BaseSpec` implements all three hooks:
  - `_check` accepts every field, and the specs that validate a field override it;
  - `_prepare` and `_run` do nothing, since a spec is a value.
- **BREAKING: `run()` returns the object.** Every verb returns `self`, and every hook returns `None`. A run keeps its
  result in an attribute ending in `_`, with one name per family:

  | Family | Result |
  | --- | --- |
  | elements, interactions, systems | `laws_` |
  | laws | `value_` |
  | compilers | `problem_` |
  | lowerings | `lowered_` |
  | backends | `functions_` |
  | solvers | `state_` |
  | kernels | `values_` |
  | searches | `pairs_` |
  | printers | `code_` |
  | observers of a compilation | `observed_` |
  | quantum lowering | `kernels_` |
  | designs | `rows_` |
  | market protocols | `trades_` |

  Analyses, experiments, simulations, observers of a run, clocks, and visuals already keep their results as
  attributes and need no new name.
- **Named inputs stay.** `Simulation.run(until=...)`, the observers' `run(frame=...)`, the clocks' `run(elapsed=...)`,
  and `BaseAnalysis.run` keep their overrides, which name the input and forward to the base verb.
- **Every caller reads the attribute.** About 321 call sites in `src` and the tests and docs that read a return value
  of `run()` read the result attribute instead.
- **Generated source has its own class.** `BaseConfig` loses `_load_source` and `CACHE_DIR`, which only four classes
  use. A `SourceLoader` in the engine takes the cache directory in `prepare` and runs on generated source, keeping
  the imported module in `module_`. `NumbaBackend`, `ExpressionPrinter`, `LatticeProblemPass`, and
  `LyapunovAnalysis` use it, and the cache stays at `~/.cache/skcomplex/kernels`.
- **The constitution's Project Profile** says the hooks are abstract and that runs keep their results as attributes.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `base-interfaces`: configurable bases declare abstract hooks, specs alone have default hooks, `run()` returns the
  object with its result in an attribute named for its family, and the root base holds only the verbs and hooks.

## Impact

- **Code:** `skcomplex.base`, `skcomplex.specs` (`BaseSpec`), and every class that implements `_run`. That is 192
  classes across `core`, `engine`, `analysis`, `experiments`, and `models`, and every caller of their `run()`.
- **API:** breaking for code that reads `run()`'s return value, such as `law.run(values)`, `compiler.run(state)`, or
  `kernel.run(state)`. Code that chains `check().prepare().run()` and reads attributes is unaffected.
- **Docs:** the user guide, the docstrings and doctests of every changed class, and the constitution's Project
  Profile.
- **Performance:** kernels and laws run once per step, and setting one attribute per run adds no measurable cost.
