# Proposal

## Why

System dynamics has decades of published models in Vensim (`.mdl`) and XMILE (Stella, iThink, Vensim exports) files,
and modellers keep their work in those tools. PySD already parses both formats into a language-neutral abstract model,
so translating that abstract model turns any of those files into a model of the library without writing a parser.

The core already holds everything a stock-and-flow model is made of, so this change adds no modelling primitive:
- `TransferInteraction` moves amounts between the states of one element, or between a state and `Source()` or
  `Sink()`, and names its flows with `flow=` and the fourth item of a leg, reported as `<element>.<flow>` when source
  and target are one element,
- `RateInteraction` adds the rest of a stock's net rate, and `RelaxationInteraction` the goal-seeking rate
  `(goal - x) / time`,
- the law utilities hold `lookup`, `where`, `step`, `ramp`, `pulse`, `pulse_train`, `delay1`, `delay3`, `smooth`,
  `distributed_delay`, `delayed`, `clip`, `minimum`, `maximum`, `total`, and the draws `uniform` and `normal`,
- `ObservableSpec` records auxiliaries, `LoopAnalysis` lists feedback loops and their dominance, and
  `Simulation(invariants=...)` checks conserved totals.

What is missing is only the translation from a file to those names.

## What Changes

- **Translating a model file:** `SystemDynamicsTranslator(path)` in a new package `skcomplex.engine.translators`
  follows the three verbs of `BaseConfig`. `run()` sets `source_` to the Python source of a model module built only
  from public names of the library, which the user saves and imports like any model, and `settings_` to the run
  settings of the file's control section.
- **Translation from PySD's abstract model:** the translator calls PySD's translators (`VensimFile`, `XmileFile`) and
  reads the `AbstractModel` they return, never the Python code PySD generates or its runtime:
  - the stocks become the states of one element class, and the constants its parameters,
  - a flow variable that leaves one stock and enters another becomes a `TransferInteraction` between the two states
    of that element, named by `flow=`, so the trajectory reports it as `<element>.<flow>`,
  - a flow variable that touches one stock becomes a `TransferInteraction` with `Source()` or `Sink()`,
  - what remains of a stock's net rate becomes a `RelaxationInteraction` when it has the goal-seeking form
    `(goal - stock) / time`, and a `RateInteraction` otherwise,
  - an auxiliary becomes a private function of the element's namespace that the laws call, and an `ObservableSpec`
    recorded under its name,
  - a lookup becomes a `lookup` table,
  - the built-in functions map to the law utilities: `IF THEN ELSE`, `ZIDZ`, and `XIDZ` to `where`, `STEP`, `RAMP`,
    `PULSE`, and `PULSE TRAIN` to `step`, `ramp`, `pulse`, and `pulse_train`, `DELAY1`, `DELAY3`, and `SMOOTH` to
    `delay1`, `delay3`, and `smooth`, `DELAY N` and `SMOOTH N` to `distributed_delay`, `DELAY FIXED` to `delayed`,
    `MIN` and `MAX` to `minimum` and `maximum`, a `MIN` of a `MAX` to `clip`, and the random functions to draws,
  - a model whose stocks are all subscripted over one range becomes a group with one member per subscript, and sums
    over the range become `total`.
- **Named errors for what does not translate:** functions without a counterpart, such as `GAME`, `ALLOCATE`, and the
  vector functions, data read from spreadsheets, macros, variables subscripted over more than one range, and models
  that mix subscripted and unsubscripted stocks raise one error that lists every unsupported construct with its
  variable.
- **Dependency:** PySD becomes a runtime dependency as `pysd>=3.14.3,<4`, since `skcomplex.engine` imports it. Its
  last release is of 2025-03-23 and its classifiers stop at Python 3.12, so a CI job checks the translator on Python
  3.13.

## Capabilities

### New Capabilities
- `system-dynamics-import`: translating Vensim and XMILE files into model modules, the mapping of stocks, flows,
  auxiliaries, constants, lookups, built-in functions, and subscripts onto the core's interactions and utilities, the
  run settings of the file, and the errors for what does not translate.

### Modified Capabilities

## Impact

- **Code:** a new package `skcomplex.engine.translators` with `SystemDynamicsTranslator`, and a frozen
  `RunSettingsSpec(start, until, dt, save_every)` in `skcomplex.specs`. The import-linter contracts place the package
  in the engine's layers. No core class changes.
- **Dependencies:** `pysd>=3.14.3,<4` in `[project] dependencies`, with its own dependencies, such as `parsimonious`,
  `xarray`, `lxml`, and `black`.
- **CI:** a job that runs the translator tests on Python 3.13.
- **Tests:** small Vensim and XMILE fixtures under `tests/fixtures`, including the classic teacup and SIR models, each
  translated, imported from a temporary directory, and compared against PySD's own run of the same file.
- **Docs:** a user-guide section on translating models and one gallery example that translates a published model,
  runs it, and plots the dominance of its loops.
