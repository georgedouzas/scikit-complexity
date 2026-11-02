# Tasks

## 1. Dependency and fixtures

- [ ] 1.1 Add `pysd>=3.14.3,<4` to `[project] dependencies`, list which of its dependencies are new to the install,
  and verify `deptry` and `pip-audit` pass
- [ ] 1.2 Add Vensim and XMILE fixtures under `tests/fixtures/system_dynamics`: teacup, SIR in both formats, a goal
  seeking stock without flow variables, a population with births and deaths, a `DELAY3` of a step, a population
  subscripted over three regions, and a file with `ALLOCATE BY PRIORITY` and `GET XLS DATA`, store PySD's run of each
  runnable fixture as reference values, and verify the references load in a test

## 2. Translator

- [ ] 2.1 Add `RunSettingsSpec` to `skcomplex.specs` and the package `skcomplex.engine.translators` with
  `SystemDynamicsTranslator`, its `check`, `prepare` reading the abstract model with PySD, and `run` writing the module
  source with snake-case names, the element class, `create_system`, `observables`, and the settings constants, update
  the import-linter contracts, and verify the teacup runs like PySD and XMILE reads like Vensim scenarios
- [ ] 2.2 Split each stock's flow into signed terms, write named `TransferInteraction`s between stocks and with
  `Source()` or `Sink()`, and gather the remaining terms in a `RelaxationInteraction` for the goal-seeking form or a
  `RateInteraction`, and verify the SIR flows are named transfers, goal seeking becomes relaxation, and loops of a
  translated model scenarios
- [ ] 2.3 Map the built-in functions through one table of syntax structures onto `lookup`, `where`, `step`, `ramp`,
  `pulse`, `pulse_train`, `delay1`, `delay3`, `smooth`, `distributed_delay`, `delayed`, `minimum`, `maximum`, `clip`,
  and the draws, shift `Time` by the start, and verify the third-order delay scenario
- [ ] 2.4 Map a model whose stocks are subscripted over one range to a `GroupElement` and `SUM` to `total`, and verify
  the regional populations scenario
- [ ] 2.5 Collect every untranslatable construct and raise one `ValidationError` that names each variable and
  construct, and verify the unsupported functions scenario

## 3. Documents and gate

- [ ] 3.1 Add a user-guide section on translating models and a gallery example that translates a published model, runs
  it, and plots the dominance of its loops with `LoopAnalysis`, and verify the documentation build runs both
- [ ] 3.2 Add a CI job in `.github/workflows/ci.yml` that installs the project on Python 3.13 and runs the translator
  tests against the fixtures, and verify it passes on a pull request
- [ ] 3.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
