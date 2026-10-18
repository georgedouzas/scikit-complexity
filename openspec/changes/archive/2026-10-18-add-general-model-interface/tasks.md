# Tasks

## 1. Interface

- [x] 1.1 Add `Element.define` and `Interaction.define`, and verify that defined types match subclassed types in
  simulation, groups, `get_params`, cloning, and pickling
- [x] 1.2 Add the boolean, categorical, and angle domains with their encodings, automatic angle wrapping, and checked
  mode for alphabets, and verify the categorical and angle scenarios
- [x] 1.3 Add `Observable` with expression and reduction forms, compile observables into one function per system, and
  rebase `Aggregate` on reductions, and verify the order parameter, group reduction, and existing aggregate tests pass

## 2. Runtime and visualization

- [x] 2.1 Record observables in trajectories under their paths, and verify the observable-by-path scenario
- [x] 2.2 Log observables to Rerun as time series, and verify the live order parameter scenario on an in-memory
  recording

## 3. Catalog and library

- [x] 3.1 Add `register_model`, `catalog`, `describe`, `create`, and `run`, and verify the browse and create scenarios
- [x] 3.2 Add registered factories for every existing ready-made model, and register the existing factories, and
  verify each factory's docstring example runs
- [x] 3.3 Add the library import check, and verify it passes on the library and fails on a planted private import
- [x] 3.4 Add the conformance suite over the catalog, and verify every registered model passes it

## 4. Documentation and gate

- [x] 4.1 Document the general interface, the domains, observables, and the catalog in the user guide, with one model
  written both with classes and with `define`, and verify the documentation test passes
- [x] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
