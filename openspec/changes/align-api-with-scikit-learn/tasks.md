## 1. Tags

- [x] 1.1 Define `Tags` in `skcomplex.core.specs` and `get_tags()` on the base of parts, with tags from state specs, spaces, and law classes
- [x] 1.2 Derive the tags of `System` from its parts, with the error for parts that disagree
- [x] 1.3 Declare accepted tags per role of every interaction, and check them in `prepare`
- [x] 1.4 Declare the supported tags of every solver and backend from their current supports
- [x] 1.5 Generate the inventory's system vocabulary from the tags of each system, and compare them in the inventory check

## 2. Systems

- [x] 2.1 Add `BaseSystem` with derived tags, nesting, and passing model parameters to parts, and make `System` its generic composite
- [x] 2.2 Add `all_systems()` with static listing by field and topics and filtering by tags, and remove the registry, `register_system`, `create_system`, and `describe`

## 3. Simulate

- [x] 3.1 Add `simulate`, which chooses default solvers from the tags and rejects mismatched solvers and backends
- [x] 3.2 Remove `create_simulation`, and read tuned run settings in `simulate` through `BaseSystem.get_run_settings()`
- [x] 3.3 Remove `Simulation`, and make compilers and problem classes private, with one hook for system packages

## 4. Analyses and experiments

- [x] 4.1 Turn every analysis class into a function with the same settings and a frozen result, and add the binding helper
- [x] 4.2 Declare the tags each analysis needs, and check them
- [x] 4.3 Turn sweeps, ensembles, scenarios, sensitivity, and basin stability into functions, keep calibration an estimator
- [x] 4.4 Move tests to the new functions, mirroring src

## 5. Ready-made systems

- [x] 5.1 Turn every factory function into a system class with model parameters and class attributes, by field
- [x] 5.2 Update every system's tests and gallery example to the class, `simulate`, and the analysis functions
- [x] 5.3 Extract general systems of the core where two fields share a structure, starting with mass-action compartments

## 6. Documentation

- [ ] 6.1 Rewrite the README and the user guide to the new API, keeping the task order
- [ ] 6.2 Replace the compatibility page with generated pages per kind of part, each with executed compatible and incompatible examples
  - Done: elements, interactions, laws, solvers, and backends. Analyses, experiments, and systems remain.
- [ ] 6.3 Add the migration page from 0.6 to 0.7
- [ ] 6.4 Add the systems page, and catalog pages with class, import path, parameters, and tags
- [ ] 6.5 Update every docstring example

## 8. Follow-ups found while converting systems

- [x] 8.1 Cache the parts `BaseSystem` builds while its parameters are unchanged, so `get_tags`, `_check`, and `prepare` do not rebuild systems that fit or generate data when built
- [x] 8.2 Add `invariants` and `schedule` to `RunSettings` and `simulate`, and restore the tuned invariants of the order book, guru order book, and goods traders systems
- [x] 8.3 Read `get_run_settings()` in the registry bridge, until the registry is removed
- [x] 8.4 Rename the classes whose catalog names end in `_system`, such as `IntelligentDriverSystemSystem` and `VoterSystemSystem`, with their gallery and inventory keys
- [x] 8.5 Fix the failures that predate the conversion: the period of `memristive_mixed_mode_oscillator` and the frozen-spec test that analysis results break
- [x] 8.6 Add `compiler` to `RunSettings` and `simulate`, the one hook through which a system brings its own form, and convert `koch_curve`, `cantor_set`, `sierpinski_substitution`, and `morse_subshift` (rewriting) to classes
- [x] 8.7 Make `simulate` defer the solver to the compiled form when a system nests subsystems, and stop `MonteCarloSolver` claiming tags its compiler rejects
- [ ] 8.8 Derive the randomness tag from draws in agent rules, random parameters, and stochastic modes, so random systems such as `GaltonWatsonSystem` and `SIRSystem(stochastic=True)` are stochastic, then make `ensemble` require it
- [ ] 8.9 Let experiments compile a system once per structure and reuse the problem across runs that change only parameters, initial states, or seeds, as before the change

## 7. Verification

- [ ] 7.1 Checks, import contracts, and layout tests pass
- [ ] 7.2 Tests and doctests pass in batches, and the inventory and gallery checks pass
- [ ] 7.3 The full suite and the strict docs build pass in CI
