## 1. Tags

- [ ] 1.1 Define `Tags` in `skcomplex.core.specs` and `get_tags()` on the base of parts, with tags from state specs, spaces, and law classes
- [ ] 1.2 Derive the tags of `System` from its parts, with the error for parts that disagree
- [ ] 1.3 Declare accepted tags per role of every interaction, and check them in `prepare`
- [ ] 1.4 Declare the supported tags of every solver and backend from their current supports
- [ ] 1.5 Generate the inventory's system vocabulary from the tags of each system, and compare them in the inventory check

## 2. Simulate

- [ ] 2.1 Add `simulate`, which chooses default solvers from the tags and rejects mismatched solvers and backends
- [ ] 2.2 Replace `create_simulation` with tuned settings of catalog entries passed to `simulate`
- [ ] 2.3 Remove `Simulation`, and make compilers and problem classes private, with one hook for system packages

## 3. Analyses and experiments

- [ ] 3.1 Turn every analysis class into a function with the same settings and a frozen result, and add the binding helper
- [ ] 3.2 Declare the tags each analysis needs, and check them
- [ ] 3.3 Turn sweeps, ensembles, scenarios, sensitivity, and basin stability into functions, keep calibration an estimator
- [ ] 3.4 Move tests to the new functions, mirroring src

## 4. Systems and catalog

- [ ] 4.1 Update every system module, its tests, and its gallery example to `simulate` and the analysis functions
- [ ] 4.2 Show capability tags in the catalog and filter by them, apart from topical tags

## 5. Documentation

- [ ] 5.1 Rewrite the README and the user guide to the new API, keeping the task order
- [ ] 5.2 Replace the compatibility page with generated pages per kind of part, each with executed compatible and incompatible examples
- [ ] 5.3 Add the migration page from 0.6 to 0.7
- [ ] 5.4 Update every docstring example

## 6. Verification

- [ ] 6.1 Checks, import contracts, and layout tests pass
- [ ] 6.2 Tests and doctests pass in batches, and the inventory and gallery checks pass
- [ ] 6.3 The full suite and the strict docs build pass in CI
