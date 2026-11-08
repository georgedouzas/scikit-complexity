## 1. Bases

- [ ] 1.1 Make `BaseSpec` a frozen dataclass base with only `_check`, not a `BaseConfig`, and remove `prepare` and `run` from specs
- [ ] 1.2 Remove `_check` and `_validated` from `BaseConfig`, and make `prepare` call `_prepare` and nothing else
- [ ] 1.3 Move the label and declared-spec checks of `BasePart` to the start of its `_prepare`

## 2. Estimators

- [ ] 2.1 Move every estimator's `_check` body to the start of its `_prepare`, in the core, family base by family base
- [ ] 2.2 Do the same in every system module of `skcomplex.systems`, by field
- [ ] 2.3 Make owners prepare the parts they own and call `_check` on the specs they read, instead of `_validated`

## 3. Callers and docs

- [ ] 3.1 Replace every `spec.prepare()` and `spec.run()` call in src, tests, docs, and gallery examples
- [ ] 3.2 Update the base tests, the guide pages that show `_check`, and the migration page
- [ ] 3.3 Update the user guide section on writing new parts

## 4. Verification

- [ ] 4.1 Checks, import contracts, and layout tests pass
- [ ] 4.2 Tests and doctests pass in batches, and the inventory and gallery checks pass
- [ ] 4.3 The full suite and the strict docs build pass in CI
