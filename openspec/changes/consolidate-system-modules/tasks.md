## 1. Preparation

- [x] 1.1 Wait for `extend-topology-requests` and `add-audited-engineering-computing-systems` to merge
- [ ] 1.2 Write the census and folding script outside the repository, and run its census on `development`
- [ ] 1.3 Add the layout test that fails on a component module serving one system, and update the import contracts to
  allow only `<field>.common`

## 2. Folding by field

- [ ] 2.1 Biology
- [x] 2.2 Physics
- [ ] 2.3 Economics
- [ ] 2.4 Social
- [ ] 2.5 Mathematics
- [ ] 2.6 Engineering
- [ ] 2.7 Computer science
- [ ] 2.8 Chemistry

## 3. Shared components

- [ ] 3.1 Move components shared within a field to its `common` package, or copy them where the uses differ
- [ ] 3.2 Move single-user modules of the existing `common` packages into their systems
- [ ] 3.3 List cross-field duplicates with a proposal for each, and act on the user's decision

## 4. Verification

- [ ] 4.1 Every subfield package exports the same public names as before
- [ ] 4.2 Ruff, black, mypy, pydoclint, the import contracts, and the layout test pass
- [ ] 4.3 The catalog, systems, and doctest tests pass in batches, and the inventory and gallery checks pass
- [ ] 4.4 The full suite and the docs build pass in CI
