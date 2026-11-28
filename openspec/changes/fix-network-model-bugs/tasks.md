## 1. Growing networks

- [ ] 1.1 Give every Holme-Kim node all its edges within its joining step (#61), and test the newest node's degree
- [ ] 1.2 Test the Holme-Kim transitivity at 1000 nodes against NetworkX for `triad` 0 and 1 (#62)
- [x] 1.3 State the default run length and node count in the growing network docstrings (#59)

## 2. Random graph ensembles

- [x] 2.1 Rewire each Watts-Strogatz lattice edge at most once, keeping its lattice end (#63), with a degree test
- [x] 2.2 Raise a `ValidationError` when the Erdős-Rényi mean degree cannot be reached in `steps` steps (#64)
- [x] 2.3 State how `random_state` seeds the draws of `ConfigurationModelSystem` (#75)

## 3. Analysis and checks

- [x] 3.1 Test `network_measures` on a Barabási-Albert trajectory (#60)
- [x] 3.2 Run black, ruff, mypy, pydoclint, interrogate, lint-imports, the conventions, and the network tests
