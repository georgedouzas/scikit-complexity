## 1. Array updates of well-mixed groups

- [ ] 1.1 Allow groups without a space in the requirement of `ArrayUpdateLaw` and lower them on the NumPy and Numba
  backends as in the network case, and verify the swap-of-paired-rows and wrong-shape scenarios in
  `tests/core/components/laws/test_array_update.py`
- [ ] 1.2 Document the well-mixed case in the docstring of `ArrayUpdateLaw` with a runnable example, and verify the
  doctest

## 2. High-order entropy

- [ ] 2.1 Add `brotli>=1.2.1` to the runtime dependencies in `pyproject.toml` and the Project Profile, and verify
  `pdm install` resolves
- [ ] 2.2 Add the kind `'high-order'` with the settings `compressor` and `quality` to `entropy`, and verify the
  random-bytes, copies, and unknown-compressor scenarios in `tests/core/analysis/test_entropy.py`
- [ ] 2.3 Read group states frame by frame as concatenated rows, and verify one value per frame on a small group
- [ ] 2.4 Add a paragraph and a runnable example to the entropy section of `docs/overview/user_guide/analyse.md`, and
  verify it runs

## 3. BFF soup

- [ ] 3.1 Add the Numba interpreter and `BffSoupSystem` in
  `src/skcomplex/systems/computer_science/evolutionary_computation/_bff_soup.py`, export the system, and verify the
  instruction-set, replicator, and defaults scenarios
- [ ] 3.2 Add the observable `high_order_entropy` and the seeds, and verify the seeded-replicator scenario, marked slow
- [ ] 3.3 Add the inventory entry `bff_primordial_soup` with the source 2406.19108 and the primitive
  `compression-entropy` with `change = "add-program-soups"`, and verify the inventory check passes
- [ ] 3.4 Add the gallery example `docs/examples/computer_science/evolutionary_computation/plot_bff_soup.py`, which
  runs a soup of 2^14 tapes with a random state for which the transition happens within the run, stated in the
  example, plots the high-order entropy over epochs, and prints the most common tape, and verify the strict gallery
  build runs it

## 4. Integration

- [ ] 4.1 Commit the work as conventional `feat` commits and verify `nox -s changelog` lists them
- [ ] 4.2 Run `nox` formatting, checks, tests of the touched modules, and the strict docs build, and fix every finding
