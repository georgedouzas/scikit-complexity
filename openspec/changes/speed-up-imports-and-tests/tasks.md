# Tasks

## 1. Lazy catalog

- [x] 1.1 Build the catalog listing by scanning the model sources for `register_model`. Reject computed name, field,
  or tag arguments with a named error, and import a model's module only in `create`, `run`, and `describe`. Verify
  with the scenarios "Catalog without imports", "Filter by field", "One model", "Unknown name", and "Computed
  registration argument".
- [x] 1.2 Stop `models/__init__.py` from importing the field packages, and add the listing-agreement test. Verify with
  the scenario "Listing matches registration", and measure the time of `import skcomplex.models` and `catalog()`.

## 2. Test session

- [x] 2.1 Move coverage from `addopts` to the `tests` session, and add the `--slow` option and `slow` marker to
  `tests/conftest.py`. Verify with the scenarios "Default run", "Gate run", "Plain run", and "Gate coverage".
- [ ] 2.2 Run the suite once with `--durations=0` on an idle machine, and mark every test above 10 seconds as slow.
  Trim every test above 60 seconds, starting with the avalanche test, keeping its tolerances. Verify with the
  scenario "Slowest test".

## 3. Gate

- [ ] 3.1 Run formatting, checks, the docs build, and tests through nox, and verify that all of them pass.
