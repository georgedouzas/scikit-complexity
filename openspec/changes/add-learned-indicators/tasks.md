# Tasks

## 1. Dependency

- [ ] 1.1 Add `equinox>=0.13.8` to the runtime dependencies in `pyproject.toml` unless an earlier change of the wave
  added it, and verify `pdm install` resolves and `python -c "import equinox"` runs

## 2. Training systems

- [ ] 2.1 Add `DiscreteNormalFormSystem` in `systems/mathematics/dynamical_systems/_discrete_normal_forms.py`, verified
  by `tests/systems/mathematics/dynamical_systems/test_discrete_normal_forms.py` (negative lag-1 autocorrelation of the
  period-doubling form, unknown kind) and its doctest
- [ ] 2.2 Add `FoxCardiacAlternansSystem` in `systems/biology/physiology/_fox_alternans.py`, verified by
  `tests/systems/biology/physiology/test_fox_alternans.py` (one value at 300, alternans at 150) and its doctest
- [ ] 2.3 Add `WesterhoffBusinessCycleSystem` in `systems/economics/macroeconomics/_westerhoff_cycle.py`, verified by
  `tests/systems/economics/macroeconomics/test_westerhoff_cycle.py` (fixed point `2 a` at 10, cycles at 22) and its
  doctest
- [ ] 2.4 Register the three systems, add their inventory entries and the `learned-indicators` primitive, and verify
  `tests/systems/test_catalog.py` and `inventory_problems()` report no problem

## 3. Transition classifier

- [ ] 3.1 Add the preprocessing (censoring, normalisation, padding, gaps channel) and the reading of runs and labels
  from arrays and experiment results, verified by unit tests in `tests/core/analysis/test_transition_classifier.py`
- [ ] 3.2 Add the equinox CNN-LSTM and the scikit-learn path behind one private wrapper, verified by tests of
  probabilities that sum to 1 and of reproducibility with one seed
- [ ] 3.3 Add `transition_classifier` and `TransitionClassifierResult` with reuse of a trained classifier and the
  errors for missing or doubled training data, verified by tests and the doctest
- [ ] 3.4 Add the probe tests of the bifurcation type of the Fox and Westerhoff runs, of null runs, of irregular tipping
  of `RampedStochasticFoldSystem`, and of the Brier score on `StommelSystem`, marking the long ones as slow

## 4. Phase indicator

- [ ] 4.1 Add `phase_indicator` and `PhaseIndicatorResult` with the three kinds and their Fisher bounds, verified by
  `tests/core/analysis/test_phase_indicator.py` on a Gaussian family with a known Fisher information and by the doctest
- [ ] 4.2 Add the Ising probe tests of the critical value and of the bound below `Var(E) / T^4`

## 5. Documentation

- [ ] 5.1 Export both functions from `skcomplex.core.analysis`, check that `applicable_analyses` lists them for series,
  and verify `tests/core/analysis/test_discovery.py` passes
- [ ] 5.2 Add a section on learned indicators to `docs/overview/user_guide/analyse.md` and verify its code blocks run
- [ ] 5.3 Add gallery examples for the three systems, the Fox example classifying its runs and the normal-form example
  building the training library with `sweep`, verified by `gallery_tools coverage --strict` and a run of each example

## 6. Checks

- [ ] 6.1 Run `nox -s checks` and `nox -s tests` with the thread limits set to 1, and verify they pass
