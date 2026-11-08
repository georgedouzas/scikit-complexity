# Design

## Context

Analyses in `src/skcomplex/core/analysis/` are functions that return a frozen `*Result` and declare `TAGS` as an
`AnalysisTags`. Series are read by `_read_samples` in `_base.py`, which accepts a trajectory, an array, or several runs.
`sweep` and `ensemble` in `core/experiments` return an `ExperimentResult`, whose `column(name)` gives design values and
whose `series(name)` gives the kept series of every run. `early_warning` lives in `systems/physics/climate/_early_warning.py`
and reads `rolling` from the core. The continuous normal forms are in
`systems/mathematics/dynamical_systems/_normal_forms.py`, `RampedStochasticFoldSystem` in
`systems/physics/climate/_tipping_types.py`, `StommelSystem` in `systems/physics/oceanography/_thermohaline.py`, and
`IsingSystem` in `systems/physics/statistical_mechanics/_spins.py`. jax and optax are runtime dependencies. equinox is
not yet.

## Goals / Non-Goals

**Goals:**

- One classifier function for bifurcation type, tipping, and tipping probability, since the three papers differ only
  in their labels and inputs.
- One indicator function for the three indicators of 2311.10710 with their Fisher bounds.
- Training libraries built from systems and `sweep`, not from a generator inside the core.

**Non-Goals:**

- The trained ewstools models, which need TensorFlow.
- Estimator classes with `fit` and `predict`. Reuse goes through the `classifier` setting.
- The quantum Fisher information and transitions in more than one parameter.
- Moving `early_warning` into the core.

## Decisions

- **Modules.** `core/analysis/_transition_classifier.py` holds `TransitionClassifierResult` and
  `transition_classifier`. `core/analysis/_phase_indicator.py` holds `PhaseIndicatorResult` and `phase_indicator`. Both
  are exported from `skcomplex.core.analysis` and declare `AnalysisTags(DataKind.SERIES)`. A new package for learned
  analyses was considered and rejected, since every analysis module is flat in `core/analysis`.
- **Signature of the classifier.**
  `transition_classifier(source, *, training=None, labels=None, classifier=None, variables=None, times=None,
  model='cnn_lstm', length=500, censor=True, validation=0.1, epochs=200, random_state=None, output=None)`. `source` and
  `training` are anything `_read_samples` reads, or an `ExperimentResult`, whose `output` series are the runs and whose
  column named by `labels` gives the labels. The result holds `classes`, `probabilities` (series by classes),
  `prediction`, `training_accuracy`, `validation_accuracy`, `brier` (NaN unless the labels are binary), and
  `classifier`.
- **Preprocessing.** Every training series is censored to a random window of length drawn in `[length / 10, length]`,
  normalised by its mean absolute value, and padded with zeros at the start to `length`, as in 2303.09669. Series to
  classify are normalised and padded the same way, without censoring. With `times`, the gaps between samples form a
  second channel, which is how 2407.18693 handles irregular sampling. Without them the gaps channel is omitted.
- **Models.** `'cnn_lstm'` is a small equinox module: `eqx.nn.Conv1d` with max pooling, two `eqx.nn.LSTMCell` layers
  scanned with `jax.lax.scan`, dropout, and `eqx.nn.Linear`. It is trained with `optax.adam` on the softmax cross
  entropy in mini-batches, with keys split from `random_state`. A scikit-learn classifier is cloned and fitted on the
  flattened padded windows. The private class `_SequenceClassifier` wraps both behind `predict_proba`, so `classifier`
  in the result is reused the same way. Writing the network in flax was rejected because flax is not a dependency and
  equinox is also needed by three other changes.
- **Labels for tipping.** Labels are given by the user. The gallery and tests label rate-induced tipping by whether the
  final state lies in the other basin, read with `basin_stability` or a threshold on the last value, and label
  bifurcation types by the design column `kind` of a `sweep` over `DiscreteNormalFormSystem`.
- **Signature of the indicator.** `phase_indicator(source, *, parameter=None, output=None, parameters=None,
  kind='confusion', model=None, width=1, warmup=0, random_state=None)`. With an `ExperimentResult`, the samples at a
  parameter value are the frames after `warmup` of the `output` series of all runs at that value of the design column
  `parameter`. With an array of shape (values, samples, ...) the values come from `parameters`. The default model is a
  scikit-learn `LogisticRegression` for the classifying kinds and `Ridge` for the prediction kind, on flattened samples.
  `width` is the number `l` of neighbouring values on each side of the confusion bipartition. The result holds
  `parameters`, `indicator`, `critical`, `fisher_bound`, and `model`. Derivatives along the parameter are central
  differences with `numpy.gradient`.
- **Fisher bound.** `fisher_bound` is the square of the bound on the square root of the Fisher information: `I1 / σ`
  for the supervised kind, `2 I2 / Δγ` for the confusion kind with `width=1`, and `I3` for the prediction kind. For a
  wider bipartition the confusion bound uses the distance between the means of the two sets.
- **Systems.** `DiscreteNormalFormSystem` goes in its own module `_discrete_normal_forms.py` of
  `mathematics/dynamical_systems`, with `kind`, `mu`, `ramp` (start and end, or `None` for a fixed parameter), `sigma`,
  `degree` of the random higher-order terms, and `random_state`. `FoxCardiacAlternansSystem` goes in
  `biology/physiology/_fox_alternans.py` and `WesterhoffBusinessCycleSystem` in
  `economics/macroeconomics/_westerhoff_cycle.py`. Each is a `BaseSystem` of one element under `DiscreteTime` with
  update laws, a noise term drawn in the law, and the ramp as a time input, as `RampedStochasticFoldSystem` writes its
  ramp. The Westerhoff element keeps the previous income as a second state.
- **Dependency.** `equinox>=0.13.8` joins the runtime dependencies in `pyproject.toml`. Its last release is 0.13.8 on
  2026-05-05, within the 24 months of `dependency-freshness`. It needs jax, jaxtyping, typing-extensions, and
  wadler-lindig. If another change of this wave lands it first, this change only uses it.

## Risks / Trade-offs

- [Training time in tests] → Tests use a few hundred short runs, a scikit-learn model or a few epochs of the CNN-LSTM,
  and mark the full CNN-LSTM probe of the Fox and Westerhoff scenarios as slow. The gallery example trains a small
  library.
- [Classifier accuracy depends on the library size] → Scenarios ask for the most probable class in most runs, not for
  the accuracies of the papers, and the user guide states the library sizes of the papers.
- [Nondeterminism of JAX on some platforms] → Training runs on the CPU in double precision with fixed keys, and the
  reproducibility test compares within a tolerance of 1e-10.
- [Confusion indicator cost] → It trains one classifier per parameter value. The default model is a linear one, and
  `width` keeps each task small.
