## Why

`early_warning` measures critical slowing down, but it cannot tell which bifurcation is coming, and it gives no
probability of tipping. Classifiers trained on simulated normal forms do both, also for untrained systems, irregular
samples, and rate-induced tipping (2303.09669, 2407.18693, 2409.07590). Machine-learning indicators of phase transitions
are lower bounds of the square root of the Fisher information along a parameter (2311.10710), so a `sweep` can locate a
transition and bound its Fisher information without an order parameter. The library has the normal forms, the ramps,
the ensembles, and `sweep`, but no analysis that learns from runs.

## What Changes

- New analysis function `transition_classifier(source, *, training=..., labels=..., ...)` in `core/analysis`. It trains
  a classifier on labelled runs, such as normal forms with a ramped parameter, and returns the class probabilities of
  every series of `source`. One function covers the type of bifurcation, tipping or not, and the probability of
  rate-induced tipping, since only the labels differ. The model is a CNN-LSTM written with Equinox and trained
  on the shared Optax loop of `add-learning-foundation`, or any scikit-learn classifier. Series may be irregularly
  sampled observed runs. A trained classifier from an earlier result can be
  passed instead of training data.
- New analysis function `phase_indicator(source, *, parameter=..., output=..., kind=...)` in `core/analysis`. It reads
  samples at every point of a `sweep` and returns the supervised, confusion, or prediction-based indicator, the
  critical point at its maximum, and the lower bound of the Fisher information that the indicator gives.
- New system `DiscreteNormalFormSystem` (`mathematics/dynamical_systems`), the noisy discrete-time normal forms of the
  period-doubling, Neimark-Sacker, fold, transcritical, and pitchfork bifurcations with random higher-order terms and a
  ramped or fixed parameter, which generate training libraries.
- New system `FoxCardiacAlternansSystem` (`biology/physiology`), the Fox map of cardiac action potential duration with
  memory, which alternates through a period-doubling bifurcation as the stimulation period falls.
- New system `WesterhoffBusinessCycleSystem` (`economics/macroeconomics`), the consumer-sentiment model of national
  income, which starts business cycles through a Neimark-Sacker bifurcation as autonomous expenditure rises.

## Capabilities

### New Capabilities

- `learned-indicators`: analyses that learn from simulated or sampled data to classify coming transitions, predict
  tipping probabilities, and locate phase transitions with a Fisher-information bound.

### Modified Capabilities

- `model-library`: the discrete normal forms, the Fox cardiac alternans map, and the Westerhoff business-cycle model,
  with their inventory entries and gallery examples.

## Impact

- Core: two new modules in `skcomplex.core.analysis` with their results, exported from the package and listed by
  `applicable_analyses`. No change to existing analyses. `early_warning` stays in `systems/physics/climate`.
- Systems: `skcomplex.systems.mathematics.dynamical_systems`, `skcomplex.systems.biology.physiology`, and
  `skcomplex.systems.economics.macroeconomics`, each with a new self-contained module.
- Dependencies: none new. Equinox comes with `add-learning-foundation`, and jax and optax are already runtime
  dependencies.
- Docs: the user guide page on analyses, three gallery examples, and the inventory with the primitive
  `learned-indicators`.
- Depends on `add-learning-foundation`: training libraries are observed runs built by `observe` from a `sweep`, with
  irregular times from the observation, the validation series are held out by `split_runs`, and the CNN-LSTM trains on
  its shared Optax loop with early stopping. Independent of the other drafts that build on it.
