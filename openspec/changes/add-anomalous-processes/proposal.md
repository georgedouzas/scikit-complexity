# Proposal

## Why

Argun et al. (2021 ch. 6) simulate anomalous diffusion with five standard models: continuous-time random walks, Lévy
walks, annealed transient time motion, fractional Brownian motion, and scaled Brownian motion. Only scaled Brownian
motion fits the framework today, because it is an ordinary SDE with a time-dependent amplitude. The other four need two
features the core lacks:
- renewal processes, where events fire after waiting times drawn from a declared distribution, often a power law, and
  reset states with random values (Ar 6.3);
- fractional Gaussian noise, whose increments are correlated over the whole run with a Hurst index `H` (Ar 6.1).

The `organize-model-library` inventory records these four models as blocked by this change.

## What Changes

- Extend the named draws of `add-stochastic-maps` with `draw(name, distribution, **parameters)`, a named draw from any
  SciPy distribution, such as `pareto`, `lomax`, or `levy_stable`, with symbolic parameters, sampled from the
  simulation's random stream.
- Accept draws in renewal laws, where a draw takes a fresh value at every renewal of every member, besides the update
  laws that `add-stochastic-maps` allows.
- Add renewal laws: `Renewal(waiting, reset, label)` fires after a waiting time drawn from a distribution, resets states
  to values that may contain draws, and draws the next waiting time. Every member of a group renews on its own clock.
  Renewal laws combine with rate and noise laws, which evolve the states between renewals.
- Add a renewal form and its stepper for systems whose only laws are renewals, which jumps from renewal to renewal
  exactly, without a time grid.
- Add fractional noise laws: `FractionalNoise(target, value, hurst)` drives a state with fractional Gaussian noise of
  Hurst index `0 < H < 1`. A Hurst index of 1/2 gives the same process as `Noise`.
- Add the `FractionalEulerMaruyama` stepper, which draws the correlated increments of the whole run at once by
  circulant embedding.
- Record each renewal in the trajectory and expose the renewal count as an observable, so it is analysed and streamed
  to Rerun like any other observable.
- Add library models in `physics/soft_matter`: continuous-time random walk, Lévy walk, annealed transient time motion,
  and fractional Brownian motion.

## Capabilities

### New Capabilities

- `anomalous-processes`: draws from SciPy distributions, renewal laws with arbitrary waiting-time distributions, the
  renewal form and stepper, fractional noise laws, and the fractional Euler-Maruyama stepper.

### Modified Capabilities

- `model-library`: anomalous diffusion models in `physics/soft_matter`.

## Impact

- `skcomplex.core.model` gains `Renewal`, `FractionalNoise`, and `draw`. `skcomplex.core.base.Form` gains
  `Form.RENEWAL`. `skcomplex.core.solvers` gains `RenewalStepper` and `FractionalEulerMaruyama`.
- `skcomplex.core.compile` gains the lowering of SciPy draws to batched sampling, of renewals to per-member scheduled
  events in the hybrid layer, and of fractional noise to a precomputed increment path.
- `skcomplex.models.physics.soft_matter` gains `continuous_time_random_walk`, `levy_walk`,
  `annealed_transient_time_motion`, and `fractional_brownian_motion`.
- No new runtime dependency. SciPy, already a dependency, supplies the distributions, including `levy_stable`, and the
  FFT of the circulant embedding.
- Depends on `add-stochastic-maps` for named draws, their per-member sampling, and the rule that the same name reads
  the same value within one firing. This change adds one draw function to that vocabulary and widens where draws are
  accepted, so the draw vocabulary lives in one place.
- Builds on the archived `add-hybrid-dynamics` for scheduled resets and event records.
