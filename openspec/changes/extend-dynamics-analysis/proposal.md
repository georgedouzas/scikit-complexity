# Proposal

## Why

Four recent papers ask the toolkit for measurements it nearly has. Fodor, Jack, and Cates (2021, arXiv 2104.06634)
sample rare trajectories of active matter with biased ensembles, which needs population dynamics (cloning) beyond the
tilted generators the toolkit has, and bound fluctuations by entropy production through the thermodynamic uncertainty
relation. Brauns and Marchetti (2023, arXiv 2306.08868) explain nonreciprocal pattern formation through the dispersion
relation of a field model, which the toolkit computes only for single equations by hand. Tian et al. (2023, arXiv
2306.05635) and Fang et al. (2025, arXiv 2511.01168) review criticality in the brain and tipping cascades, whose probe
models exercise existing primitives in combinations no implemented model has. te Vrugt and Wittkowski (2024, arXiv
2405.15751) map active matter onto the pending changes and add the nonreciprocal Vicsek model.

## What Changes

- Add a cloning sampler of rare trajectories for jump processes and stochastic dynamics, with the scaled cumulant
  generating function it estimates.
- Add the thermodynamic uncertainty relation check of a current: its precision against the entropy production.
- Add the dispersion relation of a field model: the growth rate of each wavenumber of its linearisation around a
  uniform state.
- Add the probe models `kinouchi_copelli`, `nonreciprocal_cahn_hilliard`, `nonreciprocal_vicsek`, and
  `tipping_cascade_network`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `stochastic-analysis`: cloning, and the thermodynamic uncertainty relation.
- `field-analysis`: dispersion relations of field models.
- `model-library`: four probe models.

## Impact

- `skcomplex.core.analysis` gains `cloning`, `uncertainty_relation`, and `dispersion_relation`.
