# Proposal

## Why

The survey of the thinly cited textbooks and of the papers in `references/papers` (`survey-thin-references`) found 24
models that the current core already expresses with the system vocabulary: no general primitive is missing, only the
model and, where a field needs one, a realization in the model library. Eight of them exercise combinations of
primitives that no implemented model exercises, so they probe the core first; the others repeat covered patterns and
illustrate their sources. This change implements them.

## What Changes

- Add 24 models to the library, each a registered factory under its field path with a curated gallery example:
  - physics: `swarmalators`, `ehrenfest_urn`, `directed_percolation_langevin`, `sk_game`, `active_model_b`,
    `keller_segel`, `benard_marangoni`, `charged_schrodinger`, `cahn_hilliard`, and `allen_cahn`;
  - biology: `hormone_cascade_amplifier`, `min_protein_skeleton`, `nanotube_gap_junction_exchange`,
    `closed_contour_cell_exchange`, `adaptive_response_costs`, `glucose_dual_hormone_control`,
    `coagulation_positive_feedback`, and `couzin_leadership`;
  - earth: `sutera_fraedrich_tipping` and `ramped_stochastic_fold`;
  - economics: `metaphoria` and `wilhite_trade_networks`;
  - mathematics: `koopman_slow_manifold` and `ou_driven_rossler`.
- Implement the `core` entries first, as the inventory's priorities require.
- Write any field-specific need, such as a named heuristic or a trading protocol, in the field's model library, never in
  the core.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: the surveyed models that the core already expresses.

## Impact

- `skcomplex.models` gains 24 models and the gallery 24 examples; the core is unchanged unless a model reveals a missing
  general primitive, which is then specified as its own change.
