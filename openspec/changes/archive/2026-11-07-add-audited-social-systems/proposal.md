## Why

The reference audit listed social models of Argun, Bellomo, Chesney, Jensen, Downey, Meadows, Thurner et al., and the
reviewed papers on echo chambers, multilevel selection, game transitions, and the scales of human mobility that the
current core supports but the catalog lacks. Each one is a known result the catalog can reproduce and a worked example
of the core primitives.

## What Changes

- Add `iterated_lattice_dilemma`, `two_level_replicator`, `two_level_birth_death`, and `game_transitions` to
  `social/cooperation`. The additively separable pairwise group conflict of Cooney is the option `victory` of
  `two_level_replicator`, since it reduces to a two-level replicator equation.
- Add `three_state_opinion`, `undecided_flows`, `willing_majority`, and `echo_chamber_radicalization` to
  `social/opinion_dynamics`.
- Add `kinetic_crowd` to `social/crowds`, Bellomo's walkers in a room with walls, distinct from `evacuation`.
- Add `ajanis_choice`, `supply_chain_diffusion`, `watchmakers`, `drift_to_low_performance`, `policy_resistance`,
  `container_mobility`, and `exploration_preferential_return` to `social/organizations`.
- Add `random_typewriting` to `social/language`.
- Add the option `choices` of `schelling` for the Big Sort variant of Downey's Exercise 9-1.
- Record every new system in the inventory with its sources and add its gallery example.
- Leave out the audited models the sources do not specify, listed in the design.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: new systems of cooperation, opinion dynamics, crowds, organizations, and language, and the Big Sort
  option of Schelling segregation.

## Impact

New modules under `src/skcomplex/systems/social`, their tests under `tests/systems/social`,
`docs/overview/inventory.toml`, and examples under `docs/examples/social`. No change to the core.
