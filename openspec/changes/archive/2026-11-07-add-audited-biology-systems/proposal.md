# Proposal

## Why

The audit of the reference books and articles lists biology models that the current core supports but the catalog
lacks: population maps and consumer-resource communities, compartmental and agent-based epidemics, neural masses and
plastic networks, excitable media and adaptive physiological networks, population genetics and assembly theory,
logical, Petri, and adaptation circuits, and collective herding and foraging. Implementing them exercises the general
primitives of the core on the models their sources describe, and gives each a test against a published result and a
gallery example.

## What Changes

- Add to `biology/ecology`: `ricker`, `consumer_resource` (MacArthur's substitutable resources, chemostat supply, and
  essential resources, with Gros's single-resource competition as a special case), and `penaeid_shrimp`.
- Add to `biology/epidemiology`: `zombie_outbreak`, `watts_may_concurrency`, `pair_formation_sis`,
  `record_dynamics_epidemic`, and `episims_city`.
- Add to `biology/neuroscience`: `associative_thought_process` (the dHAN), `next_generation_neural_mass`,
  `stdp_network`, and `adaptive_delay_network`.
- Add to `biology/physiology`: `barkley`, `two_layer_sepsis`, `adaptive_flow_network`, `steerable_feedback_loop`,
  `hyperbolic_chemotaxis`, and `west_brown_enquist`.
- Add to `biology/evolution`: `kimura_diffusion`, `nkcs`, `competitive_jain_krishna`, `assembly_growth`, and
  `polymer_assembly_pool`.
- Add to `biology/systems_biology`: `glycolysis_oscillations`, `three_way_fate_decision`,
  `bristle_lateral_inhibition`, `noisy_boolean_network`, `generalized_logical_network`, `petri_net`, and
  `enzyme_adaptation_circuit`.
- Add to `biology/collective_behaviour`: `ant_colony_optimization` and `shepherding` (Strömbom's heuristic).
- Every model gets its realizations in its subfield, tests against a known result, a curated gallery example, and an
  inventory entry with its sources.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: biology models of the reference audit built on the general classes of the core.

## Impact

- The biology packages of `skcomplex.systems`, their tests, the gallery, and the inventory. No class of the core
  changes and no dependency is added.
