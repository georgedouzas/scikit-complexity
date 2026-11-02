# Proposal

## Why

Three references were added to the survey:
- Padmanabhan, Meskin, and Al Moustafa 2021, *Mathematical Models of Cancer and Different Therapies*;
- Konieczny, Roterman-Konieczna, and Spólnik 2023, *Systems Biology: Functional Strategies of Living Organisms*;
- Raman 2021, *An Introduction to Computational Systems Biology*.

The library's rule is that every surveyed model is either implemented on the core interface, pending a planned change,
blocked on a specified missing feature, or recorded as out of scope. A survey of the three books found 54 models the
core expresses today:
- 47 cancer and therapy models: tumour growth laws, chemotherapy, immunotherapy, anti-angiogenic therapy, radiotherapy,
  hormone therapy, gene, viral, nanoparticle, and stem-cell therapies, and combinations;
- 7 cellular network models: oscillators, power-law systems, Boolean networks, and a microbial community.

It also found:
- 5 models pending existing changes;
- 7 constraint-based metabolic models blocked on a missing feature, now specified by `add-constraint-based-models`;
- 8 entries out of scope;
- 10 models the library already implements that gain sources.

Konieczny et al. 2023 is qualitative. Its only kinetic model is Michaelis-Menten, which the library already has.

## What Changes

- **Inventory:**
  - add the source keys `Pa`, `Ko`, and `Rm`;
  - record every surveyed model with its field path, sources, and status;
  - add the new sources of implemented models.
- **Pending changes:** list the newly pending models in their specs:
  - `probabilistic_boolean_network` in `add-stochastic-maps`;
  - `vertex_copying` and `network_attack` in `add-adaptive-networks`;
  - `optimal_chemotherapy` and `optimal_immunotherapy` in `add-optimal-control`.
- **Models:** implement the 54 models in ten groups, each with a checkable scenario, a behavioural check per model, and
  a gallery example with a Rerun thumbnail and an analysis plot. They are added to the subfields:
  - `biology/oncology` (43);
  - `biology/systems_biology` (6);
  - `biology/physiology` (2);
  - `biology/ecology` (2);
  - `chemistry/kinetics` (1).

## Capabilities

### New Capabilities

None.

### Modified Capabilities
- `model-library`: the cancer, therapy, and cellular network models of the three references, and their inventory
  record.

## Impact

- **Code:** new modules in `skcomplex.models.biology.oncology`, `systems_biology`, `physiology`, and `ecology`, and in
  `skcomplex.models.chemistry.kinetics`. The core does not change.
- **Docs:**
  - 54 new gallery examples under `docs/examples/`;
  - the inventory and coverage page gain 74 entries.
- **Dependencies:** none.
