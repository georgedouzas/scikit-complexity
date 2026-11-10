## 1. Dynamic network dilemma

- [ ] 1.1 Add `critical_benefit_cost_ratio` in `src/skcomplex/systems/social/cooperation/_dynamic_network_dilemma.py`
  and export it, and verify the dynamic-pair and static-network scenarios in `tests/systems/social/cooperation/`
- [ ] 1.2 Add `DynamicNetworkDilemmaSystem` with the structure state, its transitions, and death-birth updates on the
  current layer, export it, and verify the simulated-fixation scenario, marked slow
- [ ] 1.3 Add the gallery example `docs/examples/social/cooperation/plot_dynamic_network_dilemma.py` with the critical
  ratio over `N` for the static and dynamic networks, and verify it runs

## 2. Gossip group norms

- [ ] 2.1 Add `GossipGroupNormsSystem` in `src/skcomplex/systems/social/cooperation/_gossip_group_norms.py` with views
  on edges, norm tables, errors, and migration, export it, and verify the norm-tables scenario
- [ ] 2.2 Verify the Stern-Judging-overtakes-Scoring scenario, marked slow
- [ ] 2.3 Add the gallery example `docs/examples/social/cooperation/plot_gossip_group_norms.py` with the share of the
  Stern Judging group over time against each other norm, and verify it runs

## 3. Graph punishment game

- [ ] 3.1 Add `GraphPunishmentGameSystem` in `src/skcomplex/systems/social/cooperation/_graph_punishment_game.py`,
  export it, and verify the two-strategy scenario
- [ ] 3.2 Verify the threshold-of-punishment scenario, marked slow
- [ ] 3.3 Add the gallery example `docs/examples/social/cooperation/plot_graph_punishment_game.py` with the final share
  of defectors over `beta` and the threshold of the paper, and verify it runs

## 4. Inventory

- [ ] 4.1 Add the inventory entries `dynamic_network_dilemma`, `gossip_group_norms`, and `graph_punishment_game` under
  `[social.cooperation]` with the sources 2301.11982, 2204.10811, and 2401.11686 and the primitives of the design, and
  verify the inventory check passes
- [ ] 4.2 If a system cannot be expressed with core parts, record it as blocked in the inventory, open a change for the
  missing primitive, and verify the inventory check accepts the blocked entry

## 5. Integration

- [ ] 5.1 Commit the work as conventional `feat` commits and verify `nox -s changelog` lists them
- [ ] 5.2 Run `nox` formatting, checks, tests of the touched modules, and the strict docs build, and fix every finding
