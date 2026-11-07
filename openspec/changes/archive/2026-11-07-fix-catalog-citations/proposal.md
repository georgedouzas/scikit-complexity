## Why

The reference audit found catalog entries whose citations name sections that do not hold the rules the systems
implement, and matches between catalog systems and sources that the inventory does not record. A citation that does
not lead to the implemented rules misleads every reader who follows it.

## What Changes

- Correct the sections of entries that cite the wrong place: `creative_phase_transition` (Th 5.6.2),
  `majority_rule` (Ar 4.3), `cucker_smale_flocking` (Be 6.7.3), `holme_kim` (TC 4 and Holme and Kim 2002),
  `nicholson_bailey` (Ar Problem 14.7), `food_webs` (Ar Problems 14.3 and 14.4), and `unbounded_alphabet_automaton`
  (CC 3.3.8, whose additive rule it lifts to the integers).
- Make the entries whose rules differ from their cited chapters describe what they implement, with the primary source of
  their rules: `order_book`, `evacuation`, `fair_food`, `goods_traders`, `double_auction`, and `winners_take_all`. Their
  docstrings, registered references, inventory titles, and gallery text follow. No missing model is implemented.
- Add the source key `EX` for Ceccherini-Silberstein and Coornaert's "Exercises in Cellular Automata and Groups"
  (2023) and cite it where the audit matched a catalog system.
- Add the sections the audit matched without citing them: Je 3.4 for `tangled_nature`, Ha 4.2 for
  `threshold_adoption`, Ha 5.3 for `barter`, and the Erdős-Rényi and Watts-Strogatz sections of Ar, Je, and Th for
  `random_graph_ensembles`.
- Add the arXiv identifiers with sections of the new papers to the catalog systems they provide.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-inventory`: an entry's sources name the sections that hold the rules the system implements, or the primary
  source of those rules when no surveyed section does.

## Impact

`docs/overview/inventory.toml`, the docstrings and registered references of the affected systems under
`src/skcomplex/systems`, and the gallery examples of `fair_food`, `goods_traders`, and `order_book`. No behaviour of any
system changes.
