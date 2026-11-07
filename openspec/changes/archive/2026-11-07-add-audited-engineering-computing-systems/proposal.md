# Proposal

## Why

The reference audit found engineering and computer science models in the surveyed books and articles that the catalog
lacks although the current core can express them: circuits, mechatronic and control systems, a grain store, robots
with sensory delay, automata and machines, DNA computation, Lenia, and an evolution by circumstance. Building them
exercises the core and fills these fields with the models the references present. A crash of the histogram visual on
integer traits outside the color range of a group came up while building agent systems.

## What Changes

- Add catalog systems for the supported engineering and computer science models of the audit, each in a self-contained
  module with its components and its factory, with tests of a known result of its source, an inventory entry with its
  sources and section, and a gallery example.
- Extend `robby_ga` with Gooding's evolution by circumstance, the regime `'circumstance'`, and move the components of
  RobbyGA into its module.
- Bin every trait of the histogram visual by its own range, with the color range only starting the colored trait.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: the audited engineering and computer science systems and the regime of `robby_ga`.
- `rerun-visualization`: the histogram visual bins every trait by its own range.

## Impact

- `skcomplex.systems.engineering`: new modules in `circuits`, `mechatronics`, `control`, `processes`, and `robotics`.
- `skcomplex.systems.computer_science`: new modules in `automata`, and the module of `robby_ga` in
  `evolutionary_computation`, backward compatible.
- `skcomplex.core.engine.viz`: the histogram visual.
- The inventory, the gallery, and the tests of these subfields.
