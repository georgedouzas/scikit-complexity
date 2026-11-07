# Proposal

## Why

The reference audit found physics models in the surveyed books and articles that the catalog lacks although the
current core can express them: nonlinear oscillators, mechanics and celestial mechanics, climate and social-climate
models, synchronization, sandpile variants, lattice gases, active matter, and rigid Brownian bodies. Building them
exercises the core and fills the physics fields of the catalog with the models the references present.

## What Changes

- Add catalog systems for the supported physics models of the audit, each with its module, registration, tests of a
  known result of its source, an inventory entry with its sources and section, and a gallery example.
- Extend existing systems with an option where the source presents a variant: the three-disk open billiard in
  `billiards`, the horizontally oscillating pivot in `parametric_pendulum`, the delay-coupled Bernoulli maps in
  `aggregate_averaging_synchronization`, the Nodes-Links-Triangles coupling in `topological_kuramoto`, and the
  topological, vision-cone, and delayed alignments in `vicsek`.
- Count a reduction over a whole group as a number of no degree when the compiler checks the degrees of the exterior
  calculus, so a rate on links may scale by an order parameter of a projection on another part of a cell complex.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: the audited physics systems and the variants of existing systems.
- `link-fields`: the degree check treats group reductions as numbers.

## Impact

- `skcomplex.systems.physics`: new modules and elements in the subfields `nonlinear_oscillators`, `mechanics`,
  `celestial_mechanics`, `climate`, `synchronization`, `criticality`, `fluids`, `active_matter`, and `soft_matter`, and
  new options of five existing systems, all backward compatible.
- `skcomplex.core.engine.compile`: the degree check of `PassCompiler`.
- The inventory, the gallery, and the tests of the physics systems.
