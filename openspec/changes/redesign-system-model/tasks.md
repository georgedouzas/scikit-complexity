# Tasks

Each task writes its tests first, checks that they fail on the current code, and then implements. Every test follows
the naming and public-API rules of the constitution. Migration tasks compare trajectories before and after on fixed seeds
through the conformance suite.

## 1. Inventory gate (no code changes)

- [x] 1.1 Re-tag every inventory entry, implemented and pending, with the following, and add the inventory check:
  - time kind;
  - space kind;
  - component kinds;
  - interaction bindings;
  - law classes.
- [x] 1.2 Review every entry the check reports as unmapped. Each one either maps or gains a general primitive in this
  change's design and specs. Record the decisions in the design.
- [x] 1.3 Record a baseline of trajectories: every catalog model with random state 0, stored outside the repository for
  the migration comparisons.

## 2. Declarations and validation

- [x] 2.1 Add the requirement and support declarations to every law kind, component kind, space kind, time kind,
  solver, and backend. Add the test that fails on a missing declaration.
- [x] 2.2 Add `validate`, `ValidationReport`, and `Issue` with stable codes for each check. Verify the scenarios of
  several errors, wrong time, wrong domain, unsupported solver, unavailable backend, and mixed classes.
- [x] 2.3 Call `validate` first in `compile_system` and `Simulation`. Retire each compiler and solver raise that
  validation now covers, keeping a test per retired raise.
- [ ] 2.4 Replace the compiler's form table with derivation from solver declarations, and verify the compilation
  scenarios. This runs after 4.1: the table tells exponential from renewal jumps by law type, and solvers can declare
  the clocks they support only once `Renewal` is a `Jump` with a clock.
- [x] 2.5 Generate the compatibility table in the documentation from the declarations.

## 3. System, space, and time

- [x] 3.1 Change the signature to `System(label, components, interactions, space, time)` and make a system a component.
  Move the old arguments to their new homes:
  - observables to simulations and catalog entries;
  - history to states;
  - noise convention to noise laws;
  - phases to the schedule of discrete time.
- [x] 3.2 Add the time kinds with inference and the synchronous, random-sequential, and phases schedules, and static
  time validated; solving static systems is specified by `add-static-models`.
- [x] 3.3 Unify the space kinds across time classes:
  - boundary conditions on the space;
  - hyperedges on networks;
  - agent spaces as the same kinds.
- [x] 3.4 Add variable groups and groups on sites, edges, and hyperedges, with their validation. Reading edge states
  and rewiring are specified by `add-adaptive-networks`.
- [x] 3.5 Add the structured state domains: Hilbert space, sequence, graph, and tiling. Dissolve the core's
  quantum-element, word, patch, and host-graph classes into elements with those domains.
- [x] 3.6 Add role arities and the bindings `'members'`, `'pairs'`, `'neighbours'`, and `'hyperedges'`, with their
  validation. Replace `PairInteraction` with them, and reject states on interactions. Laws run once per hyperedge are
  specified by `add-higher-order-interactions`.

## 4. Laws and general primitives

- [ ] 4.1 Classify every law kind by time semantics. Absorb `Renewal` into `Jump` with a clock, and absorb `Energy`,
  `Move`, and `Sampler` into `Sample` with Metropolis as the only core acceptance.
- [ ] 4.2 Add `Transfer`, `Invariant`, `Match` with random pairing, and `parameter(adaptive=True)`, and verify their
  scenarios.
- [ ] 4.3 Rename `Hamiltonian` to `Generator`, and keep `Dissipator`.
- [ ] 4.4 Add the distributed-delay expression and named expressions, which replace the system-dynamics delays and
  auxiliaries.

## 5. Move realizations to the library

- [ ] 5.1 Move each group of realizations into its field module and rebuild it on core primitives:
  - markets, protocols, institutions, and learning helpers to `economics` and `social`;
  - potentials, boxes, thermostats, quantum species, and sandpile and extremal rules to `physics`;
  - turtles, tilings, substitutions, and rule codes to `mathematics`;
  - system-dynamics helpers to `engineering`.
- [ ] 5.2 Add the field-term test and the import-graph test that keep the core application agnostic, and verify that
  both pass.
- [ ] 5.3 Remove `MesaSystem` and `AgentType`, make Mesa a backend with its declared support, and port the gallery
  example that wrapped a Mesa model.

## 6. Migrate the library

- [ ] 6.1 Migrate the models field by field to the new vocabulary, comparing each against the baseline of task 1.3.
- [ ] 6.2 Migrate the gallery examples and the user guide to the new system definition, and regenerate the inventory
  coverage by category.

## 7. Pending changes and gate

- [ ] 7.1 Rebase the artifacts of the 26 pending changes onto the new vocabulary, and validate every one of them.
- [ ] 7.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass.
