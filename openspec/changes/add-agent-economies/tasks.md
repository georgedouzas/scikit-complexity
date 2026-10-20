# Tasks

## 1. Phases

- [x] 1.1 Add phases with activation policies and repeated rounds to the Mesa stepper, keeping single-phase models
  unchanged, and verify the offer and imitation scenarios and the existing agent tests

## 2. Markets

- [x] 2.1 Add `Market` with bilateral search, sample-and-switch, rationed queue, and Walrasian protocols, settled by
  transfers between the agents, and verify each protocol against a worked example
- [x] 2.2 Add the order book, call auction, and deferred acceptance protocols, the last through the `matching` library,
  and verify the double auction and priority scenarios

## 3. Institutions and learning

- [x] 3.1 Add `Institution` with recorded parameter changes, declared reporters, and per-run storage, and verify the
  interest-rate scenario
- [x] 3.2 Add memories, heuristics, logit choice, imitation, and evolutionary operators, and verify the El Farol
  scenario

## 4. Library

- [x] 4.1 Add the economic and social models that exercise the blocks in distinct ways as registered factories with
  curated gallery examples, mark them implemented in the inventory, leave the entries that repeat covered patterns
  pending, and verify the spatial cooperation scenario on the existing lattice model and the conformance suite

## 5. Documentation and gate

- [x] 5.1 Document phases, markets, institutions, and learning in the user guide, and verify the documentation test
  passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
