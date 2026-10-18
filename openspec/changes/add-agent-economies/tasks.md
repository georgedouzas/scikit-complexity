# Tasks

## 1. Phases

- [ ] 1.1 Add phases with activation policies and repeated rounds to the Mesa stepper, keeping single-phase models
  unchanged, and verify the offer and imitation scenarios and the existing agent tests

## 2. Markets

- [ ] 2.1 Add `Market` with bilateral search, sample-and-switch, rationed queue, and Walrasian protocols, settled through
  transfers, and verify each protocol against a worked example
- [ ] 2.2 Add the order book, call auction, and deferred acceptance protocols, and verify the double auction and
  priority scenarios

## 3. Institutions and learning

- [ ] 3.1 Add `Institution` with recorded parameter changes, and verify the interest-rate scenario
- [ ] 3.2 Add memories, heuristics, logit choice, imitation, and evolutionary operators, and verify the El Farol
  scenario

## 4. Library

- [ ] 4.1 Add the economic and social models as registered factories, and verify the spatial cooperation scenario and
  the conformance suite

## 5. Documentation and gate

- [ ] 5.1 Document phases, markets, institutions, and learning in the user guide, and verify the documentation test
  passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
