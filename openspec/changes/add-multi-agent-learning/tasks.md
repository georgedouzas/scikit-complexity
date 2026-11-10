# Tasks

## 1. Training

- [ ] 1.1 Train every policy of a system on its own returns from shared rollouts with one optimizer state each, and
  verify the returns-by-policy scenario in `tests/core/experiments/test_policy_training.py`
- [ ] 1.2 Add `centralised` with critics of the joint observation, and verify the critic-inputs scenario
- [ ] 1.3 Add `opponents` with a bounded league of past weights, and verify the frozen-opponent scenario

## 2. Probe

- [ ] 2.1 Add `LearningPrisonersDilemmaSystem` in `src/skcomplex/systems/social/cooperation/_learning_dilemma.py`, export
  it, and verify the payoffs-of-the-paper scenario
- [ ] 2.2 Verify the naive-learners-defect scenario, marked slow
- [ ] 2.3 Add the inventory entry `learning_prisoners_dilemma` citing Foerster et al. (2018) with the primitives
  `policies` and `groups`, and the gallery example
  `docs/examples/social/cooperation/plot_learning_prisoners_dilemma.py`, and verify the inventory check and the
  gallery build pass

## 3. Documentation and checks

- [ ] 3.1 Document multi-agent training in the experiments page of the user guide with a runnable example
- [ ] 3.2 Run `nox` (formatting, checks, tests of the touched modules with the thread limits set to 1, and the strict
  docs build) and verify every session passes
