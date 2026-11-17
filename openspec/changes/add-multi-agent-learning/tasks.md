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

## 3. Learning during a run

- [ ] 3.1 Add the rule `'epsilon-greedy'` and `epsilon` to `ChoiceInteraction` in
  `src/skcomplex/systems/general/agents/_choice.py`, and verify the share-of-the-best-option scenario in
  `tests/systems/general/agents/`
- [ ] 3.2 Add `TemporalDifferenceInteraction` in `src/skcomplex/systems/general/agents/_temporal_difference.py` with
  the four rules, `rate='average'`, and the symbolic and agent forms, export it, and verify the Q-learning-target,
  SARSA-target, and sample-averages scenarios and its doctest
- [ ] 3.3 Create the subfield `computer_science/reinforcement_learning`, approved on 2026-11-10, with its package,
  its catalog entry, and its gallery index, and add `MultiArmedBanditSystem` in `src/skcomplex/systems/computer_science/reinforcement_learning/_reinforcement_tasks.py`, export
  it, and verify the greedy-and-epsilon-greedy-learners scenario, marked slow, with a fast variant of 200 tasks
- [ ] 3.4 Add `CliffWalkingSystem` in the same module, export it, and verify the safe-and-optimal-paths scenario
- [ ] 3.5 Add `online`, `learning_rate`, and `critic` to `PolicyInteraction` with the runtime update after every
  decision, refuse online policies in `train_policy`, and verify the gradient-bandit-with-a-baseline,
  weights-as-a-trajectory, and online-policy-in-training scenarios
- [ ] 3.6 Add the inventory entries `multi_armed_bandit` and `cliff_walking` citing Sutton and Barto (2018) with the
  primitives `learning` and `discrete-states`, and the gallery examples
  `docs/examples/computer_science/reinforcement_learning/plot_multi_armed_bandit.py`, which reproduces Figure 2.2, and
  `docs/examples/computer_science/reinforcement_learning/plot_cliff_walking.py`, and add to the learning prisoner's dilemma example two
  online policies on a random game measured by `lyapunov` after Galla and Farmer (1109.4250), and verify the inventory
  check and the gallery build pass

## 4. Documentation and checks

- [ ] 4.1 Document multi-agent training and learning during a run in the experiments page of the user guide with a runnable example
- [ ] 4.2 Run `nox` (formatting, checks, tests of the touched modules with the thread limits set to 1, and the strict
  docs build) and verify every session passes
