## Why

Complex adaptive systems are populations of learners: firms, animals, traders, and agents of games adapt to one another,
so the environment of each learner moves as the others learn. `add-learning-foundation` lets a policy act for every
member of a group with shared weights, and designs independent learners and centralised critics without implementing
them. Multi-agent learning dynamics are a subject of their own in the references (2312.12252), and the classic result of
Foerster et al. (2018, arXiv 1709.04326), that naive independent learners of the iterated prisoner's dilemma end in
mutual defection, is a reference the library cannot reproduce today.

Learning is also the dynamics of such systems, not only a procedure run before them. Galla and Farmer (1109.4250) study
the strategies of two learners as a dynamical system whose attractors are fixed points, limit cycles, or chaos, Sutton
and Barto (sect. 15.10) treat the collective reinforcement learning of a population, and Carmichael et al. (sect. 2.4)
and Mobus (ch. 11) define a learning agent as a component whose decision model changes while the system runs. The
library learns inside a run only by payoff averaging and the SK game. It has no recipe for state dependent temporal
difference learning, and the weights of a `PolicyInteraction` are frozen during `simulate`.

## What Changes

- `train_policy` trains several policies of one system as independent learners, each on the returns of its own reward,
  in one call that shares the rollouts.
- `train_policy` gains `centralised=True`, where the critic of every policy reads the joint observation of all
  policies, multi-agent PPO after Yu et al. (2022, arXiv 2103.01955).
- `train_policy` gains `opponents`, a mapping from policy labels to frozen copies drawn from earlier epochs, for
  self-play and training against a league of past policies.
- `PolicyTrainingResult` gains `returns` by policy label when the system has several policies.
- New system `LearningPrisonersDilemmaSystem` in `social/cooperation`, two learners of the iterated prisoner's dilemma
  with a memory of the last joint action and the payoffs of Foerster et al. (2018).
- **Learning during a run.** New recipe `TemporalDifferenceInteraction` in `skcomplex.systems.general.agents`, an
  existing approved subpackage, which updates a table of action values or state values of every member by Q-learning,
  SARSA, expected SARSA, or TD(0) after every step (Sutton and Barto ch. 6). `ChoiceInteraction` gains the rule
  `'epsilon-greedy'`, so a learner chooses from its values with the recipes that exist.
- `PolicyInteraction` gains `online=True`, which updates its weights at every decision inside the run, by one-step
  actor-critic with a critic or by the reward against a running baseline without one (Sutton and Barto 13.5 and 2.8),
  so the weights are a trajectory that `lyapunov` and the emergence measures read.
- New systems `MultiArmedBanditSystem` and `CliffWalkingSystem` in `computer_science/reinforcement_learning`, the 10-armed testbed of
  Sutton and Barto (sect. 2.3) and their cliff walking gridworld (Example 6.6), the probes of the recipe and of online
  policies.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `reinforcement-learning`: independent learners, centralised critics, self-play, and online policies.
- `general-realizations`: temporal difference learners and epsilon-greedy choice.
- `model-library`: the learning prisoner's dilemma, the multi-armed bandit, and the cliff walk.

## Impact

- Core: `core/experiments/_policy_training.py` and `core/components/interactions/_policy.py`.
- Systems: a new module of `social/cooperation`, `general/agents/_temporal_difference.py` and `_choice.py`, and a new
  module `computer_science/reinforcement_learning/_reinforcement_tasks.py`.
- No new dependency.
- Docs: the experiments page of the user guide and a gallery example of the learning dilemma, whose learned agents
  are measured by the synergy of `extend-network-and-emergence-measures` when that change has landed.
- Depends on `add-learning-foundation`. The recipe, the epsilon-greedy rule, and the two task systems need none of the
  learning drafts, so their tasks can be applied first. After `add-learning-bridges`, whose PettingZoo environment exposes the agents of
  the new system to external learners, and after `add-off-policy-learning`, so independent learners can use every
  method.
