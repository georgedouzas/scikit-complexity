## Why

Complex adaptive systems are populations of learners: firms, animals, traders, and agents of games adapt to one another,
so the environment of each learner moves as the others learn. `add-learning-foundation` lets a policy act for every
member of a group with shared weights, and designs independent learners and centralised critics without implementing
them. Multi-agent learning dynamics are a subject of their own in the references (2312.12252), and the classic result of
Foerster et al. (2018, arXiv 1709.04326), that naive independent learners of the iterated prisoner's dilemma end in
mutual defection, is a reference the library cannot reproduce today.

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

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `reinforcement-learning`: independent learners, centralised critics, and self-play.
- `model-library`: the learning prisoner's dilemma.

## Impact

- Core: `core/experiments/_policy_training.py`.
- Systems: a new module of `social/cooperation`.
- No new dependency.
- Docs: the experiments page of the user guide and a gallery example of the learning dilemma, whose learned agents
  are measured by the synergy of `extend-network-and-emergence-measures` when that change has landed.
- Depends on `add-learning-foundation`. After `add-learning-bridges`, whose PettingZoo environment exposes the agents of
  the new system to external learners, and after `add-off-policy-learning`, so independent learners can use every
  method.
