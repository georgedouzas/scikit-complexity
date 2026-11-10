## Context

`add-learning-foundation` adds `PolicyInteraction`, whose environment carries its own reward, and `train_policy`, which
finds every policy of a system by type. A policy bound to a group acts for every member with shared weights.
`add-off-policy-learning` adds `'dqn'`, `'sac'`, and `'fqi'`. `add-learning-bridges` exposes several policies as the
agents of a PettingZoo environment. `PrisonersDilemmaTournamentSystem` in `social/cooperation/_tournaments.py` plays
fixed strategies of Axelrod's tournaments.

## Goals / Non-Goals

**Goals:**
- Several learners in one system, each with its own reward, trained together.
- Centralised training with decentralised acting, and self-play.
- A probe with a published outcome.

**Non-Goals:**
- Learning with opponent learning awareness (LOLA), which differentiates through the update of the other learner. It
  can follow as a method once independent learners exist.
- Communication channels between agents. A message is an action that another policy observes, so it needs no new part.

## Decisions

### Placement

| Part | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| Independent learners, `centralised`, `opponents` | 2, options of an experiment over any system | `core/experiments/_policy_training.py` | social systems (games, norms), economics (markets of learning traders), ecology (foraging and predator and prey learning), computer science (games) |
| `LearningPrisonersDilemmaSystem` | 3, one system | `social/cooperation/_learning_dilemma.py` | social |

### Independent learners

- Every `PolicyInteraction` of the system is a learner with its own environment and reward. One rollout records the
  draws of every policy, and each policy gets the gradient of its own returns. The optimizer state is kept per policy.
- A policy bound to a group stays one learner with shared weights, its return being the mean over members.

### Centralised critics

- With `centralised=True` and `'ppo'`, the critic of every policy reads the concatenated observations of all policies
  at the decision, and acting stays decentralised.

### Self-play

- `opponents` maps a policy label to the label of the policy whose frozen copy it plays, refreshed every
  `refresh` epochs from a league of past weights drawn uniformly.

### Probe

- `LearningPrisonersDilemmaSystem(learners='policy', memory=1, discount=0.96, random_state=None)` holds two players whose
  action is cooperate or defect, observing the last joint action, with the payoffs `(-1, -1)`, `(-3, 0)`, `(0, -3)`,
  and `(-2, -2)` of Foerster et al. (2018). Each player is a `PolicyInteraction` with its own reward. The reference
  value is the outcome of naive learners in that paper: mutual defection, near `-2` per step.

## Risks / Trade-offs

- [Independent learners do not converge in general] → The scenarios check the published outcome, and the user guide
  states that non-stationarity is the subject, not a defect.
- [Self-play leagues grow memory] → The league keeps a bounded number of past weights.

## Open Questions

- Should LOLA be a method of `train_policy` in a later change? It needs gradients through the update of the opponent,
  which the shared loop can give for differentiable policies.
