## Context

`add-learning-foundation` adds `PolicyInteraction`, whose environment carries its own reward, and `train_policy`, which
finds every policy of a system by type. A policy bound to a group acts for every member with shared weights.
`add-off-policy-learning` adds `'dqn'`, `'sac'`, and `'fqi'`. `add-learning-bridges` exposes several policies as the
agents of a PettingZoo environment. `PrisonersDilemmaTournamentSystem` in `social/cooperation/_tournaments.py` plays
fixed strategies of Axelrod's tournaments.

## Goals / Non-Goals

**Goals:**
- Several learners in one system, each with its own reward, trained together.
- Learners whose learning is part of the run: tabular temporal difference learners as a recipe, and policies that
  learn at every decision.
- Centralised training with decentralised acting, and self-play.
- A probe with a published outcome.

**Non-Goals:**
- Learning with opponent learning awareness (LOLA), which differentiates through the update of the other learner. It
  can follow as a method once independent learners exist.
- Communication channels between agents. A message is an action that another policy observes, so it needs no new
  component.

## Decisions

### Placement

| Part | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| Independent learners, `centralised`, `opponents` | 2, options of an experiment over any system | `core/experiments/_policy_training.py` | social systems (games, norms), economics (markets of learning traders), ecology (foraging and predator and prey learning), computer science (games) |
| `LearningPrisonersDilemmaSystem` | 3, one system | `social/cooperation/_learning_dilemma.py` | social |
| `online` of `PolicyInteraction` | 2, an option of a core interaction that the runtime handles by type | `core/components/interactions/_policy.py` | economics (traders that learn while trading), social systems (norms), ecology (foraging), neuroscience (learning animals), computer science (games) |
| `TemporalDifferenceInteraction` | 3, a recipe of core update laws and rules used by several fields | `skcomplex.systems.general.agents`, approved and existing | economics (learning traders and firms), social systems (norms and conventions), neuroscience (dopamine prediction errors, Sutton and Barto ch. 15), ecology (foraging), computer science (games) |
| `'epsilon-greedy'` of `ChoiceInteraction` | 3, a rule of an existing recipe | `general/agents/_choice.py` | as above |
| `MultiArmedBanditSystem`, `CliffWalkingSystem` | 3, one system each | `computer_science/reinforcement_learning/_reinforcement_tasks.py` | computer science |

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

### Temporal difference learners

- `TemporalDifferenceInteraction(label, target, values, observation, action, reward, *, rule='q-learning', rate=0.1,
  discount=0.9, form='symbolic')`. `values` names a shaped state of every member, its table of action values with one
  row per observed state and one column per action, or one row for a stateless learner. `observation` and `action`
  name integer states of the member, and `reward` is an expression of the namespace in the symbolic form or a callable
  of the agent and the system in the agent form, as in `PayoffLearningInteraction`.
- After every step it moves the value of the last observation and action towards the target at the rate: `r + discount
  max_a Q(s', a)` for `'q-learning'`, `r + discount Q(s', a')` for `'sarsa'`, the expectation under the choice
  probabilities for `'expected-sarsa'`, and `r + discount V(s')` for `'values'`, TD(0) of state values (Sutton and
  Barto 6.1 to 6.6). `discount=0` with one row is the sample average learner of a bandit when the rate is `1 / n`, which
  `rate='average'` gives.
- The choice is `ChoiceInteraction` with the utility of option `k` read from the row of the observed state, so choice
  and learning stay two components, as payoff learning and choice already are. `ChoiceInteraction` gains the rule
  `'epsilon-greedy'`: the best option, ties broken as `ties` says, with probability `1 - epsilon`, and a uniform draw
  otherwise.
- `PayoffLearningInteraction` of `economics/common/learning` is the stateless case with payoffs of every option. It
  stays where it is, since moving it is a package move that needs the maintainer's approval and nobody asked for it.
- No library: the updates are map laws on numpy states.

### Online policies

- `PolicyInteraction(..., online=False, learning_rate=0.01, critic=None)`. With `online=True` the runtime, which already
  handles the policy at every decision, also applies one gradient step to its weights after the reward of the decision
  is known: the one-step actor-critic of Sutton and Barto (13.5) when `critic` is an Equinox module of the observation,
  updated by its own TD error, and otherwise the reward minus a running mean baseline times the score of the action,
  the gradient bandit of Sutton and Barto (2.8). The weights are adaptive parameters, so the trajectory records them at
  every frame, and the run is the learning.
- A policy bound to a group updates its shared weights from the mean of the members' steps, the collective
  reinforcement learning of Sutton and Barto (15.10).
- Galla and Farmer (1109.4250) read the strategies of two learners as a dynamical system with Lyapunov exponents. With
  online policies the weights of two learners in one system are such a trajectory, and the gallery example of two
  online policies on a random game measures it with `lyapunov`.
- `train_policy` refuses a system with an online policy, since its weights change during the rollouts, and names
  `online`.

### Probe

- `LearningPrisonersDilemmaSystem(learners='policy', memory=1, discount=0.96, random_state=None)` holds two players whose
  action is cooperate or defect, observing the last joint action, with the payoffs `(-1, -1)`, `(-3, 0)`, `(0, -3)`,
  and `(-2, -2)` of Foerster et al. (2018). Each player is a `PolicyInteraction` with its own reward. The reference
  value is the outcome of naive learners in that paper: mutual defection, near `-2` per step.

### Probes of learning during a run

- `MultiArmedBanditSystem(arms=10, mean=0.0, steps=1000, random_state=None)` draws the true values of the arms from a
  normal distribution of the given mean and unit variance, and rewards of unit variance, the 10-armed testbed of
  Sutton and Barto (sect. 2.3), with one learner holding a `TemporalDifferenceInteraction` and a `ChoiceInteraction`,
  or an online policy with `learner='policy'`. The reference values, checked in
  `references/textbooks/Sutton and Barto - 2018 - Reinforcement Learning An Introduction.pdf` (pp. 29 and 30, Figure
  2.2), are averages of 2000 runs of 1000 steps: the greedy learner reaches a reward of about 1 per step against the
  best possible of about 1.54, and finds the best arm in about one third of the tasks, and epsilon 0.1 never picks the
  best arm more than 91% of the time, the bound `1 - 0.1 + 0.1 / 10`. Figure 2.5 shows that a gradient bandit with a
  baseline beats one without when the true values are near +4, with no numbers in the text, so that check is relative.
- `CliffWalkingSystem(rule='sarsa', epsilon=0.1, random_state=None)` is the 4 by 12 gridworld of Example 6.6, with
  reward -1 per move and -100 for the cliff, which sends the agent back to the start. The book reports that
  Q-learning learns the optimal path along the edge of the cliff and SARSA the safer path, and that SARSA earns more
  per episode online with epsilon 0.1.
- Both live in the new subfield `computer_science/reinforcement_learning`, approved by the user on 2026-11-10, since
  they are benchmarks of reinforcement learning. Later reinforcement learning probes join them there.

## Risks / Trade-offs

- [Online gradients make runs depend on the order of decisions of a group] → The update uses the mean of the members'
  steps, which does not depend on their order.
- [The testbed needs 2000 runs of 1000 steps] → The full reference is a slow test and the gallery example, and the fast
  test uses 200 runs with a looser tolerance.

- [Independent learners do not converge in general] → The scenarios check the published outcome, and the user guide
  states that non-stationarity is the subject, not a defect.
- [Self-play leagues grow memory] → The league keeps a bounded number of past weights.

## Open Questions

- Should LOLA be a method of `train_policy` in a later change? It needs gradients through the update of the opponent,
  which the shared loop can give for differentiable policies.
