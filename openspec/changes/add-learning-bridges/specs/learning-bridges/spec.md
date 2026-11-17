# Spec Delta

## Purpose

Lets a user run external agents on any system through Gymnasium and PettingZoo, and train policies of the library on
external environments.

## ADDED Requirements

### Requirement: Systems are Gymnasium environments
`make_environment(system)` SHALL return a Gymnasium environment whose spaces follow the environment of the system, whose
`reset` starts a fresh run from a seed, and whose `step` sets the actions, advances one decision interval, and returns
the observation, the reward, and the terminated and truncated flags.

#### Scenario: Checker passes
- **WHEN** `gymnasium.utils.env_checker.check_env` runs on the environments of `CartPoleSystem`,
  `SIRVaccinationSystem`, and `PendulumSwingUpSystem`
- **THEN** it passes for all three

#### Scenario: Same seed, same episode
- **WHEN** an environment is reset twice with `seed=3` and stepped with the same actions
- **THEN** both episodes return equal observations and rewards

#### Scenario: Policy already in the system
- **WHEN** `make_environment` is called on a system whose policy sets the actions of its environment
- **THEN** it raises `ValidationError` that names the policy and says an external agent replaces it

### Requirement: Systems are PettingZoo environments
`make_parallel_environment(system)` SHALL return a PettingZoo parallel environment with one agent per policy of the
system, or per member of a group bound to a policy, each with the spaces of its environment and its own reward.

#### Scenario: API test passes
- **WHEN** `pettingzoo.test.parallel_api_test` runs on a system of 4 learning agents bound to one policy
- **THEN** it passes, and the environment has 4 agents

### Requirement: External environments are systems
`GymnasiumSystem` and `PettingZooSystem` SHALL wrap an external environment as a system whose actions are adaptive
parameters and whose observations, rewards, and flags are states, with an environment as a class attribute, so a
`PolicyInteraction` acts on it and `train_policy` trains it.

#### Scenario: Policy trained on Gymnasium's cart-pole
- **WHEN** `train_policy` with `'ppo'` trains a policy on `GymnasiumSystem('CartPole-v1')` for 200 epochs with
  `random_state=0`
- **THEN** the mean return over 100 evaluation episodes is at least 475

#### Scenario: Not differentiable
- **WHEN** `train_policy` with `'pathwise'` is called on a `GymnasiumSystem`
- **THEN** it raises `ValidationError` that names the tag `differentiable`
