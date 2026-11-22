## Context

An agent's behaviour is an interaction on its group whose `_prepare` sets an `UpdateLaw` with a rule, `rule(agent,
system)`, and a rule of the whole system is `UpdateLaw(s, rule)` with `rule(system)`. The agent solver runs these rules
on Mesa with a system view (`SystemView` in `core/engine/solvers/_mesa.py`, typed by `SystemViewProtocol` in
`core/components/types.py`). The view already offers `rng`, `get_agents`, and `draw(name, distribution, ...)`, and
`system.simulate(record_draws=True)` keeps named draws in a private draw log of the trajectory that
`system.simulate(replay=trajectory)` forces back. Values of agents hold strings and tuples, so a memory of past rounds
or a prompt can live in a value of an agent. `NamingGameSystem` in `social/opinion_dynamics/_conventions.py` is the
classic naming game.

A system is prepared once and simulated many times. `BaseSystem.simulate` takes the arguments of one run, such as
`initial_state`, `random_state`, `observers`, `record_draws`, and `replay`, while the solver and the backend are
settings of the system.

The constitution adds three constraints. A capability with a credential or a real-world side effect lives behind an
optional extra. A credential is named, never passed. The choice of a model stays with the caller. The dependency rule
adds that a library the core uses is a runtime dependency, so the core cannot import a client of a model provider.

## Goals / Non-Goals

**Goals:**
- One core primitive for a rule that asks an outside function for an answer, with reproducibility built in: a seed per
  call, a cache, a record, and a replay.
- A provider adapter that reaches most hosted and local models through one widely used client, behind an extra.
- Probes that use the primitive in three fields, each with a rule-based player so that tests need no model.

**Non-Goals:**
- Generative agents with memory streams, reflection, and plans (2304.03442). They need an embedding oracle and a
  retrieval rule, and a later change adds them as a probe once this primitive exists.
- The donor game of 2412.10270. Its rule-based form adds nothing to the core, and its language-model form is a second
  probe of the same primitive. A later change adds it.
- Asynchronous or batched calls. Rules run in the order of the schedule, so calls run one at a time. Batching is a
  later optimisation inside the view.
- Symbolic and compiled laws. Only the rules of agents and of the whole system, which the agent solver runs with a
  system view, query oracles. Every other function of a law receives no view, so it has no way to query.

## Decisions

### Placement

| Component | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `OracleSpec`, `QueryRecordSpec` | 2, the runtime records and replays them by type | `skcomplex.core.engine.runtime`, beside `Trajectory` | social systems of language-model agents, human-in-the-loop experiments in economics and psychology, calls to a remote simulator in engineering co-simulation, pretrained policies in robotics and ecology of behaviour |
| `query` of the system view, `oracles` of `System.simulate`, the record, the cache, and the replay | 2, the engine handles them | `core/engine/solvers/_mesa.py`, `core/engine/runtime`, `core/systems/_base.py` | as above |
| `chat_oracle`, `parse_options` | 3, a recipe used by several fields | `skcomplex.systems.general.language_models` | social (language, organizations), mathematics (networks) |
| `LlmNamingGameSystem` | 3, one system | `social/language/_llm_naming_game.py` | social |
| The learned player of the naming game | 3, a `PolicyInteraction` of `add-learning-foundation` bound to the group of agents in one system | `social/language/_llm_naming_game.py` | social |
| `LlmNetworkGrowthSystem` | 3, one system | `mathematics/networks/_llm_network_growth.py` | mathematics |
| `GroupGuessingGameSystem` | 3, one system | `social/organizations/_group_guessing_game.py` | social |

The oracle is a core primitive and not a recipe because no core component can express it today. A rule can call any
Python function, but the engine would neither seed the call, nor keep its answer, nor replay it, which step 4 of the
placement rule names a missing primitive. The provider adapter is a recipe because it is only a function that returns an
`OracleSpec`, and the core never imports it (step 5).

The new concept subpackage `language_models` needs the approval of the maintainer before it is created, since every
new package does. Its module has importers in two fields, social and mathematics, as the general-realizations
requirement asks.

### The oracle

- `OracleSpec(respond, name='', cache=None)`. `respond(request, *, seed)` returns a string or a JSON value. `name`
  identifies the model and its settings, such as `'gpt-4o-mini@0.7'`, and enters the cache key, so two oracles with the
  same function and other settings do not share answers. `cache` is the path of an SQLite file or `None` for an
  in-memory cache that lives for one run.
- `system.simulate(until, ..., oracles={'player': spec})`. Oracles are an argument of the run, like `replay`, since
  the caller chooses the model and a prepared system runs with different oracles without compiling again.
- A ready-made system names the oracles its rules use in the class variable `oracles: ClassVar[tuple[str, ...]]`, and
  `simulate` raises `ValidationError` naming the missing ones before the first step. A plain `System` declares none, so
  a query of a name the run was not given raises `ValidationError` at that query, naming it and the names given.
- `system.query(name, request)`. The view draws the call seed as an integer from the run's generator, so one
  `random_state` repeats every seed. The cache key is the SHA-256 hash of the oracle name, the canonical JSON of the
  request, and the seed. A hit returns the stored answer, a miss calls `respond` and stores it.
- The record. Every query appends `QueryRecordSpec(oracle, time, agent, request, seed, answer, cached)` to the
  `queries` of the trajectory, whether or not draws are recorded, since queries are the costly part of a run. Users and
  tests read the records, so `QueryRecordSpec` is a public value and not a private engine record like the draw log.
- The replay. `system.simulate(replay=trajectory)` or `system.simulate(replay=load_queries(path))` serves the recorded
  answers in order, checks that the oracle and the request match, and calls nothing. A mismatch raises `ValidationError`
  naming
  the step, the agent, and the first differing character of the request.
- `save_queries(trajectory, path)` and `load_queries(path)` in `core/engine/runtime` write and read JSON lines. They
  hold no credential, since the oracle never puts its key in a request.

### Oracles, rules, and policies

An oracle is not a learned policy. Its answers are queried, cached, recorded, and replayed, while the actions of a
`PolicyInteraction` of `add-learning-foundation` are draws of a learned function. Both choose among the actions of the
same `EnvironmentSpec`, which every probe declares as a class attribute: the observation is the memory of the agent, the
action is the choice among the options, and the reward is the payoff of the round. The three players are therefore
interchangeable on one game, and a run of the oracle player is compared with a policy trained by `train_policy` on the
rewards of the game. Recorded queries and recorded action draws share the replay path of `simulate`.

### Seeds and determinism

Hosted models honour a seed only on a best-effort basis. The cache and the record make the run reproducible anyway: a
second run with the same seed and the same cache returns the same answers without calls, and a replay needs neither the
cache nor the model. The guide states that a fresh run against a hosted model can differ.

### The provider adapter

- `chat_oracle(model, *, base_url=None, api_key_variable='OPENAI_API_KEY', temperature=0.0, system_prompt=None,
  max_tokens=None)` returns an `OracleSpec` whose `respond` sends the request as the user message through
  `openai.OpenAI(base_url=..., api_key=os.environ[api_key_variable]).chat.completions.create(..., seed=seed)`. The key
  is read where it is used, as the constitution asks, and never stored in the value or the record.
- One client reaches OpenAI, the OpenAI-compatible endpoints of other hosted providers, and local servers such as
  Ollama, vLLM, and llama.cpp. `import_extra('openai', 'chat_oracle', 'language-models')` of `skcomplex.systems`
  raises the named `MissingExtraError` when the extra is not installed.
- `parse_options(answer, options, count=1)` returns the first `count` options that appear in the answer as whole
  tokens, in order of appearance, or raises `ValidationError` quoting the answer. The probes use it for names, letters,
  and numbers.

### Libraries surveyed

- `openai` 3.49.0, released 2026-11-10, is chosen. It is the reference client of the OpenAI chat API, which most hosted
  and local servers implement.
- `litellm` 1.108.4, released 2026-11-09, reaches more providers through their own APIs but brings a large tree of
  dependencies. It is rejected for the extra. A user who needs it passes its `completion` inside their own oracle.
- `anthropic` and other single-provider clients are rejected for the same reason: one adapter, and every other model
  through a user function.
- `diskcache` 5.6.3, released 2023-08-31, would hold the cache but is past the 24-month limit. The standard library
  `sqlite3` holds it instead.
- No library records and replays the calls of an agent-based run, so the record and replay are written in the core, on
  the existing draw log.

### Systems

- **`LlmNamingGameSystem`.** `N=24` agents, a pool of `W=10` names, a memory of `H=5` rounds, the reward `+100` for a
  match and `-50` for a mismatch, and `committed`, the number of agents that always give one name. The pool is shuffled
  for each player and round, as 2410.08948 does to remove the bias of order. `player='oracle'` builds the prompt of the
  paper from the rules, the shuffled names, and the memory, and reads the name with `parse_options`. `player='score'`
  gives the name of largest summed reward in memory, at random among ties and at random with an empty memory.
  `player='learned'` is a `PolicyInteraction` bound to the group of agents with shared weights, whose categorical
  action is the name and whose observation encodes the memory as counts of rewards per name. It is trained
  by `train_policy` with `'reinforce'` on the reward of the rounds. The
  observables are the share of the most common name and the consensus time, the first round at which that share
  reaches 0.9 over a window of rounds.
- **`LlmNetworkGrowthSystem`.** Nodes arrive one at a time with a random name from a list and link to `m` existing
  nodes. `show_degrees=True` lists the degrees beside the names, as in the first experiment of 2312.06619, and `False`
  lists only the names. `player='oracle'` asks for `m` names, and `player='preferential'` links in proportion to
  degree, the Barabási-Albert rule. The degree distribution is read with `distribution` and `fit_law`.
- **`GroupGuessingGameSystem`.** `N` agents guess integers in a range, the group hears whether the sum is above, below,
  or on the hidden target, and the game ends on a hit or after a number of rounds. `player='oracle'` uses the prompts of
  the plain, persona, and theory-of-mind conditions of 2510.05174, and `player='feedback'` moves each guess by a share
  of the error with noise. The guesses are the series that the emergence measures read.

## Risks / Trade-offs

- [Costs and rate limits of hosted models] → The cache and the replay make a second run free, the gallery replays
  recorded queries, and the guide states the number of calls of a run before it starts.
- [Answers that do not parse] → `parse_options` raises with the answer quoted, and every probe takes `retries`, the
  number of new queries with a new seed before it falls back to a random option, recorded as such.
- [Recorded answers are data from a third party] → The gallery records are labelled with the model, the endpoint, the
  date, and the seed, and are stored as data files of the docs, not as reference values of tests.
- [Seeds drawn from the run's generator shift the other draws] → The call seeds come from a child generator spawned
  for queries, so adding an oracle does not change the draws of the rest of the run.

## Migration Plan

None. Existing runs do not query and keep their trajectories.

## Open Questions

- Should the subpackage be named `language_models` or the neutral `oracles`? The recipes are specific to chat models,
  so the design proposes `language_models`.
- Should the `queries` of a trajectory also count tokens and costs when the provider reports them? The design keeps
  only what replay needs.
- Should `oracles` be an argument of `simulate`, as designed, or a setting of the system beside `solver` and `backend`?
  The design keeps it an argument of the run, since the caller and not the system chooses the model.
- Should a plain `System` declare the oracles its rules query, so that a missing one fails before the first step as it
  does for ready-made systems? The system holds only a label, its parts, a space, a solver, and a backend today, so the
  design checks a plain `System` only at the query.
