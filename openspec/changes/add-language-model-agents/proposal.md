## Why

Populations of language-model agents are now studied as complex adaptive systems. They agree on conventions in naming
games and are tipped by committed minorities (2410.08948), grow scale-free networks when they choose whom to follow
(2312.06619), pass strategies of indirect reciprocity across generations (2412.10270), live in small towns with memory
and plans (2304.03442), and coordinate as groups whose synergy can be measured (2510.05174). In every case a rule of an
agent asks an outside model for its decision. The core runs agent rules written in Python, records named draws, and
replays them, but a rule cannot ask an outside function for an answer in a way that a seed repeats, a cache keeps, and a
record replays. Without that, a run that costs money and time cannot be reproduced, shared, or analysed again.

## What Changes

- New frozen value `OracleSpec(respond, name, cache)` in `skcomplex.core.engine.runtime`, beside `Trajectory`, since
  the runtime reads it. An oracle is any function `respond(request, *, seed)` that returns a text or a JSON value, such
  as a language model, a remote simulator, or a human operator.
- `System.simulate` gains `oracles`, a mapping from the name a rule uses to an `OracleSpec`, chosen by the caller and
  never by the system. Like `replay` and `observers`, it is an argument of the run, not a setting of the system.
- The system view that agent rules receive, `SystemView` of the agent solver, gains `query(name, request)`, and
  `SystemViewProtocol` declares it. A query draws a call seed from a random stream of the run kept for queries, returns
  a cached answer for the same oracle, request, and seed when one exists, and calls the oracle otherwise.
- Every query is kept in the trajectory as a `QueryRecordSpec`, a public value beside `Trajectory`, with the oracle
  name, the time, the agent, the request, the seed, and the answer. `system.simulate(replay=...)` returns the recorded
  answers without calling the oracle and raises `ValidationError`, as a replay of draws does, when a rule sends a
  request that differs from the record. `save_queries` and `load_queries` write and read the record as JSON lines, so a
  run can be replayed on a machine without credentials.
- New concept subpackage `skcomplex.systems.general.language_models`, which needs the maintainer's approval, with
  `chat_oracle(model, *, base_url, api_key_variable, temperature, system_prompt)`, which returns an `OracleSpec` that
  calls any endpoint compatible with the OpenAI chat API, and `parse_options(answer, options, count)`, which reads the
  chosen options from a free-text answer. It sits behind the new optional extra `language-models`.
- New systems, each with a rule-based player beside the oracle player, so that tests and comparisons need no model.
  Each declares its decisions as the actions of an environment of `add-learning-foundation`, so the oracle, the rules,
  and a learned `PolicyInteraction` play the same game, and the naming game gains `player='learned'`, trained by
  `train_policy`, as a third baseline:
  - `LlmNamingGameSystem` in `social/language`, the pairwise naming game with a memory of past rounds, rewards, and a
    committed minority (2410.08948)
  - `LlmNetworkGrowthSystem` in `mathematics/networks`, a network grown by nodes that choose whom to link to from the
    names and, optionally, the degrees of the existing nodes (2312.06619)
  - `GroupGuessingGameSystem` in `social/organizations`, agents that guess numbers whose sum must hit a hidden target
    from group feedback (2510.05174)

## Capabilities

### New Capabilities

- `oracle-queries`: rules that ask an outside function for an answer, with call seeds, a cache, a record in the
  trajectory, and replay without the oracle.

### Modified Capabilities

- `agent-based-modeling`: agent rules query oracles through the system view.
- `simulation`: runs take oracles, record their queries, and replay them.
- `general-realizations`: the subpackage `language_models` with the chat oracle and the parsing of options.
- `model-library`: the language-model naming game, network growth, and group guessing game.

## Impact

- Core: `OracleSpec` and `QueryRecordSpec` in new private modules of `core/engine/runtime`, `save_queries` and
  `load_queries` in the same package, `Trajectory` in `core/engine/runtime/_trajectories.py`, `BaseSystem.simulate` in
  `core/systems/_base.py`, the agent system view in `core/engine/solvers/_mesa.py`, and `SystemViewProtocol` in
  `core/components/types.py`. No change to the law kinds, and runs without agent rules are unchanged.
- Systems: a new concept subpackage `skcomplex.systems.general.language_models`, and new modules in
  `social/language`, `mathematics/networks`, and `social/organizations`.
- Dependencies: the optional extra `language-models` holds `openai` 3.49.0, released 2026-11-10. The core uses only
  the standard library (`hashlib`, `json`, `sqlite3`) for seeds, records, and the cache.
- Docs: the agent section of the user guide, a gallery example of each probe that replays recorded queries, and the
  inventory with the primitive `oracle-queries`.
- Depends on `add-learning-foundation` only for the environments of the probes and the learned player. The oracle
  primitive, its record, and its replay do not need it.
- `extend-network-and-emergence-measures` adds the synergy measures that the gallery example of the group guessing game
  reads. The probe lands after it or shows only the practical criterion that exists today.
