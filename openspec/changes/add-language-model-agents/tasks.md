## 1. Oracle specs

- [ ] 1.1 Add `OracleSpec` and `QueryRecordSpec` in `src/skcomplex/core/specs/_oracle_spec.py` and
  `_query_record_spec.py`, export them, and verify the not-callable scenario and their doctests in
  `tests/core/specs/test_oracle_spec.py` and `test_query_record_spec.py`
- [ ] 1.2 Add `query` to `SystemViewProtocol` and to `SystemView` of `core/engine/solvers/_mesa.py`, with call seeds
  from a generator spawned for queries, and verify the rule-that-asks and other-draws-unchanged scenarios in
  `tests/core/engine/solvers/test_mesa.py`
- [ ] 1.3 Raise `ValidationError` when a symbolic law queries an oracle, and verify the scenario with a test

## 2. Runs, records, and replay

- [ ] 2.1 Add `oracles` to `simulate` and `simulations`, the class variable `oracle_names` of systems, and the check of
  missing oracles, and verify the missing-oracle scenario in `tests/core/engine/runtime/test_simulate.py`
- [ ] 2.2 Add the in-memory and SQLite caches keyed by the hash of the oracle name, the request, and the seed, and
  verify the same-seed and cache-on-disk scenarios
- [ ] 2.3 Add `Trajectory.queries` and extend `replay` to serve recorded answers and to check requests, and verify the
  trajectory-without-queries, draws-and-queries, and replay-that-does-not-match scenarios
- [ ] 2.4 Add `save_queries` and `load_queries` in `core/engine/runtime`, export them, and verify the
  replay-without-the-oracle scenario on a fake oracle and their doctests
- [ ] 2.5 Document oracles, seeds, the cache, and replay in the agent section of the user guide with a runnable
  example on a fake oracle, and verify the markdown tests run it

## 3. Language-model recipes

- [ ] 3.1 Ask the maintainer to approve the subpackage `skcomplex.systems.general.language_models`, and record the
  answer in this design before any code is written
- [ ] 3.2 Add the optional extra `language-models` with `openai>=3.49.0` in `pyproject.toml` and the Project Profile,
  and verify `pdm install -G language-models` resolves
- [ ] 3.3 Add `chat_oracle` and `parse_options` with `import_extra`, and verify the extra-not-installed, missing-key,
  options, no-option, and no-credential scenarios with a fake client in `tests/systems/general/language_models/`

## 4. Probe systems

- [ ] 4.1 Add `LlmNamingGameSystem` in `src/skcomplex/systems/social/language/_llm_naming_game.py` with both players,
  committed agents, and the consensus observables, and verify the consensus, committed-minority, and prompt scenarios
- [ ] 4.2 Once `add-learning-foundation` is applied, declare the environment of every probe and add the learned
  player of the naming game, and verify the learned-players-reach-consensus scenario, marked slow
- [ ] 4.3 Add `LlmNetworkGrowthSystem` in `src/skcomplex/systems/mathematics/networks/_llm_network_growth.py`, and
  verify the preferential-player and request scenarios
- [ ] 4.4 Add `GroupGuessingGameSystem` in `src/skcomplex/systems/social/organizations/_group_guessing_game.py` with
  the three conditions, and verify the feedback-player and prompt scenarios
- [ ] 4.5 Add the inventory entries `llm_naming_game`, `llm_network_growth`, and `group_guessing_game` with sources
  2410.08948, 2312.06619, and 2510.05174, and the primitive `oracle-queries` with
  `change = "add-language-model-agents"`, and verify the inventory check passes
- [ ] 4.6 Record one run of each probe against a named model with `save_queries`, store the logs as labelled data
  files under `docs/examples/`, and add a gallery example of each probe that replays its log and compares it with the
  rule-based player, and verify the strict gallery build runs them without network access

## 5. Integration

- [ ] 5.1 Commit the work as conventional `feat` commits and verify `nox -s changelog` lists them
- [ ] 5.2 Run `nox` formatting, checks, tests of the touched modules, and the strict docs build, and fix every finding
