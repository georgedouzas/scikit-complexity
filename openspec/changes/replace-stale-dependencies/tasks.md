# Tasks

## 1. State the limit

- [ ] 1.1 Add the 24-month rule to the Toolchain section of `openspec/constitution.md`, and a place for recorded exceptions in its Project Profile; verify `openspec validate replace-stale-dependencies --strict` still passes and the rule names the limit and the exception.
- [ ] 1.2 Add the check as a release step in `CONTRIBUTING.md`; verify running its one-off query over `pyproject.toml` names exactly `tidynamics`, `kramersmoyal`, and `matching` today, and `deeptime` too if it is still present.

## 2. Replace `tidynamics`

- [ ] 2.1 While the library is installed, record its results on fixed positions as literal reference values in `tests/analysis/test_displacement.py`: the displacement over every lag, the exponent, and the diffusion coefficient; verify the new test passes on the current code.
- [ ] 2.2 Write the FFT computation as a private method of `DisplacementAnalysis` and remove the import; verify the reference values pass within 1e-9 relative tolerance, the `agent-analysis` scenario tests pass, and the test time does not grow noticeably.

## 3. Replace `kramersmoyal`

- [ ] 3.1 Record its results on a fixed series as literal reference values in `tests/analysis/test_kramers_moyal.py`, for two bandwidths; verify the new test passes on the current code.
- [ ] 3.2 Read the source of `kramersmoyal.km` for its kernel, edges, and normalisation, write the computation as a private method of `KramersMoyalAnalysis`, and remove the `_import` call; verify the reference values pass within 1e-6 relative tolerance and the Ornstein-Uhlenbeck test passes. If they cannot, stop and ask.

## 4. Replace `matching`

- [ ] 4.1 Record the pairs and their order from the library on fixed preferences and capacities, including an unacceptable pair and an unranked player, as literal reference values in `tests/core/interactions/test_matching.py`; verify the new test passes on the current code.
- [ ] 4.2 Write requester-proposing deferred acceptance as a private method of `MatchingInteraction` and remove the import and the warning filter around it; verify the reference pairs and their order match, a stability check finds no blocking pair, and every test in `tests/core/interactions/test_matching.py` passes.
- [ ] 4.3 Run the tests of the economics models that use `StableMatchingSpec`; verify they pass unchanged.

## 5. Remove the libraries

- [ ] 5.1 Remove the three from `pyproject.toml` and the deptry list, relock, and update `docs/overview/user_guide.md` and the Project Profile; verify `grep -rnE "tidynamics|kramersmoyal|matching\.games" src tests pyproject.toml docs/overview openspec/constitution.md` finds nothing, `pdm.lock` no longer lists them, and `pdm checks` passes.
- [ ] 5.2 Install the built wheel into a new environment without the lock file; verify the three classes run and the three test files pass.
- [ ] 5.3 Push to `development`; verify every CI job passes.
