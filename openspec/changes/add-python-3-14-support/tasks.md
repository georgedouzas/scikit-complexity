# Tasks

## 1. Fix the fresh install

- [x] 1.1 Require `xgi>=1.0.0,<2` in `pyproject.toml`, read its view methods with `dtype=list`, and relock; verify `pdm lock --check` passes, `xgi` is 1.0.0 in `pdm.lock`, and the hypergraph tests pass on 3.14.

## 2. Replace `deeptime` with `pydtmc`

- [x] 2.1 While `deeptime` is installed, record its results as literal reference values in `tests/analysis/test_markov.py` and `test_passage.py`, on a fixed path and fixed chains: both fits, stationary distribution, implied timescales, metastable sets, committors, and mean first-passage times, including from an origin set; verify the new tests pass on the current code.
- [x] 2.2 Add `pydtmc` to `pyproject.toml` and move `MarkovAnalysis` to it for the non-reversible fit, stationary distribution, implied timescales, and metastable sets, keeping integer state symbols in the public results; verify the reference tests pass and the `stochastic-analysis` double-well and Schlögl scenario tests pass.
- [x] 2.3 Add the reversible maximum-likelihood fit as a private method of `MarkovAnalysis`, with the largest strongly connected set from `scipy.sparse.csgraph`; verify the reversible reference values pass within 0.001 relative tolerance.
- [x] 2.4 Move `PassageAnalysis` to `pydtmc` for committors and mean first-passage times; verify the reference values and `tests/analysis/test_passage.py` and `test_emergence.py` pass, and report their times before and after.
- [x] 2.5 Remove `deeptime` from `pyproject.toml` and the deptry list, add `pydtmc` there, update `docs/overview/user_guide.md`, the Project Profile in `openspec/constitution.md`, and any catalog import test, and relock; verify `grep -rn deeptime src tests pyproject.toml docs/overview openspec/constitution.md` finds nothing and `pdm checks` passes.

## 3. Declare Python 3.14

- [x] 3.1 Set `requires-python = ">=3.12, <3.15"` and add the 3.14 classifier in `pyproject.toml`; verify `pdm lock --python ">=3.12,!=3.14.1,<3.15" --update-reuse` succeeds, `pdm.lock` reads that target, and no locked version changes. The lock target excludes 3.14.1 because `networkx` 3.7 does.
- [x] 3.2 Add `'3.14'` to `PYTHON_VERSIONS` and a `DOCS_PYTHON = '3.13'` for the `docs` and `gallery` sessions in `noxfile.py`; verify `pdm run nox -l` lists `quick_tests-3.14` and `tests-3.14`, and lists `docs` and `gallery` on 3.13.
- [x] 3.3 Update the supported versions in `README.md`, including the C compiler `memory-allocator` needs on 3.14, and the language line of the Project Profile in `openspec/constitution.md`; verify `grep -n "3\.14" README.md openspec/constitution.md` shows both, and no text still says `<3.14` or "3.12 and 3.13".

## 4. Verify in CI

- [ ] 4.1 Set the test matrix in `.github/workflows/ci.yml` to Linux 3.12, macOS 3.13, Linux 3.14, and macOS 3.14, add 3.14 to the installed versions of the checks job, and set `python-version: '3.13'` on the `docs-release` job in `release.yml`; verify a CI run on `development` passes every job and the gallery job still reports 3.13.
- [ ] 4.2 Run `nox -s tests-3.14` once on a Linux runner from a temporary, unmerged branch; verify all three test groups pass, record the result in `design.md`, and delete the branch.
- [x] 4.3 Compare `MarkovAnalysis`, `PassageAnalysis`, and `EmergenceAnalysis` on one fixed trajectory between 3.13 and 3.14; verify the results agree within numerical tolerance.

## 5. Verify a fresh install

- [x] 5.1 Build the wheel and inspect its metadata; verify `Requires-Python` is `<3.15,>=3.12`, the 3.14 classifier is present, and no dependency names a direct URL.
- [ ] 5.2 Install that wheel without the lock file into new 3.14 environments on Linux and macOS, on a CI runner for Linux; verify the package imports, `hypergraph_kuramoto` simulates, and the fast tests pass.
- [x] 5.3 Run `pip install` of that wheel with Python 3.15; verify the installer refuses it.

## Workflow follow-up

- After this change is implemented, make a new release, as the user asked: release branch, PR to `main`, and tag. The user approved merging that release's PR with `--admin` after its CI passes.
- That release is the first real run of the `docs-release` job, and it also ships the Models tab and the `xgi` fix.
