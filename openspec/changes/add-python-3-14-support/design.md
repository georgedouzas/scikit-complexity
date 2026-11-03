# Design

## Context

See proposal.md for the motivation. Measured on 2026-11-03 on macOS with Apple silicon, in a throwaway Python 3.14
environment outside the repository, with the newest versions the ranges allow rather than the lock file:

- Resolution with wheels only fails on three packages on both platforms: `deeptime` 0.4.5, `memory-allocator` 0.1.4, and
  `pltable` 1.1.0. Allowing those three to build from source resolves the full set on macOS (151 packages) and on Linux
  (148 packages).
- `memory-allocator` 0.1.4 and `pltable` 1.1.0 build from source on macOS. `passagemath` pins `memory-allocator<0.2`, so
  the 3.14 wheels of `memory-allocator` 0.2.1 are out of reach.
- `deeptime` 0.4.5 fails to compile under the current Apple compiler. Its main branch builds, but its last release was
  April 2025.
- With `deeptime` from its main branch, the fast tests and the Markov and passage tests give 1324 passed, 82 skipped, and
  6 failed in 74 s. All 6 fail with a `TypeError` under `xgi` 1.0.0, and the affected test files pass with `xgi` 0.10.2
  (77 passed, 61 skipped).
- The published 0.2.0, installed from PyPI on Python 3.13, gets `xgi` 1.0.0, and `hypergraph_kuramoto` fails there with
  the same `TypeError`, so this failure does not come from Python 3.14.
- `mkdocs-gallery`, at its latest release, reads `ast.Str` in `py_source_parser.py`, and `ast.Str` does not exist on 3.14.

`pydtmc` 10.1.0, released October 2026, is pure Python, MIT licensed, and installs on 3.14. Compared with `deeptime` on
small chains:

| Need | `pydtmc` | Result against `deeptime` |
|---|---|---|
| Non-reversible fit | `MarkovChain.fit_sequence('mle', ...)` | First row identical to 4 decimals |
| Stationary distribution | `stationary_distributions` | Within 0.0001 |
| Implied timescales | `implied_timescales`, with a leading `inf` | Identical to 3 decimals once it is dropped |
| Committor | `committor_probabilities('forward', ...)` | Identical to 4 decimals |
| Mean first-passage time | `first_passage_times_to('mean', ...)` | Identical to 3 decimals |
| Metastable sets | `metastable_aggregation(s)`, sets in its state labels | Same partition on a 6-state double well |
| Reversible fit | None: `closest_reversible` gives a different matrix | Needs own code |

Both libraries refuse metastable sets of a non-reversible chain. Not measured yet: the full test suite on 3.14, anything
on Linux with 3.14, the lock file resolved for 3.14, mean first-passage times from an origin set, and the metastable sets
on the double-well and Schlögl scenarios of `stochastic-analysis`.

## Goals / Non-Goals

**Goals:**

- Python 3.14 declared on the supported platforms with the same features as 3.12 and 3.13.
- A fresh install from PyPI that works, whichever Python version it uses.
- Markov results that keep their values across the change of library.

**Non-Goals:**

- Dropping Python 3.12 or 3.13, or adopting 3.14-only syntax. The language floor stays 3.12.
- Free-threaded builds (`cp314t`), which the package does not declare or test even though some dependencies publish them.
- Running the documentation build on 3.14.
- Replacing the other stale dependencies, which `replace-stale-dependencies` covers.
- Pinning every dependency or adding a CI job that tests a fresh resolve on every run. This change caps `xgi` and checks
  a fresh install once.

## Decisions

### Replace `deeptime` with `pydtmc`

`pydtmc` is the only actively released library surveyed that covers the need, and it matches `deeptime` on every method
compared. Its smaller community (96 GitHub stars against 894) is the trade-off the project accepts for frequent releases.

Alternatives considered:

- Wait for a `deeptime` release: rejected because the project aims for frequently updated libraries.
- Depend on its git main branch: PyPI rejects packages whose metadata names a direct URL.
- `quantecon` (active): not confirmed to cover committors or metastable sets, so it replaces only part of the need.
- `pyGPCCA` (2022), `msmtools` (2021), `msmhelper` (2023), `PyEMMA` (archived), `MSMBuilder` (archived): all stale.
- Install it only below 3.14: three analyses would fail on 3.14, against the requirement of equal features.

### Write only the reversible fit

The reversible maximum-likelihood fit is the standard fixed-point iteration on the symmetric counts, about 15 lines in a
private `MarkovAnalysis` method. Counting transitions and restricting to the largest strongly connected set use NumPy and
`scipy.sparse.csgraph.connected_components`, since `fit_sequence` works on the full list of states and takes no lag.
`closest_reversible` is not used, because it solves a different problem and gives a different matrix.

The non-reversible fit is the counts normalized by row, and the stationary distribution comes from
`MarkovAnalysis._stationary_distribution`, which the `'stationary'` kind already uses, so `pydtmc` holds only the implied
timescales, the metastable sets (its default mode is PCCA+), the committors, and the mean first-passage times. It refuses
one set and one set per state, which `deeptime` allowed, so those two cases stay in `MarkovAnalysis`. It also gives no
mean first-passage time on a reducible chain, which raises an `AnalysisError`, where `deeptime` raised a `LinAlgError` or
a `ValueError`. A path with fewer than two connected states raises an `AnalysisError`, where `deeptime` raised a
`ValueError`.

### Pin behaviour with reference values from `deeptime`

Before removing `deeptime`, its results on fixed paths and chains are recorded as literal numbers in the tests, for both
fits and for every public result. The tests then need neither library to state what is correct, and any difference shows
up as a failure rather than a silent change.

### Accept source builds of `memory-allocator` and `pltable`

Both build today, and `pltable` needs no compiler. `memory-allocator` needs a C compiler on 3.14 until `passagemath`
allows 0.2, which the README states. CI on both platforms confirms the build, because Linux is unmeasured.

### Cap `xgi` below 1.0

`xgi>=0.10.2,<1` restores the version the lock file and CI already use. Adapting the hypergraph code to 1.0 is a separate
piece of work with its own review of the API changes, and the cap fixes the install at once.

### Test every version, and the newest on both platforms

The matrix becomes Linux 3.12, macOS 3.13, Linux 3.14, and macOS 3.14. Every declared version runs, and 3.14 runs on both
platforms because source builds are specific to it. This adds two jobs to each run, and does not change the fast test set.

### Keep the documentation sessions on Python 3.13

`docs` and `gallery` in `noxfile.py` use `PYTHON_VERSIONS[-1]`, which would become 3.14 and break `mkdocs-gallery`. They
get a named `DOCS_PYTHON = '3.13'`. The other `PYTHON_VERSIONS[-1]` sessions move to 3.14 with the list. CI's gallery job
already sets 3.13. The `docs-release` job in `release.yml` sets no Python, so it gets `python-version: '3.13'` too, or a
wider `requires-python` could move it onto 3.14.

### Run the full suite once on a CI runner

The full suite is too heavy for the 16 GB development laptop, which ran out of memory on it earlier. A temporary branch,
not merged, runs `nox -s tests-3.14` on a Linux runner once, and any failure there returns to this design.

## Risks / Trade-offs

- [`pydtmc` disagrees on a larger chain or a scenario not compared yet] → The reference values and the `stochastic-analysis`
  scenarios catch it. A real disagreement pauses the change for a decision rather than loosening a tolerance.
- [`pydtmc` is slower, being pure Python] → Check `tests/analysis/test_markov.py` and the emergence tests do not grow
  noticeably, and report the times.
- [`pydtmc` needs `matplotlib` at runtime] → Accepted if it is already in the resolved set. Check when relocking.
- [State labels are strings in `pydtmc`] → The adapter maps them back to the integer state symbols the public results use,
  with the metastable sets parsed from the labels `metastable_aggregation` returns.
- [Relocking moves other dependencies] → Lock with `--update-reuse` to move only what 3.14 needs, and rely on the fast
  tests on all three versions and the full suite to catch a regression.
- [Dependencies drift past their ranges again] → The lock file hides this from CI. The fresh install check in the tasks
  catches it once, and a recurring check is out of scope.
- [Third-party `SyntaxWarning`s on 3.14, as from `nolds`] → They do not fail the import or the tests. Leave them upstream.

## Migration Plan

1. Record the reference values while `deeptime` is still installed, then swap the library on 3.12 and 3.13.
2. Declare 3.14, relock, and extend CI on `development`, after the temporary full-suite run passes.

To roll back, restore the version range, `deeptime`, and the lock file.
