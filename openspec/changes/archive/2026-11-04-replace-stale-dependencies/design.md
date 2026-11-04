# Design

## Context

See proposal.md for the motivation. Release dates come from PyPI on 2026-11-03. The three libraries touch little of the
package:

- `tidynamics.msd` is one call per particle in `DisplacementAnalysis._run`, over all lags of unwrapped positions.
- `kramersmoyal.km` is one call in `KramersMoyalAnalysis._run`, loaded through `BaseAnalysis._import`, taking `bins`,
  `powers = [0, 1, 2]`, and `bw`. The drift and the diffusion are moments 1 and 2 divided by `dt`.
- `matching.games.HospitalResident` runs in one method of `MatchingInteraction`, with `clean=True` and
  `optimal='resident'`, for `StableMatchingSpec`. Several economics models reach it through that interaction.

The survey covered the libraries that could wrap each algorithm. None was implemented or run for this design:

| Need | Considered | Outcome |
|---|---|---|
| Mean squared displacement | `freud-analysis` (September 2026), `MDAnalysis` (October 2026) | Active, but each adds a large particle-simulation library for one formula |
| Kramers-Moyal coefficients | `pydaddy` (2022), `statsmodels` (August 2026) | `pydaddy` is stale, and `statsmodels` has no Kramers-Moyal coefficients that were identified |
| Stable matching | `pygambit` (October 2026), `nashpy` (November 2025), `pymatching` (May 2026) | Game equilibria or error-correcting codes, not deferred acceptance |

## Goals / Non-Goals

**Goals:**

- No runtime dependency past 24 months without a recorded exception.
- The same public results from the three classes.

**Non-Goals:**

- Replacing `deeptime`, which `add-python-3-14-support` covers.
- Replacing libraries inside the limit, however close, or changing their ranges.
- An automatic check in CI. PyPI lookups in the gate make it fail for reasons unrelated to the code.

## Decisions

### A 24-month limit with recorded exceptions

Twelve months would also flag `mrestimator`, `pydmd`, and `igraph`, and a small library can be slow because it is finished.
Twenty-four months catches libraries that have stopped without forcing churn on the rest. A recorded exception covers a
library that has no replacement, such as a finished algorithm with no open problems.

The limit lives in the constitution's Toolchain section, with the exceptions in its Project Profile. `CONTRIBUTING.md`
gains the check as a release step, as a short one-off query of PyPI over the manifest, with no new tool in the repository.

### Compute the mean squared displacement with SciPy

`tidynamics.msd` is the FFT-based algorithm of Calandrini and others, about ten lines with `scipy.signal.fftconvolve`
or `scipy.fft`. A private method of `DisplacementAnalysis` holds it. Wrapping `freud` or `MDAnalysis` would swap a small
stale dependency for a large one to call a single formula.

### Compute the Kramers-Moyal coefficients with NumPy

The computation is a histogram of the increments weighted by a kernel over the positions. `kramersmoyal.km` has its own
choices of kernel, edges, and normalisation, and the reference values must reproduce them, so the first step reads its
source before writing the method. The Epanechnikov kernel and the convolution step are the likely match, and the reference
tests settle it. A private method of `KramersMoyalAnalysis` holds it, and the `_import` call goes away.

### Write deferred acceptance as a private method

The requester-proposing algorithm is about 30 lines, and `MatchingInteraction` already holds algorithms as private methods
next to its other rules. `clean=True` drops unacceptable pairs and unranked players, which the new method must also do. The
requester-optimal stable matching is unique, so equality with the library is a real test and not a coincidence of ties.

### Correct the catalog spec rather than defer the imports

`_catalog.py` imports `Simulation` for `run`, and the engine imports `mesa` and `lammps` when it loads. `skcomplex.specs`
imports `qutip`. Deferring them would take a lazy import of a required dependency, which the constitution forbids, and
`run` would lose its return type, since `TYPE_CHECKING` imports are forbidden too. It would save about 0.5 s of a 7 s
import, as the core still loads Sage. So the scenario drops the named libraries and keeps what it can promise, that no
model module loads. This loosens a spec, by decision of the user, and a fast catalog is a change of its own if wanted.

## Risks / Trade-offs

- [The project now owns three algorithms] → Each is short and well specified, and reference values plus the stability
  check cover them. This is a deliberate exception to wrapping mature libraries, recorded in the constitution.
- [`km` has a detail that the reference values cannot reach within tolerance] → The change pauses for a decision between
  recording an exception for `kramersmoyal` and accepting changed values. A tolerance is not loosened to pass.
- [A model relies on the library's order of pairs] → The trades of a matching settle in solution order, and quantities can
  depend on that order. Keep the order of hospitals and residents the library returns, and compare the affected model
  tests before and after.
- [Slower than the library] → `tidynamics` and `kramersmoyal` are also Python and NumPy, so no large change is expected.
  Compare the times of the three test files.

## Migration Plan

Each replacement lands on its own commit on `development`, after its reference values, so any one can be reverted alone
by restoring the library and the method.

## Open Questions

- Whether this change or `add-python-3-14-support` lands first does not matter to the code. The 3.14 change comes first
  because a release waits on it, and these libraries do not block 3.14.
