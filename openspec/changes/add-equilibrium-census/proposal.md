## Why

`stability` finds the fixed points of a flow symbolically, and only falls back to 50 Newton starts that keep the roots
they happen to reach. It caps groups at 64 members and rejects callable laws. Large random systems, such as the
relaxation dynamics under a random force field of Lacroix-A-Chez-Toine, Le Doussal, and Ros (2026, arXiv 2611.08324),
have exponentially many equilibria. Their study needs a census: all the fixed points one can find, each with its
number of unstable directions, counted over many realizations of the quenched disorder, so that the typical
(quenched) and average (annealed) counts can be compared.

## What Changes

- `stability` gains the kind `'census'`. It searches the fixed points of a flow or a map numerically, from many
  starting points, with deflation so a root found once is not found again. Every point gets its index, the number of
  eigenvalues of the Jacobian with positive real part for a flow, or of modulus above one for a map.
- The census runs over quenched realizations. With `realizations` above one, it redraws the disorder of a system that
  has a `random_state` and repeats the search, then reports the counts by index of every realization, their mean (the
  annealed count), and the exponential of the mean of `log(1 + n)` (the typical, or quenched, count), with both
  complexities per unit.
- The census reads the numeric drift and Jacobian of the compiled problem, not the symbolic equations, so it has no
  member limit and accepts callable laws.
- The numeric fallback of the kind `'fixed-points'` uses the same deflated search, so it no longer misses roots that
  share a basin of the Newton iteration.
- New system `RandomForceFieldRelaxationSystem` in `physics/statistical_mechanics`: units that relax to zero under a
  quenched Gaussian random force field, with a gradient part and a divergence-free part mixed by the reciprocity `tau`.

## Capabilities

### New Capabilities

- `equilibrium-census`: a numeric census of the fixed points of a flow or a map by stability index, with deflation,
  averaged over quenched realizations of the disorder into annealed and quenched counts.

### Modified Capabilities

- `model-library`: adds the random force field relaxation system.

## Impact

- Code: `src/skcomplex/core/analysis/_stability.py` (new kind, `StabilityResult` fields, deflated root search that
  replaces `_numeric_points`). A new module in `src/skcomplex/systems/physics/statistical_mechanics/`.
- No new dependencies. SciPy gives `optimize.root` and `differentiate.jacobian`. NumPy draws the realizations.
- Docs: the stability section of the analysis guide, a gallery example, and an inventory entry.
- Not breaking. The kind `'fixed-points'` keeps its results where Sage solves the equations, and its numeric fallback
  finds the same roots or more.
