# Design

## Context

The issues were filed against 0.9.0 by someone writing tutorials. Each is checked on the current `development` branch
before it is fixed, and one that no longer reproduces is closed with a note.

## Goals / Non-Goals

Goals: fix or document every issue, each with a test.

Non-goals: new analysis kinds, or changes to systems the issues do not name.

## Decisions

1. **Documentation where the behaviour is right.** The Simon Yule exponents, the Poisson draw times of the urns, the
   percolation strength, and the contact process threshold are documented. The exponents are checked by a short run
   in the tests. The contact process docstring says that the threshold approaches the continuous value 1.64872 of the
   square lattice as `tau` shrinks and is found from the decay of the density at the critical point, without a
   number for one `tau` unless a test measures it.
2. **Discrete urns.** `time='discrete'` follows the name `DynamicalSystem` uses. One step draws one ball in every urn.
3. **Galton Watson.** `offspring` is `'poisson'`, `'geometric'`, or a sequence of probabilities `p_k` that sums to 1,
   with `m` the mean of the Poisson and geometric laws. With `individuals=False` the generation size is one count
   whose next value is drawn as the sum of the offspring of every individual, a Poisson draw of mean `n m`, a negative
   binomial draw, or a multinomial draw over the counts. The default stays `individuals=True`, so existing runs keep
   their outputs. No constructor seed: the seed rule of `fix-open-issues` gives a system its own `random_state` only
   for what it draws when prepared, and the docstring says that `simulate(random_state=)` seeds the runs.
4. **S0 location.** The fit keeps SciPy's parametrisation internally and converts the location to S0 with
   `loc0 = loc1 + beta * scale * tan(pi * alpha / 2)` for alpha not 1, and `loc0 = loc1 + beta * scale * 2 / pi *
   log(scale)` at alpha 1. At alpha within `1e-6` of 2 the skewness is reported as 0. The docstring names S0.
5. **Quiet mrestimator.** Its standard output is redirected for the call, and lines it prints that are warnings are
   raised again through `warnings.warn`.
6. **West Brown Enquist.** Profile depth 6 to 8, find the step that grows faster than the node count, and fix it.
7. **Box counting of points.** An array of shape `(N, d)` with `N > d` and more than one column is read as points when
   `points=True` is given or the source is not a grid of occupancies, which the design states explicitly: a 2-D array
   whose values are not all 0 and 1 is points. For points, every box size counts the distinct boxes `floor((x - min) /
   size)`. A grid whose smallest side is shorter than the largest box raises an `AnalysisError` naming the side.
8. **Unique windows.** The default windows are `np.unique` of the rounded log-spaced sizes.
9. **Ragged values.** For a group whose size changes, `get_values` of a per-agent state returns a list of one array per
   frame, in the order of the members at that frame, instead of raising. Groups of fixed size keep returning arrays.

## Risks / Trade-offs

- The S0 location changes the results of existing stable fits.
- Reading 2-D arrays as points by their values could misread a grid of real numbers. An explicit `points` argument
  removes the doubt, so the guess applies only when it is not given.

## Open Questions

None.
