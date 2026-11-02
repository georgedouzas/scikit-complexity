# Tasks

## 1. Dependency and form

- [x] 1.1 Add `shapely>=2.1` to the runtime dependencies and the Project Profile, and verify `pdm install`, `deptry`,
  and `pip-audit` pass
- [x] 1.2 Add `Form.REWRITING` and `RewritingError` to `skcomplex.core.base`, accept integer `dt` for the form in
  `Simulation`, and verify the fractional generation scenario

## 2. Words and string productions

- [x] 2.1 Add `Word`, `Module`, and `Production` with the symbol counts and size as states, and the `Rewriter` stepper
  registered as the form's default solver, and verify the generations as steps and Parikh vector scenarios
- [x] 2.2 Add stochastic productions with the probability check and draws from the simulation's stream, and verify
  the stochastic choice and probability sum scenarios
- [x] 2.3 Add context-sensitive productions with ignored symbols and the Numba bracket-aware context search, and
  verify the context scenario and the bracketed examples of Prusinkiewicz and Lindenmayer (1990 §1.8)
- [x] 2.4 Add parametric productions with conditions and successor parameters lowered from Sage on NumPy and Numba,
  and verify the parametric growth scenario on both backends
- [x] 2.5 Add the size guard with `max_size`, and verify the exponential growth scenario and the benchmark target of
  one generation of 10⁶ symbols
- [x] 2.6 Write productions in their written form with `repr` (`equations_of` returns relations of states, so
  rewriting systems have none there), document words and productions in the user guide with a runnable example, and
  verify the documentation test passes

## 3. Turtle and substitution

- [x] 3.1 Add `Turtle` in two and three dimensions with the NumPy path and the Numba stack walk, and verify the Koch
  length and branch scenarios
- [x] 3.2 Add `Patch`, `Prototile`, `Affine`, and `Substitution` with the batched matrix composition, and verify the
  Cantor and Sierpiński area scenarios and a chair tiling that covers its chair without overlap
- [x] 3.3 Add the measures `length`, `area`, `parts`, `bounds`, and `raster` on Shapely, recorded as reporters, and
  verify the gasket box-counting scenario
- [x] 3.4 Document turtles, patches, and measures in the user guide with runnable examples, and verify the
  documentation test passes

## 4. Graph rewriting

- [x] 4.1 Add `HostGraph` and `GraphProduction` with NetworkX matching, parallel independent matches, and the
  dangling condition, and verify the edge subdivision and dangling edge scenarios
- [x] 4.2 Expose node counts by label as states, document graph rewriting in the user guide, and verify a seeded run
  is reproducible and the documentation test passes

## 5. Runtime and visualization

- [x] 5.1 Add `Frame.structures` and `Trajectory.structure` with recorded copies, and verify the structure of a
  generation and unknown element scenarios
- [x] 5.2 Add `GeometryVisual` as the default visual of rewriting elements with fixed bounds, line strips, and the
  filled image, and verify the Koch curve in Rerun and filled gasket scenarios
- [x] 5.3 Make the geometry view the primary view of the default blueprint for rewriting systems, and verify the
  blueprint test picks it

## 6. Library models

- [x] 6.1 Add `skcomplex.models.mathematics.fractals` factories `koch_curve`, `cantor_set`, and
  `sierpinski_substitution` with their references and defaults, and verify the Koch, Cantor, and carpet scenarios and
  the conformance suite
- [x] 6.2 Move the three inventory entries from out of scope to pending this change and then to implemented, and
  verify the inventory check passes
- [x] 6.3 Add the gallery examples `docs/examples/mathematics/fractals/plot_koch_curve.py`, `plot_cantor_set.py`, and
  `plot_sierpinski_substitution.py` with captions, a Rerun thumbnail of the last generation, and the measure and
  box-counting figures, and verify the gallery coverage, budget, and screenshot checks pass
- [x] 6.4 Add the changelog entry for rewriting systems and the three models as the conventional `feat` commit that
  the changelog session renders, and verify the changelog session passes

## 7. Gate

- [x] 7.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
