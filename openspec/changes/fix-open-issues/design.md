# Design

## Context

The issues were filed against 0.9.0 and were reproduced on 0.10.0 with `system.prepare().simulate(...)`. Ready-made
systems draw their initial state and random structure in `_prepare` from their own seed. Most call it `random_state`,
with a default of 0, and 35 call it `seed`. `ensemble` derives one run seed per replicate and passes it to
`simulate`, so replicates of a system that draws its initial state in `_prepare` share that state.

## Goals / Non-Goals

Goals: close every open issue with a fix, a test, and docs, and finish the two loose ends of the redesign.

Non-goals: drawing initial states at run time inside the engine, and a general frame stride on `simulate`.

## Decisions

1. **One seed name, two roles.** A system's `random_state` seeds what it draws when prepared. The run's
   `random_state` seeds the dynamics. Defaults stay as they are, so no recorded output changes. This keeps the
   scikit-learn name and avoids moving initial draws of every system into the engine.
2. **Ensembles vary the system seed.** When the system has a `random_state` parameter, `ensemble` clones it per
   replicate and sets that parameter to a seed spawned from the experiment's seed, next to the run seed. A replicate is
   then independent in both its start and its dynamics, as the ensemble requirement already promises. Preparing per
   replicate costs a compile per replicate only for such systems. A user who wants one fixed start passes
   `initial_state`.
3. **Schedules follow the lattice.** With the solver `'auto'`, a Metropolis sampler on a periodic lattice with an odd
   side uses the random sequential schedule. An explicit checkerboard schedule on such a lattice still raises, and the
   message shows `solver=MetropolisSolver(schedule='random-sequential')`.
4. **Frame 0 holds derived states.** States computed from the structure, such as a degree, are evaluated on the
   initial structure before the first frame is recorded.
5. **Fixed points are sorted.** Points are sorted by their values in the order of `names`, using values rounded to 10
   decimals for the key only. Exact Sage solutions are converted with full precision, and numerical roots are polished
   with one Newton step before rounding is considered.
6. **Period 0 means aperiodic.** The period array stays integer, and 0 marks orbits whose kept values never repeat
   within the rounding.
7. **Blocks in `coarse_grain`.** `blocks` gives the block side, one per axis or one for all, and `rule` is `'majority'`
   or `'mean'`. Majority ties are broken by `tie`, `'random'` with a `random_state` or a value. The source is a lattice
   array or a trajectory with `state`. The result holds the block configuration per frame. This keeps one public
   function for coarse graining, which is cross-domain (images, lattices, grids).
8. **Normal forms need no new API.** `DynamicalSystem` already takes states and equations in one call. The docstring of
   `stability` and the guide show `dm/dt = a m - b m^3` this way.
9. **Performance by measurement.** On 0.10.0 separate runs of the logistic map take about 20 ms each, so 1200 values
   take about 25 s, while the vectorized run takes under a second with the same orbits. Separate runs stay the default,
   because not every system replicates, and the docstring points large scans to `vectorized=True`.
   `CurieWeissSystem` gets a default `dt` of 1, one sweep per frame.
10. **`LoweredFunctionsSpec` becomes `_LoweredFunctionsSpec`.** It is only used inside the engine and in its test,
    which moves to the private name through the backend.

## Risks / Trade-offs

- Renaming `seed` breaks user code. The release notes name the rename.
- Changing the Curie Weiss default `dt` changes its recorded frames. Its tests and docs follow.

## Open Questions

None.
