# Tasks

## 1. Random draws

- [ ] 1.1 Add `draw(name, distribution, **parameters)` to the named draws of `add-stochastic-maps`, with name and
  keyword validation against `scipy.stats`, and verify the unknown distribution scenario
- [ ] 1.2 Sample SciPy draws with vectorised `rvs` calls on the simulation's `Generator`, and as kernel buffers in
  update laws on NumPy and Numba, and verify the draw statistics scenario
- [ ] 1.3 Widen the draw placement check to accept renewal laws, and verify the shared draw within a renewal and
  reproducible draws scenarios, and that a draw in a rate law is still rejected
- [ ] 1.4 Document `draw` with a runnable docstring example and a user guide section, and verify the doctest run and the
  documentation test pass

## 2. Renewal laws

- [ ] 2.1 Add `Renewal` with the nonnegative support check from `scipy.stats` `support()`, and verify the negative
  waiting time scenario
- [ ] 2.2 Compile renewals to per-member next-time arrays that act as breakpoint sources in the hybrid layer, with
  event records carrying the member index, and verify the ballistic flights scenario
- [ ] 2.3 Add `Form.RENEWAL` and `RenewalStepper`, register it as the solver for the form, and verify the renewal form,
  state between renewals, exponential waiting times, and power-law waiting times scenarios
- [ ] 2.4 Advance non-interacting groups with per-member steps and interacting groups with split steps, and verify the
  independent member clocks scenario on both paths
- [ ] 2.5 Add the `renewals` observable and stream renewal records to Rerun, and verify the renewal count observable
  scenario and a recording test against a Rerun memory sink
- [ ] 2.6 Document renewal laws and the renewal form in the user guide with a runnable example, and verify the
  documentation test passes

## 3. Fractional noise

- [ ] 3.1 Add the Davies-Harte adapter over `scipy.fft` with the Cholesky fallback from `scipy.linalg`, and verify that
  sample covariances of 10⁴ paths match the fractional Gaussian noise covariance within 3 standard errors for
  `H` in 0.2, 0.5, and 0.8
- [ ] 3.2 Add `FractionalNoise` with the Hurst range and state-independence checks, lowering `H = 1/2` to an ordinary
  noise row, and verify the Hurst index out of range, state-dependent amplitude, and Hurst index of one half scenarios
- [ ] 3.3 Add `FractionalEulerMaruyama` with the horizon, the buffer size limit, and the Numba kernel with precomputed
  increments, and make other SDE steppers reject fractional rows, and verify the increment correlation,
  superdiffusion, beyond the horizon, and wrong stepper scenarios
- [ ] 3.4 Pass the end time of `Simulation.run` as the horizon, and verify that a run without an explicit horizon
  completes and two runs with random state 0 are equal
- [ ] 3.5 Document fractional noise and its stepper in the user guide with a runnable example, and verify the
  documentation test passes

## 4. Library models

- [ ] 4.1 Add the continuous-time random walk and the Lévy walk to `physics/soft_matter` with their sources, and verify
  the subdiffusive random walk and superdiffusive Lévy walk scenarios
- [ ] 4.2 Add annealed transient time motion and fractional Brownian motion with their sources, and verify the
  annealed transient time motion and fractional Brownian motion scenarios
- [ ] 4.3 Mark the four models implemented in the model inventory, and verify the layout test, the conformance suite
  for registered models, and that no inventory row is blocked by `add-anomalous-processes`

## 5. Gate

- [ ] 5.1 Add a changelog entry for the new laws, form, steppers, and models, and verify the changelog session passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
