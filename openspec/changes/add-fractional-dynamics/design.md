# Design

## Context

ODE solvers advance from the current state alone. The history buffer of `add-hybrid-dynamics` keeps past states for
delays. Fractional derivatives need a weighted sum over all past states, with weights that decay as a power law.

## Goals / Non-Goals

**Goals:**

- Fractional models written like ordinary ones, with the order as a parameter.
- Accuracy checked against Mittag-Leffler solutions.

**Non-Goals:**

- Orders above 1 and Riemann-Liouville initial conditions. Caputo derivatives of order up to 1 cover the references.
- Fast convolution with FFT-based sum-of-exponentials kernels. The short-memory window covers long runs, and faster
  kernels can follow.

## Decisions

### Laws

`FractionalRate(target, value, order)` declares `D^α x = f(t, x)`. A system mixing fractional and ordinary rate laws
compiles to the fractional form, treating ordinary states as order 1. `order` is a parameter, so sweeps can study it.

### Solvers

`GrunwaldLetnikov(dt, memory=None)` uses the binomial weights `w_j = (-1)^j C(α, j)`, computed by recursion.
`L1Scheme(dt, memory=None)` uses the L1 weights for the Caputo derivative. `FractionalABM(dt)` is the
predictor-corrector of Diethelm, Ford, and Freed. `memory` limits the sum to a window for the short-memory principle.
Histories are stored in a fixed-size array per state, and the weighted sums are dot products. The Numba backend fuses
them.

### Mittag-Leffler

`mittag_leffler(alpha, beta, z)` evaluates `E_{α,β}(z)` by series for small `|z|` and by its integral representation
otherwise. `solve_symbolically` returns `x0 E_α(-λ t^α)` for linear fractional relaxation.

## Risks / Trade-offs

- [Quadratic cost of the full memory] → The short-memory window bounds it, and the user guide states the error
  trade-off.
- [Low order of accuracy] → The predictor-corrector gives order `1 + α`, and tests set tolerances per scheme.
