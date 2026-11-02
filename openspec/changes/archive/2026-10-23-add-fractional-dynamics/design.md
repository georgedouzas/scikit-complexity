# Design

## Context

This design is rebased on `redesign-system-model`. A state declares its `order`, rate laws target its highest
derivative, and solvers and backends declare what they support, which validation compares with the system before
compilation. The inventory gate resolved Caputo derivatives as a non-integer order of a state.

ODE solvers advance from the current state alone. The history buffer of delayed states keeps past states. Fractional
derivatives need a weighted sum over all past states, with weights that decay as a power law.

## Goals / Non-Goals

**Goals:**

- Fractional models written like ordinary ones: a `Rate` law on a state declared with a non-integer order.
- Accuracy checked against Mittag-Leffler solutions.
- Invalid uses of fractional orders reported by validation, not by the solvers.

**Non-Goals:**

- Orders above 1 and Riemann-Liouville initial conditions. Caputo derivatives of order up to 1 cover the references.
- Fast convolution with FFT-based sum-of-exponentials kernels. The short-memory window covers long runs, and faster
  kernels can follow.
- A fractional law kind. The order belongs to the state, and the rate law is unchanged.

## Decisions

### Fractional orders of states

`state(order=α)` with `0 < α < 1` declares that the `Rate` laws on the state give `D^α x = f(t, x)` in the Caputo
sense. Rate laws on the state superpose as for any state. A system mixing fractional and ordinary states lowers to the
internal fractional form, treating ordinary first-order states as order 1. The form is a lowering detail of
`skcomplex.core.compile`, as for every form.

The order is fixed when the element type is declared. The library factories take the order as an argument, so sweeps
over models study it by building one system per order.

### Declarations and validation

The vocabulary of `skcomplex.core.base` gains `OrderKind` with the terms `'integer'` and `'fractional'`, and `Supports`
gains an `orders` field over them. Every existing solver and backend declares `orders={'integer'}`. The new solvers
declare:

```text
Supports(time={continuous}, space=every kind, laws={rate, equation, boundary, transfer},
         domains=numeric scalar domains, components={element, group, site-group, system},
         orders={'integer', 'fractional'})
```

and the NumPy and Numba backends declare fractional orders for them.

`validate()` reports:
- `fractional-order`: an order that is not an integer and not in `(0, 1)`, naming the state, with the fix of an order
  in that interval;
- `fractional-time`: a fractional state in a system whose time is not continuous, with the fix of continuous time;
- `order-unsupported`: a fractional state with a chosen solver or backend that does not declare fractional orders, with
  the fix listing the fractional solvers;
- the usual law issues for laws the fractional solvers do not support, such as a noise law on a fractional state.

### Solvers

`GrunwaldLetnikov(dt, memory=None)` uses the binomial weights `w_j = (-1)^j C(α, j)`, computed by recursion.
`L1Scheme(dt, memory=None)` uses the L1 weights for the Caputo derivative. `FractionalABM(dt)` is the
predictor-corrector of Diethelm, Ford, and Freed. `memory` limits the sum to a window for the short-memory principle.
Histories are stored in a fixed-size array per state, and the weighted sums are dot products. The Numba backend fuses
them.

### Mittag-Leffler

`mittag_leffler(alpha, beta, z)` in `skcomplex.core.analysis` evaluates `E_{α,β}(z)` by series for small `|z|` and by
its integral representation otherwise. `solve_symbolically` returns `x0 E_α(-λ t^α)` for linear fractional relaxation.

## Risks / Trade-offs

- [Quadratic cost of the full memory] → The short-memory window bounds it, and the user guide states the error
  trade-off.
- [Low order of accuracy] → The predictor-corrector gives order `1 + α`, and tests set tolerances per scheme.
- [The order is not a run-time parameter] → Sweeps rebuild the system per order, which the factories make one call.

## Open Questions

- Resolved: the order of a state is fixed per built system. `state(order=...)` takes a number, not a parameter symbol,
  so the memory weights are computed once per run, and a sweep over orders builds one system per order, which the
  library factories make one call. A run-time order can follow as its own change if a model needs it.

## Implementation Notes

- The window of `memory` keeps the exact weights for the last `memory / dt` steps, and the older history enters
  through about 40 exponential modes from a trapezoidal rule of the Laplace representation of the weights, updated by
  one recursion per step. A plain truncation of the sum stays 2.4e-2 away from the full memory for relaxation of
  order 0.6 at time 20 with a window of 10, far from the 1e-3 the short memory scenario asks for, and the modes bring
  it to 1e-8. The modes fold only the history past the window, so they are not the FFT-based fast convolution of the
  non-goals.
- The explicit L1 scheme sums the differences of the L1 weights over the displacements from the initial state, which
  decay as `j^(-1 - alpha)` like the Grünwald-Letnikov weights, so both schemes share one stepper base.
- A change of a fractional state between frames starts a new history from the changed state, and states of order 1
  take Euler or Heun steps from the state given at every frame, so co-simulation partitions and interventions work.
- The steppers run in Python with NumPy dot products over the history and the drift of the backend, compiled by Numba
  with the Numba backend. The Numba backend has no fused stepper loop for the fractional form.
- `mittag_leffler` sums the power series in arbitrary precision with Sage's MPFR fields for `|z|^(1/alpha) <= 50`,
  enough bits for the cancellations, and takes the asymptotic expansion beyond, instead of the integral
  representation. No maintained Python library provides the function: SciPy and Sage lack it, mpmath has no
  Mittag-Leffler function, and the small `mittag-leffler` package on PyPI is unmaintained.
- `OrderKind` is in the vocabulary of `skcomplex.core.base` and in `Supports`, but not in `VOCABULARY`, the categories
  of the system tags of the model inventory, since the inventory would then need an order tag on every entry.
- The delay history buffer keeps interpolants over adaptive steps, so the fractional steppers keep their own history
  on the fixed grid instead of reusing it.
