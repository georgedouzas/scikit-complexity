# Tasks

## 1. Order and companions

- [x] 1.1 Add `order` and `initial` to `state`, with generated `<state>_d<j>` fields in classes and `define`, and verify
  the initial angular velocity and first-order scenarios
- [x] 1.2 Add the order-reduction stage, so laws on order-`k` states drive the `k`-th derivative and order assertions
  are checked, and verify the Newton's law, third-order, and order mismatch scenarios, and that velocity Verlet runs a
  position of order 2

## 2. Equations

- [x] 2.1 Add `derivative` and `Equation`, with joint symbolic solving for highest derivatives and the numeric fallback,
  and verify the pendulum-as-written, coupled equations, and unsolvable scenarios

## 3. Difference equations

- [x] 3.1 Add `lags` on discrete states, `lag`, and companion shifting in the iteration, and verify the
  multiplier-accelerator and Fibonacci scenarios

## 4. Analysis

- [x] 4.1 Return equations in their written order, and verify the pendulum equation scenario
- [x] 4.2 Solve equations of any order with initial derivatives, reject unevaluated results, and pass parameter signs,
  and verify the small-angle, third-order, nonlinear pendulum, and plain positive numbers scenarios

## 5. Library and documentation

- [x] 5.1 Add the higher-order models as registered factories, and verify the circuit and oscillator scenario and the
  conformance suite
- [x] 5.2 Document orders, laws of any order, equations, lags, and the analysis of higher-order equations in the user
  guide, with Newton's second law as the first example, and verify the documentation test passes
- [x] 5.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
