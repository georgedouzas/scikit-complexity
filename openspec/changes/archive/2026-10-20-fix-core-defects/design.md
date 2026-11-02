# Design

## Context

The defects come from the issue log of the model-implementation agents and from the numerics audit in
`organize-model-library`. Each defect below was reproduced on the current code with `.venv/bin/python`, except where
the evidence is the code itself:
- A particle at 0.5 with speed 1 in a reflecting box of side 1 ends at `x = -3.5` after one frame of duration 5. The
  boundary is applied only in `Simulation._loop` and in the co-simulation partition loop, never inside a stepper.
- A rule that records `model.time` sees 0 at every step. `MesaStepper.advance` calls `agents.shuffle_do('step')`
  directly and never `model_.step()`, so Mesa's clock never moves.
- `where(x < 0, 1, 2).diff(y)` is `where(x < 0, 0, 0)`, which is not `is_trivial_zero()`. The linearity check in
  `_solve_equations` then rejects any equation with `where` or a driver, such as `step` or `pulse`, on its right side.
  `bool(where(x < 0, 0, 0) == 0)` also crashes inside Maxima.
- `Rate(x, -delayed(x, 1e-4))` under `ScipyODE` raises `IndexError: list index out of range`. SciPy's
  `select_initial_step` probes the field at `t0 + h0`, which reads the history past its recorded end when `h0 > τ`.
- `W.group('w', 1, location=[(1, 1)])` raises `unhashable type: 'list'`, and `W.group('w', 1, n=[3.0])` gives the
  agent `n = [3.0]`. `_member_value` and `_member_location` split per-member sequences only when `size > 1`.
- `np.float64(2.0) * v` for a Sage vector `v` is a NumPy object array, `np.float64(2.0) + v` broadcasts where `2.0 + v`
  raises, and `(np.float64(2.0) * v) * v` is the element-wise product instead of the dot product. NumPy scalars convert
  a Sage vector to an array before Sage sees the operation.
- An `Observable` with a lambda fails only when the system is pickled, for example by the conformance suite or a
  process executor.
- The compiled Gillespie SIR run with `NumbaBackend(parallel=True)` takes 0.28 s, against 0.014 s with
  `parallel=False`. Every event calls `prange` kernels that start a thread pool.
- `IMEX2.advance` called twice from the same state on Kuramoto-Sivashinsky gives results 8e-6 apart. `_previous` is
  reset only when the parameters change.
- Viscous Burgers on 200 finite-volume cells with `ν = 0.5` under `SSPRK` reaches `max |u| = 13` from an initial
  maximum of 2.5. `note_limit` records Laplacian limits only for finite-difference fields, and `SSPRK._step_size` uses
  the CFL condition alone.
- `run(until=0)`, then `update_parameters({'w.rate': 5.0})`, then `run(until=2)` uses the rate 1. `run` calls
  `_prepare`, which resets the time, the state, and the parameters. The same happens for ODE parameters.
- `Jump(s.r, {s.n: s.k})` raises `TypeError: unable to simplify to float approximation` from Sage.
- The NumPy drift of 20 populations with 20 logistic interactions takes 550 µs. `np.prod` called from `Entry.size`,
  `Entry.columns`, and `Entry.view` takes about 40% of that time.
- `ScipyODE.advance` calls `solve_ivp` once per frame, so every frame repeats the initial-step selection and the
  first function evaluations.
- `DEFAULT_DT = 0.01` gives 100001 frames for `run(until=1000)`.
- `where(x > 0, 1/x, 0)` on NumPy emits `RuntimeWarning: divide by zero`. `np.where` evaluates both branches.
- An agent state named `position` makes `_location_of` read the state as a location, so every snapshot raises
  `TypeError: iteration over a 0-d array`.

## Goals / Non-Goals

**Goals:**

- Fix each confirmed defect at its source, with a regression test that fails before the fix.
- Keep every fix inside the libraries the core already adapts, which are SciPy's `OdeSolver`, Mesa's model clock,
  NumPy's error state, and Numba's compile options. No new dependency.

**Non-Goals:**

- New features from the same log, which other changes own:
  - group reductions readable by laws in their own partition and observables in laws, owned by `add-group-reductions`;
  - variadic roles, interactions over several groups, member offsets, scoped rule parameters, `size` as a state
    name, and the space size in views, owned by `extend-interaction-roles`;
  - shared and correlated Wiener increments, owned by `add-correlated-noise`.
- The finite-volume wave speeds from `np.linalg.eigvals`, owned by `adopt-pyclaw-finite-volumes`, and the Jacobian
  sparsity of `ScipyODE`, owned by `adopt-scipy-sparse-jacobians`.
- Other items of the log that are features or tuning, not defects:
  - groups that start empty;
  - a `Domain` for probabilities;
  - symbolic `Jump` increments;
  - simultaneous events;
  - checkerboard rules that read their own site;
  - live parameters in `pulse_train`;
  - faster default SDE steps.
- Items that could not be confirmed:
  - The crash of a Laplacian with a state-dependent coefficient in `note_limit` did not reproduce for
    `Rate(u, u * laplacian(u))` with default or array initial values.
  - The collapse of per-member parameters to the first value in `ModelView.parameters` and `AgentView` does not
    arise. Agent group parameters have one value, and a per-member value is rejected with a named error.
  - Per-site arrays reaching `Drive` functions were not checked.

## Decisions

### Boundaries after every internal step

Every stepper applies `problem.boundary` after each internal step. This covers the NumPy loops and the fused Numba loops
of `RK4`, `Euler`, `Heun`, `EulerMaruyama`, `Milstein`, `VelocityVerlet`, `SSPRK`, the split steppers, and `Iteration`,
and Gillespie after each event. The Numba loops receive the compiled boundary as one more kernel argument.

`ScipyODE` treats a boundary like an event. After each accepted step, and at each frame time inside a step, it
applies the boundary. When the boundary changes the state, it finds the first time in the step at which the boundary
changes the dense output by bisection, applies the boundary there, and restarts the integrator from that time, so long
adaptive steps cannot carry a particle through two walls. `VelocityVerlet` applies the boundary after it moves the
positions and before it computes the accelerations of the second half kick, so a step still costs one drift
evaluation. `Simulation._loop` and the co-simulation loop stop
applying the boundary per frame, so a boundary that is not idempotent, such as a counter, is not applied twice.

The alternative of shortening frames to the internal step was rejected. It changes what observers receive, and it
still lets adaptive steps cross a wall.

### Mesa's clock

The stepper builds its model from a `mesa.Model` subclass whose `step` activates the agents and runs the model rules.
`advance` calls `model_.step()`, so Mesa's own wrapper moves `model.time` and runs any scheduled events.
`ModelView.time` keeps reading `model_.time`. During the step that produces frame `k`, it equals `k`, as Mesa defines.

### Conditions

`_Where._eval_` returns `then` when `then` and `otherwise` are the same expression, so `where(c, 0, 0)` is `0` and
derivatives of conditions vanish where both branch derivatives do. Driver functions build their expressions on
`where`, so they gain the same simplification.

The NumPy printer keeps `np.where`, and `numpy_function` runs the whole body of a function whose code holds a condition
inside `np.errstate(divide='ignore', invalid='ignore', over='ignore')`. A helper that takes the branches as values
cannot silence them, because Python evaluates them before the call, and the common subexpressions of a branch are bound
to names before the condition. A masked evaluation of each branch on its own members was rejected, because every input
of the branch would need to be broadcast and indexed. The scalar Numba and Lepton
printers already emit conditionals and stay as they are.

### NumPy values in laws

Law constructors check their values once. A NumPy scalar becomes a Sage real or integer. A NumPy array raises a named
`ValidationError` that names the law's target and asks for `float(...)` on the NumPy scalar that produced it. Coercing
the array back to a Sage vector was rejected, because the product of two vectors has already gone wrong before the law
sees it.

### Jump increments

`Jump` checks at construction that every increment is a number. An increment that holds a symbol raises a named
`ValidationError` that names the state and the increment and asks for a number. Symbolic increments are a feature and
stay out of scope.

### Observable pickling

`Observable.__init__` stores its arguments unmodified and then checks that `expression` and a callable `reduce` pickle
with the standard library. When they do not, it raises a named `ValidationError` that names the observable and asks
for a module-level function or a symbolic expression. This does not modify any argument, so the scikit-learn contract
holds, and `clone` repeats the same check.

Pickling functions by value with `cloudpickle` was rejected. Element types already pickle by reference, and code
pickled by value breaks across versions.

### One-member groups and reserved agent names

`_member_value` and `_member_location` treat a value as per-member when its leading length equals the group size, for
every size, including 1. A location is per-member when it is a list of locations whose length is the group size.

`AgentType` rejects state names that Mesa or the stepper read from agents, which are `position`, `pos`, `cell`,
`unique_id`, and `model`. They join the reserved names of agent types, so the type raises the same named `LabelError`
as for the names of the agent view when it is defined.

### Runs that keep their parameters and can be extended

`_prepare` takes the parameters from the previous run when there was one of the same system object with the same
parameter names and sizes, so values set with `update_parameters` stay in force. `run` still starts from time 0 and
the initial state.

`Simulation.extend(until)` continues from the last frame. It keeps the stepper, the clock, the observers, and the
recorder, applies pending parameter changes, and appends frames with steps numbered after the last one. It raises a
`SimulationError` when the simulation has not run or `until` is not later than the current time.

Making `run` itself continue was rejected, because runs that restart with a new clock or new observers, as the Game of
Life benchmark does, would change meaning.

### Default frame duration

When `dt` is `None` for a continuous form, the frame duration is the smallest value of the form `{1, 2, 5} × 10^k`
that is at least 0.01 and keeps `until / dt` at most 1000. An unbounded background run keeps 0.01. `_check_dt`
receives `until` from `run` and `start`, and `extend` keeps the frame duration of the first run.

### Persistent SciPy integrator

`ScipyODE` keeps the `OdeSolver` of `_advance_hybrid` between `advance` calls. Plain ODEs use the same path, with no
breakpoints and no events. The frame state comes from the dense output at the frame time.

The solver restarts when:
- the parameters change;
- the incoming state or time differs from the state and time it returned;
- a breakpoint, an event, or a boundary changes the state;
- a frame ends and a condition on time jumps at a time that depends on parameters, such as `step(t, s.step_time)`,
  which `breakpoints_of` cannot place and marks as `unresolved`. These problems keep the restart per frame they had
  before, so their frames still see the left limit of the input at a frame time.

`first_step` is set to at most half of the smallest delay, so SciPy's initial-step probe stays inside the recorded
history. Steps are at most the smallest delay, or a hundredth of the frame when the delay is shorter, and the history
of `ScipyODE` reads a time after its last step from the extension of that step's dense output. Bounding steps by a
delay of 1e-6 alone takes about a minute for a run until 0.2. The first three multiples of every delay after the start
are breakpoints of the persistent integrator, where the slope inherited from the history jumps. Breakpoints gain
`next_after(time)`, which finds the next periodic breakpoint without an end time. When `adopt-scipy-sparse-jacobians`
lands, the persistent solver receives the Jacobian or its sparsity once,
when it is built.

### Equations with floating point numbers

`_solve_equations` solves the linear system of the highest derivatives with `solve_right(check=False)` after checking
that its determinant is not zero. Sage's check fails on Python floats inside expressions, such as the height `1.0` of
`step(t, 1)` or the coefficient of `3.0 * x`.

### Serial event loops

`NumbaBackend` compiles every kernel that the fused event loops of Gillespie, `Iteration`, and lattice updates call per
event with `parallel=False`, whatever its `parallel` argument. The compiled lattice updates already run serially. It
keeps `parallel` for the drift and pair kernels of
groups of at least 1000 members, the same threshold as the automatic backend choice. Fused reductions from
`add-group-reductions` that run inside an event loop follow the same rule.

### Compiled layout

`Entry` is a frozen dataclass. Its `size`, `indices`, `columns`, and `rows` are computed once with `math.prod` in
`__post_init__` and stored as private fields. Kernels resolve every reference to a slice and a reshape shape when they
are built, so `_gather` slices and reshapes without calling layout properties. This removes the layout arithmetic from
every evaluation. It does not batch per-instance kernels into one, which `extend-interaction-roles` addresses for the
models that need it.

## Risks / Trade-offs

- [Errors in the chosen branch of `where` are silenced too] → The selected value still carries `inf` or `nan` into the
  state, and `checked=True` runs name the state that left its domain.
- [Rejecting lambda observables breaks existing code] → The error message names the fix, and the changelog lists it
  as breaking. The docstring example of `ObservableEvaluator`, the observable tests, and the user guide move to
  module-level functions.
- [Boundaries inside fused Numba loops add a call per step] → The boundary kernel is small and compiled. The
  real-time benchmarks of the interacting gas and the ideal gas must still pass.
- [Parameters that persist across runs surprise a user who expects a fresh start] → `run` documents that it keeps the
  current parameters. A fresh start is `clone(simulation).run(until)`.
- [The diffusive limit depends on parameters that change during a run] → The compiler keeps the limit as an expression
  of the parameters, and `SSPRK` evaluates it again whenever the parameters change.
