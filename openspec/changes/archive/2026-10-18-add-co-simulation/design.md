# Design

## Context

After the four preceding changes, a compiled `Problem` has one form, one stepper, and one state representation: a flat
vector for ODE, SDE, jump, and discrete forms, a LAMMPS instance for the LAMMPS backend, and a Mesa model with snapshots
for agents. `_infer_form` rejects any set of law kinds outside `FORMS`. Coupling different forms needs a layer above
single problems, and the standard approach is co-simulation as in the FMI standard: each part advances with its own
solver, and values are exchanged at synchronisation points.

## Goals / Non-Goals

**Goals:**

- Any combination of forms in one system, partitioned at subsystem boundaries.
- Read-only coupling through inputs and aggregates, with a known first-order splitting error.
- No change for systems that fit one form.

**Non-Goals:**

- Higher-order coupling schemes, such as extrapolated inputs, or adaptive synchronisation intervals. The scheduler's
  scheme parameter leaves room for them.
- Instantaneous writes across partitions, such as an agent rule setting an ODE state. Effects cross partitions through
  laws that read inputs.
- Running partitions in parallel processes. Jacobi allows it later.

## Decisions

### Partitioning at subsystem boundaries

`compile_system` first infers one form for the whole system as today. If that fails, it builds a partition tree: a
subsystem whose laws fit one form becomes a partition, and a subsystem that does not is split into its own elements,
which must fit one form, and its child subsystems, recursively. A law joins the partition that owns its target, so an
interaction in a parent may contribute to several partitions. A leaf subsystem whose own laws mix kinds raises
`IncompatibleLawsError` with the subsystem path and the suggestion to split it.

- Alternative considered: partitioning per element. Rejected because it would split coupled ODE elements into
  separate solvers and add splitting error where none is needed.

### Inputs

Every partition is compiled with the state layout of the whole system and only the laws of its own elements, so the
states of other partitions appear in its vector without laws. They stay constant while it advances, which is exactly a
held input, and the scheduler writes their latest values before every advance. The lowered functions and the
backends need no change. Agent rules read the values through `model.inputs[name]`.

- Alternative considered: a fourth argument `u` of the lowered functions. Rejected because both backends and every
  stepper would change, while the held states give the same behaviour.

### Aggregates

`Aggregate(label, group=..., reduce='count' | 'sum' | 'mean' | callable, variable=None)` is an element whose single
state `value` is computed from its group at every synchronisation point, not integrated. Its partition is the partition
of its group, and other partitions read `label.value` as an input.

### Scheduler

A partitioned problem is a `Problem` of the form `Form.COUPLED` whose `partitions` hold every `Partition` in system
order, with its own problem, the state indices it owns, and its owner paths. `CoSimulation` is a `Stepper` for
partitioned problems, so `Simulation` uses it through `resolve_solver` without other changes. It takes the per-partition
solvers by partition path, `sync_dt`, and `scheme`.

- Gauss-Seidel advances partitions in order, each reading the inputs already advanced by those before it.
- Jacobi advances every partition from the same snapshot of inputs.
- `sync_dt` defaults to the frame duration and must divide it. Discrete and agent partitions advance whole steps, so
  for them `sync_dt` must be a multiple of their step.

### Frames and trajectories

The frame of a partitioned problem holds the flat states of the vector partitions under their dotted names, and the
agent snapshots of agent partitions, so observers and `Trajectory` see one namespace. Live parameter updates are routed
to the partition that owns the parameter.

## Risks / Trade-offs

- [Splitting adds error of order `sync_dt`] → The convergence requirement pins the rate, and the user guide states the
  trade-off between `sync_dt` and speed.
- [Tightly coupled stiff parts become unstable when split] → Partitioning only happens across forms, so parts of one
  form stay in one solver, and the error message points to a smaller `sync_dt`.
- [Gauss-Seidel results depend on partition order] → The order is the system declaration order, which is deterministic
  and documented.

## Implementation Notes

- A rule of an interaction acts for the first agent group among its roles, so an interaction may return a rate for an
  ODE state and a rule for agents.
- Aggregates are computed before the first frame and at every synchronisation point, through the stepper hook
  `initial_state`, and only in partitioned problems.
- The symbolic `equations` of a partition leave out interactions that sum over a group, as for single problems.
- Each partition draws from its own generator spawned from the simulation's random state, so runs are reproducible.
