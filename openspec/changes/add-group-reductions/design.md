# Design

## Context

See proposal.md for the models that need reductions and the workarounds they use now.

The compiler resolves every symbol of a law to a `Reference` whose `Source` says how a kernel reads it: `MEMBER`,
`SINGLE`, `FIRST`, `SECOND`, `NEIGHBOR`, or `DERIVED`. A role bound to a group applies the interaction's laws to every
member, and a law whose target is a single element and whose value reads members adds up over the members
(`has_group_sums`). That linear sum is the only reduction a law can express. The sum cannot sit inside a nonlinear
function, and a member cannot read it back in the same evaluation.

`Observable` already holds the reduction table `REDUCTIONS` and `reduce_values`, and `ObservableEvaluator` computes
observables at every frame. A role can bind to an observable path, and `_with_held_observables` turns that observable
into an `Aggregate` whose value a co-simulation refreshes at synchronisation points. `compile_system` rejects such a
read when the system has one form, so a law cannot read an observable of its own partition.

The symbolic analysis in `skcomplex.core.analysis` rejects every system with groups, so the vector-state workaround is
also what keeps `fixed_points` working on replicator dynamics.

`add-adaptive-networks` specifies `neighbors(expression, over=..., reduce=...)`, a reduction whose argument reads both
the reading member and its neighbour, such as `sin(s.other.theta - s.theta)`. This change specifies the reductions
whose argument reads only the reduced member.

## Goals / Non-Goals

**Goals:**

- Write a mean-field law as its textbook equation, with the reduction inside any expression.
- Evaluate a reduction once per evaluation, at a cost linear in the group size, and read it in every member kernel.
- One reduction table for observables, aggregates, and laws.

**Non-Goals:**

- Reductions whose argument reads the reading member, such as pair sums with a kernel. They are `neighbors` in
  `add-adaptive-networks`, and a complete graph covers the all-to-all case.
- Segmented reductions by a member label, such as one mean per node of a network. `neighbors` lowers those as segment
  reductions over CSR rows.
- Reductions over the axes of a phase-space field. `add-phase-space-fields` specifies `moment`.
- Reductions inside Mesa agent rules, which read model reporters in Python.
- Low-rank Jacobian updates for implicit solvers. SciPy's BDF and Radau take a dense or sparse matrix.

## Decisions

### Libraries surveyed

The feature is the lowering of symbolic reductions into the compiled kernels. It is built on the numerical libraries
the project already uses, and the in-house part is the adapter that lowers the Sage function calls to them:
- **NumPy:** `np.add.reduce`, `np.mean`, `np.maximum.reduce`, `np.minimum.reduce`, `np.var`, and masked reductions
  through `np.where` evaluate a reduction over the member axis on the NumPy backend. `np.add.at` scatters weighted
  contributions when the weights are given as index pairs.
- **SciPy sparse:** a weight matrix is held as a CSR matrix, and a weighted reduction is the product `W @ values`.
- **Numba:** the Numba backend fuses the reduction into one `prange` loop with a reduction variable for sums and
  counts, and a serial loop for the maximum and minimum, before the member loop.

Libraries rejected:
- **Sage's symbolic `sum`, `max_symbolic`, and `min_symbolic`:** they sum over a symbolic index or compare a fixed
  list of expressions, and know nothing of member arrays. The reductions are Sage functions so that laws stay Sage
  expressions, and the lowering replaces them.
- **SymPy `Sum` over `IndexedBase`:** Sage is the symbolic layer, and SymPy values are rejected in laws.
- **JAX `segment_sum` and PyTorch `scatter_reduce`:** heavy dependencies that fit neither backend.
- **pandas `groupby` and `numbagg`:** built for labelled tables and moving windows, with a per-call overhead far above
  a drift evaluation.

### Reduction functions

`total`, `mean`, `maximum`, `minimum`, `variance`, and `count` are Sage `BuiltinFunction`s in
`skcomplex.core.model._functions`, next to `where` and `mod`, and their names map to the keys of `REDUCTIONS`.
`count` adds to that table, which already holds it as a special case. The signature is
`total(expression, over=None, where=None, weights=None)`, and `count(over=None, where=None)` takes no expression:
- Inside the call, the symbols of a member stand for the reduced member, and outside it they stand for the reading
  member. So `Rate(s.x, -s.x + tanh(total(s.x)))` reads the group total inside a member's own law.
- `over` defaults to the one group whose symbols the argument reads. In an interaction, the argument reads the symbols
  of the role bound to the reduced group, such as `mean(pulse(s.neurons.theta))`. An argument that reads symbols of two
  groups is rejected with a named error that names both.
- `where` is a condition on the reduced member, such as `s.x > s.theta_z`. The reductions of no selected members are 0,
  except `count`, as for observables.
- `weights` is a constant array or SciPy sparse matrix of shape `(readers, members)`, so the reading member `i` gets
  `sum_j W_ij a_j`. It is fixed when compiling, like a topology, and a change needs a new compile. It is the payoff
  matrix of replicator dynamics and the catalytic links of hypercycles. It is allowed with `total` and `mean` only.

Alternatives considered:
- A lambda over a member namespace, `total(lambda m: m.x, over='population')`. It spells the reduced member
  explicitly, but every law must then name the group, and a role already binds it.
- Python `sum` over the members. It unrolls the group into the expression, which `Groups compile independently of
  their size` forbids.

### Element sets

A role binds to a list of paths of single elements of one type, such as `units=['unit_0', 'unit_1']`. The compiler
gathers the listed elements into a group layout for that interaction, so the role behaves like a role bound to a group,
and reductions over it reduce over the listed elements. The elements keep their own paths in the state vector, and the
gather is an index array. Elements of different types in one set are rejected with a named error.

### Observable reads

A role that binds to an observable reads it as `s.<role>.value`, as it does today. When the observable is symbolic, an
expression of states, parameters, and earlier observables, or a reduction of a group with a variable or a symbolic
expression, and the reading law is in the observable's partition, the compiler lowers it into the reduction pass and the
law reads its value at the current state. A Python callable observable cannot be lowered, so reading it in its own
partition raises a named error that tells the user to write it symbolically or move the reading law to another
partition. Across partitions, the observable is held at synchronisation points as today.

`ObservableEvaluator` and the reduction pass compute the same values from the same table. The trajectory records the
observable from the frame's state, which equals the value a law read at that state.

### Lowering and ordering

The compiler collects every reduction call from the lowered law expressions, as it collects `delayed` calls, and
deduplicates them by their kind, group, argument, condition, and weights. Each becomes a reduction spec and a new
`Source.REDUCED` reference. Kernels read an unweighted reduction like a `SINGLE` value and a weighted one like a
`MEMBER` value of the reading group.

Every evaluation of a partition runs in this order:
1. read the state, and solve the algebraic `Equation` laws as today;
2. evaluate the reductions in topological order, so a reduction whose argument holds another reduction, such as the
   mean fitness that sums each member's weighted payoff, comes after it;
3. evaluate the member, pair, field, and interaction kernels, which read the reductions.

A reduction reads states, parameters, time, observables, and other reductions, never a rate, so the order has no cycle.
A reduction over a group in another partition reads the state held at the last synchronisation point, as an aggregate
does. Discrete maps evaluate the reductions on the state at step `n`, so every member updates from the same values.
Jump processes recompute the reductions their propensities read after every event. On Numba, sums, means, and counts
are updated by the change of the members the event touched, and maxima and minima are recomputed. Boundary and event
laws evaluate the reductions on the state they receive.

### Jacobian structure

A reduction makes every reading member depend on every reduced member. The chain rule gives
`J_ij = ∂F_i/∂x_i δ_ij + ∂F_i/∂R · ∂R/∂x_j`, with `∂R/∂x_j` equal to `∂a/∂x_j` for `total`, divided by the size for
`mean`, `W_ij ∂a/∂x_j` with weights, the indicator of the first extreme member for `maximum` and `minimum`, the
derivative of the variance for `variance`, and 0 for `count` and for the condition.

For `adopt-scipy-sparse-jacobians`, `Source.REDUCED` states its pattern: the block of the reading rows against the rows
of the reduced members for every state the argument reads, or the nonzeros of `W` with weights. For the exact sparse
Jacobian, the partial derivative by a reduction compiles as one more output of the kernel, and the scatter forms the
rank-one blocks. When a dense block exceeds the pattern's size limit, the pattern is omitted and `ScipyODE` estimates a
dense Jacobian, which the user guide states with the advice to use an explicit stepper for large mean-field groups.

### Symbolic analysis of small groups

`dynamics_of` expands a group of at most 64 members whose laws read reductions into one equation per member, with each
reduction written as the explicit sum, mean, maximum, or minimum of the member symbols. `jacobian`, `fixed_points`, and
`stability` then work on the expanded system. Larger groups keep the named error, which now states the limit.

## Risks / Trade-offs

- [Dense Jacobian blocks for large groups] → The pattern is omitted above a size limit, and the user guide recommends
  explicit steppers, which mean-field models rarely need to avoid.
- [Ties in maximum and minimum] → The derivative takes the first extreme member, the same convention as NumPy's
  `argmax`. It matters only for the exact Jacobian.
- [Changed model states] → Rebuilt models change state names and shapes, such as `population.x` becoming a group
  state. The changelog lists them, and the inventory check keeps the names.
- [Two places that compute observables] → The reduction pass and `ObservableEvaluator` share `REDUCTIONS`, and a test
  checks that they agree on every library model with observables.

## Migration Plan

The rebuilt models keep their factory names and parameters. Code that reads `synapse.s` of `theta_neurons` or the
vector components of `population.x` reads the group states instead, which the changelog states.
