# Design

## Context

See proposal.md for the workarounds this change removes. The relevant code today:
- `role()` in `skcomplex.core.base._variables` declares a required string field. `_resolve_role` in the compiler
  resolves it to one `_Block`, an element or a group, relative to the interaction's scope.
- `_interaction_spec` raises `RoleError` when roles reach two groups, and evaluates the interaction once per member of
  its one group, with `Source.MEMBER` references for the group and `Source.SINGLE` references for single elements.
  Agent systems compile through the same role collection, so rule-only interactions inherit the limit.
- The field lowering already reads a member's neighbour through `NeighborRead`, an index table per offset with a
  periodic, zero-flux, or fixed mode, and lowers it to NumPy gathers and to Numba loops.
- `Rule` functions of interactions receive one `ModelView`, which is not scoped. `ModelView.parameters` is the flat
  mapping of dotted parameter names, and `ModelView.agents` and `AgentView.agents_here` resolve a group by its path or
  by a label relative to another group.
- `Element.group(label, size, *, visual=None, **values)` and `Element.lattice(label, space, *, visual=None, **values)`
  take their own arguments by keyword, so a state of the same name cannot appear in `**values`.

## Goals / Non-Goals

**Goals:**

- Every model the proposal names is written without a padding role, a hard-coded group path, a duplicated space
  parameter, or one interaction per group member.
- Laws stay symbolic, so variadic roles and member offsets compile once per group, whatever its size.

**Non-Goals:**

- A role whose sequence length differs per group member, such as a group of Boolean nodes with a different number of
  inputs each. The Boolean networks stay one element and one interaction per node, which is cheap at their sizes.
- Reads of arbitrary members by a per-member index table. Network neighbours of a group already go through network
  spaces, and the offset read covers ordered chains and rings.
- Field reads at physical displacements, which `add-nonlocal-operators` adds as `shifted`. `shift_member` counts
  members, not distances, and keeps a distinct name.
- Geographic agent spaces. `extend-agent-spaces` adds them, and their views report no `dimensions`.

## Decisions

### Library survey

No new dependency is needed. Every capability lowers to libraries the core already uses:
- **NumPy, chosen for member offsets.** An offset read is a gather by an index table, the operation `NeighborRead`
  already performs. A one-dimensional `LatticeSpace` of the group size builds the table with its existing
  `index_table`, so no index code is written twice.
- **Mesa 3.5.1, chosen for the space geometry.** Mesa's continuous space holds its bounds and torus flag and computes
  torus-corrected difference vectors. Its orthogonal grids hold their dimensions and torus flag. The views only read
  them.
- **NetworkX, chosen for network distances.** It already backs `NetworkSpace`, and `shortest_path_length` gives the hop
  distance between nodes.
- **Boolean network packages, rejected.** PyBoolNet, CANA, and the CoLoMoTo notebook tools are mature simulators of
  Boolean networks with their own model formats. They do not give an interaction a variable number of roles, which is
  a feature of the modelling layer that every field uses, not a Boolean network solver.
- **JAX `vmap` and `xarray` shifts, rejected.** Neither is a dependency, and both would bypass the NumPy and Numba
  backends that the kernels already target.

### Variadic roles

`role(variadic=True)` declares a field that takes a sequence of strings. Each entry is an element path, or a member
reference `path[index]` to one member of a group, both relative to the interaction's scope. The compiler resolves every
entry to a block and, for member references, to a column offset of one member, with `Source.SINGLE` references. Inside
`laws`, `s.<role>` is a tuple of namespaces, one per entry, in the given order, so `len(s.inputs)`, indexing, and
iteration work in plain Python while the law is built.

`parameter(per='<role>')` declares a parameter with one value per entry of a variadic role. The constructor stores the
sequence unmodified, the layout gives it one column per entry, and `s.<parameter>` is a tuple of symbols. A length that
differs from the role's raises `ValidationError` at compile time, naming the interaction, the parameter, and both
lengths. Each component of such a parameter is addressed as `<path>.<name>[<i>]`, so it updates live like any other.

Alternative: one generic `inputs` role per node of a network element. It would need a network-aware law kind that the
Boolean networks alone use, while a variadic role serves every interaction with a variable arity, such as reaction
networks with any number of reactants.

### Several groups per interaction

The compiler drops the one-group check for interactions. When roles reach several groups, every group must have the
same size, the interaction runs once per member index, and row `i` of every group is read together. A group of another
size raises `RoleError` naming the groups and their sizes, and the message points to pair interactions for couplings
between all members. Single elements stay shared by every row as today.

Agent groups have no fixed size, so rule-only interactions are exempt from the size check. Their roles bind any number
of agent groups, and the rule reads each one through `model.agents('<role>')`.

### Member offset reads

`shift_member(expression, offset, *, boundary='periodic', fill=0.0)` returns a symbolic call, like `lag` and `delayed`,
that the compiler replaces before lowering. The expression must read states or parameters of one group through a role,
and `offset` is a whole number of members along the group order. The boundary names follow `LatticeSpace`:
- `'periodic'` wraps past the last member to the first, which closes a ring;
- `'reflecting'` reads the end member itself past the end;
- `'fixed'` reads `fill` past the end.

The lowering builds the index table of a one-dimensional lattice of the group size with that boundary, and replaces each
member symbol by a symbol with a `NeighborRead` reference, the path that lattice and network laws already take. Both
backends therefore vectorise it, and the ring road is one kernel over the group. Offsets nest by adding.

The name is a function, not a method, because Sage expressions take no new methods and because a method on the
namespace would hide a state of the same name. `s.x.shift(1)` therefore becomes `shift_member(s.cars.x, 1)`.

Alternative: a group placed on a one-dimensional `LatticeSpace` with `neighbors`. That function serves the local
energies and rules of lattice states, and a car ring is an ordered group of ordinary elements whose rate laws read the
car ahead. `shift_member` works in every law kind and on every group, and shares the lowering.

Positions on a ring jump by the ring length where they wrap, so the headway of the last car is written with the
remainder function the modelling layer already has, `mod(shift_member(s.cars.x, 1) - s.cars.x, length)`. No per-car
offset parameter is needed.

### Rules scoped to their owner

The stepper passes every rule a `ModelView` scoped to the component that returned it, the interaction's path or the
agent group's path. The scoped view adds:
- `parameter(name)`, the current value of the owner's parameter, with `name[i]` for a component of a per-entry
  parameter;
- `agents(name)` and `count(name)`, which resolve a role of the owner first, then a path relative to the owner's
  scope, then an absolute path, and raise `SimulationError` listing the roles and groups otherwise.

`ModelView.parameters` stays the flat mapping by dotted path. `AgentView.group` stays the full path, the identifier
trajectories use, and `AgentView.group_label` returns the group's label alone.

Alternative: `self.parameter('name')` on the component. A component is a plain estimator that may be cloned or placed
in two systems, and it does not know its compiled path or the running stepper. Reading through it would need hidden
ambient state, which the constitution forbids. The scoped view passes the same information explicitly.

### Space dimensions and displacements

`ModelView.dimensions` returns the extent of the agent space per axis, `(width, height)` for grids and continuous
spaces, with a third entry when `extend-agent-spaces` adds a depth, and `None` for networks. `ModelView.torus` returns
whether the space wraps. `AgentView.displacement_to(other)` takes an agent view or a location and returns the vector to
it, the shortest image on a torus, from Mesa's difference vector in continuous spaces and from the grid dimensions on
grids. `AgentView.distance_to(other)` returns its Euclidean norm, or the hop count from NetworkX on networks, where
`displacement_to` raises `SpaceError`.

### Positional group arguments and reserved names

`Element.group(label, size, /, *, visual=None, **values)` and `Element.lattice(label, space, /, *, visual=None,
**values)` make their first two arguments positional-only, so `size` and `space` reach `**values` when an element
declares them. The few calls that pass `size=` are rewritten. `visual` stays a keyword, because it is the class
variable of every element and is reserved.

`Component.__init_subclass__` checks declared names against a reserved set: `label` and `visual` for every element,
and `location` and the names `AgentView` defines for agent types. A clash raises `LabelError` when the class or
`define` creates the type, naming the field and the reserved names.

Alternative: rename `size` to `n_members`. It touches every group call in the library, the docs, and the tests, and
still leaves `visual` and `space` to collide, so positional-only arguments plus one reserved set is the smaller change.

## Risks / Trade-offs

- [Positional-only `size` breaks callers that pass it by keyword] → The changelog marks it breaking, and the
  library, docs, and tests are migrated in the same change.
- [A group whose order is not meaningful still accepts offset reads] → The docstring of `shift_member` states that the
  order is the member index, and the ring scenario checks the trajectories against the per-car model.
- [Members of different sizes bound to one interaction are a common mistake] → The error names both sizes and points
  to pair interactions.
- [Scoped resolution could pick a role over a same-named group] → Roles win by rule, and the error lists both kinds.

## Migration Plan

Each model rebuild keeps its registered name and arguments, except that `random_boolean_network` accepts any `k` up to
`n_nodes` and `trafficking` and `ring_road_drivers` no longer pass the space size to their interactions and agents.
Their per-car or per-node trajectories are compared with the old versions before the old code is removed.

## Reconciliation and Implementation Notes

`add-group-reductions` landed before this change was applied. It added element sets, reductions, `Source.REDUCED`, and
observable reads, and it relaxed the one-group check. The decisions above hold with these adjustments:
- **Element sets and variadic roles stay two mechanisms.** Both bind a list of paths. An element set, a list bound to
  an ordinary role, applies the interaction's laws once per element, each read like a single element, and reductions
  over the role reduce over the set. A variadic role, declared with `role(variadic=True)`, gives one law the whole
  sequence as a tuple of namespaces, the fan-in of a Boolean node. A variadic role cannot be expressed as an element
  set, since a set maps laws over its elements instead of joining them, so the declaration tells the two apart. The
  resolution of plain paths is shared through `_resolve_path`. An entry of a variadic role is a single element or a
  member reference, and a bare group path raises `RoleError` that suggests a member reference. Laws combine the
  entries with Python's `sum` over the tuple, since the number of entries is fixed per interaction, and reductions do
  not take a variadic role.
- **The one-group check was already partly relaxed.** After `add-group-reductions`, `_settle_rows` raised only when
  laws read the members of two groups outside reductions, and rule-only interactions read no symbols, so they already
  passed. This change replaces that error with the equal-size check and keeps every group read outside reductions as
  an aligned group of the kernel, so `_prune_foreign` keeps their references. Rule-only interactions stay exempt
  because rules read no symbols, and a test covers two agent groups of 42 and 43 agents.
- **Member references reuse the single read.** A member reference `nodes[3]` is read with `Source.SINGLE` and a column
  counted from the group's first component, `3 * columns + column`, instead of a new source. The Numba backend and the
  Jacobian already index single reads by `start + column`, and NumPy single reads and writes now index the flat
  vector the same way. The member's symbols carry the suffix `__m<i>`.
- **Per-entry parameters are vectors of the layout.** A parameter declared `per=<role>` has the shape `(k,)`, so its
  components are named `<path>.<name>[<i>]` by the existing component names. `Layout.locate` resolves a variable or
  one component, and `Simulation.update_parameters`, `schedule`, and `ModelView.parameter` take component names. A
  scalar value is shared by every entry. Laws read the parameter as a tuple of symbols.
- **`shift_member` lowers in its own module.** `skcomplex.core.compile._members` replaces each call, innermost first,
  by a symbol with a `Source.NEIGHBOR` reference whose `NeighborRead` holds the table of a one-dimensional
  `LatticeSpace` of the group size, the sizes `(rows,)`, the step `(offset,)`, and the mode of the boundary. Sage
  functions take neither keywords nor strings, so the Python function `shift_member` passes the boundary as its index
  in `('periodic', 'reflecting', 'fixed')` and the fill as a fourth operand. The lowering runs right after the
  operator lowering, through one helper, and reads only member symbols of one group.
- **Group resolution is relative first everywhere.** `MesaStepper.resolve_group` now tries the path relative to the
  owner's system before the path from the root, so `agents_here` and `create` resolve like the scoped view.
  `ModelView.parameter` returns a number, and a parameter with several components must be read by component.
- **Displacements use NumPy on Mesa's geometry.** Mesa's `calculate_difference_vector` takes agents, not locations, so
  `displacement_to` applies the same minimum image rule with NumPy to the extent and the torus flag that the Mesa
  space holds. `ModelView.dimensions` reads the Mesa grid's dimensions and the continuous space's width and height.
- **Reserved names are a class variable.** `Component.reserved_names` holds `label`, elements add `visual`, and agent
  types add every public name of `AgentView`. A test checks that the agent view's public names are reserved.
- **The meadow grazes in an interaction.** The cow's rule read the pasture with `agents_here('pasture')`, a group named
  by path. Grazing moves into the interaction `Grazing`, which binds `cows` and `pasture` like `Herding`. The week now
  regrows the grass, then the cows graze, then they move, where before regrowth and grazing interleaved in the random
  activation. The recorded cow counts of random state 0 are unchanged, and the nested scenario compares the nested
  run with the root run.
- **The layout check covers interactions.** The check reads the source of every model. It rejects a formatted string
  that starts with `{self.label}.`, and in every interaction a call `model.agents`, `model.count`, or `model.create`
  whose group is a string or `self.<attribute>` that is not one of its roles. Agent types have no roles, and their
  rules read groups of their own system by relative label through `agents_here` and `create`, which resolves inside
  the agent's system, so the check leaves them. Eight interactions named their group by a class variable or a plain
  string field, and their groups become roles: `Redesign.components`, `Invention.technologies`,
  `CournotAdjustment.firms`, `CarWashOperation.cars` and `till`, `DebtRankCascade.banks`, `ReflectingFloor.firms`,
  `Reproduction.population`, and `Evolution.population`. Their factories bind the same paths as before.
- **The ring roads name the ring length `road_length`.** `OptimalVelocity` and `IntelligentDriver` bind the role
  `cars` and take `road_length`, and `IntelligentDriver.length` stays the car length.
- **Recorded trajectories live in the tests.** The yeast states and the trafficking trips and bills of random state 0
  are literals in the model tests. The ring roads are compared with a per-car rebuild of the old interactions inside
  the test, which runs the old laws instead of storing arrays.
- **The changelog comes from commit messages.** `CHANGELOG.md` is generated by the changelog session from the commit
  history, so the breaking changes go into the commit message instead of an edited changelog.

### Right-hand side timings

One evaluation of the drift of the optimal velocity ring with NumPy:

| Cars | One interaction per car | One group |
| --- | --- | --- |
| 20 | 1.27 ms | 0.047 ms |
| 200 | 12.3 ms | |
| 1000 | | 0.057 ms |

The per-car ring of 1000 cars was not timed, since its compile time grows with the number of interactions.

