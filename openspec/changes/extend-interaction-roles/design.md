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
