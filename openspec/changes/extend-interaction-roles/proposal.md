# Proposal

## Why

Writing the model library exposed six gaps in how interactions reach their elements, and each one forced a workaround
into a model:
- A role binds exactly one element. `random_boolean_network` and `yeast_cell_cycle` declare five fixed input roles,
  bind the unused ones to the target, and cap the number of inputs `k` at five.
- An interaction acts on at most one group, which the compiler enforces with the `RoleError` "acts on at most one
  group", also for interactions whose only laws are agent rules. `trafficking` passes the premises group as a plain
  string, and `Herding` reads the pasture group by the hard-coded path `'pasture'`.
- A law of a group member cannot read another member. The ring roads of `optimal_velocity` and
  `intelligent_driver_model` need one element and one interaction per car, so the right-hand side of 20 cars takes
  about 1.5 ms instead of one vectorised kernel.
- Interaction rules read their own parameters as `model.parameters[f'{self.label}.name']`, which is wrong as soon as
  the interaction sits in a subsystem, and `AgentView.group` returns a full path, so comparisons by label break there.
- Rules cannot read the dimensions of the agent space or a torus-aware displacement, so `ring_road_drivers` and
  `trafficking` repeat the road length and the grid size as parameters.
- `Element.group(label, size, **values)` takes `size` as a keyword, so an element type with a state named `size`
  cannot set it per member.

## What Changes

- Add variadic roles: a role declared variadic binds a sequence of element paths or group member references, laws
  read it as a sequence of namespaces, and a parameter can hold one value per bound entry.
- Let one interaction bind roles to several groups. Laws of groups of equal size run member by member, and agent rule
  interactions bind any groups and read each through its role.
- Add member offset reads, `shift_member(s.cars.x, 1)`, which read a member expression at the member a given number of
  places along the group order, with periodic, reflecting, or fixed ends. They compile into the group's kernel.
- Give rules a model view scoped to the component that owns them: `model.parameter('name')` reads the owner's
  parameter, `model.agents('role')` reads the group bound to a role, and `AgentView.group_label` gives the group's own
  label.
- Add `ModelView.dimensions`, `ModelView.torus`, `AgentView.displacement_to`, and `AgentView.distance_to`, torus-aware
  on grids and continuous spaces and shortest-path on networks.
- **BREAKING:** `label` and `size` of `Element.group`, and `label` and `space` of `Element.lattice`, become
  positional-only, so states named `size` or `space` take per-member values. State and parameter names that clash with
  a component's own attributes raise a named error when the type is defined.
- Rebuild `random_boolean_network` and `yeast_cell_cycle` on variadic roles without the cap on `k`, the ring roads of
  `optimal_velocity` and `intelligent_driver_model` as one group and one interaction, `trafficking` and
  `meadow_capacity` with roles on both groups, and `ring_road_drivers` and `trafficking` on the space dimensions.
- **BREAKING:** the ring roads report `cars.x` and `cars.v` with one column per car instead of `car_<i>.x` and
  `car_<i>.v`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: variadic roles, roles on several groups, member offset reads, positional group arguments, and
  reserved names.
- `agent-based-modeling`: rules scoped to their component, space dimensions, and torus-aware displacements.
- `model-library`: Boolean networks without a cap on inputs, ring roads as one group, and agent models that read
  their groups through roles and the space through the model view.

## Impact

- `skcomplex.core.base`: `role` gains `variadic`, `parameter` gains `per`, and `LabelError` covers reserved names.
- `skcomplex.core.model`: `Element.group` and `Element.lattice` take positional-only arguments, and the new symbolic
  function `shift_member` is re-exported.
- `skcomplex.core.compile`: role resolution for sequences and member references, several groups per interaction, and
  member offset reads lowered through the existing `NeighborRead` tables of the field lowering.
- `skcomplex.core.solvers`: `ModelView` gains `parameter`, `dimensions`, and `torus`, and rule views are scoped to the
  rule's owner. `AgentView` gains `group_label`, `displacement_to`, and `distance_to`.
- Models in `biology/systems_biology`, `engineering/traffic`, `social/organizations`, and `biology/ecology`, and every
  call of `Element.group` that passes `size=` by keyword.
- No new dependency. NumPy, Mesa, and NetworkX, already dependencies, cover the index tables, the space geometry, and
  the network distances.
- Composes with `extend-agent-spaces`: dimensions and displacements cover a third dimension when that change adds one,
  and geographic spaces report no dimensions.
