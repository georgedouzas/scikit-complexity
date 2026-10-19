# Tasks

## 1. Positional group arguments and reserved names

- [x] 1.1 Make `label` and `size` of `Element.group`, and `label` and `space` of `Element.lattice`, positional-only, and
  rewrite every call in `src`, `tests`, and `docs` that passes them by keyword, and verify the state named size scenario
- [x] 1.2 Check declared names against the reserved names of elements and agent types when a type is created, raising
  `LabelError`, and verify the state named visual and agent state named location scenarios

## 2. Variadic roles and per-entry parameters

- [x] 2.1 Add `variadic` to `role` and `per` to `parameter` in `skcomplex.core.base`, with docstrings and doctests, and
  verify the doctest run passes
- [x] 2.2 Resolve variadic entries and member references `path[index]` in the compiler, and give laws a tuple of
  namespaces, and verify the sum over a variable number of inputs, member reference, and missing member scenarios
- [x] 2.3 Lay out per-entry parameters with one column per entry, addressed as `<path>.<name>[<i>]`, and check their
  length, and verify the wrong length and live per-entry parameter scenarios

## 3. Several groups per interaction

- [x] 3.1 Replace the one-group check of `_interaction_spec` with the equal-size check for symbolic laws, and verify the
  two groups member by member and groups of different sizes scenarios on the NumPy and Numba backends
- [x] 3.2 Exempt rule-only interactions from the size check, and verify the two agent groups scenario

## 4. Member offset reads

- [x] 4.1 Add `shift_member` to `skcomplex.core.model` with its periodic, reflecting, and fixed ends, re-export it, and
  verify its doctest
- [x] 4.2 Lower `shift_member` through `NeighborRead` tables of a one-dimensional lattice of the group size on both
  backends, and verify the ring of members, fixed end, same result on both backends, and single element scenarios

## 5. Agent views

- [x] 5.1 Scope the `ModelView` of every rule to its owner, with `parameter`, and role-first `agents` and `count`, and
  add `AgentView.group_label`, and verify the nested parameter, group by role, unknown name, and group label scenarios
- [x] 5.2 Add `ModelView.dimensions` and `ModelView.torus`, and `AgentView.displacement_to` and `distance_to` on Mesa's
  grids, continuous space, and the NetworkX graph, and verify the grid dimensions, seam, and network scenarios

## 6. Models

- [x] 6.1 Record the trajectories of `yeast_cell_cycle`, `optimal_velocity`, `intelligent_driver_model`, `trafficking`,
  and `meadow_capacity` with their current code as test fixtures before rebuilding them
- [x] 6.2 Rebuild `BooleanFunction`, `ThresholdFunction`, `random_boolean_network`, and `yeast_cell_cycle` on one
  variadic `inputs` role and per-entry weights, and verify the eight inputs, too many inputs, and yeast scenarios
- [x] 6.3 Rebuild `optimal_velocity` and `intelligent_driver_model` as one group `cars` with one interaction using
  `shift_member` and `mod`, and verify the one group, same dynamics, and jam scenarios, and record the time of one
  right-hand side evaluation for 20 and 1000 cars in the change notes
- [x] 6.4 Bind `premises` as a role of `DailyMovements` and read the grid size from `model.dimensions`, bind `pasture`
  as a role of the meadow interactions, and drop `road_length` from `Driver`, and verify the trafficking, nested
  meadow, and ring road scenarios
- [x] 6.5 Replace every `model.parameters[f'{self.label}.<name>']` in the library with `model.parameter('<name>')`, add
  the layout test check, and verify the no rule reads its own label scenario
- [x] 6.6 Run the model conformance, inventory, and layout tests, and verify every rebuilt model keeps its registered
  name and stays implemented in the inventory

## 7. Documentation and gate

- [x] 7.1 Document variadic roles, several groups, `shift_member`, scoped rule views, and the space geometry in the user
  guide with runnable examples, and verify the documentation test passes
- [x] 7.2 Write the breaking changes, positional-only `size`, the renamed trajectory variables of the ring roads, and
  the new required roles, as the notes of the commit message that the changelog session builds `CHANGELOG.md` from
- [x] 7.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
