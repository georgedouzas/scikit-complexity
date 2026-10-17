# Design

## Context

`compile_system` declares one symbol per element variable and lowers every law with `lambdify` over the flat state
vector. A system of N elements therefore has N copies of every expression, so compile time grows with N and each step
evaluates a Python list of N scalar results. A prototype ideal gas of 2000 molecules took 62 seconds to compile. The
same law written once and evaluated with NumPy broadcasting over arrays of a million elements takes 5 ms per step.

## Goals / Non-Goals

**Goals:**

- Groups of 10^5 to 10^6 members that compile in about the time of one member and step with array operations.
- One description for a single element and a group, so every existing element type and interaction works on groups.
- Pair interactions with a cutoff and a neighbour search, so short-range forces cost O(N) per step.

**Non-Goals:**

- Members that appear or disappear during a run. Varying populations come with `add-mesa-agents`.
- Numba kernels and compiled steppers. They come with `add-numba-acceleration`, which plugs into the lowering added
  here.
- Long-range all-pairs forces faster than O(N^2), such as Barnes-Hut or Ewald sums.

## Decisions

### A group is a sized element

`Element.group(label, size, **values)` returns a `Group` that holds the element type, the label, the size, and the
values unmodified. A single element is a group of size one in the compiled problem, so the compiler has one path.

- Alternative considered: a `size` field on every element. Rejected because a field would also be a constructor
  argument of every element type and would mix the description of one member with the population.

### Laws are written per member and evaluated per group

The `laws` namespace of a group member holds scalar symbols, exactly as for a single element. The compiler lowers each
law once to a function of arrays: every member symbol stands for a column of length N, and parameters that are shared
stay scalars. `lambdify` with NumPy broadcasts the expression over the columns.

- The state vector stays flat. A group variable occupies `N * size` contiguous components, laid out member-major, and
  `Layout` gains the group shape so `read` returns `(N,)` or `(N, n_dim)` arrays.
- A per-member parameter is a group parameter given as an array, and lowers to a column like a state.

### Pair interactions

`PairInteraction` subclasses implement `pair_energy(s, r)` or `pair_laws(s)`, where `s.first` and `s.second` are the
two members and `r` their distance. Forces come from `-dV/dr` along the separation, derived symbolically once. At run
time a neighbour search returns the pairs within `cutoff`, the pair kernel is evaluated on the pair arrays, and the
results are summed into the members with `np.add.at`.

- The neighbour search uses `scipy.spatial.cKDTree.query_pairs`, which is O(N log N), handles periodic boxes through
  `boxsize`, and is already a dependency. `add-numba-acceleration` can replace it with cell lists.

### Boundary laws

`Boundary(target, value)` maps a state to a new value after every solver step. The compiler lowers boundaries to one
function `boundary(t, x, p)` and the runtime applies it after every `advance`. Boundaries are excluded from form
inference.

- Reflection is applied per frame step, so a member that hits a wall within a step is mirrored exactly for straight
  motion and approximately when forces act. The ideal gas spec uses straight motion, so its pressure is exact.
- Alternative considered: event detection in `solve_ivp`. Rejected because events stop the integrator per crossing,
  which costs one restart per wall hit and does not scale to many members.

### Group visuals

`Element.visual` keeps naming a registered visual. Visuals receive a group with its arrays and log one archetype per
state, so `PointsVisual` logs one `Points2D` or `Points3D` with N points. A `color_by` option maps a scalar state or a
vector norm to colors on a fixed scale.

## Risks / Trade-offs

- [The symbolic equations of a group are written for one member] → `equations_of` returns them with a member index,
  which documents the law of every member.
- [Pair kernels with `np.add.at` are slower than compiled loops] → The numba change replaces them behind the same
  lowering.
- [Boundary laws are not differentiable] → They are excluded from `solve_symbolically`, which raises the existing
  analysis error when a system has one.

## Migration Plan

Existing systems compile unchanged because single elements become groups of size one, and their dotted names stay the
same. The scratchpad ideal gas becomes `models.physics.ideal_gas` with a group, a reflecting box, and `Particle`.
