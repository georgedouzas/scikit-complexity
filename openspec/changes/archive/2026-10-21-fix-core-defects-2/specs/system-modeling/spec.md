# Spec Delta

## ADDED Requirements

### Requirement: Reductions appear anywhere in interaction laws
An interaction law SHALL accept a reduction in any position of its value, including a denominator or the argument of
a function undefined at zero, and SHALL compile to the same values as the equivalent element law.

#### Scenario: Share of a group total
- **WHEN** an interaction bound to a group of 4 members with `x = 1, 2, 3, 4` declares
  `Rate(s.cells.x, s.cells.x / count(over=s.cells))`
- **THEN** the system compiles, and the rate of member `i` is `x_i / 4`

#### Scenario: Weighted reduction in a denominator
- **WHEN** an interaction bound to a group declares `Rate(s.cells.x, s.cells.x / total(s.cells.x, weights=W))` with a
  matrix `W` of positive entries
- **THEN** the system compiles on the NumPy and the Numba backends, and both drifts equal `x_i / sum_j W_ij x_j`

### Requirement: Observable member expressions read reductions
The member expression of a group observable SHALL accept reductions over its group, with or without weights, with the
meaning they have in laws. A reduction over another group SHALL raise the named error that laws raise.

#### Scenario: Local field energy
- **WHEN** a group of 4 members declares the observable `energy` as the sum over members of `x_i * total(x, weights=W)`
- **THEN** the trajectory returns `x · W x` at every frame

#### Scenario: Plain reduction
- **WHEN** a group observable's member expression is `m.x - mean(m.x)` with reduction `'var'`
- **THEN** the trajectory returns the variance of `x` at every frame, with no error about a missing NumPy function

### Requirement: Plain roles bind group members
A plain role SHALL bind a member reference `group[i]`, and the interaction's laws SHALL read and change the states of
that member as those of a single element. A reference past the group's size SHALL raise a named role error that names
the interaction, the role, the reference, and the group size.

#### Scenario: Hub reads one site
- **WHEN** an interaction binds its plain role `site` to `sites[3]` of a group with `x = 0, 1, 2, 3, 4` and declares
  `Rate(s.hub.y, s.site.x)`
- **THEN** the rate of `hub.y` is 3, the same as with a variadic role bound to `['sites[3]']`

#### Scenario: Member past the end
- **WHEN** a plain role binds `sites[7]` of a group of 5 members
- **THEN** compiling raises a named role error that names the interaction, the role, `sites[7]`, and the size 5

### Requirement: Jumps change members at an offset
A jump of a group law SHALL target a member at an offset, written with the offset read and its periodic, reflecting,
or fixed ends, and SHALL compile to one channel per member. Past a fixed end the jump SHALL change only its other
targets, so the quantity leaves the group.

#### Scenario: Conserved hop on a ring
- **WHEN** a group of 50 sites on a ring declares `Jump(s.n, {s.n: -1, shift_member(s.n, 1): 1})` with 150 particles
- **THEN** every run keeps 150 particles, and each event moves one particle from a site to the next one

#### Scenario: Open end
- **WHEN** a group of 5 sites with fixed ends declares the same hop and holds one particle at the last site
- **THEN** the hop of the last site removes the particle and changes no other site

#### Scenario: Same law as per-link interactions
- **WHEN** a zero-range process of 20 sites is written once with an offset jump and once with one interaction per
  link
- **THEN** the mean occupation profiles of 200 runs agree within their statistical error, on NumPy and on Numba

#### Scenario: Many sites
- **WHEN** a zero-range process of 1000 sites with an offset jump is compiled and run until 5
- **THEN** it compiles one jump kernel, and compiling and running take less than 10 seconds on the Numba backend

### Requirement: Compile time grows linearly with the components
Compiling a system SHALL take time that grows linearly with its number of elements, interactions, and variables.

#### Scenario: Many one-link interactions
- **WHEN** zero-range processes written with one interaction per link are compiled with 400 and with 800 links on the
  NumPy backend
- **THEN** the larger one takes at most 2.5 times as long as the smaller one
