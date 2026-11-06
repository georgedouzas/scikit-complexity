# Spec Delta

## ADDED Requirements

### Requirement: Link states take values in a matrix group
A state SHALL be declarable with a compact matrix group, `SU(n)` or `U(n)`, which stores the `2 n^2` real components of
the matrix of every member, the real and imaginary part of every entry in row order. The group SHALL convert matrices
to and from their components, draw matrices from the Haar measure of the group and near its identity, and project
matrices onto the group, the unitary factor of their polar decomposition, divided by the principal root of its
determinant for `SU(n)`. A matrix group state SHALL be neither a vector nor a tensor, of order 1 without lags.

#### Scenario: SU(3) links store their components
- **WHEN** a group of links of the complex of a periodic 4 × 4 lattice declares an SU(3) state with the identity as
  default
- **THEN** the state holds 18 components for each of the 32 links, which are the identity matrix of every link

#### Scenario: Draws stay in the group
- **WHEN** matrices of SU(2), SU(3), U(1), or U(2) are drawn from the Haar measure or near the identity
- **THEN** they are unitary, and of unit determinant for the special unitary groups, within 1e-12

#### Scenario: Projection undoes rounding
- **WHEN** matrices of a group are perturbed by 1e-6 and projected
- **THEN** the projected matrices lie in the group and within 1e-5 of the original matrices

### Requirement: Holonomies are path-ordered products of link matrices
The matrix group SHALL give the holonomy of paths of oriented edges, the product of the matrix of every edge along its
orientation and of its inverse against it, from the first edge of a path to the last, with padding that contributes the
identity. A `CellComplexSpace` SHALL list the edges around every face in the cyclic order of its nodes with the sign of
their traversal, whose signed sum is the row of the face in the edge-to-face incidence.

#### Scenario: Ordered products do not commute
- **WHEN** the holonomy of a path of three SU(2) links is computed, once forward and once along the reversed path
- **THEN** the first is the ordered product of the links with the inverse of the link traversed backward, and the second
  is its inverse

#### Scenario: Gauge-invariant trace
- **WHEN** a random gauge transformation `g_x U g_y^-1` is applied to the SU(3) links of a closed loop
- **THEN** the trace of the holonomy of the loop is unchanged within 1e-12

#### Scenario: Face boundaries agree with the incidence
- **WHEN** the boundaries of the faces of a periodic 3 × 4 × 2 lattice, a fixed 4 × 4 lattice, a clique complex, or a
  cycle basis of a network are read
- **THEN** every boundary is a closed walk, and its signed edges sum to the row of the face in the incidence

### Requirement: Monte Carlo samples matrix link states
`SamplingInteraction` SHALL sample a matrix group state on the edges of a cell complex with a holonomy score, a class
function of the holonomy of every face, whose local objective of an edge sums the score over the holonomies of the faces
around it started at the edge. A move SHALL multiply a link by a random matrix of the group near the identity, draw it
from the Haar measure, or be a function of the link matrices, accepted by the Metropolis or the Glauber rule, with a
colored or random-sequential schedule. Every sweep SHALL project the links onto the group. A drive SHALL change a state
of several components per member as a whole, with the components on a last axis.

#### Scenario: Two-dimensional SU(2)
- **WHEN** the Wilson action `-(beta / 2) Re tr U_p` with `beta = 2` is sampled on a periodic 16 × 16 lattice for 300
  sweeps after 100 sweeps of transient
- **THEN** the mean of `Re tr U_p / 2` is within 0.01 of the exact value `I_2(2) / I_1(2)`

#### Scenario: Two-dimensional SU(3)
- **WHEN** the Wilson action `-(beta / 3) Re tr U_p` with `beta = 4` is sampled on a periodic 12 × 12 lattice
- **THEN** the mean of `Re tr U_p / 3` is within 0.01 of the one-plaquette integral over the Haar measure of SU(3)

#### Scenario: Links stay in the group
- **WHEN** SU(3) links are sampled for 20 sweeps with the colored or the random-sequential schedule
- **THEN** they are unitary of unit determinant within 1e-12

#### Scenario: Moves of scalars are refused
- **WHEN** a holonomy score is given the flip move, the heat-bath rule, or the checkerboard schedule
- **THEN** checking the interaction raises a `ValidationError` that names the moves, rules, or schedules it takes
