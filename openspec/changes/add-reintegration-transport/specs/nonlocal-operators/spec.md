## ADDED Requirements

### Requirement: Fields move along a flow by reintegration
Laws of a `FieldElement` on a lattice SHALL update an expression of the field by `reintegrate(expression, flow, dt,
spread, reach)`, which moves the value of every site to a square of half side `spread` sites centred at the site plus
`dt` times the flow there, and gives every site the sum of the shares of the squares that overlap it, each share being
the overlap area over the area of the square. Displacements longer than `reach` sites minus `spread` SHALL be clipped
to that length.

#### Scenario: Shift by one site
- **WHEN** a field on a periodic 2D lattice of 32 by 32 sites is reintegrated with a uniform flow `(1, 0)`, `dt = 1`,
  and `spread = 0.5`
- **THEN** the result equals the field shifted by one site along the first axis within 1e-12

#### Scenario: Half a site spreads evenly
- **WHEN** a field that is 1 at one site and 0 elsewhere is reintegrated with a uniform flow `(0.5, 0)`, `dt = 1`, and
  `spread = 0.5`
- **THEN** the site and its neighbour along the first axis each hold 0.5 within 1e-12

### Requirement: Reintegration conserves the total
The sum of a reintegrated field over the sites SHALL equal the sum before it on periodic lattices and on lattices with
fixed edges, where the centres of the squares are clipped so that the squares stay inside the lattice.

#### Scenario: Random flow on a torus
- **WHEN** a random nonnegative field on a periodic lattice of 64 by 64 sites is reintegrated 100 times with random
  flows of magnitude up to 3, `spread = 0.65`, and `reach = 5`
- **THEN** its sum stays equal to the first sum within a relative error of 1e-12

#### Scenario: Flow towards a fixed edge
- **WHEN** a field is reintegrated with a uniform flow towards a fixed edge until all its mass reaches the edge
- **THEN** its sum stays equal to the first sum within a relative error of 1e-12

### Requirement: Carried fields move with the mass
The ratio of the reintegration of a product `A P` to the reintegration of `A`, guarded where `A` vanishes, SHALL give
the mass-weighted mean of a carried field `P` of equation 9 of Plantec et al. (2023, arXiv 2212.07906), and it SHALL
lower on every backend that lowers `reintegrate`.

#### Scenario: Two parameters meet
- **WHEN** two blocks of equal mass, one carrying `P = 0` and one carrying `P = 1`, are moved into the same cells
- **THEN** the carried field in those cells is 0.5 within 1e-12

### Requirement: Reintegration is differentiable on JAX
On the JAX backend the gradient of a function of a reintegrated field with respect to a parameter of the flow SHALL
match its finite-difference estimate.

#### Scenario: Gradient of the travelled distance
- **WHEN** the centre of mass of a blob after 20 reintegrations with the uniform flow `(v, 0)` is differentiated with
  respect to `v`
- **THEN** the gradient on JAX matches the central finite difference within a relative error of 1e-5
