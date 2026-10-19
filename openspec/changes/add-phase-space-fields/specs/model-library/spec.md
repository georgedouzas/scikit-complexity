# Spec Delta

## ADDED Requirements

### Requirement: Kinetic equations of gases and plasmas
The library SHALL provide the spatially inhomogeneous Boltzmann equation, its BGK model, the Vlasov-Poisson equation
with Landau damping and two-stream set-ups, and the local Enskog equation as registered factories under
`physics/fluids` (Bellomo et al. 2017 §2.2–2.3). The inventory SHALL record them as implemented.

#### Scenario: Sod shock tube near the fluid limit
- **WHEN** the BGK model runs the Sod shock tube in one position and one velocity dimension at a Knudsen number of 1e-4
- **THEN** the density moment at time 0.2 matches the exact Euler solution within 2% in the L1 norm

#### Scenario: Landau damping model
- **WHEN** the Vlasov-Poisson factory runs with its default Landau damping set-up
- **THEN** the measured damping rate of the electric energy lies within 5% of 0.1533, and the model passes the
  conformance suite

### Requirement: Kinetic theory of flocking
The library SHALL provide the Bertin-Droz-Grégoire kinetic theory of flocking over position and orientation as a
registered factory under `physics/active_matter` (Marchetti et al. 2013, arXiv 1207.2929 §V.2.3). The inventory SHALL
record it as implemented.

#### Scenario: Flocking onset
- **WHEN** the model runs from a homogeneous disordered state with a small polar perturbation, at densities 20% below
  and 20% above the transition density that Bertin, Droz, and Grégoire give in closed form for the chosen noise
- **THEN** the polarization moment decays below the transition and grows above it
