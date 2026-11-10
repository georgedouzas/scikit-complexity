## ADDED Requirements

### Requirement: Flow-Lenia
The library SHALL provide `FlowLeniaSystem` in `computer_science/automata` (Plantec et al. 2023, arXiv 2212.07906):
`C` channels of nonnegative mass, kernels from a source channel to a target channel with their growth maps, the
affinity `U` of every channel, the flow `F_i = (1 - alpha) grad U_i - alpha grad A_sum` with
`alpha = clip((A_sum / theta_A)^n, 0, 1)` and Sobel gradients, and the update of every channel by `reintegrate`.

#### Scenario: Mass conserved
- **WHEN** the system runs 500 steps from a 40 by 40 patch of uniform random mass with parameters drawn from Table 1
  of the paper
- **THEN** the mass of every channel equals its initial mass within a relative error of 1e-10

#### Scenario: Patterns stay localised
- **WHEN** the system runs 1000 steps from such a patch on a world of 128 by 128 sites for 10 seeds
- **THEN** in at least 8 runs the sites above 1e-3 cover less than half of the world

#### Scenario: Defaults of the paper
- **WHEN** a user builds the system with its defaults
- **THEN** `spread` is 0.65, `n` is 2, and `dt` is 0.2

### Requirement: Kernel weights travel with the matter
`FlowLeniaSystem(localized=True)` SHALL hold the kernel weights `h` as fields carried with the mass by the mass-weighted
mean of equation 9 of the paper, and SHALL add Gaussian noise to them in a random square zone at a given rate.

#### Scenario: Two species
- **WHEN** two patches with different weights run side by side without mutation until they touch
- **THEN** the carried weights in every cell lie between the two initial values, and the total mass is conserved

### Requirement: Flow-Lenia is recorded in the inventory
The inventory SHALL record `flow_lenia` as implemented with the source 2212.07906 and the primitive `mass-transport`,
and the gallery SHALL hold its example.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for `flow_lenia`
