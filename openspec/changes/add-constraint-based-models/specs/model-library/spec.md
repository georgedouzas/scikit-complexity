# Spec Delta

## ADDED Requirements

### Requirement: Flux balance model
The library SHALL provide `flux_balance`, which runs FBA, FVA, and parsimonious FBA on Raman's toy network of Table 8.2
by default without the extra, and on the E. coli core model with the extra. It SHALL accept a ramp of an uptake bound
over the frames.

#### Scenario: Toy network growth
- **WHEN** `flux_balance` runs with its defaults
- **THEN** with all bounds at 1000 and only `A` taken up, the growth flux is 50 within 1e-9, as Raman Eq. 8.9 gives, and
  FVA at that optimum ranges `v1`, `v2`, and `v3` over [0, 600] and fixes `v4 = 550`, `v5 = 50`, `v6 = 1000`,
  `v7 = 100`, `b1 = 600`, and `b2 = 1000`, as Raman Eq. 8.12 gives

#### Scenario: E. coli core aerobic and anaerobic growth
- **WHEN** `flux_balance` runs on the E. coli core model while the oxygen uptake bound ramps from 0 to 20
- **THEN** growth is 0.2117 h⁻¹ within 1e-4 at no oxygen, rises with oxygen, and reaches 0.8739 h⁻¹ within 1e-4 once
  oxygen no longer limits

### Requirement: Metabolic perturbations model
The library SHALL provide `metabolic_perturbations`, which applies gene and reaction knock-outs, MoMA, ROOM, OptKnock,
synthetic lethal scans, and FSEOF to the E. coli core model.

#### Scenario: Essential gene
- **WHEN** the `gapA` gene (b1779) is knocked out
- **THEN** growth is zero within 1e-6

#### Scenario: Single-deletion scan
- **WHEN** every gene of the E. coli core model is deleted in turn
- **THEN** exactly 7 deletions give growth below 1e-6

### Requirement: Regulatory FBA model
The library SHALL provide `regulatory_fba`, where Boolean gene rules alternate with FBA (Raman Eq. 9.7) in a batch
culture on glucose and acetate.

#### Scenario: Glucose represses acetate uptake
- **WHEN** `regulatory_fba` runs until both substrates are exhausted
- **THEN** the acetate uptake flux is zero while glucose is above the repression threshold. After depletion it becomes
  positive, and biomass grows in two phases with a lag between them.

### Requirement: Dynamic FBA model
The library SHALL provide `dynamic_fba`, a batch culture of the E. coli core model with Michaelis-Menten glucose uptake
and the static optimisation approach (Raman Eqs. 9.8–9.9). The direct approach SHALL be available as an option.

#### Scenario: Batch culture
- **WHEN** `dynamic_fba` runs from biomass 0.01 g/L and glucose 10 mM until 10 h
- **THEN** glucose decreases monotonically to below 1e-3 mM, and biomass grows exponentially at 0.874 h⁻¹ within 1%
  while glucose exceeds ten half-saturation constants. Biomass then stays at 0.01 + 0.0874 · 10 g/L within 2%, with
  one growth phase and no diauxic shift.

### Requirement: Dynamic multispecies FBA model
The library SHALL provide `dynamic_multispecies_fba`, the dynamic multi-species metabolic model of microbes competing
for a shared medium, each species with its own network, uptake kinetics, and biomass.

#### Scenario: Competition for glucose
- **WHEN** two E. coli core strains that differ only in maximal glucose uptake grow on one medium
- **THEN** the share of the faster strain rises monotonically. The glucose consumed equals the time integral of each
  strain's uptake flux times its biomass, summed over the strains, within 1%.

### Requirement: COMETS model
The library SHALL provide `comets`, dynamic FBA on a 2D lattice in which metabolites and biomass diffuse, and only sites
with biomass above a threshold are solved.

#### Scenario: Expanding colony
- **WHEN** `comets` runs from a central inoculum on uniform glucose
- **THEN** the colony radius grows with a nearly constant speed after a transient. The glucose consumed over the lattice matches the
  integral of uptake flux times biomass over sites and time within 1%, and sites without biomass have zero fluxes.

### Requirement: Community FBA model
The library SHALL provide `community_fba`, which models a microbial community by joint FBA, community FBA with
abundances, SteadyCom, and, with the extra, the cooperative trade-off (Raman Eq. 11.3).

#### Scenario: Cross-feeding mutants
- **WHEN** two E. coli core members with complementary knock-outs that each make the other's missing metabolite are
  joined
- **THEN** each member alone has zero growth, and the community has positive growth with both members active

### Requirement: Elementary flux modes model
The library SHALL provide `elementary_flux_modes`, which enumerates the elementary flux modes and extreme pathways of a
metabolic network (Raman 2021 §9.8), Raman's toy network of Table 8.2 by default, ordered by growth yield on the
substrate, with one frame per mode. It SHALL need the extra `polyhedra`.

#### Scenario: Toy network modes and the FBA optimum
- **WHEN** `elementary_flux_modes` runs with its defaults
- **THEN** it records 2 modes, each with growth yield `1/12` on `A` and `v6 = 20 · vbio`, and the FBA optimum of growth
  50 is their non-negative combination within 1e-9, so its maximum is set by the bound 1000 on `v6`

#### Scenario: Yield of a knock-out
- **WHEN** `elementary_flux_modes` runs with `v1` knocked out
- **THEN** it records 1 mode, through `v2` and `v3`, with the same growth yield `1/12`

### Requirement: Carbon-13 metabolic flux analysis model
The library SHALL provide `carbon_13_mfa`, which estimates the fluxes of a network with atom transitions from
labelling data of parallel tracer experiments (Raman 2021 §9.7), with confidence intervals and a goodness-of-fit test.
By default it SHALL use the TCA example network of Antoniewicz et al. (2007) with data simulated from known fluxes plus
seeded noise, and it SHALL accept user networks and measurements. It SHALL run without any extra.

#### Scenario: Recovered fluxes
- **WHEN** `carbon_13_mfa` runs with its defaults, noise of standard deviation 0.002, and seed 0
- **THEN** the estimated split of α-ketoglutarate, `v3` and `v4`, is within 2 of the true 50 and 50, every net flux's
  95% interval contains its true value, and the chi-square test accepts the fit at the 5% level

#### Scenario: Published labelling pattern
- **WHEN** `carbon_13_mfa` simulates glutamate with the true fluxes and the tracer of Antoniewicz et al. (2007)
- **THEN** the distribution is `[0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039]` within 5e-5

### Requirement: Constraint-based models have gallery examples
Each model of this change SHALL have a gallery example at `docs/examples/biology/<subfield>/plot_<name>.py` that runs
within the gallery budget. It SHALL show its Rerun primary view as the thumbnail and at least one analysis plot from
its own run, with a caption stating the behaviour shown.

#### Scenario: Dynamic FBA example
- **WHEN** a user opens the `dynamic_fba` example page
- **THEN** its thumbnail is the Rerun series of biomass and glucose. Below it, a plot of the logarithm of biomass
  against time shows a fitted slope of 0.874 h⁻¹ and the plateau after glucose depletion.

#### Scenario: Carbon-13 flux analysis example
- **WHEN** a user opens the `carbon_13_mfa` example page
- **THEN** its thumbnail is the Rerun pool-process graph with the estimated fluxes at the last iteration. Below it, a
  plot shows the estimated net fluxes with their 95% intervals against the true fluxes, and a plot shows the measured
  and simulated glutamate distributions.

#### Scenario: Coverage of the new models
- **WHEN** the gallery coverage check runs after this change
- **THEN** each of the nine models has exactly one example, and the check names any that does not

### Requirement: Models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `flux_balance` in `biology/systems_biology`
  - `metabolic_perturbations` in `biology/systems_biology`
  - `regulatory_fba` in `biology/systems_biology`
  - `dynamic_fba` in `biology/systems_biology`
  - `dynamic_multispecies_fba` in `biology/ecology`
  - `comets` in `biology/ecology`
  - `community_fba` in `biology/ecology`
  - `elementary_flux_modes` in `biology/systems_biology`
  - `carbon_13_mfa` in `biology/systems_biology`

#### Scenario: Models that need the extra
- **WHEN** a user without the extra `metabolism` creates a model that needs it
- **THEN** a named error names the model and the extra, and `flux_balance` with its default toy network still runs

#### Scenario: Models outside the inventory's earlier scope
- **WHEN** the inventory check runs after this change
- **THEN** `elementary_flux_modes` and `carbon_13_mfa`, which the inventory recorded as out of scope, are recorded as
  implemented with the sources `Rm 9.8` and `Rm 9.7`, and the `polyhedra` extra is named in the docstring of
  `elementary_flux_modes`
