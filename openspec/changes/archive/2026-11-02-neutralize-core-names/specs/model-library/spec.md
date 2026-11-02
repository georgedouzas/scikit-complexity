## MODIFIED Requirements

### Requirement: Models are organized by scientific field
The library SHALL organize models into packages per scientific field, `physics`, `chemistry`, `biology`,
`economics`, `social`, `engineering`, `mathematics`, and `computer_science`, each split into subfield packages. Every
model SHALL live in exactly one subfield.

#### Scenario: Import by field
- **WHEN** a user imports `skcomplex.models.physics.fluids`, `skcomplex.models.biology.ecology`, or
  `skcomplex.models.chemistry.pattern_formation`
- **THEN** each package exports the factories and component types of its subfield, such as `burgers`,
  `lotka_volterra`, and `gray_scott`

#### Scenario: Technique packages are gone
- **WHEN** a user imports `skcomplex.models.fluids`, `skcomplex.models.lattices`, `skcomplex.models.fields`, or
  `skcomplex.models.nonlinear`
- **THEN** the import fails, and every model of those packages is found in its scientific subfield

#### Scenario: One home per model
- **WHEN** the catalog is listed in full
- **THEN** every model name appears once, with a field path of the form `<field>/<subfield>`

### Requirement: Stock-flow models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries, which probe flows, sources and sinks, auxiliaries, lookups, delays, smoothing, loops, and
  discrete flows in distinct ways, are registered under their field paths and marked implemented:
  - `stock_flow_consistent_economy` in `economics/macroeconomics`
  - `commodity_cycle` in `economics/markets`
  - `beer_game` in `economics/markets`
  - `bass_diffusion` in `economics/innovation`
  - `success_to_the_successful` in `social/organizations`
  - `shifting_the_burden` in `social/organizations`
  - `escalation` in `social/organizations`
  - `rice_milling` in `engineering/processes`
  - `solid_waste` in `engineering/processes`
  - `food_relief` in `engineering/processes`
  - `irrigation` in `engineering/processes`

#### Scenario: Repeated patterns registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries, which repeat patterns of stocks and flows that the models above probe, are registered under
  their field paths and marked implemented: `energy_return` in `physics/climate`, and `cocoa`, `hilsa_fishery`, and
  `food_security` in `economics/resources`

#### Scenario: Meadows' resource models rebuilt
- **WHEN** the change is applied
- **THEN** Meadows' nonrenewable and renewable resource models stay registered as `oil_economy` and
  `fishery_capital`, rebuilt on flows, and the duplicate survey entries `renewable_resource` and
  `nonrenewable_resource` are removed from the inventory

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Specific realizations live in the model library
The library SHALL hold every specific realization of a core primitive or a general analysis in its own private module
of the model package that uses it, next to the model definitions, exported from that package. A realization that
several packages of one field share SHALL live in `skcomplex.models.<field>.common`, grouped in family packages such as
`particles`, `oscillators`, `excitable`, or `forecasting`, with no loose modules beside the families, and only that
field's packages import it. A realization SHALL be a subclass of a general core class, even a pure renaming, and no field
SHALL define the same realization twice. A field whose packages share no code SHALL have no `common`
package. A model class whose mechanism the core provides SHALL use the core class and keep only what is its own.

#### Scenario: Order book from the library
- **WHEN** the finance models build their order book, and an engineering model pairs riders with cars
- **THEN** both use the core `MatchingInteraction`, the order book binds it to bids, asks, money, and goods with
  `ContinuousMatchingSpec`, and the earlier of two same-limit sell orders trades first

#### Scenario: Potential from the library
- **WHEN** a user imports the Lennard-Jones potential from `skcomplex.models.physics.common.particles`
- **THEN** it is a realization built on the core `PairCouplingInteraction`, and the existing LAMMPS agreement tests pass
  with it

#### Scenario: Realization of one package
- **WHEN** only the climate models use `EarlyWarningAnalysis`
- **THEN** it lives in `skcomplex.models.physics.climate`

#### Scenario: No copy across fields
- **WHEN** the ecology meadow and the economics commons models are built
- **THEN** both build foraging and regrowth from the core `TransferInteraction`, `LifeCycleInteraction`, and
  `MapInteraction`

### Requirement: Nonlocal field models
The library SHALL provide, as registered factories built on the general interactions and the nonlocal operators:
- `amari_neural_field` under `biology/neuroscience` (2112.12147 §1.3.1);
- `double_gyre` under `physics/oceanography` (1910.00583 §III.6.1);
- `gkt_traffic` under `engineering/traffic` (cond-mat/0012229 §III.E.4);
- `active_viscoelastic_gel` under `physics/active_matter` (2306.08868 VI).

#### Scenario: Amari bump
- **WHEN** the Amari neural field `tau u_t = -u + convolve(H(u), w) + h` runs on a line with a lateral-inhibition
  kernel `w`, a Heaviside firing rate, and a resting level `h < 0`, from a localized excitation
- **THEN** a stationary bump forms, and its width is within 5% of the root `a` of `W(a) + h = 0` with `w(a) < 0`,
  where `W(a)` is the integral of `w` from 0 to `a`

#### Scenario: Double gyre circulation
- **WHEN** the quasi-geostrophic double gyre runs in a closed square basin from rest under the symmetric double-gyre
  wind stress `-tau0 cos(2 pi y / L)` in its viscous, steady regime
- **THEN** the streamfunction forms two gyres of opposite sign, one on each side of the mid-basin latitude, with each
  extremum in the western fifth of the basin, and along the central latitude of each gyre the interior streamfunction
  is within 10% of the Sverdrup balance `beta psi_x = curl(tau)`

#### Scenario: Stop-and-go waves
- **WHEN** the gas-kinetic-based traffic model runs on a ring road with a small local density perturbation, once at a
  density of 15 vehicles per kilometre and once at a density in its unstable range
- **THEN** the perturbation decays at the low density, and at the unstable density it grows into congested clusters
  whose downstream fronts move upstream at between 10 and 25 kilometres per hour

#### Scenario: Contractile gel
- **WHEN** the active viscoelastic gel runs from a small perturbation of uniform cell density above its contractile
  threshold
- **THEN** the density concentrates into peaks, and the velocity at every step solves the force balance within 1e-10

### Requirement: Kinetic equations of gases and plasmas
The library SHALL provide the spatially inhomogeneous Boltzmann equation, its BGK model, the Vlasov-Poisson equation
with Landau damping and two-stream set-ups, and the local Enskog equation as registered factories under
`physics/fluids` (Bellomo et al. 2017 §2.2–2.3), built on `PhaseSpace`, `AdvectionInteraction`,
`EncounterInteraction`, and `RelaxationInteraction`. The inventory SHALL record them as implemented.

#### Scenario: Sod shock tube near the fluid limit
- **WHEN** the BGK model runs the Sod shock tube in one position and one velocity dimension at a Knudsen number of 1e-4
- **THEN** the density moment at time 0.2 matches the exact Euler solution within 2% in the L1 norm

#### Scenario: Landau damping model
- **WHEN** the Vlasov-Poisson factory runs with its default Landau damping set-up
- **THEN** the measured damping rate of the electric energy lies within 5% of 0.1533, and the model passes the
  conformance suite
