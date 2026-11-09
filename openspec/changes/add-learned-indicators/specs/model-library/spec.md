## ADDED Requirements

### Requirement: Discrete normal forms of bifurcations
The library SHALL provide `DiscreteNormalFormSystem` in `mathematics/dynamical_systems`, the noisy discrete-time normal
form of a period-doubling, Neimark-Sacker, fold, transcritical, or pitchfork bifurcation chosen by `kind`, with random
higher-order terms drawn from a seed, and a bifurcation parameter that is fixed or ramped linearly to the bifurcation
over the run, as in 2303.09669.

#### Scenario: Period-doubling form
- **WHEN** the period-doubling form runs with a small noise and its parameter ramped from -1.5 to 0
- **THEN** the lag-1 autocorrelation of the detrended series over the last fifth of the run is negative

#### Scenario: Unknown kind
- **WHEN** the system is built with a kind that is not one of the five
- **THEN** preparing it raises `ValidationError` naming the kinds

### Requirement: Fox cardiac alternans map
The library SHALL provide `FoxCardiacAlternansSystem` in `biology/physiology`, the Fox map of the action potential
duration and a memory variable from beat to beat, with additive noise and a stimulation period that is fixed or ramped
over the run, with the defaults of 2303.09669 (A = 88, B = 122, C = 40, D = 28, τ = 180, α = 0.2).

#### Scenario: Alternans below the bifurcation
- **WHEN** the map runs without noise at the stimulation periods 300 and 150
- **THEN** the duration settles to one value at 300 and alternates between two values that differ by more than 1 at 150

### Requirement: Westerhoff business-cycle model
The library SHALL provide `WesterhoffBusinessCycleSystem` in `economics/macroeconomics`, the second-order map of
national income driven by consumer sentiment, with additive noise and an autonomous expenditure that is fixed or ramped,
with the defaults of 2303.09669 (b = 0.45, c = 0.1, d = 0), whose Neimark-Sacker bifurcation lies at a = 20.

#### Scenario: Fixed point and cycles
- **WHEN** the model runs without noise at the autonomous expenditures 10 and 22
- **THEN** income settles at the fixed point `2 a` at 10, and keeps oscillating at 22 with an amplitude above 1

### Requirement: Learned-indicator systems are recorded in the inventory
The inventory SHALL record `discrete_normal_form`, `fox_cardiac_alternans`, and `westerhoff_business_cycle` as
implemented with their sources and primitives, among them `learned-indicators`, and the gallery SHALL hold an example of
each.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for the three systems
