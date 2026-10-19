# Design

## Context

Surveys of the three references classified every model they pose:

| Book | Implementable | Pending other changes | Blocked | Out of scope | Duplicates |
|---|---|---|---|---|---|
| Padmanabhan et al. 2021 | 47 | 2 (`add-optimal-control`) | 0 | 1 | 2 |
| Konieczny et al. 2023 | 0 | 0 | 0 | 2 | 1 |
| Raman 2021 | 7 | 3 (`add-stochastic-maps`, `add-adaptive-networks`) | 7 (`add-constraint-based-models`) | 5 | 8 |

Nearly all Padmanabhan models are ODEs or delay-ODEs with drug inputs, which the core expresses through rates, delays,
events, and the drivers `step`, `pulse`, `pulse_train`, and `series`:
- impulsive doses and threshold switching are events;
- Heaviside switches use `where`.

The Raman Boolean networks use the variadic `BooleanFunction` that `extend-interaction-roles` added. The generalized
Lotka-Volterra community reads its interaction matrix through the weighted reductions of `add-group-reductions`.

## Goals / Non-Goals

**Goals:**
- Every model of the three references recorded in the inventory with its sources and status.
- The 54 implementable models built only from public core names, each with a scenario and a gallery example.

**Non-Goals:**
- Parameter estimation, identifiability, and global sensitivity analysis (Padmanabhan 2.3.6 and 2.4, Raman 6). These
  are methods, not models. SALib already covers sensitivity, and a later `add-parameter-estimation` change could wrap
  scipy.optimize and pyPESTO.
- Model predictive or reinforcement-learning control of therapies (Padmanabhan 10). The book names no concrete
  controller. `add-optimal-control` covers the two optimal dosing problems.
- Flux balance and other constraint-based models, which `add-constraint-based-models` specifies.

## Decisions

- **Subfields:**
  - The cancer models go to `biology/oncology`.
  - The pharmacokinetic and toxicity models go to `biology/physiology`.
  - The Droop cell quota and the generalized Lotka-Volterra community go to `biology/ecology`.
  - The nanoparticle corona goes to `chemistry/kinetics`.
  - A `biology/pharmacology` subfield was considered and deferred. Two models do not justify it, and the taxonomy rule
    is that a model lives where the science that poses it lives.
- **Shared components within `biology/oncology`:** tumour populations, effector cells, drug compartments with infusion
  inputs, and log-kill and saturating kill interactions. The ten groups reuse them, in line with the rule that
  components are shared within a subfield package.
- **Gaps in the book's equations.** The survey found six models whose equations the book leaves incomplete. Each is
  implemented by the decision below, stated in its docstring:
  - `vascular_core_periphery`: increments read as rates, checked against the Hahnfeldt-type reference the book cites.
  - `prostate_androgen_receptor`: activated receptor concentrations as parameters.
  - `her2_targeted_therapy`: free antibody and free paclitaxel as drivers.
  - `nanoparticle_corona`: free nanoparticle and protein balances from mass action.
  - `stem_cell_differentiation`: the random mutation factor drawn once per run as a parameter.
  - `erythropoiesis_mercaptopurine`: the garbled equation 3.36 checked against the PDF.
- **Goodwin oscillator:** the book points to BioModels and prints no equations. The model uses the standard Goodwin
  1965 form with three variables, whose oscillation threshold for equal degradation rates is a Hill coefficient of 8.

## Risks / Trade-offs

- **Many similar ODE models.** The groups share components, and the gallery examples show what distinguishes each.
- **Scenario thresholds taken from the book's figures may depend on parameters given only in the cited papers.** Where
  the book omits a parameter, the cited paper's value is used and named in the docstring.

## Decisions as built

The models follow the book, and where its equations or parameters are incomplete or garbled the implementation takes
the decision below, stated in the model's docstring.

- **Growth and kill:**
  - Exponential-linear growth is `dn/dt = min(r0 n, r)`, equivalent to the book's switch at `tau`.
  - Gompertz uses the GGo2 parameters, `r = 0.0719` and `K = 1964`.
  - Doses are rectangular exposure pulses.
  - `drug_toxicity` adds a plasma compartment and implements equation 2.26, which reduces to 2.25 at `eta = 0`.
- **Shared components:** shared components carry the single-compartment models. Each multi-compartment model is one
  element holding all its states.
- **Chemotherapy:**
  - `de_pillis_radunskaya` starts from the paper's initial state (BioModels BIOMD0000000909), because the book's lies
    in the basin of the tumour-free equilibrium.
  - Resistance switching rates default to 0, because the book gives no values.
  - The cell-cycle agents act multiplicatively, after Świerniak.
  - Erythropoiesis uses Mackey's form of equation 3.36.
- **Immunotherapy:**
  - The CD8 maturation term of `delayed_dc_vaccine` saturates in the dendritic cells, like every other dendritic term,
    because the literal term diverges within a day.
  - `anti_pd1_checkpoint` holds free PD-1 at zero once it would go negative.
- **Radiotherapy:**
  - The periodic radiation kill is per capita.
  - The linear-quadratic damaged-cell chain follows the Simeoni structure of the cited paper, because the printed
    equations make an untreated tumour shrink.
- **Gene, viral, and stem-cell therapies:**
  - `oncolytic_virus` implements the dimensionless mass-action form of equations 8.9–8.10. The ratio-dependent form of
    equations 8.11–8.12 is garbled in the book and is not implemented.
  - The arrival term of equation 8.39 takes a plus sign.
  - The stem-cell mutation factor is drawn once per run from the seed.
- **Combinations:** dose schedules are finite sums of pulses. With the book's parameters the combination of
  chemovirotherapy beats each therapy alone up to about day 64, after which the virus alone clears the tumour slightly
  faster, so its scenario uses the default run of 64 days.
- **Parameter names:** names follow ruff's naming rule, for example `du` for the book's `dU`. Each docstring gives the
  book's symbol.

## Open Questions

- **`chemoimmunotherapy` decay rates:** the model swaps the decay rates of book Table 9.4 to the values of de Pillis,
  Gu, and Radunskaya 2006 as recalled, which needs checking against the paper.
- **Parameters the cited papers may settle:**
  - `delayed_dc_vaccine`: regulatory T cells suppress the response, so the tumour grows with or without the vaccine.
  - `tumour_immune_suppression`: the memory pools are assumed constants of 1e4.
  - `vascular_core_periphery`: not yet checked against Csercsik et al. 2017.
  - `prostate_androgen_receptor`: the book's PSA clearance keeps serum PSA near 0.05 ng/mL.
  - `drug_resistance_switching`: the switching rates are missing from the book.
- **Core findings:** these do not block any model.
  - `Simulation` built directly ignores a model's registered default solver.
  - System observables cannot read time.
  - `delayed(x, 2 * s.tau)` needs its minimum delay passed explicitly.

