# Design

## Context

- `add-learning-foundation` is a prerequisite. It adds `learned_function` and `learned_weights` in
  `components/laws/utils/_learned.py`, whole-number sizes of `ParameterSpec.shape`, `ObservedRunsSpec` with one column
  per member for a group and NaN for missing samples, the losses `'squares'`, `'log'`, and `'likelihood'`, the shared
  `_OptaxLoop` of `GradientDescentSolver`, and `identify(system, data, *, trained, ...)` with its
  `IdentificationResult`. The Diffrax solver, tolerances, and adjoint are module constants that the solver, `identify`,
  and `calibrate(method='variational')` read.
- `DosingInteraction` (`systems/biology/common/pharmacokinetics/_dosing.py`) doses a state by infusions, a rate law
  with `pulse_train`, so it already lowers to JAX. The NumPy path respects the breakpoints of pulses, but the Diffrax
  calls of the JAX path do not pass them to the step size controller, so a short infusion can be stepped over.
  `ImpulseInteraction` resets states through `CrossingLaw`, which has no JAX form.
- Pharmacokinetics lives in `systems/biology/physiology` (`_pharmacokinetics.py`), with shared components in
  `systems/biology/common/pharmacokinetics`. There is no `biology/pharmacology` subfield and none is created.

## Goals / Non-Goals

**Goals:**
- Shared and per-member parameters in one `identify` call, with the prior of the effects as a penalty.
- Differentiable dosing on JAX.
- Probes of hybrid laws, covariate maps, latent neural ODEs, and learned functions on lattices.

**Non-Goals:**
- A second law function for networks or a second fitting function. Both come from `add-learning-foundation`.
- Amortised encoders of variational autoencoders. The latent form of 2602.03215 is fitted here by maximum a posteriori
  effects per patient, which is the empirical Bayes estimate that NLME practice uses.
- Estimating the variances of the random effects. They are given as priors. Their estimation belongs with the Bayesian
  methods of `calibrate`.
- The PBPK surrogate of 2310.09167. The reference form of `oral_iv_pk_hybrid` is a reduced compartmental model, which
  the paper names as an option.
- Graph networks of chemical structure. Descriptors are given as numbers per compound.

## Decisions

### Placement

| Part | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| Breakpoints of pulse inputs on JAX | 2, solver support | `core/engine/solvers/_gradient_descent.py` and the shared Diffrax constants | pharmacology (doses), ecology (stocking), engineering (pulsed inputs), epidemiology (vaccination campaigns) |
| `individual` and `effects` of `identify` | 2, an option of an experiment over any system | `core/experiments/_identification.py` | pharmacology (patients), ecology (sites), epidemiology (regions), psychology and economics (subjects and firms) |
| `TacrolimusPopulationPKSystem` | 3, one system | `biology/physiology/_tacrolimus.py` | biology |
| `OralIVPKHybridSystem` | 3, one system | `biology/physiology/_oral_iv_pharmacokinetics.py` | biology |
| `GrowingNeuralCellularAutomatonSystem` | 3, one system | `computer_science/automata/_neural_cellular_automaton.py` | computer science |

### Doses on JAX

- Doses are infusions of the existing `DosingInteraction`, rate laws that JAX already lowers. Bolus resets by
  `ImpulseInteraction` stay on the NumPy path, since a reset has no JAX form and a short infusion approximates a bolus
  as closely as needed.
- The breakpoints of `pulse` and `pulse_train` inputs, which the compiled problem already collects for the SciPy
  integrators, are passed as `jump_ts` to `diffrax.PIDController` through the shared Diffrax constants, so
  `GradientDescentSolver`, `identify`, and `calibrate` never step across the start or the end of an infusion.

### Mixed effects in identify

- `identify(..., individual=None)` takes a mapping from per-member parameters, declared
  `ParameterSpec(per=True, adaptive=True)`, to the standard deviation of their Gaussian prior. The loss of the shared
  loop adds `sum(eta^2) / (2 sigma^2)` divided by the number of samples, so its minimum is the maximum a posteriori
  estimate. A group compiles once for any size, so all members solve in one call.
- `IdentificationResult` gains `effects`, a mapping from each individual parameter to one value per member, and
  `estimates` keeps only the shared values.
- With `individual` set, `validation` holds out members rather than runs. It fits only the effects of the held-out
  members with the shared values fixed, and reports their error.
- Prediction for new patients is `identify` with `trained=()` and `individual` set, which keeps the shared values of
  the system and needs no method on the result.
- A covariate map is a `learned_function` of per-member covariates inside a law, so it needs no option. Its weights
  are shared parameters named in `trained`.

### Systems

- **Placement.** Both pharmacokinetic systems go in `systems/biology/physiology`, where pharmacokinetics lives, each in
  its own module. Both dose through `DosingInteraction` of `biology/common/pharmacokinetics`. Components that both
  share, such as a first-order transfer chain of compartments, go to `biology/common/pharmacokinetics` beside it, and
  components that only one uses stay in its module.
- **`tacrolimus_population_pk`.** One `BaseSystem` with `form='linear'`, `'saturable'`, or `'latent'`, `n_patients`,
  `hematocrit`, and `random_state`. The group `patients` holds the states `A1` to `A4`, `Ac`, and `Ap`, and the
  per-member covariates `CYP` and `ST` and effects `eta_Ktr`, `eta_CL`, `eta_Vc`, `eta_Q`, and `eta_Vp`, drawn at
  construction from the values of the supplementary tables of 2602.03215. Doses are a `DosingInteraction` into `A1`
  every 12 hours, an infusion of 0.1 hours with a per-member dose. The observable is `Cc = 1000 Ac / Vc`. The latent
  form holds a depot that receives the doses, latent states whose rate is a `learned_function` of the latent states,
  the depot, the covariates, and per-member effects, and a linear readout as `Cc`. Primitives: `rates`, `time-inputs`,
  `groups`, `draws`, `observables`, and `learned-functions` for the latent form.
- **`oral_iv_pk_hybrid`.** One `BaseSystem` with `form='reference'` or `'learned'`, `n_compounds`, and
  `random_state`. The group `compounds` has per-member descriptors and two arms, intravenous (central and peripheral)
  and oral (gut, liver, central, and peripheral). The absorption rate, the absorbed fraction, and the hepatic and renal
  clearances follow from the descriptors through a fixed smooth map in the reference form and through one
  `learned_function` in the learned form, whose last layer at zero gives the typical rates. Areas under the curves are
  states with the concentration as rate, and the observable `F` is their dose-scaled ratio. Primitives: `rates`,
  `time-inputs`, `groups`, `observables`, and `learned-functions`.
- **`growing_neural_cellular_automaton`.** Added from the survey of complex adaptive systems, since the neural
  cellular automata that Hartl, Levin, and Pio-Lopez (2025, arXiv 2509.11131) review need exactly `learned_function`
  on the sites of a lattice. One `BaseSystem` with `n_channels=16`, `hidden=128`, `size`, `fire_rate=0.5`, `target`, and
  `random_state`. A site element holds the channels, of which the first four are colour and alpha. One update
  interaction perceives every channel through `convolve` with the identity and the two Sobel kernels, feeds the
  perception to one `learned_function` of two layers whose last layer starts at zero, multiplies the change by a
  `bernoulli` draw of each cell with probability `fire_rate`, and keeps a cell alive when the largest alpha in its 3 by
  3 neighbourhood, a `maximum` reduction, exceeds 0.1 before and after the step. Training is the growing experiment of
  Mordvintsev et al.: from one seed cell, a fixed number of steps, and the squared error of the colour and alpha
  channels against the target at the last step, through `TrainingInteraction` on the unrolled run, whose `data` is one
  observed run of the target at the last step. The sample-pool and damage experiments are left out. The default target
  is a synthetic ring of colours drawn by the system and labelled as such. Primitives: `lattices`,
  `nonlocal-operators`, `updates`, `draws`, `reductions`, `learning`, and `learned-functions`.
- The inventory primitive is `learned-functions` of `add-learning-foundation`. This change adds no primitive.

## Risks / Trade-offs

- [Fits of 200 patients over many epochs are slow in tests] → Tests use smaller groups and fewer epochs with relaxed
  thresholds, and the full scenarios run in the gallery examples, which the docs build caches.
- [Maximum a posteriori effects ignore their uncertainty] → The result documents it, and the Bayesian methods of
  `calibrate` give posteriors when needed.

## Open Questions

- Should a general latent ODE element move to `systems/general` once a second field needs one? Today only the latent
  form of tacrolimus uses it.
