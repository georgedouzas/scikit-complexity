# Design

## Context

The inventory records the four models as pending this change, with tags that predate the general rewrite:
`szabo_cells` and `active_passive_flocking` carry a network and an edge group, which the systems they build do not
have. The general classes they now build on are listed in the proposal. `ModeDecompositionAnalysis` fits the exact DMD
of the one-sample map `g(t + dt) = A g(t)` with PyDMD, on the variables or on the observables of a scikit-learn
dictionary, and its `system_` follows `dg/dt = L g` under continuous time. It takes no inputs.

## Goals / Non-Goals

**Goals:**
- Build the four models from the general classes, with tests of their known behaviour.
- Add the one general operation the Koopman controller needs, a mode decomposition with control inputs.

**Non-Goals:**
- A receding-horizon solver in the engine. The controller is a discrete-time partition of the model, which
  `CoSimulationSolver` already runs beside the plant.
- Any other new core class. A model that turns out to need one stops, and its need becomes a change of its own.

## Decisions

### Mode decomposition with control

`ModeDecompositionAnalysis(inputs=None)` takes the names of the variables that are control inputs. With inputs, it fits
`g(t + dt) = A g(t) + B u(t)` with PyDMD's `DMDc` on the pairs of consecutive samples, where `g` holds the observables
of the dictionary computed from the other variables, and it sets `operator_` and `input_matrix_`, the matrices `A` and
`B` of the one-sample map. Its `system_` follows `dg/dt = L g + M u` with `L` from the eigenvalues of `A` as today and
`M` the input matrix that gives the same one-sample map under an input held over the sample, and it declares every input
as a bounded adaptive parameter of the surrogate element, bounded by the range of the inputs in the samples. PyDMD is
already a dependency of the analysis layer, and its `DMDc` is the mature implementation, so nothing is written in house
beyond the adapter. The audit of 2026-11-01 recorded that PyDMD's last release, 2025.8.1, is of 2025-08-01; the
manifest already requires `pydmd>=2025.8.1`, and the risk is recorded below.

### szabo_cells

Szabó et al. (2006) cells in a periodic `ContinuousSpace`: a group of cells whose position `x`, a vector of order 1,
moves at `v0 n + mu F`, where `n` is the unit vector of the heading `theta` and `F` the sum of the pair forces over the
neighbours within twice the cell radius, `neighbors(force, over=RadiusNeighborhood(2 R))`. The heading turns towards the
velocity, `theta' = asin((n × v / |v|) · e_z) / tau`, with angular noise. Both are `RateInteraction`s, the second with
`noise`, and both read the same reduction, so the velocity needs no state of its own. The force is the linear
repulsion and attraction of the paper, a function of the pair namespace written in the package.

### active_passive_flocking

The model of Charbonneau (2017, section 10.4), taken from Silverberg et al. (2013): one group of agents in a periodic box
with a mark that tells active from passive, since the flocking force of an active agent reads the velocities of both
kinds and a reduction reads one group. All agents feel a soft repulsion, a `PairCouplingInteraction` with a potential,
and a propulsion towards a target speed, a `RateInteraction`, with the target `v0` when active and 0 when passive. Active
agents also feel a flocking force, a `RateInteraction` reading `neighbors(velocity, over=RadiusNeighborhood(r_f))`, and
noise. The system runs under continuous time with Euler-Maruyama steps.

### koopman_model_predictive_control

Korda and Mezić (2018), as Brunton et al. (2021, §6.1.1 and §6.3) review it, for the Duffing oscillator, the Van der Pol
oscillator, and a bilinear DC motor, chosen by the variant `plant`:
- The factory runs the plant with a bounded adaptive control input set to a random binary value every sample, and
  fits `ModeDecompositionAnalysis(dictionary=..., inputs=('plant.u',))` to the runs.
- The returned system nests the plant under `ContinuousTime` and a controller element under `DiscreteTime` with the
  sample step. The controller's `MapInteraction` sets its state `u` to a `CallSpec` of the plant states. The called
  function lifts the states with the dictionary, condenses the quadratic tracking cost over the horizon with `A` and
  `B`, and solves the bounded least-squares problem with `scipy.optimize.lsq_linear`, returning the first input.
- The plant reads `u` from the controller through a role, and `CoSimulationSolver` holds it between samples.

The predictive problem lives in the model package as a private class, because it is the one model that uses it.

### next_generation_reservoir

Gauthier et al. (2021): an element whose variables keep `delay - 1`
lags, `StateSpec(lags=...)`, and one `MapInteraction` per variable in the increment mode, whose function is the ridge
readout of the constant, linear, and quadratic monomials of the variables and their lags. The factory does not fit a
second readout. It fits `ReservoirAnalysis(kind='next_generation', delay=..., order=..., ridge=..., standardize=False)` of
`skcomplex.analysis` on a trajectory of the Lorenz flow, which the package builds from `RateInteraction`s since model
packages do not import each other, or on a series the caller passes. That analysis already fits ReservoirPy's `NVAR`
features and `Ridge` readout in `src/skcomplex/analysis/_reservoirs.py`, and the factory reads the fitted readout from
its `step_`: the weights and bias of the `Ridge` node, the standardisation `mean` and `scale`, and the order of the
`NVAR` monomials. It writes them as the symbolic increment of every variable over its lags, so the model is a readable
system of the core while the fit stays in one place. A test checks that the forecast of the system agrees with the
`forecast_` of the same analysis.

## Findings against the sources

Recorded on 2026-11-04 after reading the sources. Pages are those of the printed book or paper. The sources marked
fetched are absent from `references/` and were read from arXiv.

### active_passive_flocking

Charbonneau (2017), `textbooks/`, section 10.1 pages 225 to 228, figures 10.1 and 10.3 pages 229 and 231, table 10.1 page
236, section 10.4 pages 240 to 242, and Silverberg et al. (2013, arXiv 1302.1886, fetched), supplement equation A.1.

- **What gathers.** The scenario "with flocking off" was wrong. The book shows active agents flocking into streams with
  the flocking amplitude 1 (figure 10.7, pages 240 and 241), and exercise 3 asks at which lower amplitude flocks cease.
  Silverberg et al. state that "any non-zero flocking force" causes self-segregation from random positions. No source
  says that active agents gather without the force. The scenario now runs with flocking on, and a second scenario holds
  that the agents stay mixed with flocking off.
- **What was wrong in the model.** Both the model and the spec were wrong.
  - The force exponent was 5/2, the value of Silverberg et al. The book has 3/2 in equation 10.1 and figure 10.3. The
    default is now 3/2 and `exponent` takes the other.
  - The flocking force read only active agents and included the agent itself. Equation 10.3 and figure 10.3 sum over all
    agents with `j != k`, so an agent with no neighbour feels no force, while the old one pushed along its own velocity.
  - The parameters were those of Silverberg et al. with `alpha = 2` and `noise = 0.5`, not one published set. They are
    now the column of figure 10.7: `r_0 = 0.025`, `epsilon = 25`, `r_f = 0.1`, `alpha = 1`, `v0 = 0.02`, `mu = 10`,
    `eta = 0.1`, 45 active agents, `dt = 0.01`, 50 time units, in the unit box. The old `mu = 0.05` is Silverberg's.
  - The start was a jittered grid. Figure 10.1 lines 25 to 29 draw uniform positions and velocity components uniform in
    `[-1, 1]`.
- **Signs and propulsion.** The sign of the repulsion (`r_jk` points from `k` to `j`) and the propulsion `mu (v0 - v) v`
  were right.
- **Passive agents do not flock, which is a judgement.** Table 10.1 lists one `alpha = 1` for the column of figure 10.7,
  and page 240 says active agents differ only in their target speed and their random force. Taken literally the passive
  agents flock too. A NumPy transcription of figures 10.1 to 10.3 shows the result: with passive `alpha = 1` every agent
  moves in one direction at about `5 v0` at all four compactnesses, with polarization 0.998 and no segregation, against
  the book's own figure and its text that passive agents remain at rest. With passive `alpha = 0`, passive speeds fall to
  0.4 to 0.9 `v0`, active speeds are 3 to 5 `v0`, and the snapshots and polar diagrams look like figure 10.7. Silverberg et
  al. state `v0 = alpha = eta = 0` for passive agents. The model follows them.
- **Noise.** The book draws a uniform force in `[-eta, eta]` every step, which depends on the step. The model uses a
  Wiener process with the same variance per step, of amplitude `eta sqrt(dt / 3)`. The law differs from uniform, and the
  book's Euler step updates the velocity first.
- **Measured.** Over four seeds, the mean distances over the last 10 time units with flocking on are, active against
  passive: at `C = 0.224`, 0.33 to 0.36 against 0.38 to 0.39; at `C = 0.672`, 0.33 to 0.36 against 0.38; at `C = 0.895`,
  0.37 to 0.38 against 0.38, where the book reports active agents trapped in the crystal. With flocking off they stay
  within 0.007 of each other. A uniform draw gives 0.383. Polarization is 0.86 to 0.90 at `C = 0.224` and 0.37 to 0.61 at
  `C = 0.672`, where figure 10.7 shows a tighter fan. The book gives no number to compare, and this snapshot comparison
  is by eye.
- **Silverberg's own claim is not reproduced.** Their supplement says that any non-zero flocking force segregates agents
  from random positions on a timescale of `10^3 r_0 / v_0` time steps. With their values (`N = 500`, 150 active,
  `L = 40.8`, packing 0.94, `epsilon = 100`, `mu = 0.05`, `r_0 = 1`, `r_f = 4`, exponent 5/2) and the corrected force, the
  model stays mixed for 3000 time units at `dt = 0.05`: mean distances 15.6 and 15.6, and mixed snapshots at `t` = 200,
  1000, and 3000, for `alpha = 1` with noise 0.5 (polarization 0.1) and `alpha = 5` with noise 0.1 (polarization 0.98).
  The noise values are our choice within their range, and the paper does not give its time step or how many time units
  a step is. The scenario therefore rests on the book's setup, and this claim stays open.
- **Why the earlier search failed.** About 20 parameter sets with flocking off could not gather, as the runs above
  confirm. The run with flocking on used the packing 0.94, near the compactness at which the book reports active agents
  trapped, with the flocking force of the old model. Which of these stopped it was not separated.

### next_generation_reservoir

Gauthier et al. (2021, arXiv 2106.07688, fetched), section V and the supplement, and Cestnik and Martens (2025,
`papers/2509.11338`).

- **The two papers differ.** Cestnik and Martens use a pseudorandom projection, normalize to `[0, 1]`, add 1% noise, and
  sample the Lorenz flow at 0.04 with `lambda = 0.01` or none (sections III.B to III.F, IV.A). None of this is what the
  model does. The model, its monomials, and the increment target (equation 11) follow Gauthier et al. The value 2.5e-6
  appears only there.
- **Values of Gauthier et al.** `dt = 0.025`, `k = 2`, `s = 1`, 400 training points after 2 warm-up points, 28 features,
  and `alpha = 2.5e-6` on features that are not standardized (figure 2, supplement table 1). The supplement names an
  explicit Runge-Kutta 3(2) integrator. The model used 800 samples, `1e-4`, standardized features, and DOP853. The
  defaults are now those of the paper, with a new setting `standardize` on `ReservoirAnalysis`.
- **The integrator matters, measured.** With raw features and 2.5e-6 the forecast from RK23 data at SciPy's default
  tolerances holds 10% of the attractor size for a median of 4.5 Lyapunov times over 20 windows, minimum 2.4, all stable
  over 4000 steps, which agrees with the paper's "~5". The paper does not state the tolerances. With DOP853 or RK45 data
  the same fit diverges within 0.1 to 0.4 Lyapunov times, and standardized features then hold for 2.2 to 6.7. A test
  records this, and the docstring states it.
- **The start.** The factory started at `(1, 1, 1)`, whose 400 samples after the transient stay on one wing, and the forecast
  then diverged. The paper says the window must cover both wings. The start is now `(-8, 8, 27)`.
- **Scenario.** The forecast scenario, one Lyapunov time, stays as a lower bound, and the agreement scenario holds at the
  new defaults.

### koopman_model_predictive_control

Brunton et al. (2021, `papers/2102.12086`), section 6.3, table 6.1 and figure 6.6 pages 83 and 84, and Korda and Mezić
(2018, arXiv 1611.03537, fetched), section 8.2.

- **Duffing input, wrong.** Figure 6.6 has `+ (3 + cos(x_1)) u`. The model had `+ u`. The mean squared error of the
  output falls from 0.214 to 0.002.
- **Training input, wrong.** The review applies a random binary signal at the limits of the range, and the prediction
  panels of figure 6.6 switch every few samples, without a stated hold. The model held uniform values over 1 to 20
  samples. It now draws the binary value every sample, which is our reading of the figure. Errors after this and the Duffing fix, controlled against free: Duffing 0.0020 and 0.82,
  Van der Pol 0.0018 and 0.95, motor 0.19 and 2.8e4. The motor was 0.076 with the uniform input, so this source value
  costs the motor some error. Holding the binary input over up to 20 samples gave 4.5 to 6.4.
- **Training size, wrong.** Section 6.3 uses 100 initial conditions and 1000 samples per trajectory. The model used 20 and
  200. The defaults are now those of the review. The errors barely change, and a factory call takes about 30 to 50 s.
- **Correct as inferred.** The Van der Pol equation, the motor equations and parameters, `y = x_2` for all three, the
  Duffing parameters, the time steps, the domains, the references, the horizons, and the weights agree with table 6.1
  and figure 6.6.
- **Motor input limit.** The table says `[-1, 1]`, and the model multiplies by 4. Korda and Mezić, section 8.2, say the
  physical range is `[-4, 4]`, scaled to `[-1, 1]`. Figure 6.6 agrees with the scaling: holding the output near 0 needs a
  physical input near 1.2, which is the plotted 0.3 times 4 and outside `[-1, 1]`. The limit of 4 stays.
- **Not implemented.** Table 6.1 constrains the output, such as `-0.8 <= x_2 <= 0.8`, which `lsq_linear` cannot express, so
  the controlled output follows the reference past the bound where the review clips it. The default dictionary is the
  monomials up to degree 3 on the full state. The review uses delay coordinates and 100 thin plate splines (`p = 103`) on
  the output. Both are reported as open, since the first needs another solver and the second another dictionary.

### szabo_cells

Szabó et al. (2006, arXiv q-bio/0611045, fetched), equations 1 to 4, and Vicsek and Zafeiris (2012, arXiv 1010.5017,
fetched), section 5.1.2, equations 45 and 46.

- **Correct.** The motion, the heading rule, the noise variance `eta^2 / 12`, the values `v0 = mu = tau = R_0 = 1`,
  `R_eq = 5 / 6`, `F_adh = 0.75`, `F_rep = 30`, and `rho_c = 0.18` at `eta = 0.6` agree. The inventory citation of section
  5.1.2 is right. Equation 4 as printed gives an attractive force below `R_eq` with `e_ij` from `j` to `i`, against the
  paper's text. The model follows the text, repulsive below `R_eq`.
- **Fixed.** The test said it measured the mean normalized velocity of equation 1 but did not normalize. It does now. The
  observable `polarization` reads the headings, which is a different quantity from equation 1, as its name allows.

## Risks / Trade-offs

- [Fitting inside a factory makes it slow] → The default runs are short, and the factory takes the training series or
  the fitted readout as arguments to skip the fit.
- [The readout depends on the feature layout of ReservoirPy's `NVAR`] → The factory reads the monomial order from the
  fitted node rather than assuming it, and the agreement scenario fails if the layouts diverge.
- [PyDMD has not released since 2025-08-01] → `DMDc` is a small, stable part of PyDMD, the adapter is the one place
  that calls it, and the linear system scenario detects a break.
- [The surrogate of a controlled plant may track poorly] → The scenario compares the controlled with the uncontrolled
  plant rather than fixing an error bound, as the review does.
