## Context

`GradientInteraction` in `skcomplex.systems.general.gradients` gives the rate `RateLaw(x, -U.diff(x))` of a potential
`U` written as a Sage expression. Laws differentiate with Sage's `.diff`, and the library defines no derivative
function of its own. The reductions `total`, `mean`, `variance`, `maximum`, and others in
`core/components/laws/utils/_reductions.py` are instances of one private `_ReductionFunction`, a symbolic function of
an expression, a set `over`, and a condition `where`, and the backends lower them over neighbours of radius, network,
or lattice sets. The class defines no `_derivative_` hook, so Sage cannot differentiate an expression that holds them.
The Vicsek system of `physics/active_matter` already reads neighbours in a vision cone through `where` on angles.

Every part follows the current API. An element holds its values, an interaction owns the parameters of the relation
and sets its laws in `laws_` from `_prepare`, and a ready-made system sets `system_` from `_prepare` and is used as
`system.prepare().simulate(until=...)`.

## Goals / Non-Goals

**Goals:**
- Differentiate linear reductions over neighbours with respect to the reading member, the one missing primitive of
  continuous active inference in groups.
- The flock of 2307.14804 with its single-sector equivalence to social forces, its set point, and its collective states.

**Non-Goals:**
- Active inference over discrete states (POMDP agents with expected free energy). pymdp covers it, and no surveyed
  model needs it yet.
- Derivatives with respect to the states of the neighbours, which a reading member does not control.
- Derivatives of `maximum`, `minimum`, and argmax reductions, which have no derivative where the maximiser changes.
- The target-navigation experiment of the paper with informed agents. It is an option of the system for a later change.

## Decisions

### Placement

| Component | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `_derivative_` of `_ReductionFunction` for `total`, `mean`, `variance` | 2, operators the compiler lowers, whose derivative is part of the operator | `core/components/laws/utils/_reductions.py` | active inference in collective behaviour, gradient-based crowd models in social systems, mean-field games in economics, energy minimisation of particle clusters in physics, belief dynamics on networks in opinion dynamics |
| `FreeEnergyInteraction` | 3, one system | `biology/collective_behaviour/_active_inference_flock.py` | biology |
| `ActiveInferenceFlockSystem` | 3, one system | same module | biology |

`FreeEnergyInteraction` is a recipe. It writes the free energy with the core reductions, differentiates it with
`.diff`, and sets `UpdateLaw` laws in `laws_`, so it needs no engine support. It moves to `skcomplex.systems.general`
when a
system of another field, such as predictive coding in neuroscience, uses it.

### Derivatives of reductions

- `total(e, over, where)` differentiates to `total(de/dx, over, where)`, and `mean` to `mean(de/dx, over, where)`, where
  `x` is a state of the reading member and `de/dx` treats the states of the reduced member as constants. The count of
  `mean` does not depend on `x`, since `over` and `where` are held fixed.
- `variance(e)` differentiates to `2 mean((e - mean(e)) de/dx)`.
- The one hook of `_ReductionFunction` dispatches on the name of the reduction. It returns a reduction of a derived
  expression, so the backends print it with no new code. A derivative with respect to a value of the reduced member,
  or of a reduction without a derivative, raises `ValidationError` when the system is prepared.
- Holding `where` fixed is the choice of the paper, whose sector membership is piecewise constant in the heading, and of
  every gradient model of neighbours. The docstring states it.

### The flock

- The agents are one `Element` with `size=` members on a periodic `ContinuousSpace`. Values of an agent: its position,
  named after the axes of the space, its heading `v` (a unit vector), and its beliefs `mu` and `mu1` per sector, the
  first two generalised coordinates. `mu1` is a value of its own and not the time derivative `mu'`, since the beliefs
  move in discrete time. Observations per sector `l`, with `r` the position: `y_l = mean_{j in N_l} |r_j - r| + z` and
  `y1_l = mean_{j in N_l} (r - r_j) . (v - v_j) / |r - r_j| + z'`, with `N_l` the neighbours within the sensing radius
  whose bearing relative to `v` lies in sector `l`, read through `mean(..., over=RadiusNeighborhood(...), where=...)`,
  and `z`, `z'` Gaussian draws.
- Generative model: `f(x) = -alpha (x - eta)` on every sector, so `mu1` is predicted as `-alpha (mu - eta)`. Free
  energy: `F = 1/2 sum_l [pi_z (y_l - mu_l)^2 + pi_z' (y1_l - mu1_l)^2 + pi_w (mu1_l + alpha (mu_l - eta))^2 +
  pi_w' (alpha mu1_l)^2]`, with the precisions built from `Gamma_z` and `lambda_z` as `pi_z' = 2 Gamma_z lambda_z^2`.
  The precisions, `alpha`, and `eta` are parameters of `FreeEnergyInteraction`, since they belong to the relation.
- Updates are `UpdateLaw` laws in discrete time with the step `dt = 0.01`: `mu <- mu + dt (mu1 - k_mu dF/dmu)`,
  `mu1 <- mu1 - dt k_mu dF/dmu1`, `v <- normalise(v - dt k_a dF/dv + noise)`, `r <- r + dt s0 v`. `dF/dv` reaches `v`
  only through `y1_l`, so it is a mean over neighbours that the new derivatives give.
- Observables: polarization `|mean v|` and the normalised angular momentum about the centre of the group.
- The defaults follow the default configuration of the SI of the paper: 50 agents, 4 sectors of 60 degrees, `eta = 1`,
  `alpha = 0.5`, and 1500 steps of 0.01 seconds. Values the SI gives for the precisions and speed are copied, and any
  value it leaves open is recorded in the docstring with its source in the authors' code.
- Primitives: `continuous-spaces`, `groups`, `updates`, `angles`, `draws`, `reductions`, and the new
  `reduction-derivatives`.

### Libraries surveyed

- `inferactively-pymdp` 1.0.5, released 2026-11-07, implements discrete active inference on JAX. The flock is
  continuous, so it does not apply.
- The authors' flocking code is a research repository on JAX, not a package. SPM, which holds generalised filtering, is
  MATLAB. No Python package differentiates symbolic reductions over neighbours, which is what the core needs.

## Risks / Trade-offs

- [Derivatives of large reductions grow the expressions] → The hook differentiates the reduced expression only, so the
  size grows like the expression, not like the group.
- [Collective-state scenarios are noisy] → They compare two configurations over 20 seeds instead of fixing thresholds,
  and are marked slow.
- [Parameters the paper leaves open] → The docstring cites where each default comes from.

## Migration Plan

None. Expressions without derivatives of reductions compile as before.

## Open Questions

- Should derivatives with respect to the neighbours' states be added for adjoint gradients of a whole group? No model
  needs them yet.
