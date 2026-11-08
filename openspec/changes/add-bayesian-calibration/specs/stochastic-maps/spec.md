## ADDED Requirements

### Requirement: Discrete draws take a continuous relaxation
`bernoulli` and `categorical` SHALL take `temperature`, a positive number or `None`. With a temperature, the JAX
lowering SHALL replace the draw by its concrete relaxation of the uniform value of the step (Maddison et al. 2017, Jang
et al. 2017): `sigmoid((logit p - logit u) / temperature)` for a Bernoulli draw, and `sum_k sigmoid((u - c_k / W) / temperature)`
over the cumulative weights `c_k` before the last and the total weight `W` for a categorical draw, the relaxation of
its inverse distribution function of the same uniform value.

#### Scenario: Gradient through a relaxed Bernoulli draw
- **WHEN** the mean over fixed uniform values of a relaxed Bernoulli draw of probability `p` at temperature 0.5 is
  differentiated in `p` on JAX
- **THEN** the gradient agrees with central finite differences of step 1e-6 within 1e-6 relative error

#### Scenario: Relaxed draws converge to exact draws
- **WHEN** a relaxed Bernoulli draw of probability 0.3 is lowered on JAX at temperature 0.01 for 20000 uniform values
- **THEN** the mean of the values lies within 0.02 of 0.3

### Requirement: Relaxed draws take straight-through hard values
`bernoulli` and `categorical` SHALL take `hard`. With `hard=True` and a temperature, the JAX lowering SHALL give the exact
draw in the forward pass and the gradient of the relaxation, as a straight-through estimator.

#### Scenario: Hard values
- **WHEN** a relaxed Bernoulli draw with `hard=True` is lowered on JAX for 1000 uniform values
- **THEN** every value is 0 or 1 and the gradient in `p` is the gradient of the relaxation

### Requirement: Other backends draw relaxed draws exactly
Every backend other than JAX SHALL draw a Bernoulli or categorical draw exactly and ignore its relaxation.

#### Scenario: Exact draws on NumPy
- **WHEN** a map with a relaxed Bernoulli draw runs on the NumPy backend
- **THEN** every drawn value is 0 or 1

### Requirement: Discrete draws without a relaxation are not differentiable
The JAX lowering SHALL report a `not-differentiable` warning that names a Bernoulli or categorical draw without a
temperature, and differentiable runs SHALL stop at it.

#### Scenario: Discrete draw without a temperature
- **WHEN** a system whose update draws `bernoulli('flip', p)` without a temperature is lowered on JAX for a
  differentiable run
- **THEN** the report holds a `not-differentiable` issue that names the draw `flip` and asks for a temperature
