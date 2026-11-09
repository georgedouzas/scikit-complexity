# Spec Delta

## ADDED Requirements

### Requirement: Laws call neural networks
A law SHALL call a neural network through `neural_function(module, *inputs, weights=...)`, which returns one expression
per output of the network. The weights SHALL be one vector parameter of the part, read at every evaluation, so the
same network gives different outputs after `set_params` or a scheduled change of the weights.

#### Scenario: Network in a rate law
- **WHEN** a rate law reads the output of a network with two inputs, one hidden layer of 8 units, and one output
- **THEN** the compiled rate equals the network evaluated on the same inputs and weights within 1e-12 on the NumPy and
  JAX backends

#### Scenario: Weights changed
- **WHEN** the weights parameter is set to new values through `set_params` and the system is compiled again
- **THEN** the rate equals the network evaluated with the new weights within 1e-12

#### Scenario: Weights of the wrong length
- **WHEN** the weights parameter holds fewer values than the network has weights
- **THEN** the compiler's report holds an issue with the code `neural-weights` that names the parameter, the expected
  length, and the given length

### Requirement: Backends without a neural form fall back
A backend that cannot lower a neural network SHALL evaluate the law holding it in Python and SHALL warn once, naming the
backend, the law, and the reason, as for any kernel that falls back.

#### Scenario: Numba backend
- **WHEN** a group of 2000 members with a neural rate law is compiled with `backend='numba'`
- **THEN** compilation warns that the neural law runs in Python, and the rates equal those of the NumPy backend within
  1e-12

### Requirement: Neural laws are differentiable
A system whose laws call neural networks SHALL validate as differentiable, so `GradientStepLaw` and
`neural_identification` train its weights through the unrolled run.

#### Scenario: Gradient of the weights
- **WHEN** the gradient of the loss of a system with a neural rate law is taken with respect to its weights
- **THEN** it agrees with central finite differences of step 1e-6 within 1e-6 relative error

### Requirement: Parameters take fixed shapes
`ParameterSpec(shape=...)` SHALL accept whole numbers as fixed sizes beside role names, so a parameter holds an array of
a given shape that is addressed by path, recorded with its shape, and changed or trained like any parameter.

#### Scenario: Weight vector
- **WHEN** a part declares `ParameterSpec(shape=(97,), adaptive=True)` and the system is simulated
- **THEN** the trajectory records the parameter with shape `(97,)` at every frame

#### Scenario: Shape not positive
- **WHEN** a part declares `ParameterSpec(shape=(0,))`
- **THEN** `prepare` raises `ValidationError` that names the shape and says sizes are positive whole numbers or role
  names

### Requirement: Infusion schedules are differentiable
The JAX path of the ODE solvers SHALL pass the breakpoints of pulse and pulse-train inputs to the step size controller,
so a system dosed by short infusions is simulated and differentiated on JAX without stepping over an infusion.

#### Scenario: Dosing schedule on JAX
- **WHEN** a one-compartment system receives an infusion of width 0.1 hours every 12 hours through a dosing interaction
  and runs on the NumPy and JAX backends until 72 hours
- **THEN** both trajectories agree within 1e-6 and match the analytic response to the infusions within 1e-6

#### Scenario: Gradient through doses
- **WHEN** the gradient of the squared error of that system with respect to its elimination rate is taken on JAX
- **THEN** it agrees with central finite differences within 1e-6 relative error
