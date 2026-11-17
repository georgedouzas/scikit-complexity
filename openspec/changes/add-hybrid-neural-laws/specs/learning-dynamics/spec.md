# Spec Delta

## ADDED Requirements

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
