## MODIFIED Requirements

### Requirement: Any differentiable system can be fitted
A gradient-step law SHALL accept any nested system that validates as differentiable, not only neural networks, so the
parameters of a system of any field can be fitted to observed series by gradient descent.

#### Scenario: Lotka-Volterra fit
- **WHEN** a gradient-step law with Adam fits the four adaptive rate parameters of `LotkaVolterraSystem`, started 30% off, to
  noise-free prey and predator series of 50 points over 20 time units
- **THEN** after 500 epochs every parameter is within 1% of its true value and the loss has fallen by at least four
  orders of magnitude
