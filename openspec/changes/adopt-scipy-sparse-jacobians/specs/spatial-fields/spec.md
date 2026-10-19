# Spec Delta

## MODIFIED Requirements

### Requirement: Explicit steps are checked for stability
The compiler SHALL warn when an explicit stepper's step exceeds the diffusive stability limit of a field, and name the
largest stable step and the implicit SciPy methods BDF and Radau as the stiff path. An adaptive explicit method SHALL
warn once when one call spans more than 1000 times the stability limit.

#### Scenario: Too large a step
- **WHEN** a user runs diffusion with RK4 and a step four times the stability limit
- **THEN** a warning names the limit before the run starts

#### Scenario: Stiff path named
- **WHEN** the stability warning is raised for a finite-difference or finite element field
- **THEN** it names BDF and Radau of the SciPy ODE solver

#### Scenario: Adaptive explicit method on a stiff field
- **WHEN** Fisher-KPP on 4000 points with a stability limit near 1e-3 runs with RK45 and frames of 2
- **THEN** one warning names BDF and Radau on the first frame, and later frames do not repeat it
