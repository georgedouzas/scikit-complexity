## MODIFIED Requirements

### Requirement: Specs live in one package
Every spec SHALL be a frozen `BaseSpec` in a module of its own, in the lowest package that can import every type its
fields name: `skcomplex.specs` for specs built from root types, `skcomplex.core.specs` for specs that name core
classes, the engine package whose classes they name, such as `skcomplex.engine.problems` for the problem and kernel
specs, and `skcomplex.experiments.specs` for experiment specs. Each of these packages SHALL export its public specs.

#### Scenario: Specs from one place
- **WHEN** a user imports `StateSpec` and `SupportSpec` from `skcomplex.specs`, `NodeSpec` from
  `skcomplex.core.specs`, and `ProblemSpec` and `FrameSpec` from `skcomplex.engine.problems`
- **THEN** every import succeeds and each class is a frozen `BaseSpec`

#### Scenario: No scattered dataclass
- **WHEN** a test lists every dataclass defined in `skcomplex.core`, `skcomplex.engine`, `skcomplex.analysis`, and
  `skcomplex.experiments`
- **THEN** every one of them is a frozen `BaseSpec`
