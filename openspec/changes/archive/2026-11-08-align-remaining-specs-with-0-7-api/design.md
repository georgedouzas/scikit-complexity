# Design

## Decisions

- The renames of system and solver names are mechanical: a backticked snake-case name maps to the system class whose
  name, without the `System` suffix and underscores, matches it case-insensitively, and only text outside headings
  changes, so every requirement and scenario keeps its name. The duplicate inventory entries `renewable_resource` and
  `nonrenewable_resource`, which a requirement names as removed, keep their old names.
- `koopman_system_predictive_control`, whose name an earlier rename of model to system broke, maps to
  `KoopmanModelPredictiveControlSystem`.
- "A system compiler is not a pass" keeps its subject, since `BaseCompiler` still is the base of whole-system compilers
  and stays unrelated to `BasePass`.
- Requirements whose whole subject is gone and needs a decision, such as the `skcomplex.formalisms` packages or the
  `skcomplex.core.components.problems` package, are left for a later change and reported.
