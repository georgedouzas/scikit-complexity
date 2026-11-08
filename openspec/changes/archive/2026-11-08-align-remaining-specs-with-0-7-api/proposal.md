# Proposal

## Why

After `align-specs-with-0-7-api`, the main specs still named APIs that the 0.7 code removed or made private: the
`calibration experiment` with unnamed fitted attributes, `SINDySurrogate` as an estimator, a public `RewritingCompiler`,
`BaseSystemCompiler` and `QuantumCompiler`, the snake-case catalog names of ready-made systems that are now classes,
and solver names without their `Solver` suffix.

## What Changes

- experiments: "Calibration fits parameters to data" names `Calibration` and its fitted attributes `system_`,
  `estimates_`, `r2_`, `rmse_`, `mae_`, and `mape_`, and the calibration scenario of "Experiments are functions, except
  calibration" reads `system_`.
- base-interfaces: "Surrogates are estimators" is renamed to "Surrogates are analysis functions" and maps to
  `mode_decomposition`, `sparse_identification`, and `reservoir` with their frozen result types.
- rewriting-systems and package-layout: the compiler of the rewriting form is private and chosen by `simulate`.
- package-layout: "A system compiler is not a pass" names `BaseCompiler` and the compiler `simulate` chooses for a
  quantum system.
- Every spec: the snake-case names of ready-made systems become their class names, such as `lotka_volterra` to
  `LotkaVolterraSystem`, and `EulerMaruyama`, `ScipyODE`, and `VelocityVerlet` become `EulerMaruyamaSolver`,
  `ScipyODESolver`, and `VelocityVerletSolver`.
- analysis-interface: the Purpose describes analyses as functions that return frozen results. A Purpose is no
  requirement, so it is edited in the main spec directly.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `base-interfaces`, `experiments`, `package-layout`, `rewriting-systems`: the requirements listed above.
- `curved-spacetimes`, `declared-state-elements`, `hybrid-dynamics`, `learning-dynamics`, `model-catalog`,
  `model-gallery`, `model-inventory`, `model-library`, `python-support`, `rerun-visualization`, `simulation`,
  `spatial-fields`, `system-modeling`, `system-validation`, `trajectory-analysis`: system and solver names follow the
  0.7 classes.

## Impact

Specs only. No code changes.
