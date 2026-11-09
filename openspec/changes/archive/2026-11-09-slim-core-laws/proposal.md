## Why

The laws of the core and the functions of `laws.utils` hold several parts that are only parametrised versions of
another core part. `FractionalNoiseLaw` is a `NoiseLaw` with a Hurst index, `EmissionLaw` is a `TransitionLaw` whose
row a second state selects, `EncounterLaw` is a `BilinearLaw` with a matching loss, and `lookup`, `delay1`, `delay3`,
`smooth`, and `inverse_laplacian` are calls of `series`, `distributed_delay`, and `inverse_helmholtz` with fixed
arguments. So a reader meets two names for one operation, and the engine tests both classes everywhere. `laws.utils`
also re-exports helpers that only the compiler reads, and it re-exports the draw functions from a spec module.
`DriveLaw` names an effect, not the operation, and the kinetic `cross_section` is an argument of `RateLaw` that one
utility reads. This change keeps one way for each operation, as the user decided for the whole core.

## What Changes

- **BREAKING** `FractionalNoiseLaw` is removed. `NoiseLaw` takes `hurst=None`, and a Hurst index makes it the
  amplitude of a fractional Brownian motion. The fractional stepper is chosen by a new noise kind, not by a law class.
- **BREAKING** `EmissionLaw` is removed. `TransitionLaw` takes `source=None`, and a source state selects the row of a
  rectangular matrix, read after the same step.
- **BREAKING** `EncounterLaw` is removed. `BilinearLaw` takes `conserved=False`, and a conserved law also takes the
  matching loss from its first density, so the total over the classes stays constant. Its second density may be
  `None` for a linear redistribution.
- **BREAKING** `DriveLaw` is renamed `ArrayUpdateLaw`, with the same arguments.
- **BREAKING** `RateLaw` loses `cross_section`. The utility `collision` becomes `collision_integral(density, kernel)`,
  which carries its kernel `B(speed, cosine)` in the expression.
- **BREAKING** `lookup` is removed. `series(x, points, values, kind)` takes any value as its first argument.
- **BREAKING** `delay1`, `delay3`, and `smooth` are removed. Laws call `distributed_delay` with an order and
  `conserved`.
- **BREAKING** `inverse_laplacian` is removed. `inverse_helmholtz` takes the screening `0` by default.
- **BREAKING** `laws.utils` stops re-exporting the helpers the compiler reads: the `*_in` functions, `delay_parts`,
  `has_operator`, `distribute_operators`, `coordinates_of`, and the constants `COORDINATE_PREFIX`,
  `INVERSION_BOUNDARIES`, `MAX_NEIGHBOR_HYPEREDGE_SIZE`, and `NEIGHBOR_REDUCTIONS`. `random_pairing` becomes a
  private method of `MatchLaw`.
- The draw functions `uniform`, `normal`, `bernoulli`, `categorical`, and `draw` move from `specs/_draw_spec.py` into
  `laws/utils/_draws.py`, so `laws.utils` re-exports them where they live. `DrawSpec` stays in `specs`.
- The empty module `laws/utils/_tables.py` is removed.
- Kept as named math vocabulary: `step`, `ramp`, `pulse`, `pulse_train`, `series`, `clip`, `sign`, `hill`,
  `logistic`, `circular_mean`, `dagger`, `operator_product`, `distributed_delay`, `inverse_helmholtz`. `ExtremalLaw`,
  `Source`, and `Sink` stay.
- No realization subclass is added. The removed laws are emitted by core interactions only, and the removed
  utilities are pure math conveniences, so systems call the surviving core part directly.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `anomalous-processes`: fractional Gaussian noise is a noise law with a Hurst index, validated against a process and
  a conserved noise.
- `stochastic-maps`: hidden Markov sources are transition laws with a source state, and the matrix check covers them.
- `kinetic-models`: conservative encounters are bilinear laws with a conserved loss.
- `general-mechanisms`: the bilinear law covers conserved encounters, the law utilities lose their parametrised
  copies and the compiler's helpers, and the draw functions have one import path.
- `phase-space-fields`: the collision integral carries its kernel in the expression.
- `system-dynamics-import`: lookups and delays translate to `series` and `distributed_delay`.
- `molecular-structures`: the replay of an ensemble runs through an array update law.
- `model-library`: the fuzzy oil drop system names the array update law.
- `tensor-fields`: the Leray projection is built from `inverse_helmholtz` with no screening.

## Impact

- Code: `src/skcomplex/core/components/laws` and `laws/utils`, the interactions that emit the removed laws
  (`_rate`, `_relaxation`, `_transition`, `_encounter`, `_sampling`, `_cascade`), the import sites of the private
  helpers in `_grouping` and `_gradient`, `specs/_draw_spec.py`, `specs/_interaction_requirement_spec.py`,
  `core/types.py` (`NoiseKind`), `engine/types.py`, the compile passes, solvers, and backends that name the removed
  classes, `experiments/_replicate.py`, and the system dynamics translator.
- Systems: the modules that call `lookup`, `delay3`, `smooth`, `inverse_laplacian`, or `DriveLaw`, in economics,
  social, physics, engineering, biology, and chemistry.
- Tests: the law tests of `tests/core/components/laws`, the utility tests, and the engine tests that name the removed
  classes.
- Documents: the user guide pages under `docs/overview/user_guide/build_a_custom_system`.
- Dependencies: it lands after `add-system-package`. It shares the interaction emit sites with
  `slim-core-interactions` and `tests/core/components/elements/test_density.py` with the elements and spaces change.
- Users: code that imports a removed name must use the surviving part. The project makes breaking changes in 0.x
  without shims.
