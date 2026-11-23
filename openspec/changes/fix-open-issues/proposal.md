# Proposal

## Why

Ten GitHub issues are open against 0.9.0, and each still happens on 0.10.0: seeds that do not do what users expect,
two lattice and network bugs, analysis results that are hard to reproduce or read, a missing lattice coarse graining,
slow defaults, and wrong docstrings. The redesign of the core API also left two loose ends: the Project Profile of the
constitution still names removed parts, and the engine record `LoweredFunctionsSpec` is still public.

## What Changes

- **Seeds (#23, #24):** every ready-made system names its seed `random_state`, and the 35 systems that call it `seed`
  are renamed. Its docstring says it seeds what the system draws when prepared, such as the initial state and a random
  network, while `simulate(random_state=)` seeds the dynamics. `ensemble` gives every replicate its own system seed,
  derived from the experiment's seed, so replicates start from different initial states. **BREAKING:** `seed=` is gone.
- **Ising sizes (#26):** a lattice system on an odd periodic lattice uses the random sequential schedule when its solver
  is `'auto'`, and the checkerboard error names how to choose a schedule through the `solver` setting.
- **Coevolving voters (#27):** the degree of every voter starts at its degree in the initial network, so frame 0
  records it.
- **Fixed points (#25):** `stability` returns fixed points in a fixed order, sorted by their values, and exact solutions
  convert to floats without solver noise.
- **Orbit periods (#29):** `orbit_diagram` reports a period of 0 when no period is found among the kept values.
- **Block coarse graining (#30):** `coarse_grain` takes `blocks` and `rule` to coarse grain a lattice configuration over
  blocks by majority, with a tie rule, or by mean.
- **Plain equations in stability (#31):** `stability` is shown on a normal form written as a `DynamicalSystem` in one
  line, in its docstring and the user guide.
- **Performance (#32):** `orbit_diagram` over many parameter values runs fast enough for a 1200 value scan, and
  `CurieWeissSystem` records one frame per sweep by default.
- **Docstrings (#28):** `IdealGasSystem`, `GalamMajoritySystem.share`, and the states of `BernoulliProcessSystem`.
- **Cleanup:** the constitution's Project Profile names the current parts, and `LoweredFunctionsSpec` becomes private.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `experiments`: ensembles vary the seed of each replicate's system.
- `model-catalog`: ready-made systems name their seed `random_state` with one meaning, and lattice systems choose a
  schedule that fits their size, and coevolving networks start with their initial degrees.
- `trajectory-analysis`: stable fixed point order, aperiodic orbits, and block coarse graining.

## Impact

- Code: `core/experiments/_ensemble.py`, `_orbit_diagram.py`, `core/analysis/_stability.py`, `_coarse_grain.py`,
  `core/engine/backends`, the Metropolis schedule choice, the 35 systems with `seed`, `CurieWeissSystem`,
  `CoevolvingVoterSystem`, `IdealGasSystem`, `GalamMajoritySystem`, `BernoulliProcessSystem`.
- Docs: the user guide pages on analysis and experiments, `openspec/constitution.md`.
- Breaking: `seed=` on ready-made systems becomes `random_state=`. `orbit_diagram` periods of aperiodic orbits change
  from `keep` to 0. Ensemble replicates of systems with a seed start from different initial states.
