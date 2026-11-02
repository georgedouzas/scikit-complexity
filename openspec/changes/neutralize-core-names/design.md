# Design

## Context

See proposal.md. `Simulation(compiler=...)` already lets a model package bring its own compiler, as
`models/mathematics/fractals/_rewriting_compiler.py` does, while `PassCompiler` still imports `QuantumCompiler` from
`engine/compile/passes` to report a quantum system. `_defaults.py` picks quantum solvers, and `_numpy.py`, `_numba.py`,
`_compile.py`, `_observables.py`, and `analysis/_symbolic.py` name quantum parts. The import contracts forbid the core,
engine, analysis, and experiments from importing the models.

## Goals / Non-Goals

**Goals:**
- A core with no field vocabulary, and one way to do each thing.
- Unchanged trajectories, apart from rounding.

**Non-Goals:**
- New capabilities or new models.
- Aliases for the old names: the library is pre-1.0 and the project keeps one way.
- `HamiltonianInteraction` and `EulerLagrangeInteraction`, which the maintainer ruled pass the core rule. `hill` and
  `logistic` stay as named functions in general use.

## Decisions

- **Decided on 2026-11-02.** Only true particle groups take `StateSpec(role='position')`, with the first derivative of a
  position-role state as its velocity; other groups stop being read as agent snapshots, and each changed default is
  reported. Also renamed: `MatchSpec.price` and its reporter to `value`, `budget` in matching to `allowance`,
  `ConeNeighborhood(heading=)` to `direction=`, `StateSpec(role='heading')` to `role='direction'`, and the local
  `temperature` names in the engine and analysis. `limit` and `clearing` stay.

### Sampling reads an objective at an intensity
`ChoiceInteraction` and `ImitationInteraction` already call the same idea `intensity`, so acceptance is
`exp(-intensity * change)`. `cooling` multiplies one over the intensity, so a schedule keeps its meaning.
`models/physics/common` holds a thin sampling subclass with a `temperature` parameter and an `energy` argument, which
gives the core `1 / temperature`, so sweeps of `<label>.temperature` keep working. Alternative: a neutral `scale` equal to
the temperature, rejected because it gives two words for one idea beside `intensity`.

### Matching
Offer algorithms keep their behaviour and take operation names. `GreedyMatchingSpec` merges into `PartnerSpec`, whose
feasibility gains the offer limits and quantities greedy matching reads, keeping draws identical where a test can check.

### One bilinear law
`BilinearLaw(target, first, second, rate)` with a signed rate replaces both laws, in `EncounterInteraction`, the backends,
and the kinetics solver.

### Collisions on encounters
`EncounterInteraction(cross_section=, factor=)` lowers to the existing collision kernel. The two-velocity limit stays with
its error. Boltzmann, BGK, and Enskog keep their names as models.

### Spreading
A thin `TransitionInteraction` subclass in each field using it, in `<field>/common/<family>` if several packages share it.

### Quantum moves as a compiler plugin
Everything quantum moves to `models/physics/common/quantum`. Engine code naming quantum becomes general hooks: the default
solver comes from the compiler a system names, `PassCompiler` reports a law it cannot lower by the law's declared compiler
instead of importing `QuantumCompiler`, and the backends' quantum branches move behind the plugin's kernel. If a backend
branch has no general hook, the task stops and reports rather than leaving quantum names in the engine. `qutip` stays a
runtime dependency unless only models use it, in which case the dependency placement rule makes it a `quantum` extra.

### Earth to physics
`climate`, `oceanography`, and `seismology` become siblings of `fluids`. Catalog names stay, so factories keep working.

## Risks / Trade-offs

- [Many models change in one change] → Mechanical renames per group, each with the full model tests, and no other agents
  in those packages meanwhile.
- [The quantum move is the largest part and may need hooks the engine lacks] → It is its own last group and can stop with a
  report without blocking the renames.
- [Open changes name old paths] → Updated in the same change.

## Migration Plan

Pre-1.0 with no aliases: the changelog commits are marked breaking. Each group merges with its tests green.
