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

- **Decided on 2026-11-02, after the renames.** Removing the name guess must not lose analysis data: group states count as
  series, and phases take `role='phase'`, so alignment reads them. Analyses of particle models keep their previous results
  and shapes where the guess gave a correct reading. `IrreversibilityAnalysis` takes neutral arguments, and the remaining
  field words in names go.

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

### Operator dynamics stay in the core, named for their mathematics
A linear operator generating an evolution, a Lindblad term of a Markovian master equation, and local operators on a tensor
product are general mathematics, so they pass the core rule, and `qutip` stays a runtime dependency. Decided on
2026-11-02 in place of moving them to the models, which would need new engine hooks for no change in behaviour. The
names become `Operator*`, docstrings state the operation, and physics keeps qubits, spins, and modes in
`models/physics/common/quantum`. `GeneratorLaw` still asks for a Hermitian operator, and classical generators usually are
not, so non-Hermitian generators are a recorded limit that a later change can lift.

### Earth to physics
`climate`, `oceanography`, and `seismology` become siblings of `fluids`. Catalog names stay, so factories keep working.

## Risks / Trade-offs

- [Many models change in one change] → Mechanical renames per group, each with the full model tests, and no other agents
  in those packages meanwhile.
- [Open changes name old paths] → Updated in the same change.

## Migration Plan

Pre-1.0 with no aliases: the changelog commits are marked breaking. Each group merges with its tests green.
