# Kinds of dynamics

This page shows where the classes of each kind of dynamics live, and how `prepare` turns the laws of a system into a
problem that a [solver](../glossary.md#solver) runs. A kind of dynamics is the form of the change, such as rates in
continuous time, local rules on a lattice, or operators on a quantum state.

The tags of a system name its kind. Here the SIR epidemic of 990 susceptible and 10 infected people runs with its
default settings, 160 days in steps of one day.

```python
from skcomplex.systems.biology.epidemiology import SIRSystem

system = SIRSystem()
trajectory = system.prepare().simulate()
sorted(system.get_tags().laws), sorted(system.get_tags().time), trajectory.get_values('R.x')[-1].round(1)
```

The tags read `['rate']` and `['continuous']`, so the system returns rates in continuous time and `prepare` picks a
solver of ordinary differential equations. After 160 days, 941.2 people have recovered.

Some kinds of dynamics, such as lattice dynamics or operator dynamics, serve few systems. None of them is a package of
its own. Its laws, elements, and spaces are components of the core, and its compiler passes, solvers, and visuals are
parts of the engine, beside those of every other kind.

## Where each kind lives

The table below finds the classes that a kind of dynamics needs. The column "Core" names the components you build a
system from, and the column "Engine" names what compiles and runs them.

| Kind | Core | Engine |
| --- | --- | --- |
| agents | [`UpdateLaw`](../core_components/laws.md#updatelaw) with a rule on a group, `pairs` | the agents passes, `MesaSolver` |
| lattices | [`UpdateLaw`](../core_components/laws.md#updatelaw) with a neighbourhood, [`relax`](../core_components/laws.md#relax), [`Element`](../core_components/elements.md#element) with `space=`, [`LatticeSpace`](../core_components/spaces.md#latticespace), `ObjectiveSpec`, `SamplerSpec` | `LatticeSolver`, `MonteCarloSolver` |
| finitely generated groups | `CayleySpace` and `MemoryNeighborhood` of [`skcomplex.systems.computer_science.common.automata`][skcomplex.systems.computer_science.common.automata] | `CayleyVisual` of the same package |
| rewriting | [`ProductionLaw`](../core_components/general.md#productionlaw), [`SubstitutionLaw`](../core_components/general.md#substitutionlaw), [`WordElement`](../core_components/general.md#wordelement), [`PatchElement`](../core_components/general.md#patchelement) of `skcomplex.systems.general.rewriting` | the rewriting compiler, `RewritingSolver`, `GeometryVisual` |
| operator | `local_operator`, `dagger`, and the `GeneratorLaw`, `DissipatorLaw`, and `HilbertSpec` of the [quantum package](quantum_systems.md) | the operator compiler, and `UnitarySolver`, `LindbladSolver`, and `JumpTrajectoriesSolver` of the quantum package |
| fields | [`Element`](../core_components/elements.md#element) with `space=`, `FiniteVolumeSpec`, [`CellComplexSpace`](../core_components/spaces.md#cellcomplexspace) of a mesh, `laplacian` and the other operators | the fields passes, `IMEXEulerSolver`, `CrankNicolsonSolver`, `SSPRKSolver` |
| kinetics | [`RateLaw`](../core_components/laws.md#ratelaw) of [`bilinear`](kinetic_systems_of_active_particles.md#the-bilinear-form), `moment` | the encounter passes, `KineticSolver` |
| fractional | [`fractional`](../core_components/laws.md#fractional)`(x, alpha)` with `alpha` in `(0, 1)`, [`NoiseLaw`](../core_components/laws.md#noiselaw)`(..., hurst=)` | the fractional passes, `FractionalABMSolver`, `GrunwaldLetnikovSolver`, `FractionalEulerMaruyamaSolver` |
| topology | `TopologyRequestSpec` | the topology passes |
| particles | a vector position `x` with laws of `x.diff(t, 2)`, `neighbors` within a `RadiusNeighborhood`, [`GradientInteraction`](../core_components/general.md#gradientinteraction) | `VelocityVerletSolver`, `Yoshida4Solver`, `LammpsSolver` |
| control | allowed values in `Annotated`, [`ObjectiveLaw`](../core_components/laws.md#objectivelaw)`(cost, controls=...)` | `PontryaginSolver`, `CollocationSolver` |

`prepare` compiles a system into the problem of its form, and `simulate` runs it. You never name the compiler of a
form that the library ships. An element can name the compiler of its system, as the words and patches of
`skcomplex.systems.general.rewriting` name the rewriting compiler. A system of operator elements compiles with the
operator compiler, and every other system with the pass compiler. A pass is one step of that compiler, and it
lowers each law to the problem of its form, such as a right-hand side of an ordinary differential equation.

A package of your own can bring a form of its own. Its compiler derives from `BaseCompiler` of
`skcomplex.core.engine.compile`, with passes on `BasePass` and lowerings on `BaseLowering`. An element or a value of
the package names it, as `HilbertSpec` names the operator compiler, and `prepare` chooses it by itself. This is the one
hook to the compiled form. The compiler has two main public methods.

- `prepare(system)` validates its settings and the system, and sets `report_`, the issues it found.
- `run(initial_state)` raises the errors of the report, sets `problem_`, the problem, and sets `solver_`, the solver
  it names for the problem.

## Writing a law

This section writes a law class of your own and shows why a system cannot run it without a compiler that knows it. A
new law derives from [`BaseLaw`](../core_components/laws.md#baselaw). It is a frozen dataclass whose fields are its
arguments, and it implements three methods.

- `_get_targets` returns the states it changes.
- `_get_requirements` returns a `RequirementSpec` of the kinds of space it runs in.
- `_get_value` returns its value at given values of the symbols it reads.

Here a decay law returns the rate `-k n` and runs in any space.

```python
from dataclasses import dataclass

from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BaseLaw, RequirementSpec
from skcomplex.core.errors import UnregisteredLawError
from skcomplex.core.systems import System


@dataclass(frozen=True)
class DecayLaw(BaseLaw):
    target: object
    rate: object
    label: str = ''

    def _get_targets(self):
        return (self.target,)

    @classmethod
    def _get_requirements(cls):
        return RequirementSpec()

    def _get_value(self, inputs):
        return float((-self.rate * self.target).subs(**inputs))


n, k = var('n k')
law = DecayLaw(n, k)
law.target, law.rate, law == DecayLaw(n, k)
```

The law targets `n` at the rate `k`, and equals any decay law with the same fields.

The passes of the compilers that the library ships are fixed lists of the engine, so no form takes a law of a new
class inside a system. An [`Interaction`](../core_components/interactions.md#interaction) that sets the law shows
this. `prepare` reports it as an `UnregisteredLawError` with the code `law-unregistered`.

```python continuation
atom = Element('atom', values={'n': 1.0, 'k': 1.0})
decay = Interaction('decay', roles={'target': 'atom'}, laws=lambda s: [DecayLaw(s.target.n, s.target.k)])
try:
    System(elements=[atom], interactions=[decay]).prepare()
except UnregisteredLawError as error:
    message = str(error)
assert '[law-unregistered]' in message
message
```

The message reads "The law class DecayLaw has no form the engine knows" for `atom`, and advises to write the law
with a class the core provides. A new kind of dynamics is a change to the engine, or a compiler of your own that an
element or a value of your package names.
