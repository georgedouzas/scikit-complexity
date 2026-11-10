# Kinds of dynamics

This page shows where the parts of each kind of dynamics live, and how `simulate` turns the laws of a system into a
problem that a [solver](../glossary.md#solver) runs. A kind of dynamics is the form of the change, such as rates in
continuous time, local rules on a lattice, or operators on a quantum state.

The tags of a system name its kind. Here the SIR epidemic of 990 susceptible and 10 infected people runs with its
default settings, 160 days in steps of one day.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import SIRSystem

system = SIRSystem()
trajectory = simulate(system)
sorted(system.get_tags().laws), sorted(system.get_tags().time), trajectory.get_values('R.x')[-1].round(1)
```

The tags read `['rate']` and `['continuous']`, so the system returns rates in continuous time and `simulate` picks a
solver of ordinary differential equations. After 160 days, 941.2 people have recovered.

Some kinds of dynamics, such as lattice dynamics or operator dynamics, serve few systems. None of them is a package of
its own. Its laws, elements, and spaces are parts of the core, and its compiler passes, solvers, and visuals are parts
of the engine, beside those of every other kind.

## Where each kind lives

The table below finds the parts that a kind of dynamics needs. The column "Core" names the parts you build a system
from, and the column "Engine" names what compiles and runs them.

| Kind | Core | Engine |
| --- | --- | --- |
| agents | [`RuleLaw`](../core_components/laws.md#rulelaw), [`MatchLaw`](../core_components/laws.md#matchlaw) | the agents passes, `MesaSolver` |
| lattices | [`LocalRuleLaw`](../core_components/laws.md#localrulelaw), [`LocalRuleInteraction`](../core_components/interactions.md#localruleinteraction), [`ArrayUpdateLaw`](../core_components/laws.md#arrayupdatelaw), [`CascadeLaw`](../core_components/laws.md#cascadelaw), [`ExtremalLaw`](../core_components/laws.md#extremallaw), [`GroupElement`](../core_components/elements.md#groupelement), [`LatticeSpace`](../core_components/spaces.md#latticespace), `ObjectiveSpec`, `SamplerSpec` | `LatticeSolver`, `MonteCarloSolver` |
| finitely generated groups | [`CayleySpace`](../core_components/spaces.md#cayleyspace) | `CayleyVisual` |
| rewriting | [`ProductionLaw`](../core_components/laws.md#productionlaw), [`SubstitutionLaw`](../core_components/laws.md#substitutionlaw), [`WordElement`](../core_components/elements.md#wordelement), [`PatchElement`](../core_components/elements.md#patchelement) | the rewriting compiler, `RewritingSolver`, `GeometryVisual` |
| operator | [`GeneratorLaw`](../core_components/laws.md#generatorlaw), [`DissipatorLaw`](../core_components/laws.md#dissipatorlaw), `HilbertSpec` | the operator compiler, `UnitarySolver`, `LindbladSolver`, `JumpTrajectoriesSolver` |
| fields | [`FieldElement`](../core_components/elements.md#fieldelement), [`FluxLaw`](../core_components/laws.md#fluxlaw), [`MeshSpace`](../core_components/spaces.md#meshspace), `laplacian` and the other operators | the fields passes, `IMEXEulerSolver`, `CrankNicolsonSolver`, `SSPRKSolver` |
| kinetics | [`BilinearLaw`](../core_components/laws.md#bilinearlaw), `moment` | the encounter passes, `KineticSolver` |
| fractional | `StateSpec(order=...)` in `(0, 1)`, [`NoiseLaw`](../core_components/laws.md#noiselaw)`(..., hurst=)` | the fractional passes, `FractionalABMSolver`, `GrunwaldLetnikovSolver`, `FractionalEulerMaruyamaSolver` |
| topology | `TopologyRequestSpec` | the topology passes |
| particles | `StateSpec(vector=True, order=2)`, [`GradientInteraction`](../core_components/interactions.md#gradientinteraction), [`PairInteraction`](../core_components/interactions.md#pairinteraction) | `VelocityVerletSolver`, `Yoshida4Solver`, `LammpsSolver` |
| control | `ParameterSpec(adaptive=True, bounds=...)` | `PontryaginSolver`, `CollocationSolver` |

`simulate` compiles a system into the problem of its form and runs it. You never name the compiler of a form that the
library ships. A system of operator elements compiles with the operator compiler, a system of words or patches with
the rewriting compiler, and every other system with the pass compiler. A pass is one step of that compiler, and it
lowers each law to the problem of its form, such as a right-hand side of an ordinary differential equation.

A package of your own can bring a form of its own. Its compiler derives from `BaseCompiler` of
`skcomplex.core.engine.compile`, with passes on `BasePass` and lowerings on `BaseLowering`. A system hands it to
`simulate` through the `compiler` setting, or tunes it as the class attribute `compiler`. This is the one hook to the
compiled form. The compiler has two main public methods.

- `prepare(system)` validates its settings and the system, and sets `report_`, the issues it found.
- `run(initial_state)` raises the errors of the report, sets `problem_`, the problem, and sets `solver_`, the solver
  it names for the problem.

## Writing a law

This section writes a law class of your own and shows why a system cannot run it without a compiler that knows it. A
new law derives from [`BaseLaw`](../core_components/laws.md#baselaw). It stores its arguments under their own names
and implements three methods.

- `_get_targets` returns the states it changes, which `prepare` sets as `targets_`.
- `_get_requirements` returns a `RequirementSpec` of the kinds of time and space it runs in.
- `_run` sets `value_`, its value at given values of the symbols it reads, which `prepare` sets as `reads_`.

Here a decay law returns the rate `-k n` and runs only in [`ContinuousTime`](../core_components/spaces.md#continuoustime).

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import BaseLaw
from skcomplex.core.components.spaces import ContinuousTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RequirementSpec, StateSpec


class DecayLaw(BaseLaw):
    def __init__(self, target, rate, *, label=''):
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _get_targets(self):
        return (self.target,)

    @classmethod
    def _get_requirements(cls):
        return RequirementSpec(time=(ContinuousTime,))

    def _run(self, inputs):
        self.value_ = float((-self.rate * self.target).subs(**inputs))


n, k = var('n k')
law = DecayLaw(n, k).prepare('atom')
law.targets_, sorted(map(str, law.reads_)), law.owner_, law.run({'n': 2.0, 'k': 0.5}).value_
```

The law targets `n`, reads `k` and `n`, belongs to `atom`, and returns -1.0 at `n = 2` and `k = 0.5`. It checks,
prepares, and runs on its own.

The passes of the compilers that the library ships are fixed lists of the engine, so no form takes a law of a new
class inside a system. An [`Element`](../core_components/elements.md#element) that returns the law shows this.
`simulate` reports it with the code `laws-mixed`.

```python continuation
class NucleusElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0, k: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.n = n
        self.k = k

    def _get_laws(self, s):
        return [DecayLaw(s.n, s.k)]


try:
    simulate(CompositeSystem(elements=[NucleusElement('atom')]), until=1)
except ValidationError as error:
    message = str(error)
assert '[laws-mixed]' in message
message
```

The message reads "No form takes the laws of this system together (DecayLaw: atom)" and advises to move the
components of each kind of law into their own nested system. For a law of a new class that advice does not help,
since no form takes it at all. A new kind of dynamics is a change to the engine, or a compiler of your own passed
through the `compiler` setting.
