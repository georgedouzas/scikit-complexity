# Linear programs

Set some states of a system to the optimum of a linear program, as flux balance analysis sets the fluxes of a cell.
`LinearProgramLaw` sets algebraic states, its variables, to an optimal solution, and HiGHS solves the program through
`highspy`, in the default install.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import LinearProgramLaw
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import LinearObjectiveSpec, ParameterSpec, StateSpec


class RoutesElement(Element):
    def __init__(
        self,
        label,
        direct: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        detour: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        capacity: Annotated[float, ParameterSpec()] = 3.0,
    ):
        super().__init__(label)
        self.direct = direct
        self.detour = detour
        self.capacity = capacity

    def _get_laws(self, s):
        constraints = [s.direct <= s.capacity, s.detour <= 4, s.direct >= 0, s.detour >= 0]
        objective = LinearObjectiveSpec(s.direct + s.detour)
        return [LinearProgramLaw([s.direct, s.detour], constraints, [objective])]


system = System(elements=[RoutesElement('flows')], time=StaticTime())
trajectory = Simulation(system).prepare().run(0.0).trajectory_
trajectory.values('flows.objective_0')[0]
```

The objective reaches 7, three on the direct route at its capacity and four on the detour. The rules of a program are
these:

- The constraints are Sage relations, `==`, `<=`, or `>=`, linear in the variables. A coefficient of a variable reads no
  state, and the other terms may read states, parameters, and the time. A relation in one variable is a bound of that
  variable.
- The objectives are `LinearObjectiveSpec`s, linear in the variables or a sum of their absolute values. They solve in
  order, and each optimum is kept within a relative 1e-9 while the next one solves.
- `active` is a condition. Where it is false the solve is skipped and the variables are zero.
- `infeasible` is `'raise'`, `'zero'`, or `'hold'`. `InfeasibleError` names the program, the time, the member, and the
  constraints and variables of an irreducible infeasible subset.
- A run records the variables under their paths, and the objective values under `<path>.objective_<k>`. With
  `duals=True` it also records the duals of the constraints under `<path>.dual_<k>`. The dual of a constraint is the
  change of the first objective with its right side.

## Merged programs

The laws whose variables overlap, from any element or interaction, form one program. Their variables unite, their
constraints add, and at most one of them holds objectives. So an interaction adds a bound to the program of an element
by returning a law over the variable it bounds, and the tighter of two bounds holds.

## Switches

A switch turns a part of a program on or off, such as a gene knock-out. It is a parameter that a bound reads, such as
`s.v <= 1000 * either(s.g1 > 0.5, s.g2 > 0.5)`. A knock-out sets the parameter to 0 in one of three ways:

- at construction,
- between runs, with `update_parameters`,
- for an adaptive parameter, by an `UpdateLaw` of an interaction that reads a Boolean node.

## The forms of a program

A linear program law does not decide the form of the system that holds it. The program solves wherever the form needs
it:

- Beside rate laws under continuous time, the system keeps the ODE form, and the drift solves the program first at each
  evaluation. This is the direct approach. Its right side is only piecewise smooth, so use fixed steps, such as
  `RK4Solver`, rather than an adaptive integrator.
- Alone or beside update laws under discrete time, the system keeps the map form, and each step solves the program
  first.
- Alone under static time, the system has the static form, and each frame solves the program from the current
  parameters.
- In a discrete-time nested system beside a sibling with rate laws, the program solves at every synchronisation point of
  `CoSimulationSolver`, whatever the length of the interval, and the sibling reads its variables as held inputs. This is
  the static optimisation approach.
- In a group on the sites of a lattice, the program solves per member where `active` holds, each with its own warm
  start.

## Issues of linear programs

`PassCompiler` reports these issues before it compiles:

- `linear-program-form`: a relation is not linear in the variables, or a coefficient of a variable reads a state.
- `linear-program-objectives`: two merged laws hold objectives.
- `linear-program-variable`: a variable is not declared with `StateSpec(algebraic=True)`.

A program under any other time gives the issue `law-time`.

## Elementary modes

`ElementaryModeAnalysis` enumerates exactly the generators of the steady-state cone of a program, with the Parma
Polyhedra Library. The kind `'modes'` gives the elementary modes, and the kind `'pathways'` the extreme pathways. The
number of modes grows combinatorially, so the enumeration stops after `timeout` seconds with `EnumerationLimitError`.

## Metabolism

The metabolic systems of the catalog, such as `flux_balance`, `dynamic_fba`, `community_fba`, and `comets`, are built
from the parts above. A metabolic network is an element whose data are its reactions. It has one algebraic state per
reaction, one adaptive switch per gene, and one lower and one upper bound parameter per reaction.

`elementary_flux_modes` runs on the default install. The systems built on the E. coli core model, which COBRApy bundles,
need the extra `metabolism`: `flux_balance`, `dynamic_fba`, `regulatory_fba`, `metabolic_perturbations`,
`community_fba`, `dynamic_multispecies_fba`, and `comets`. Install it with `pip install
'scikit-complexity[metabolism]'`. Without it they raise `MissingExtraError`.

The extra holds COBRApy, StrainDesign, and MICOM. COBRApy brings GLPK into the environment, which is under the GPL-3.0.
The package imports no module of GLPK, and it points COBRApy to the `hybrid` solver interface. The extra also holds
pandas below version 3, since COBRApy caps it.

## Isotope labelling

Only the system `carbon13_mfa` uses isotope labelling. It takes the atom transitions of the reactions, the isotopomer
fractions of a substrate, and the measured fragments. From these it simulates the mass isotopomer distributions by
elementary metabolite units. It fits the fluxes to the measurements from seeded starts, with 95% intervals and a
chi-square test. Select it with `create_system('carbon13_mfa')`.
