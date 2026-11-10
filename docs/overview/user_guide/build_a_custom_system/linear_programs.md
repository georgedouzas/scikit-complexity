# Linear programs

This page sets some states of a system to the optimum of a linear program, as flux balance analysis sets the fluxes of
a cell. A linear program maximises a linear objective over variables that satisfy linear equalities and inequalities.
The [`LinearProgramLaw`](../core_components/laws.md#linearprogramlaw) sets algebraic states, its variables, to an
optimal solution. An algebraic state is one that a law sets directly at every evaluation, rather than integrating a
rate. HiGHS solves the program through `highspy`, which the default install holds.

## A first program

This section routes traffic over two roads and maximises the total flow. The direct road carries at most its capacity
of 3, and the detour at most 4. The [`Element`](../core_components/elements.md#element) declares both flows with
`StateSpec(algebraic=True)` and returns the program from `_get_laws`. Under a
[`StaticTime`](../core_components/spaces.md#statictime), a run solves the program once and gives one frame.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import LinearProgramLaw
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
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


system = CompositeSystem(elements=[RoutesElement('flows')], time=StaticTime())
trajectory = simulate(system, until=0.0)
trajectory.get_values('flows.objective_0')[0]
```

The objective reaches 7.0, three on the direct road at its capacity and four on the detour. A program follows these
rules.

- The constraints are Sage relations, `==`, `<=`, or `>=`, linear in the variables. A coefficient of a variable reads
  no state, and the other terms may read states, parameters, and the time. A relation in one variable is a bound of
  that variable.
- The objectives are `LinearObjectiveSpec`s, linear in the variables or a sum of their absolute values. They solve in
  order, and each optimum is kept within a relative 1e-9 while the next one solves.
- `active` is a condition. Where it is false the solve is skipped and the variables are zero.
- `infeasible` is `'raise'`, `'zero'`, or `'hold'`. `InfeasibleError` names the program, the time, the member, and the
  constraints and variables of an irreducible infeasible subset, a smallest set of constraints that cannot hold
  together.
- A run records the variables under their paths, and the objective values under `<path>.objective_<k>`. With
  `duals=True` it also records the duals of the constraints under `<path>.dual_<k>`. The dual of a constraint is the
  change of the first objective with its right side.

## Merged programs

This section adds a constraint to a program from another part of the system. The laws whose variables overlap, from
any element or interaction, form one program. Their variables unite, their constraints add, and at most one of them
holds objectives. So an [`Interaction`](../core_components/interactions.md#interaction) adds a bound to the program of
an element by returning a law over the variable it bounds, and the tighter of two bounds holds. Here a road closure
limits the detour to 1.

```python continuation
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.specs import RoleSpec


class ClosureInteraction(Interaction):
    def __init__(self, label, *, routes: Annotated[str, RoleSpec()], limit: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.routes = routes
        self.limit = limit

    def _get_laws(self, s):
        return [LinearProgramLaw([s.routes.detour], [s.routes.detour <= s.limit])]


closure = ClosureInteraction('closure', routes='flows')
closed = CompositeSystem(elements=[RoutesElement('flows')], interactions=[closure], time=StaticTime())
simulate(closed, until=0.0).get_values('flows.objective_0')[0]
```

The objective falls to 4.0, three on the direct road and one on the detour. The bound 1 of the closure is tighter than
the bound 4 of the element, so it holds.

## Switches

This section turns a part of a program on or off, such as a gene knock-out in a metabolic network. A switch is a
parameter that a bound reads, such as `s.v <= 1000 * either(s.g1 > 0.5, s.g2 > 0.5)`. The capacity of the direct road
acts as such a switch, and setting it to 0 at construction closes the road.

```python continuation
switched = CompositeSystem(elements=[RoutesElement('flows', capacity=0.0)], time=StaticTime())
simulate(switched, until=0.0).get_values('flows.objective_0')[0]
```

The objective falls to 4.0, the flow of the detour alone. A knock-out sets the parameter to 0 in one of three ways.

- At construction, as above.
- Between runs, with `set_params` of the element that holds it.
- For an adaptive parameter, by an [`UpdateLaw`](../core_components/laws.md#updatelaw) of an interaction that reads a
  Boolean node.

## The forms of a program

This section explains where the program solves in systems of each form. A linear program law does not decide the form
of the system that holds it. The program solves wherever the form needs it.

- Beside rate laws under continuous time, the system keeps the ODE form, and the drift solves the program first at each
  evaluation. This is the direct approach. Its right side is only piecewise smooth, so fixed steps, such as
  `RK4Solver`, suit it better than an adaptive integrator.
- Alone or beside update laws under discrete time, the system keeps the map form, and each step solves the program
  first.
- Alone under static time, the system has the static form, and each frame solves the program from the current
  parameters.
- In a discrete-time nested system beside a sibling with rate laws, the program solves at every synchronisation point
  of `CoSimulationSolver`, whatever the length of the interval, and the sibling reads its variables as held inputs.
  This is the static optimisation approach.
- In a group on the sites of a lattice, the program solves per member where `active` holds, each with its own warm
  start.

## Issues of linear programs

This section lists the issues that stop a program before it runs. `simulate` checks a program first and raises a
`ValidationError` that lists each issue with its code. A variable declared without `algebraic=True` is one of them.

```python continuation
from skcomplex.core.errors import ValidationError


class StockElement(Element):
    def __init__(self, label, level: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.level = level

    def _get_laws(self, s):
        return [LinearProgramLaw([s.level], [s.level <= 2], [LinearObjectiveSpec(s.level)])]


try:
    simulate(CompositeSystem(elements=[StockElement('stock')], time=StaticTime()), until=0.0)
except ValidationError as error:
    message = str(error)
message
```

The message says that the law sets `stock.level`, which is not an algebraic state, and asks to declare it with
`StateSpec(algebraic=True)`, under the code `linear-program-variable`. The codes are these.

- `linear-program-form`: a relation is not linear in the variables, or a coefficient of a variable reads a state.
- `linear-program-objectives`: two merged laws hold objectives.
- `linear-program-variable`: a variable is not declared with `StateSpec(algebraic=True)`.

A program under any other time gives the issue `law-time`.

## Elementary modes

The analysis `elementary_modes` enumerates exactly the generators of the steady-state cone of a program, with the Parma
Polyhedra Library. The steady-state cone holds the flux vectors that leave every internal metabolite unchanged. The
kind `'modes'` gives the elementary modes, and the kind `'pathways'` the extreme pathways. The number of modes grows
combinatorially, so the enumeration stops after `timeout` seconds with `EnumerationLimitError`.

## Metabolism

The metabolic systems of the catalog, such as `FluxBalanceSystem`, `DynamicFBASystem`, `CommunityFBASystem`, and
`CometsSystem`, are built from the parts above. A metabolic network is an element whose data are its reactions. It has
one algebraic state per reaction, one adaptive switch per gene, and one lower and one upper bound parameter per
reaction.

`ElementaryFluxModesSystem` runs on the default install. The systems built on the E. coli core model, which COBRApy
bundles, need the extra `metabolism`: `FluxBalanceSystem`, `DynamicFBASystem`, `RegulatoryFBASystem`,
`MetabolicPerturbationsSystem`, `CommunityFBASystem`, `DynamicMultispeciesFBASystem`, and `CometsSystem`. The command
`pip install 'scikit-complexity[metabolism]'` installs it. Without it they raise `MissingExtraError`.

The extra holds COBRApy, StrainDesign, and MICOM. COBRApy brings GLPK into the environment, which is under the GPL-3.0.
The package imports no module of GLPK, and it points COBRApy to the `hybrid` solver interface. The extra also holds
pandas below version 3, since COBRApy caps it.

## Isotope labelling

Only `Carbon13MFASystem` uses isotope labelling. It takes the atom transitions of the reactions, the isotopomer
fractions of a substrate, and the measured fragments. From these it simulates the mass isotopomer distributions by
elementary metabolite units. It fits the fluxes to the measurements from seeded starts, with 95% intervals and a
chi-square test. It is imported from `skcomplex.systems.biology.systems_biology`.
