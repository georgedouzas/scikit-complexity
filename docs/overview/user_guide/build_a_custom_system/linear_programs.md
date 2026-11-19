# Linear programs

This page sets some values of a system to the optimum of a linear program, as flux balance analysis sets the fluxes of
a cell. A linear program finds the least value of a linear objective over variables that satisfy linear equalities and
inequalities. An [`ObjectiveLaw`](../core_components/laws.md#objectivelaw) linear in its targets, with
[`ConstraintLaw`](../core_components/laws.md#constraintlaw) relations on them, is solved as a linear program, and the
targets take an optimal solution at every time of a run. HiGHS solves the program through `highspy`, which the default
install holds.

## A first program

This section routes traffic over two roads and makes the total flow as large as it can be. The direct road carries at
most its capacity of 3, and the detour at most 4. The [`Element`](../core_components/elements.md#element) holds both
flows, and the interaction `Routing` gives the objective, minus the total flow, and one constraint per bound. A
constraint names the value it bounds in `target`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, ObjectiveLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Routing(Interaction):
    def __init__(self, label, *, routes: Role = 'flows', capacity=3.0):
        super().__init__(label)
        self.routes = routes
        self.capacity = capacity

    def _prepare(self, s):
        super()._prepare(s)
        direct, detour = s.routes.direct, s.routes.detour
        self.laws_ = [
            ObjectiveLaw(-(direct + detour), [direct, detour]),
            ConstraintLaw(direct <= s.capacity, target=direct),
            ConstraintLaw(detour <= 4, target=detour),
            ConstraintLaw(direct >= 0, target=direct),
            ConstraintLaw(detour >= 0, target=detour),
        ]


routes = Element('flows', values={'direct': 0.0, 'detour': 0.0})
system = System(elements=[routes], interactions=[Routing('routing')]).prepare()
trajectory = system.simulate(until=0.0)
trajectory.get_values('flows.objective_0')[0], trajectory.get_values('flows.direct')[0]
```

The objective reaches -7.0, so the total flow is 7, three on the direct road at its capacity and four on the detour. A
program follows these rules.

- The constraints are Sage relations, `==`, `<=`, or `>=`, linear in the targets, each naming the value it bounds in
  `target`. A coefficient of a target reads no state, and the other terms may read states, parameters, and the time. A
  relation in one target is a bound of that target.
- The objective is minimised, so a value to make as large as it can be, such as a flow or a growth, is minimised with
  its sign changed. It is linear in the targets or a sum of their absolute values. A list of objectives solves in
  order, and each optimum is kept within a relative 1e-9 while the next one solves, such as the smallest total flux at
  the greatest growth.
- `active` is a condition. Where it is false the solve is skipped and the targets are zero.
- `infeasible` is `'raise'`, `'zero'`, or `'hold'`. `InfeasibleError` names the program, the time, and the constraints
  and variables of an irreducible infeasible subset, a smallest set of constraints that cannot hold together.
- A run records the targets under their paths, and the objective values under `<path>.objective_<k>`, the path being
  that of the element that holds the targets. With `duals=True` it also records the duals of the constraints under
  `<path>.dual_<k>`. The dual of a constraint is the change of the first objective with its right side, such as the
  shadow price of a metabolite in flux balance analysis.

## Duals

This section reads how much the optimum gains from a looser constraint. `ObjectiveLaw(..., duals=True)` records the dual
of each constraint of its interaction, in the order of the constraints.

```python continuation
class PricedRouting(Routing):
    def _prepare(self, s):
        super()._prepare(s)
        direct, detour = s.routes.direct, s.routes.detour
        self.laws_[0] = ObjectiveLaw(-(direct + detour), [direct, detour], duals=True)


priced = System(elements=[routes], interactions=[PricedRouting('routing')]).prepare().simulate(until=0.0)
[float(priced.get_values(f'flows.dual_{k}')[0]) for k in range(4)]
```

The duals are -1.0, -1.0, 0.0, and 0.0. Each unit of capacity on either road lowers the objective by one, a unit more of
total flow, while the lower bounds at zero hold no flow back.

## Merged programs

This section adds a constraint to a program from another interaction of the system. The constraints whose targets are
the targets of an objective, from any interaction, join its program. So an
[`Interaction`](../core_components/interactions.md#interaction) adds a bound to a program by returning a constraint on
the value it bounds, and the tighter of two bounds holds. Here a road closure limits the detour to 1.

```python continuation
class Closure(Interaction):
    def __init__(self, label, *, routes: Role = 'flows', limit=1.0):
        super().__init__(label)
        self.routes = routes
        self.limit = limit

    def _prepare(self, s):
        super()._prepare(s)
        detour = s.routes.detour
        self.laws_ = [ConstraintLaw(detour <= s.limit, target=detour)]


closed = System(elements=[routes], interactions=[Routing('routing'), Closure('closure')]).prepare()
closed.simulate(until=0.0).get_values('flows.objective_0')[0]
```

The objective rises to -4.0, a total flow of 4, three on the direct road and one on the detour. The bound 1 of the
closure is tighter than the bound 4 of the routing, so it holds.

## Switches

This section turns a part of a program on or off, such as a gene knock-out in a metabolic network. A switch is a
parameter that a bound reads, such as `v <= 1000 * either(g1 > 0.5, g2 > 0.5)`. The capacity of the direct road acts as
such a switch, and setting it to 0 between runs closes the road.

```python continuation
system.set_params(routing__capacity=0.0).prepare().simulate(until=0.0).get_values('flows.objective_0')[0]
```

The objective rises to -4.0, the flow of the detour alone. A knock-out sets the parameter to 0 in one of three ways.

- At construction, such as `Routing('routing', capacity=0.0)`.
- Between runs, with `set_params`, as above.
- During a run, by an [`UpdateLaw`](../core_components/laws.md#updatelaw) of an interaction that reads a Boolean
  node.

## The forms of a program

This section explains where the program solves in systems of each form. A linear objective does not decide the form of
the system that holds it. The program solves wherever the form needs it. Here a store fills at the total flow of the
roads, once with a rate and once with a step.

```python continuation
from skcomplex.core.components.laws import RateLaw, UpdateLaw

store = Element('store', values={'stock': 0.0})
fill = Interaction(
    'fill',
    roles={'store': 'store', 'routes': 'flows'},
    laws=lambda s: [RateLaw(s.store.stock.diff(s.t), s.routes.direct + s.routes.detour)],
)
step = Interaction(
    'step',
    roles={'store': 'store', 'routes': 'flows'},
    laws=lambda s: [UpdateLaw(s.store.stock, s.store.stock + s.routes.direct + s.routes.detour)],
)
filled = System(elements=[routes, store], interactions=[Routing('routing'), fill]).prepare()
stepped = System(elements=[routes, store], interactions=[Routing('routing'), step]).prepare()
flowing = filled.simulate(until=2.0, dt=1.0).get_values('store.stock')
flowing.round(3).tolist(), stepped.simulate(until=2).get_values('store.stock').tolist()
```

Both stores hold 7 after one unit of time and 14 after two, since the roads carry 7 at every time.

- Beside rate laws, the system flows in continuous time, and the drift solves the program first at each evaluation.
  This is the direct approach. Its right side is only piecewise smooth, so fixed steps, such as `RK4Solver`, suit it
  better than an adaptive integrator.
- Beside update laws, the system steps in discrete time, and each step solves the program first.
- Alone, each frame solves the program from the current parameters.
- In a nested system that steps beside a sibling with rate laws, the program solves at every synchronisation point of
  `CoSimulationSolver`, whatever the length of the interval, and the sibling reads its targets as held inputs. This is
  the static optimisation approach.
- In a group on the sites of a lattice, the program solves per member where `active` holds, each with its own warm
  start.

## Issues of linear programs

This section lists the issues that stop a program before it runs. `prepare` checks a program first and raises a
`ValidationError` that lists each issue. An inequality without its `target` is one of them, since a program adds the
bounds of one target from every interaction.

```python continuation
from skcomplex.core.errors import ValidationError


class OpenClosure(Closure):
    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [ConstraintLaw(s.routes.detour <= s.limit)]


try:
    System(elements=[routes], interactions=[Routing('routing'), OpenClosure('closure')]).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the inequality of `closure` holds in a linear program, so it names the value it bounds in
`target`, such as `ConstraintLaw(flux >= -uptake, target=flux)`. The other issues are these, with their codes.

- `linear-program-form`: a relation is not linear in the targets, or a coefficient of a target reads a state.
- `linear-program-objectives`: two interactions give objectives to one program. One `ObjectiveLaw` with a list of
  objectives holds them instead.
- `linear-program-variable`: a target of the program is bounded by no constraint, so the program has no bounded
  optimum. A `ConstraintLaw` that bounds it, such as `ConstraintLaw(v >= 0, target=v)`, fixes it.

## Elementary modes

The analysis `elementary_modes` enumerates exactly the generators of the steady-state cone of a program, with the Parma
Polyhedra Library. The steady-state cone holds the flux vectors that leave every internal metabolite unchanged. The
kind `'modes'` gives the elementary modes, and the kind `'pathways'` the extreme pathways. The number of modes grows
combinatorially, so the enumeration stops after `timeout` seconds with `EnumerationLimitError`.

## Metabolism

The metabolic systems of the catalog, such as `FluxBalanceSystem`, `DynamicFBASystem`, `CommunityFBASystem`, and
`CometsSystem`, are built from the components above. A metabolic network is an element whose data are its reactions. It
has one algebraic state per reaction, one adaptive switch per gene, and one lower and one upper bound parameter per
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
