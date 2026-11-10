# Stocks and flows

This page builds stocks that change only through flows, as in system dynamics, checks that what flows is conserved, and
finds the feedback loops that drive them. A stock is an amount that accumulates, such as the people of a town or the
water in a bath, and a flow is a rate at which an amount enters or leaves a stock. In the library a stock is a state
tagged with a quantity, such as `StateSpec(quantity='people')`.

## Stocks and their flows

This section builds a population whose only changes are births and deaths. A
[`TransferLaw`](../core_components/laws.md#transferlaw) moves an amount of the quantity from one stock to another, and
it exchanges the quantity with the environment through a [`Source`](../core_components/laws.md#source) and a
[`Sink`](../core_components/laws.md#sink). The population is an [`Element`](../core_components/elements.md#element)
with ten people, a birth rate of 3 percent a year and a death rate of 1 percent. The birth rate falls to zero between
100 and 200 people through `series(x, points, values)`, a table function that is piecewise linear between its points
and flat outside them.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import Sink, Source, TransferLaw
from skcomplex.core.components.laws.utils import series
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ObservableSpec, ParameterSpec, StateSpec


def birth_rate(s):
    return s.b * s.n * series(s.n, (0, 100, 200), (1.0, 1.0, 0.0))


class PeopleElement(Element):
    def __init__(
        self,
        label,
        *,
        n: Annotated[float, StateSpec(quantity='people')] = 10.0,
        b: Annotated[float, ParameterSpec()] = 0.03,
        d: Annotated[float, ParameterSpec()] = 0.01,
    ):
        super().__init__(label)
        self.n = n
        self.b = b
        self.d = d

    def _get_laws(self, s):
        return [
            TransferLaw(Source(), s.n, birth_rate(s), label='births'),
            TransferLaw(s.n, Sink(), s.d * s.n, label='deaths'),
        ]


crowd = CompositeSystem(elements=[PeopleElement('crowd')])
observed = [ObservableSpec('birth_rate', birth_rate, element='crowd')]
trajectory = simulate(crowd, until=10, dt=1.0, observables=observed)
round(float(trajectory.get_values('crowd.births')[-1]), 3), round(float(trajectory.get_values('crowd.deaths')[-1]), 3)
```

After ten years the births run at 0.366 people a year and the deaths at 0.122. Trajectories record every flow under
`<element path>.<label>`, such as `crowd.births`. What the amount of a `TransferLaw` means depends on the laws around
it.

- In a system of rate laws the amount is a rate.
- In a system of update laws it is moved once per step.
- With `rate=...` it is the size of a jump with that propensity.

An intermediate expression, such as the birth rate, is a function of the namespace that the laws call.
`ObservableSpec(label, function, element=element)` records the same function as the observable `<element>.<label>`,
which Rerun logs under the element.

```python continuation
round(float(trajectory.get_values('crowd.birth_rate')[-1]), 3)
```

The observable `crowd.birth_rate` reads 0.366 at ten years, the same value as the flow `crowd.births`.

## Conservation

This section checks that the stocks change only through flows and shows how much of each quantity enters from
outside. A rate or update law outside flows that changes a tagged state raises `ConservationError` when the system
compiles. The analysis `symbolic(system, kind='conservation')` returns the net inflow of every quantity from the
environment, after checking that the internal flows cancel.

```python continuation
from skcomplex.core.analysis import loops, symbolic
from skcomplex.core.engine.runtime import ConservationObserver

symbolic(crowd, kind='conservation').inflows
```

The inflow of `people` is the birth rate times the population, `b_crowd*n_crowd` times the table of the birth rate,
minus the deaths `d_crowd*n_crowd`. The compiler accumulates this inflow in the hidden state `__boundary.<quantity>`,
and the observer `ConservationObserver` checks at every frame that each total changed only by it. The analysis
`loops(trajectory, kind='flow-matrix', frame=frame)` gives the transaction-flow matrix of a frame of a trajectory, with
one row per flow and one column per holder.

```python continuation
monitored = simulate(crowd, until=10, dt=1.0, observers=[ConservationObserver()])
loops(monitored, kind='flow-matrix').matrix.sum(axis=1).tolist()
```

The rows of the matrix sum to `[0.0, 0.0]`, since every flow leaves one holder and enters another, the environment
included.

`distributed_delay(value, time, order, conserved=True)` is a material delay of any order. It is a chain of hidden
stocks in series, and it returns their outflow. With `conserved=False` it is an information smoothing. It adds hidden
stocks `<element path>.<kind>_delay<order>_<index>` to the rate laws of single elements and interactions, and they
start in equilibrium with their input. The `DELAY1` and `DELAY3` functions of system dynamics are
`distributed_delay(value, time, 1)` and `distributed_delay(value, time, 3)`, and `SMOOTH` is
`distributed_delay(value, time, conserved=False)`.

## Feedback loops

This section finds the feedback loops of a system and ranks them over a run. A feedback loop is a closed chain of
dependencies, through which a change in a state comes back to it. It is reinforcing when the change returns with the
same sign and balancing when it returns with the opposite sign. `loops(system)` derives the signed dependency graph of
a system from its laws and lists its feedback loops with their polarity.

```python continuation
feedback = loops(crowd)
[(polarity, ' -> '.join(nodes)) for nodes, polarity in zip(feedback.loops, feedback.polarities, strict=True)]
```

The population has a reinforcing loop through its births and a balancing loop through its deaths.
`loops(trajectory, kind='dominance')` ranks the loops by their gains at every frame of a trajectory.

```python continuation
dominance = loops(trajectory, kind='dominance')
dominance.polarities[int(dominance.dominant[-1])]
```

The reinforcing loop dominates at the last frame, since the births still outrun the deaths at ten years.

An edge of the graph goes from one state to another when the rate of the second depends on the first, with the sign of
the entry of the Jacobian. In systems that declare flows, the edges also go through the flows. They go from a state to
every flow it drives, and from a flow to its sink with sign `+` and to its source with sign `-`.

The graph is that of any system with a vector field or a map, not only of stocks and flows. At one prey and one
predator, prey growth closes a reinforcing self-loop, predator death a balancing one, and predation a balancing loop
through both populations.

```python continuation
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

predation = loops(LotkaVolterraSystem(), at=[1.0, 1.0])
dict(zip(predation.loops, predation.polarities, strict=True))
```

The loops read `('prey.x',)` reinforcing, `('predators.x',)` balancing, and `('predators.x', 'prey.x')` balancing.

## Conserved quantities between agents

This section moves money and claims between agents so that the totals stay constant, and lets the run check it. Agents
move conserved quantities between each other inside rules. `transfer(source, sink, amount, state)` moves an amount
between two agents at once. A negative holding is an amount owed, and the total holdings stay constant. Here three
traders in a [`GroupElement`](../core_components/elements.md#groupelement) each hold 10 units of cash, and at every
step each one lends a unit to the first trader in exchange for a claim, through a
[`RuleLaw`](../core_components/laws.md#rulelaw). A run checks that an expression keeps its first value with
`simulate(system, invariants={name: tolerance})`, where the name is an observable of the run, here the total net worth.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.laws.utils import transfer
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ObservableSpec, StateSpec


def net_worth(agent):
    return agent.cash + agent.claims - agent.equity


def lend(agent, system):
    borrower = system.get_agents('traders')[0]
    if borrower != agent:
        transfer(agent, borrower, 1.0, 'cash')
        transfer(borrower, agent, 1.0, 'claims')


class TraderElement(Element):
    def __init__(
        self,
        label,
        *,
        cash: Annotated[float, StateSpec(quantity='money')] = 10.0,
        claims: Annotated[float, StateSpec(quantity='claims')] = 0.0,
        equity: Annotated[float, StateSpec()] = 10.0,
    ):
        super().__init__(label)
        self.cash = cash
        self.claims = claims
        self.equity = equity

    def _get_laws(self, s):
        return [RuleLaw(lend)]


balance = ObservableSpec('net_worth', net_worth, group='traders', reduce='sum')
market = CompositeSystem(elements=[GroupElement('traders', element=TraderElement, size=3)])
traders = simulate(market, until=3, observables=[balance], invariants={'net_worth': 1e-9})
float(traders.get_agents('traders', 3).values['cash'].sum())
```

The cash of the three traders still sums to 30.0 after three steps, and the net worth keeps its first value within
`1e-9`. At every frame the run compares the observable with its value at the first frame. It raises
`ConservationError`, naming the observable and the first frame that breaks it. Removing an agent that still holds a
tagged state also raises `ConservationError`, until its holdings are settled with transfers.

## Translating systems

This section turns a Vensim or Stella file into a system that runs and that the analyses read.
`SystemDynamicsTranslator(path)` reads a Vensim (`.mdl`) or XMILE (`.xmile`, `.xml`, `.stmx`) file and writes the
source of a system module. PySD reads the file. The translator writes `source_` from the abstract model of PySD, and
`settings_` from the control section of the file. The source is saved and imported like any module. The file below
holds a population with births and deaths.

```python
import importlib
import sys
import tempfile
from pathlib import Path

from skcomplex.core.analysis import loops
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerSolver
from skcomplex.core.engine.translators import SystemDynamicsTranslator

text = """
Population = INTEG(Births - Deaths, 100) ~ People ~ |
Births = Population * Birth Fraction ~ People/Year ~ |
Deaths = Population / Lifetime ~ People/Year ~ |
Birth Fraction = 0.03 ~ 1/Year ~ |
Lifetime = 70 ~ Years ~ |
Doubling Share = Population / 200 ~ Dmnl ~ |
INITIAL TIME = 0 ~ Years ~ |
FINAL TIME = 100 ~ Years ~ |
TIME STEP = 0.25 ~ Years ~ |
SAVEPER = 1 ~ Years ~ |
"""
folder = Path(tempfile.mkdtemp())
(folder / 'people.mdl').write_text(text)
translator = SystemDynamicsTranslator(folder / 'people.mdl').prepare().run()
(folder / 'translated_people.py').write_text(translator.source_)
sys.path.insert(0, str(folder))
module = importlib.import_module('translated_people')
translator.settings_.until, module.DT
```

The run lasts 100 years with steps of 0.25 years, as the control section of the file says. The births and deaths of
the population are its only flows, so the module defines two things.

- The system class `PeopleSystem`, named after the file, a subclass of
  [`StockFlowSystem`](../core_components/general.md#stockflowsystem) from `skcomplex.systems.general.dynamics`. Its
  keyword arguments are the constants, with the values of the file as defaults. It returns the stocks from
  `_get_stocks`, the constants from `_get_parameters`, and the flows from `_get_flows`. Its tuned run settings are the
  length of the run, the saving interval, and one observable per auxiliary.
- The constants `START`, `UNTIL`, `DT`, and `SAVE_EVERY`.

The class runs like any system, with its constants as model parameters.

```python continuation
system = module.PeopleSystem(birth_fraction=0.04)
trajectory = simulate(system)
trajectory.get_values('people.births')[0], trajectory.get_values('people.doubling_share')[0]
```

With a birth fraction of 0.04, the births start at 4.0 people a year and the doubling share at 0.5, a population of
100 against the 200 of its definition. The loops of the translated system are those of the population above.

```python continuation
loops(system).polarities
```

The block returns `['reinforcing', 'balancing']`, the births and the deaths. The class is a subclass of the recipe,
and its model parameters are the constants of the file.

```python continuation
from skcomplex.systems.general.dynamics import StockFlowSystem

issubclass(module.PeopleSystem, StockFlowSystem), list(system.get_params())
```

The block returns `True` and the parameters `['birth_fraction', 'label', 'lifetime']`. A file whose stocks also change
outside flow variables, or are subscripted, keeps the shape of a custom system. It gets one element class, whose states
are the stocks and whose parameters are the constants, and a system class on
[`BaseSystem`](../core_components/systems.md#basesystem) that builds the element and its interactions.

Time in the module runs from zero, and `Time` in the file becomes `t + START`. The parts of the file become these.

- A flow variable that leaves one stock and enters another is a flow between them. A flow variable of one stock is a
  flow with the environment, a [`TransferInteraction`](../core_components/interactions.md#transferinteraction) with
  `Source()` or `Sink()`. The trajectory reports each flow as `<element>.<flow>`.
- The rest of the net rate of a stock is a [`RateInteraction`](../core_components/interactions.md#rateinteraction).
  When it has the form `(goal - stock) / time` or `(goal - stock) * rate`, its rate is `(goal - stock) / time` of a
  goal function and a time function.
- An auxiliary is a private function `_name(s, t)` of the module that the laws call, and an `ObservableSpec` under its
  snake-case name.
- A lookup is a `series` of its input through its points. The built-in functions are `where`, `step`, `ramp`, `pulse`,
  `pulse_train`, `distributed_delay` for the delays and smoothings, `delayed`, `minimum`, `maximum`, `clip`, and the
  draws `uniform` and `normal`.
- A system whose stocks are all subscripted over one range is a `GroupElement` with one member per subscript. `SUM`
  over the range is `total`, and a read of one subscript is a `total` over that member.

The default solver finds the exact solution. Vensim and Stella integrate with Euler steps by default. A run with
`solver=EulerSolver(dt=module.DT)` and `dt=module.DT` reproduces their run step by step. A run with the default solver
differs from it by the Euler error of the time step of the file.

```python continuation
stepped = simulate(system, until=module.UNTIL, dt=module.DT, solver=EulerSolver(dt=module.DT))
final = stepped.get_values('people.population')[-1]
round(float(final), 3)
```

The Euler run ends with 1297.726 people after 100 years.

One `ValidationError` lists every construct that does not translate, with its variable. These are `GAME`, `ALLOCATE`,
the vector functions, `TREND`, `FORECAST`, `SAMPLE IF TRUE`, data read from spreadsheets, macros, variables subscripted
over more than one range, systems that mix subscripted and unsubscripted stocks, delays and smoothings whose initial
value differs from their input, and `DELAY FIXED` of anything other than a stock.

The library sets these limits on what a translated module does.

- The core takes random draws only in update and renewal laws. A module with a random function translates, but raises
  `IncompatibleLawsError` when its system is prepared.
- A run cannot record an auxiliary that reads `DELAY FIXED`, or a delay that no law reads, so `observables` leaves it
  out.
- A trajectory records the flow of one member from the values of that member. So in a group, a flow variable that
  reads `SUM` or one named subscript stays in the rate of its stock, without a flow, and its value stays an observable.

The systems of the catalog built on flows cover several fields.

- Engineering processes: `BathtubSystem`, `RiceMillingSystem`, `SolidWasteSystem`, `FoodReliefSystem`, and
  `IrrigationSystem`.
- Markets: `CarDealerSystem`, `CommodityCycleSystem`, and `BeerGameSystem`.
- Innovation: `BassDiffusionSystem`.
- Macroeconomics: `StockFlowConsistentEconomySystem`.
- Organizations: `SuccessToTheSuccessfulSystem`, `ShiftingTheBurdenSystem`, and `EscalationSystem`.
