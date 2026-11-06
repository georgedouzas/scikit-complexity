# Stocks and flows

Build stocks that change only through flows, as in system dynamics, check that what flows is conserved, and find the
feedback loops that drive them. A stock is a state tagged with a quantity, such as `StateSpec(quantity='people')`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import Sink, Source, TransferLaw
from skcomplex.core.components.laws.utils import lookup
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ObservableSpec, ParameterSpec, StateSpec


def birth_rate(s):
    return s.b * s.n * lookup(s.n, [(0, 1.0), (100, 1.0), (200, 0.0)])


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


crowd = System(elements=[PeopleElement('crowd')])
observed = [ObservableSpec('birth_rate', birth_rate, element='crowd')]
trajectory = Simulation(crowd, dt=1.0, observables=observed).prepare().run(until=10).trajectory_
round(float(trajectory.values('crowd.births')[-1]), 3), round(float(trajectory.values('crowd.deaths')[-1]), 3)
round(float(trajectory.values('crowd.birth_rate')[-1]), 3)
```

A `TransferLaw(source, sink, amount, label=...)` moves an amount of the quantity from one stock to another. It exchanges
the quantity with the environment through `Source()` and `Sink()`. What the amount means depends on the laws around it:

- In a system of rate laws the amount is a rate.
- In a system of update laws it is moved once per step.
- With `rate=...` it is the size of a jump with that propensity.

Trajectories record every flow under `<element path>.<label>`, such as `crowd.births`.

An intermediate expression, such as the birth rate, is a function of the namespace that the laws call.
`ObservableSpec(label, function, element=element)` records the same function as the observable `<element>.<label>`,
which Rerun logs under the element. `lookup(x, points)` is a table function, piecewise linear between its points and
flat outside them.

## Conservation and loops

Check that the stocks change only through flows, and see how much of each quantity enters from outside.

```python continuation
from skcomplex.core.analysis import LoopAnalysis, SymbolicAnalysis
from skcomplex.core.engine.runtime import ConservationObserver

SymbolicAnalysis('conservation').prepare(crowd).run().inflows_
monitored = Simulation(crowd, dt=1.0, observers=[ConservationObserver()]).prepare().run(until=10).trajectory_
LoopAnalysis('flow-matrix').prepare(monitored).run().matrix_.sum(axis=1).tolist()
```

- A rate or update law outside flows that changes a tagged state raises `ConservationError` when the system compiles.
- `SymbolicAnalysis('conservation')` returns the net inflow of every quantity from the environment, after checking that
  the internal flows cancel.
- The compiler accumulates that inflow in the hidden state `__boundary.<quantity>`. The observer
  `ConservationObserver` checks at every frame that each total changed only by it.
- `LoopAnalysis('flow-matrix', frame=frame)` gives the transaction-flow matrix of a frame of a trajectory, with one row
  per flow and one column per holder. Its rows sum to zero.

`distributed_delay(value, time, order, conserved=True)` is a material delay of any order. It is a chain of hidden stocks
in series, and it returns their outflow. With `conserved=False` it is an information smoothing. It adds hidden stocks
`<element path>.<kind>_delay<order>_<index>` to the rate laws of single elements and interactions, and they start in
equilibrium with their input. The system dynamics names `delay1`, `delay3`, and `smooth` of
`skcomplex.core.components.laws.utils` are its cases.

`LoopAnalysis('loops')` derives the signed dependency graph of a system from its laws, and lists its feedback loops with
their polarity. `LoopAnalysis('dominance')` ranks the loops by their gains at every frame of a trajectory.

```python continuation
analysis = LoopAnalysis().prepare(crowd).run()
[(polarity, ' -> '.join(nodes)) for nodes, polarity in zip(analysis.loops_, analysis.polarities_, strict=True)]
dominance = LoopAnalysis('dominance').prepare(trajectory).run()
dominance.polarities_[int(dominance.dominant_[-1])]
```

An edge of the graph goes from one state to another when the rate of the second depends on the first, with the sign of
the entry of the Jacobian. In systems that declare flows, the edges also go through the flows: from a state to every
flow it drives, and from a flow to its sink, with sign `+`, and to its source, with sign `-`.

The graph is that of any system with a vector field or a map, not only of stocks and flows. At one prey and one
predator, prey growth closes a reinforcing self-loop, predator death a balancing one, and predation a balancing loop
through both populations.

```python continuation
from skcomplex.systems import create_system

predation = LoopAnalysis(at=[1.0, 1.0]).prepare(create_system('lotka_volterra')).run()
dict(zip(predation.loops_, predation.polarities_, strict=True))
```

## Conserved quantities between agents

Move money or goods between agents so that the totals stay constant, and let the run check it.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.laws.utils import transfer
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ObservableSpec, StateSpec


def net_worth(agent):
    return agent.cash + agent.claims - agent.equity


def lend(agent, system):
    borrower = system.agents('traders')[0]
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
market = System(elements=[GroupElement('traders', element=TraderElement, size=3)])
simulation = Simulation(market, observables=[balance], invariants={'net_worth': 1e-9})
traders = simulation.prepare().run(until=3).trajectory_
float(traders.agents('traders', 3).values['cash'].sum())
```

Agents move conserved quantities between each other inside rules. `transfer(source, sink, amount, state)` moves an
amount between two agents at once. A negative holding is an amount owed, and total holdings stay constant.

A run checks that an expression keeps its first value with `Simulation(invariants={name: tolerance})`, where the name is
an observable of the run. At every frame the run compares the observable with its value at the first frame. It raises
`ConservationError`, naming the observable and the first frame that breaks it. Removing an agent that still holds a
tagged state also raises `ConservationError`, until its holdings are settled with transfers.

## Translating systems

Turn a Vensim or Stella file into a system you can run and analyse. `SystemDynamicsTranslator(path)` reads a Vensim
(`.mdl`) or XMILE (`.xmile`, `.xml`, `.stmx`) file and writes the source of a system module.

```python
import importlib
import sys
import tempfile
from pathlib import Path

from skcomplex.core.analysis import LoopAnalysis
from skcomplex.core.engine.runtime import Simulation
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
_ = (folder / 'people.mdl').write_text(text)
translator = SystemDynamicsTranslator(folder / 'people.mdl').prepare().run()
_ = (folder / 'translated_people.py').write_text(translator.source_)
sys.path.insert(0, str(folder))
module = importlib.import_module('translated_people')
translator.settings_.until, module.DT

system = module.create_system()
simulation = Simulation(system, dt=module.SAVE_EVERY, observables=module.observables(system))
trajectory = simulation.prepare().run(until=module.UNTIL).trajectory_
trajectory.values('people.births')[0], trajectory.values('people.doubling_share')[0]
LoopAnalysis().prepare(system).run().polarities_
```

PySD reads the file. The translator writes `source_` from PySD's abstract model, and `settings_` from the control
section of the file. You save the source and import it like any module. The module defines:

- one element class, whose states are the stocks and whose parameters are the constants,
- the functions `create_system()` and `observables(system)`,
- the constants `START`, `UNTIL`, `DT`, and `SAVE_EVERY`.

Time in the module runs from zero, and `Time` in the file becomes `t + START`. The parts of the file become these:

- A flow variable that leaves one stock and enters another is a `TransferInteraction` between the two states. A flow
  variable of one stock is a `TransferInteraction` with `Source()` or `Sink()`. The trajectory reports each flow as
  `<element>.<flow>`.
- The rest of the net rate of a stock is a `RelaxationInteraction` when it has the form `(goal - stock) / time` or
  `(goal - stock) * rate`, and a `RateInteraction` otherwise.
- An auxiliary is a private function `_name(s, t)` that the laws call, and an `ObservableSpec` under its snake-case
  name.
- A lookup is a `lookup` table. The built-in functions are `where`, `step`, `ramp`, `pulse`, `pulse_train`, `delay1`,
  `delay3`, `smooth`, `distributed_delay`, `delayed`, `minimum`, `maximum`, `clip`, and the draws `uniform` and
  `normal`.
- A system whose stocks are all subscripted over one range is a `GroupElement` with one member per subscript. `SUM` over
  the range is `total`, and a read of one subscript is a `total` over that member.

The default solver finds the exact solution. Vensim and Stella integrate with Euler steps by default. A run with
`solver=EulerSolver(dt=module.DT)` and `dt=module.DT` reproduces their run step by step. A run with the default solver
differs from it by the Euler error of the time step of the file.

```python continuation
stepped = Simulation(system, solver=EulerSolver(dt=module.DT), dt=module.DT)
final = stepped.prepare().run(until=module.UNTIL).trajectory_.values('people.population')[-1]
round(float(final), 3)
```

One `ValidationError` lists every construct that does not translate, with its variable. These are `GAME`, `ALLOCATE`,
the vector functions, `TREND`, `FORECAST`, `SAMPLE IF TRUE`, data read from spreadsheets, macros, variables subscripted
over more than one range, systems that mix subscripted and unsubscripted stocks, delays and smoothings whose initial
value differs from their input, and `DELAY FIXED` of anything other than a stock.

The library sets these limits on what a translated module does:

- The core takes random draws only in update and renewal laws. A module with a random function translates, but raises
  `IncompatibleLawsError` when its system is prepared.
- A run cannot record an auxiliary that reads `DELAY FIXED`, or a delay that no law reads, so `observables` leaves it
  out.
- A trajectory records the flow of one member from the values of that member. So in a group, a flow variable that reads
  `SUM` or one named subscript stays in the rate of its stock, without a flow, and its value stays an observable.

The catalog systems built on flows are these:

- engineering processes: `bathtub`, `rice_milling`, `solid_waste`, `food_relief`, and `irrigation`,
- markets: `car_dealer`, `commodity_cycle`, and `beer_game`,
- innovation: `bass_diffusion`,
- macroeconomics: `stock_flow_consistent_economy`,
- organizations: `success_to_the_successful`, `shifting_the_burden`, and `escalation`.
