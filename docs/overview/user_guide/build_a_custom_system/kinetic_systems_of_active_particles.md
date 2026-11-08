# Kinetic systems of active particles

Describe a population by how its members spread over an activity, such as wealth, an opinion, or the progression of a
tumour cell, and let encounters between members reshape that spread. This is the kinetic theory of active particles of
Bellomo et al. Encounters change the distribution, and its moments give the macroscopic quantities.

## Distributions as site groups

Hold a distribution over nine wealth classes as a group with one member per class:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.specs import StateSpec


class WealthDensityElement(Element):
    def __init__(self, label, *, f: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0):
        super().__init__(label)
        self.f = f


classes = np.linspace(0.0, 1.0, 9)
profile = np.exp(-((classes - 0.5) ** 2) / 0.05)
wealth = GroupElement('wealth', element=WealthDensityElement, size=9, on='sites', values={'f': profile / profile.sum()})
wealth.values['f'].round(3).tolist()
```

A distribution over `n` activity values is a group on the sites of a lattice of `n` sites, one member per activity,
`GroupElement(label, element=E, size=n, on='sites')`. The element declares one real state, the density at the member's
activity. The activity values are an array that the laws and the moments receive. No distribution component is needed,
and a population of several functional subsystems is several such groups.

## Encounters

Declare who meets whom and where they end up, and let the compiler derive the equations. Traders exchange wealth, and
cooperation and competition move them one class up or down:

```python continuation
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import EncounterLaw
from skcomplex.core.components.spaces import ContinuousTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, RoleSpec


def exchange_table(cooperation, competition):
    table = np.zeros((9, 9, 9))
    for candidate in range(9):
        for field in range(9):
            gap = int(np.sign(field - candidate))
            for outcome, probability in ((candidate + gap, cooperation), (candidate - gap, competition)):
                if gap and 0 <= outcome < 9:
                    table[candidate, field, outcome] += probability
            table[candidate, field, candidate] += 1 - table[candidate, field].sum()
    return table


class ExchangeInteraction(Interaction):
    def __init__(self, label, *, population: Annotated[str, RoleSpec()], eta: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.population = population
        self.eta = eta

    def _get_laws(self, s):
        return [EncounterLaw(s.population.f, s.population.f, s.eta, exchange_table(0.2, 0.3))]


economy = System(
    elements=[wealth],
    interactions=[ExchangeInteraction('exchange', population='wealth')],
    space=LatticeSpace(9, boundary='fixed'),
    time=ContinuousTime(),
)
densities = simulate(economy, until=100, dt=10.0).values('wealth.f')
densities.sum(axis=1)[[0, -1]].round(12).tolist(), densities[-1].round(3).tolist()
```

`EncounterLaw(candidate, field, rate, transition, test=None)` declares that a candidate particle at activity `h` meets a
field particle at activity `k` at the rate `eta[h, k]`, and ends at activity `i` with the probability `A[h, k, i]`. An
interaction returns it with roles bound to the candidate and field groups. The compiler derives the gain and loss
equations

`df_i/dt = sum_hk eta[h, k] A[h, k, i] f_h g_k - f_i sum_k eta[i, k] g_k`

without any derivation by hand. When `test` is given, the candidate moves into the `test` group, such as a cell that
mutates into another subsystem.

- `BilinearLaw(target, first, second, rate, outcomes=None)` changes the target at the signed rate `r[h, k]`. A positive
  rate creates a particle without removing the first density, at its activity or at an activity drawn from `outcomes`.
  A negative rate with the first density as the target removes it.
- Rates and probabilities are numbers or parameter expressions, scalars or arrays. Only the nonzero probabilities enter
  the lowering, as sparse contractions on NumPy and one fused loop on Numba.
- The two laws need continuous time, a lattice space, and real states. Validation reports a law whose system lacks
  them, with the fix to declare them.
- Before compilation, validation reports the encounters whose probabilities do not sum to one over the outcomes, with
  the candidate and field activities and the sum.
- A system with encounters has no exact Jacobian, so implicit solvers estimate it by finite differences.

## External actions

Add a policy that acts on the population from outside. External actions need no primitive of their own.

- A micro-scale action, such as a welfare policy that acts on the transitions of the wealth classes, is an encounter
  whose field role binds a group with fixed states, a group no law changes.
- A macro-scale action, such as a recruitment of security forces proportional to the mean criminality, is a `RateLaw`
  that reads the moments of a distribution.

A welfare policy that lifts the poorest classes and taxes the richest:

```python continuation
class PolicyInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        population: Annotated[str, RoleSpec()],
        government: Annotated[str, RoleSpec()],
        tax: Annotated[float, ParameterSpec()] = 0.2,
    ):
        super().__init__(label)
        self.population = population
        self.government = government
        self.tax = tax

    def _get_laws(self, s):
        rates, table = np.zeros((9, 1)), np.zeros((9, 1, 9))
        for candidate in range(9):
            outcome = candidate + 1 if candidate < 3 else candidate - 1 if candidate >= 6 else candidate
            rates[candidate, 0] = float(outcome != candidate)
            table[candidate, 0, outcome] = 1.0
        return [EncounterLaw(s.population.f, s.government.f, rates * s.tax, table)]


welfare = System(
    elements=[wealth, GroupElement('government', element=WealthDensityElement, size=1, values={'f': 1.0})],
    interactions=[
        ExchangeInteraction('exchange', population='wealth'),
        PolicyInteraction('policy', population='wealth', government='government'),
    ],
    space=LatticeSpace(9, boundary='fixed'),
    time=ContinuousTime(),
)
simulate(welfare, until=100, dt=10.0).values('wealth.f')[-1].round(3).tolist()
```

## Moments

Record the mean and the variance of the distribution as it changes:

```python continuation
from skcomplex.core.components.laws.utils import moment
from skcomplex.core.specs import ObservableSpec


def mean_wealth(s):
    return moment(s.f, 1, classes, normalized=True)


def wealth_variance(s):
    return moment(s.f, 2, classes, central=True)


observables = [
    ObservableSpec('mean_wealth', mean_wealth, group='wealth'),
    ObservableSpec('wealth_variance', wealth_variance, group='wealth'),
]
trajectory = simulate(economy, until=100, dt=10.0, observables=observables)
variance = trajectory.values('wealth_variance')
round(float(trajectory.values('mean_wealth')[-1]), 3), variance[[0, -1]].round(3).tolist()
```

`moment(expression, order, coordinates)` is the reduction `sum_j c_j^r f_j` over the members of a group, with the
coordinate `c_j` of every member, such as its activity.

- The order-0 moment is the density.
- `normalized=True` divides by the density, so the normalized order-1 moment is the mean activity.
- `central=True` takes the moment about the mean, so the central order-2 moment is the variance.

Moments are expressions of `total` with a shared row of weights, a vector with one weight per reduced member. So laws
read them and observables record them. An observable of a group receives the namespace of one member, and its function
must be defined at module level so that it pickles.

## The particle solver

Run the same declarations with individual particles instead of densities, and compare:

```python continuation
from skcomplex.core.engine.solvers import KineticSolver

sampled = simulate(economy, until=100, dt=10.0, solver=KineticSolver(n=20_000), random_state=0)
round(float(np.abs(sampled.values('wealth.f')[-1] - densities[-1]).sum()), 2)
policed = simulate(welfare, until=100, dt=10.0, solver=KineticSolver(n=20_000), random_state=0)
policed.values('wealth.f')[-1].round(2).tolist()
```

`KineticSolver(n, dt)` runs the declarations by direct simulation Monte Carlo. It holds every density as counts of
particles of one mass, with `n` particles for the largest initial density. In every step:

- a candidate particle encounters with the probability `1 - exp(-lambda_h dt)`, where `lambda_h = sum_k eta[h, k] g_k`,
- it picks its field activity and outcome in proportion to `eta[h, k] A[h, k, i] g_k`,
- a proliferation adds a Poisson number of particles, and a destruction removes a binomial number.

The state holds the empirical densities, so trajectories match the ODE form, and the error against the ODE solution
falls as one over the square root of `n`. The welfare system, whose policy is an encounter too, runs on it as well, as the
last line shows. The solver supports continuous time, lattice spaces, site groups, and
encounter laws only. Validation reports a rate law with it and lists the solvers that run it.

## Levels

Move between individual agents and their distribution, so one level of a system can feed another:

```python continuation
from skcomplex.core.analysis import coarse_grain


class TraderElement(Element):
    def __init__(self, label, *, w: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.w = w


profile = densities[-1] / densities[-1].sum()
wealths = classes[np.random.default_rng(0).choice(classes.size, size=1000, p=profile)]
traders = GroupElement('traders', element=TraderElement, size=1000, values={'w': wealths}, variable=True)
density = coarse_grain(traders.values['w'], over=classes).field
binned = GroupElement('wealth', element=WealthDensityElement, size=classes.size, on='sites', values={'f': density})
round(float(density.sum()), 6), round(float(density @ classes), 3), round(float(traders.values['w'].mean()), 3)
```

- `coarse_grain(source, over=values)` coarse-grains in state space. It bins the values of a state of a group's members,
  from their initial values or from a frame of a trajectory, over given values. Every member counts at the nearest
  value, so the densities sum to 1 and start a distribution site group.
- The other way round, members drawn from a distribution are values sampled with NumPy in proportion to the densities,
  and the group's constructor builds them.
- A system can then run one level in a partition of a co-simulation and feed another level its moments.

The library has `KineticWealthWelfareSystem`, `ImmuneCompetitionSystem`, `KineticOpinionSystem`, `EvacuationSystem`,
and `CriminalitySystem`. `EvacuationSystem` holds a crowd on a network of rooms as a site group on a two-dimensional lattice indexed by room and speed
class, whose walking encounters read the doors of the building.

## Automata on groups

Run a cellular automaton on the elements of a group instead of a square grid, such as a free group, whose Cayley graph
is a tree. A majority rule on the ball of radius 4 of the free group on two generators:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_gap import PermutationGroup, libgap
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import CayleySpace, GeneratorNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec, TotalisticSpec

tree = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=4).prepare()
memory = GeneratorNeighborhood(tree.generators_)


class MajorityElement(Element):
    def __init__(self, label, *, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive

    def _get_laws(self, s):
        return [LocalRuleLaw(s.alive, memory, TotalisticSpec({3: 1, 4: 1, 5: 1}))]


cells = (np.random.default_rng(0).random(tree.size_) < 0.5).astype(float)
majority = System(elements=[LatticeElement('cells', element=MajorityElement, space=tree, values={'alive': cells})])
tree.ball_sizes_, simulate(majority, until=3, dt=1).values('cells.alive').shape
CayleySpace(PermutationGroup([[(1, 2)], [(1, 2, 3)]])).prepare().words_
```

A cellular automaton needs only a group. Its cells are the elements, and a cell at `g` reads the cells at `g s` for the
elements `s` of a finite memory set (Ceccherini-Silberstein and Coornaert 2023).

- `CayleySpace(group, ball=r)` is the Cayley graph of a finitely generated group on the ball of radius `r` in its
  generators and their inverses.
- `CayleySpace(group)` without a ball is the whole of a finite group, such as a finite quotient, where configurations
  are periodic.
- A group space is a space with sites, so `LocalRuleLaw`, `SampleLaw`, and the other lattice laws run on it with the
  NumPy and Numba backends, and a run reads only its neighbour table.
- Preparing the space enumerates its sites, numbered by word length. `generators_` names the generators, `words_` gives
  the shortest word of every site, and `ball_sizes_` gives the growth of the balls. Cells outside a ball read the
  space's `fill`.
- The memory set is a `MemoryNeighborhood` of words such as `'a'`, `'a^-1'`, or `'a*b'`, with `'1'` for the cell
  itself. `GeneratorNeighborhood(generators)` gives the identity followed by every generator and its inverse.

The group is a native Sage or GAP group. So any presentation works whose word problem a library solves:

- A free group, `libgap.FreeGroup(['a', 'b'])`, reduces its words directly. A finitely presented group,
  `free / relators`, is solved by Knuth-Bendix completion with KBMAG in GAP. A presentation it cannot complete raises
  `SpaceError`, which asks for a finite quotient.
- An automaton group of the AutomGrp package, which `libgap.LoadPackage('automgrp')` loads, has its elements told apart
  by their action on a level of the tree. An example is `libgap.AutomatonGroup('t = (1, t)(1, 2)')` for the adding
  machine, and `libgap.PermGroupOnLevel(group, n)` is its finite quotient on level `n`.
- A Sage `PermutationGroup` or `MatrixGroup` compares its elements directly.

GAP comes with the `passagemath-gap` dependency, and starts on the first ball that needs it. A ball whose size bound
exceeds `max_sites`, 100,000 by default, raises `SpaceTooLargeError` before GAP is called. `simulate` reports it with the
issue `space-too-large`, which names the largest radius that fits.

A group space lacks the regular geometry of a square lattice and an image layout. So validation reports a checkerboard
schedule as `schedule-unsupported` and an image visual as `visual-unsupported`. Groups on group spaces stream to Rerun
with `CayleyVisual`: nodes on rings by word length, colored by state, with the edges of the generators logged once.

The core names no specific group, so you write a named group in one of these forms. The Grigorchuk group acts on the
binary tree by `a = (1, 1)(1, 2)`, `b = (a, c)`, `c = (a, d)`, `d = (1, b)`, so it is an automaton group. It is not
finitely presented, and its balls grow faster than any polynomial and slower than any exponential. Its action on
level 3 of the tree is a finite quotient:

```python
from sage.all__sagemath_gap import libgap

from skcomplex.core.components.spaces import CayleySpace

libgap.LoadPackage('automgrp')
grigorchuk = libgap.AutomatonGroup('a = (1, 1)(1, 2), b = (a, c), c = (a, d), d = (1, b)')
ball = CayleySpace(grigorchuk, ball=6).prepare()
quotient = CayleySpace(libgap.PermGroupOnLevel(grigorchuk, 3)).prepare()
ball.ball_sizes_, quotient.size_
```

**Theory of automata.** Ask whether an automaton is injective or surjective, find a pattern no configuration reaches,
or invert a reversible rule:

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.analysis import automaton
from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace, MooreNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import OuterTotalisticSpec, StateSpec, TableSpec


class CellElement(Element):
    def __init__(self, label, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive


def cellular_automaton(space, neighborhood, rule):
    cells = LatticeElement('cells', element=CellElement, space=space)
    interaction = LocalRuleInteraction('rule', target='cells', state='alive', rule=rule, neighborhood=neighborhood)
    return System(elements=[cells], interactions=[interaction])


life_rule = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})
life = cellular_automaton(LatticeSpace((6, 6)), MooreNeighborhood(), life_rule)
automaton(life, kind='surjective').surjective
automaton(life, kind='garden-of-eden', random_state=0).pattern is not None
ring = LatticeSpace(8)
shift = cellular_automaton(ring, LatticeNeighborhood(((1,),)), TableSpec({(1,): 1}))
back = automaton(shift, kind='inverse').inverse
composition = automaton(shift, kind='compose', second=back).composition.interactions[0]
law = LocalRuleLaw(var('alive'), composition.neighborhood, composition.rule).prepare()
law.run({'space': ring, 'alive': [1, 0, 1, 1, 0, 0, 0, 1]}).value_.tolist()

step = LocalRuleLaw(var('alive'), shift.interactions[0].neighborhood, shift.interactions[0].rule).prepare()
cells = [1, 0, 1, 1, 0, 0, 0, 0]
for _ in range(2):
    cells = step.run({'space': ring, 'alive': cells}).value_
start = LatticeElement('cells', element=CellElement, space=ring, values={'alive': [1, 0, 1, 1, 0, 0, 0, 0]})
started = System(elements=[start], interactions=shift.interactions)
frames = simulate(started, until=2, dt=1).values('cells.alive')
cells.tolist() == frames[-1].astype(int).tolist()
```

`skcomplex.core.analysis` decides the properties the theory studies on finite universes, where they are decidable:
periodic configurations on a torus or a finite quotient, and patterns on windows. The checks do not decide them on
infinite groups. A cellular automaton is a `System` with a `LocalRuleInteraction`, and the analysis `automaton` reads
the system. The `run` of a `LocalRuleLaw` is one step of the map on configurations, so several steps are a loop over
that `run` or a `simulate` of the system. The two give the same configuration, as the last line shows.

- The kinds `'injective'` and `'surjective'`, which agree on a finite universe, enumerate the configurations of small
  universes and solve a SAT encoding with PySAT on larger ones.
- `'pre-injective'` asks whether two configurations that differ only inside a `window` can share their image.
- `'garden-of-eden'` finds a pattern without a preimage, by enumeration or by SAT on random patterns.
- `'inverse'` finds the rule of the inverse of a reversible automaton with the smallest memory up to `max_radius`.
- `'compose'` gives the automaton of its composition with a `second` one.

**Linear automata and subshifts.** Decide a linear rule by algebra, and restrict configurations to those without a
forbidden pattern, such as the golden mean shift, where no two neighbouring cells are both 1:

```python
from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.analysis import automaton
from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import LinearRuleSpec, StateSpec, SubshiftSpec, TableSpec


class CellElement(Element):
    def __init__(self, label, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive


class GoldenCellElement(Element):
    def __init__(
        self,
        label,
        alive: Annotated[float, StateSpec(domain=SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),)))] = 0,
    ):
        super().__init__(label)
        self.alive = alive


def golden(space):
    return System(elements=[LatticeElement('cells', element=GoldenCellElement, space=space)])


laplacian = System(
    elements=[LatticeElement('cells', element=CellElement, space=LatticeSpace(5))],
    interactions=[
        LocalRuleInteraction(
            'laplacian',
            target='cells',
            state='alive',
            rule=LinearRuleSpec((1, -2, 1), 'GF(2)'),
            neighborhood=LatticeNeighborhood(((-1,), (0,), (1,))),
        ),
    ],
)
automaton(laplacian, kind='injective').injective
automaton(laplacian, kind='surjective', infinite=True).surjective
[automaton(golden(LatticeSpace(length, boundary='fixed')), kind='count').count for length in range(1, 8)]
windows = [LatticeSpace(24, boundary='fixed')]
system = golden(windows[0])
automaton(system, kind='entropy', windows=windows).entropy.round(3)

forbidden = [1, 1, 0, 0, 0]
cells = LatticeElement('cells', element=GoldenCellElement, space=LatticeSpace(5), values={'alive': forbidden})
keep = LocalRuleInteraction(
    'keep',
    target='cells',
    state='alive',
    rule=TableSpec({(1,): 1}),
    neighborhood=LatticeNeighborhood(((0,),)),
)
held = System(elements=[cells], interactions=[keep])
automaton(held, kind='admissible', configuration=forbidden).admissible
simulate(held, until=2, dt=1).values('cells.alive')[-1].astype(int).tolist()
```

The rule `LinearRuleSpec(coefficients, field)` is the element `sum_s a_s s` of the group ring over `'GF(p)'` or
`'QQ'`.

- On a finite universe it is a matrix, whose rank decides injectivity, surjectivity, and pre-injectivity together.
- On `Z^d`, with `infinite=True`, its Laurent polynomial decides them. It is invertible when the polynomial is a
  monomial, and surjective when it is not zero.
- The kinds `'matrix'` and `'polynomial'` of `automaton` give them.

`SubshiftSpec(neighborhood, forbidden)` is the domain of a state whose configurations hold no forbidden pattern.
Admissibility is an analysis kind, and a run does not enforce it. A run checks only that every cell holds a value of
the alphabet, so it runs from a configuration with a forbidden pattern, as the last line shows.

- The kinds `'admissible'` and `'violations'` check a configuration.
- `'count'` counts the admissible configurations of a window, and `'sample'` samples them by a Markov chain of
  single-cell moves.
- `automaton(system, kind='entropy', windows=windows)` gives `log |X_F| / |F|` along windows `F`.
- `automaton(system, kind='difference')` gives the cells where two runs differ after one cell changes.
- `entropy(frames, kind='rate')` of the frames of a run as columns gives the spatial entropy rate of every frame, which never
  increases under an automaton.

The library has `FreeGroupAutomatonSystem`, `LamplighterAutomatonSystem`, `HeisenbergAutomatonSystem`,
`BaumslagSolitarAutomatonSystem`, `GrigorchukAutomatonSystem`, the Laplacian `LinearRingAutomatonSystem`, and the golden
mean shift `SubshiftsSystem`, in `skcomplex.systems.computer_science.automata`. Their groups are written in the forms above.
