# Kinetic systems of active particles

This page describes a population by how its members spread over an activity, such as wealth, an opinion, or the
progression of a tumour cell, and lets encounters between members reshape that spread. This is the kinetic theory of
active particles of Bellomo et al. Encounters change the distribution, and its moments give the macroscopic quantities,
such as the mean wealth or its variance.


## Distributions on a lattice

This section holds a distribution over nine wealth classes as an element with one member per class. An
[`Element`](../core_components/elements.md#element) given a [`LatticeSpace`](../core_components/spaces.md#latticespace)
with `space=` has one member on each site, so a distribution over `n` activity values is an element on a lattice of `n`
sites. Its value `f` is the density at the activity of each member, and the allowed values in `Annotated` keep it from
going negative. Here the wealth starts as a bell around the middle class.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.spaces import LatticeSpace


class WealthDensity(Element):
    def __init__(self, label, *, f: Annotated[float, RealSet.closed_open(0, oo)] = 0.0, space=None):
        super().__init__(label, space=space)
        self.f = f


classes = np.linspace(0.0, 1.0, 9)
profile = np.exp(-((classes - 0.5) ** 2) / 0.05)
wealth = WealthDensity('wealth', f=profile / profile.sum(), space=LatticeSpace(9, boundary='fixed'))
wealth.f.round(3).tolist()
```

The densities read 0.002, 0.019, 0.09, 0.231, 0.315, 0.231, 0.09, 0.019, and 0.002 from the poorest class to the
richest, and they sum to one. The activity values are the array `classes`, which the laws and the moments receive. No
distribution component is needed, and a population of several functional subsystems is several such elements.

## Encounters

This section says who meets whom and where they end up, and lets the compiler derive the equations. An encounter is a
meeting of a candidate particle at activity `h` with a field particle at activity `k`. Traders here exchange wealth,
and cooperation moves the candidate one class towards its partner while competition moves it one class away. A
[`RateLaw`](../core_components/laws.md#ratelaw) whose rate is the form `bilinear` of
`skcomplex.core.components.laws.utils` states the rate of the encounters and the table of their outcomes, and an
[`Interaction`](../core_components/interactions.md#interaction) sets it in `_prepare`, with its role bound to the
population. A rate law runs in continuous time, so the system needs no setting for it.

```python continuation
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import bilinear
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


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


class Exchange(Interaction):
    def __init__(self, label, *, population: Role = 'wealth', eta=1.0):
        super().__init__(label)
        self.population = population
        self.eta = eta

    def _prepare(self, s):
        super()._prepare(s)
        f = s.population.f
        self.laws_ = [RateLaw(f, bilinear(s.eta, exchange_table(0.2, 0.3), f, f, conserved=True))]


economy = System(elements=[wealth], interactions=[Exchange('exchange')]).prepare()
densities = economy.simulate(until=100, dt=10.0).get_values('wealth.f')
densities.sum(axis=1)[[0, -1]].round(12).tolist(), densities[-1].round(3).tolist()
```

The total density stays at 1.0 from the start to the end, since every encounter moves a trader without creating one.
After 100 time units the wealth spreads to the two ends, with 0.173 in the poorest and the richest classes and 0.078 in
the middle class. Competition, at the probability 0.3, outweighs cooperation at 0.2, so the traders drift apart.

`RateLaw(target, bilinear(rate, outcomes, first, second, conserved=True))` says that a candidate of the first density
at activity `h` meets a particle of the second density at activity `k` at the rate `eta[h, k]`. It ends at activity `i`
with the probability `A[h, k, i]`. The compiler derives the gain and loss equations

`df_i/dt = sum_hk eta[h, k] A[h, k, i] f_h g_k - f_i sum_k eta[i, k] g_k`

with no derivation by hand. When the target of a form that is not conserved is another density than the first, the
candidate moves into the target element, such as a cell that mutates into another subsystem. The form follows these
rules.

- The form is the whole rate of its rate law, and the rate laws of one target add up.
- Without `conserved`, the form changes the target at the signed rate `r[h, k]`. A positive rate creates a particle
  without removing the first density, at its activity or at an activity drawn from `outcomes`. A negative rate with the
  first density as the target removes it.
- Rates and probabilities are numbers or parameter expressions, scalars or arrays. Only the nonzero probabilities enter
  the compiled code, as sparse contractions on NumPy and one fused loop on Numba.
- Bilinear forms need rate laws, elements on a lattice, and real values. Validation reports a form whose system lacks
  them, with the fix.
- Before compilation, validation reports the encounters whose probabilities do not sum to one over the outcomes, with
  the candidate and field activities and the sum.
- A system with encounters has no exact Jacobian, so implicit solvers estimate it by finite differences.

## The bilinear form

This section writes encounters of other fields with the same form. `bilinear(rate, outcomes, first, second)` is an
operator of `skcomplex.core.components.laws.utils`, kept unevaluated in the rate of a `RateLaw` until the compiler
lowers it to sparse contractions of the densities. Immune cells that meet tumour cells destroy them at the rate
`0.5 f g`, where `f` and `g` are the two densities. A negative rate on the first density, with no outcome table, is a
loss.

```python continuation
class Killing(Interaction):
    def __init__(self, label, *, prey: Role = 'tumour', hunter: Role = 'immune', k=0.5):
        super().__init__(label)
        self.prey = prey
        self.hunter = hunter
        self.k = k

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [RateLaw(s.prey.f, bilinear(-s.k, None, s.prey.f, s.hunter.f))]


tumour = Element('tumour', values={'f': 1.0}, space=LatticeSpace(1))
immune = Element('immune', values={'f': 0.4}, space=LatticeSpace(1))
body = System(elements=[tumour, immune], interactions=[Killing('killing')])
body.prepare().simulate(until=5, dt=5.0).get_values('tumour.f').round(3).tolist()
```

No law changes the immune cells, so they act as a fixed input, and the tumour falls from 1 to 0.368 after five units of
time, which is `exp(-0.5 * 0.4 * 5)`.

People shift their opinion after a conversation. Three classes hold the shares of people against, undecided, and for
a proposal, and a person who talks with someone of another class moves one class towards them with probability one
half. The form is conserved, so every person who leaves a class enters another.

```python continuation
class Persuasion(Interaction):
    def __init__(self, label, *, people: Role = 'opinions'):
        super().__init__(label)
        self.people = people

    def _prepare(self, s):
        super()._prepare(s)
        outcomes = np.zeros((3, 3, 3))
        for h in range(3):
            for k in range(3):
                outcomes[h, k, h + int(np.sign(k - h))] += 0.5
                outcomes[h, k, h] += 0.5
        f = s.people.f
        self.laws_ = [RateLaw(f, bilinear(1.0, outcomes, f, f, conserved=True))]


opinions = Element('opinions', values={'f': [0.6, 0.1, 0.3]}, space=LatticeSpace(3))
debate = System(elements=[opinions], interactions=[Persuasion('persuasion')])
shares = debate.prepare().simulate(until=10, dt=10.0).get_values('opinions.f')
shares.round(3).tolist(), shares.sum(axis=1).round(12).tolist()
```

The shares move from 0.6, 0.1, and 0.3 to 0.338, 0.625, and 0.038, and their total stays 1. The undecided gain from both
sides, and the minority for the proposal shrinks most.

Particles of smoke coagulate when they collide. A cluster of `h + 1` particles that meets a cluster of `k + 1` merges
into a cluster of `h + k + 2`, so the equations of Smoluchowski are two forms on the same density. The gain is
`kernel / 2` into the merged class, and the loss is `-kernel` with no outcome table. The rate arrays leave out the
merges past the largest class, so the mass stays the same.

```python continuation
class Coagulation(Interaction):
    def __init__(self, label, *, particles: Role = 'clusters', kernel=1.0):
        super().__init__(label)
        self.particles = particles
        self.kernel = kernel

    def _prepare(self, s):
        super()._prepare(s)
        fits = (np.add.outer(np.arange(6), np.arange(6)) < 5).astype(float)
        merged = np.zeros((6, 6, 6))
        for h, k in zip(*np.nonzero(fits)):
            merged[h, k, h + k + 1] = 1.0
        n = s.particles.f
        gain = bilinear(fits * s.kernel / 2, merged, n, n)
        loss = bilinear(-fits * s.kernel, None, n, n)
        self.laws_ = [RateLaw(n, gain), RateLaw(n, loss)]


clusters = Element('clusters', values={'f': [1.0, 0, 0, 0, 0, 0]}, space=LatticeSpace(6))
smoke = System(elements=[clusters], interactions=[Coagulation('coagulation')])
sizes = smoke.prepare().simulate(until=1, dt=1.0).get_values('clusters.f')
sizes[-1].round(3).tolist(), round(float(sizes[-1] @ np.arange(1, 7)), 3)
```

After one unit of time the single particles fall to 0.445 and the pairs reach 0.148, the exact solution
`(1 + t / 2)^-2` and `(t / 2) (1 + t / 2)^-3` for a constant kernel, and the mass stays 1.

A conserved form needs its table of outcomes, because without one its gain and loss cancel, and the call raises a
`ValidationError`.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError

f, g = var('f g')
try:
    bilinear(1.0, None, f, g, conserved=True)
except ValidationError as error:
    message = str(error)
message
```

The message says that the conserved bilinear form of `f` has no outcome table, so its gain and loss cancel, and asks
for the outcome probabilities `B[h, k, i]`. The fix is an array `outcomes` of shape `(classes, classes, classes)`.

## External actions

This section adds a policy that acts on the population from outside. External actions need no component of their own. A
micro-scale action, such as a welfare policy that acts on the transitions between wealth classes, is an encounter whose
field role binds an element that no law changes. A macro-scale action, such as a recruitment of security forces in
proportion to the mean criminality, is a [`RateLaw`](../core_components/laws.md#ratelaw) that reads the moments of a
distribution. The policy below lifts the three poorest classes by one and taxes the three richest down by one, through
encounters with a government of fixed density one.

```python continuation
class Policy(Interaction):
    def __init__(self, label, *, population: Role = 'wealth', government: Role = 'government', tax=0.2):
        super().__init__(label)
        self.population = population
        self.government = government
        self.tax = tax

    def _prepare(self, s):
        super()._prepare(s)
        rates, table = np.zeros((9, 1)), np.zeros((9, 1, 9))
        for candidate in range(9):
            outcome = candidate + 1 if candidate < 3 else candidate - 1 if candidate >= 6 else candidate
            rates[candidate, 0] = float(outcome != candidate)
            table[candidate, 0, outcome] = 1.0
        form = bilinear(rates * s.tax, table, s.population.f, s.government.f, conserved=True)
        self.laws_ = [RateLaw(s.population.f, form)]


government = Element('government', values={'f': 1.0}, space=LatticeSpace(1))
welfare = System(elements=[wealth, government], interactions=[Exchange('exchange'), Policy('policy')])
welfare.prepare().simulate(until=100, dt=10.0).get_values('wealth.f')[-1].round(3).tolist()
```

With the policy, the poorest and the richest classes fall from 0.173 to 0.049, and the wealth peaks at 0.19 in the
classes on either side of the middle. The middle class holds 0.168, since the policy moves no one into it from below or
above.

## Moments

This section records the mean and the variance of the distribution as it changes. The function
`moment(expression, order, coordinates)` sums `c_j^r f_j` over the members of an element, with the coordinate `c_j` of
every member, here its wealth class. Each moment enters an `Observable`, which records it at every output time.

```python continuation
from skcomplex.core.components import Observable
from skcomplex.core.components.laws.utils import moment


def mean_wealth(s):
    return moment(s.f, 1, classes, normalized=True)


def wealth_variance(s):
    return moment(s.f, 2, classes, central=True)


observables = [
    Observable('mean_wealth', mean_wealth, group='wealth'),
    Observable('wealth_variance', wealth_variance, group='wealth'),
]
trajectory = economy.simulate(until=100, dt=10.0, observables=observables)
variance = trajectory.get_values('wealth_variance')
round(float(trajectory.get_values('mean_wealth')[-1]), 3), variance[[0, -1]].round(3).tolist()
```

The mean wealth ends at 0.5, since the exchange is symmetric about the middle class. The variance grows from 0.025 to
0.133 as the wealth spreads to the ends. The options of `moment` give the usual quantities.

- The moment of order 0 is the total density.
- `normalized=True` divides by the density, so the normalized moment of order 1 is the mean activity.
- `central=True` takes the moment about the mean, so the central moment of order 2 is the variance.

Moments are expressions of `total` with a shared row of weights, a vector with one weight per reduced member. So laws
read them and observables record them. An observable of an element with many members receives the namespace of one
member, and its function must be defined at module level so that it pickles.

## The particle solver

This section runs the same laws with individual particles in place of densities and compares the two.
`KineticSolver(n, dt)` runs them by direct simulation Monte Carlo. It holds every density as counts of particles of one
mass, with `n` particles for the largest initial density. It is the `solver` setting of the system.

```python continuation
from skcomplex.core.engine.solvers import KineticSolver

particles = System(elements=[wealth], interactions=[Exchange('exchange')], solver=KineticSolver(n=20_000))
sampled = particles.prepare().simulate(until=100, dt=10.0, random_state=0)
round(float(np.abs(sampled.get_values('wealth.f')[-1] - densities[-1]).sum()), 2)
```

The final densities of the particles differ from those of the equations by 0.02 in total over the nine classes. In
every step of the solver three things happen.

- A candidate particle encounters with the probability `1 - exp(-lambda_h dt)`, where `lambda_h = sum_k eta[h, k] g_k`.
- It picks its field activity and outcome in proportion to `eta[h, k] A[h, k, i] g_k`.
- A proliferation adds a Poisson number of particles, and a destruction removes a binomial number.

The state holds the empirical densities, so the trajectories match the form of the equations, and the error falls as
one over the square root of `n`. The welfare system, whose policy is an encounter too, runs on the solver as well.

```python continuation
policed = welfare.set_params(solver=KineticSolver(n=20_000)).prepare()
policed.simulate(until=100, dt=10.0, random_state=0).get_values('wealth.f')[-1].round(2).tolist()
```

The particles give 0.05, 0.07, 0.11, 0.19, 0.17, 0.19, 0.11, 0.07, and 0.05, the densities of the equations to two
decimals. The solver supports rate laws, elements on a lattice, and bilinear forms only. Validation reports any other
rate law with it and lists the solvers that run it.

## Levels

This section moves between individual agents and their distribution, so one level of a system can feed another. The
analysis `coarse_grain(source, over=values)` bins the values of the members of an element over given values. Every
member counts at the nearest value, so the densities sum to 1 and start a distribution on a lattice. Here 1000 traders
are drawn from the final distribution of the exchange and binned back.

```python continuation
from skcomplex.core.analysis import coarse_grain

profile = densities[-1] / densities[-1].sum()
wealths = classes[np.random.default_rng(0).choice(classes.size, size=1000, p=profile)]
traders = Element('traders', values={'w': wealths}, size=1000)
density = coarse_grain(traders.values['w'], over=classes).field
binned = WealthDensity('wealth', f=density, space=LatticeSpace(classes.size, boundary='fixed'))
round(float(density.sum()), 6), round(float(density @ classes), 3), round(float(traders.values['w'].mean()), 3)
```

The binned densities sum to 1.0, and their mean wealth of 0.524 equals the mean wealth of the 1000 traders. The two
levels connect in both directions.

- `coarse_grain` reads the initial values of a group, as here, or a frame of a trajectory.
- The other way round, members drawn from a distribution are values sampled with NumPy in proportion to the densities,
  and the constructor of the group builds them, as the line that draws `wealths` does.
- A system can then run one level in a partition of a co-simulation and feed another level its moments.

The library has `KineticWealthWelfareSystem`, `ImmuneCompetitionSystem`, `KineticOpinionSystem`, `EvacuationSystem`, and
`CriminalitySystem`. `EvacuationSystem` holds a crowd on a network of rooms as an element on a two-dimensional lattice
indexed by room and speed class, whose walking encounters read the doors of the building.

## Automata on groups

This section runs a cellular automaton on the elements of a group in place of a square grid. A cellular automaton
needs only a group. Its cells are the elements, and a cell at `g` reads the cells at `g s` for the elements `s` of a
finite memory set (Ceccherini-Silberstein and Coornaert 2023). A free group on two generators `a` and `b` has as
elements all reduced words in the generators and their inverses, and its Cayley graph, the graph that joins `g` to
`g s` for every generator `s`, is a tree. The package
[`skcomplex.systems.computer_science.common.automata`][skcomplex.systems.computer_science.common.automata] holds the
space and the neighbourhood of such automata. Its `CayleySpace` holds a ball of the tree, the words up to a length, and
its `MemoryNeighborhood` lists the memory set as words, such as the identity `'1'` followed by every generator and its
inverse. `FreeGroupAutomatonSystem` of `skcomplex.systems.computer_science.automata` builds them into an
[`UpdateLaw`](../core_components/laws.md#updatelaw) with a neighbourhood, a local rule that applies a majority rule
over these five cells on the ball of radius 4.

```python
from skcomplex.systems.computer_science.automata import FreeGroupAutomatonSystem

majority = FreeGroupAutomatonSystem(radius=4).prepare()
tree = majority.system_.elements[0].space.prepare()
tree.ball_sizes_, majority.simulate(until=3, dt=1).get_values('cells.alive').shape
```

The balls of radius 0 to 4 hold 1, 5, 17, 53, and 161 elements, so the number of words triples at every length. The
run returns four frames of 161 cells. A cell turns alive when more than half of itself and its neighbours inside the
ball are alive.

A `CayleySpace` without a ball is the whole of a finite group, such as a finite quotient, where configurations are
periodic. The symmetric group on three letters, generated by a swap and a cycle, has six elements, and
`SelfIndexedShiftSystem` of the same subfield takes it as its group.

```python
from sage.all__sagemath_gap import PermutationGroup

from skcomplex.systems.computer_science.automata import SelfIndexedShiftSystem

shift = SelfIndexedShiftSystem(PermutationGroup([[(1, 2)], [(1, 2, 3)]]), radius=None)
shift.prepare().system_.elements[0].space.prepare().words_
```

The shortest words of the six elements are `'1'`, `'a'`, `'b'`, `'b^-1'`, `'a*b'`, and `'a*b^-1'`. The group spaces
follow these rules.

- A group space is a space with sites, so local rules and the other lattice laws run on it with the NumPy and Numba
  backends, and a run reads only its neighbour table.
- Preparing the space enumerates its sites, numbered by word length. `generators_` names the generators, `words_` gives
  the shortest word of every site, and `ball_sizes_` gives the growth of the balls. Cells outside a ball read the
  `fill` of the space.
- The memory set is a `MemoryNeighborhood` of words such as `'a'`, `'a^-1'`, or `'a*b'`, with `'1'` for the cell
  itself.

The group is a native Sage or GAP group, so any presentation works whose word problem a library solves. The word
problem asks whether two words name the same element.

- A free group, `libgap.FreeGroup(['a', 'b'])`, reduces its words directly. A finitely presented group,
  `free / relators`, is solved by Knuth-Bendix completion with KBMAG in GAP. A presentation it cannot complete raises
  `SpaceError`, which asks for a finite quotient.
- An automaton group of the AutomGrp package, which `libgap.LoadPackage('automgrp')` loads, tells its elements apart
  by their action on a level of the tree. An example is `libgap.AutomatonGroup('t = (1, t)(1, 2)')` for the adding
  machine, and `libgap.PermGroupOnLevel(group, n)` is its finite quotient on level `n`.
- A Sage `PermutationGroup` or `MatrixGroup` compares its elements directly.

GAP comes with the `passagemath-gap` dependency and starts on the first ball that needs it. A ball whose size bound
exceeds `max_sites`, 100,000 by default, raises `SpaceTooLargeError` before GAP is called. `simulate` reports it with
the issue `space-too-large`, which names the largest radius that fits.

A group space lacks the regular geometry of a square lattice and an image layout. So validation reports a checkerboard
schedule as `schedule-unsupported`, and a member drawn as an image is drawn as the group space instead. Groups on group
spaces stream to Rerun with `CayleyVisual` of the same package, which places the nodes on rings by word length,
colours them by state, and logs the edges of the generators once.

The core names no group space, so a named group is written in one of the forms above. The Grigorchuk group acts on
the binary tree by `a = (1, 1)(1, 2)`, `b = (a, c)`, `c = (a, d)`, `d = (1, b)`, so it is an automaton group. It is not
finitely presented, and its balls grow faster than any polynomial and slower than any exponential. Its action on
level 3 of the tree is a finite quotient.

```python
from sage.all__sagemath_gap import libgap

from skcomplex.systems.computer_science.automata import GrigorchukAutomatonSystem, SelfIndexedShiftSystem

libgap.LoadPackage('automgrp')
grigorchuk = libgap.AutomatonGroup('a = (1, 1)(1, 2), b = (a, c), c = (a, d), d = (1, b)')
ball = GrigorchukAutomatonSystem(radius=6).prepare().system_.elements[0].space.prepare()
level = SelfIndexedShiftSystem(libgap.PermGroupOnLevel(grigorchuk, 3), radius=None)
quotient = level.prepare().system_.elements[0].space.prepare()
ball.ball_sizes_, quotient.size_
```

The balls of radius 0 to 6 hold 1, 5, 11, 23, 40, 68, and 108 elements, far fewer than the balls of the free group.
The quotient on level 3 has 128 elements.

## Theory of automata

This section asks whether an automaton is injective or surjective, finds a pattern that no configuration reaches, and
inverts a reversible rule. The analysis `automaton` of `skcomplex.core.analysis` decides these properties on finite
universes, where they are decidable: periodic configurations on a torus or a finite quotient, and patterns on windows.
It does not decide them on infinite groups. A cellular automaton is a `System` whose interaction sets an
[`UpdateLaw`](../core_components/laws.md#updatelaw) with a neighbourhood, and the analysis reads the system. The cells
allow the values 0 and 1, written in `Annotated`. The first system is the Game of Life on a torus of 6 by 6 cells,
whose rule a [`RadiusNeighborhood`](../core_components/spaces.md#radiusneighborhood) of the eight surrounding cells
reads.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var

from skcomplex.core.analysis import automaton
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import OuterTotalisticSpec, TableSpec, UpdateLaw
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace, RadiusNeighborhood
from skcomplex.core.systems import System


class Cells(Element):
    def __init__(self, label, *, alive: Annotated[int, {0, 1}] = 0, space=None):
        super().__init__(label, space=space)
        self.alive = alive


def cellular_automaton(space, neighborhood, rule, alive=0):
    step = Interaction(
        'rule',
        roles={'target': 'cells'},
        laws=lambda s: [UpdateLaw(s.target.alive, rule, neighborhood=neighborhood)],
    )
    return System(elements=[Cells('cells', alive=alive, space=space)], interactions=[step])


life_rule = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})
life = cellular_automaton(LatticeSpace((6, 6)), RadiusNeighborhood(1, metric='chebyshev'), life_rule)
automaton(life, kind='surjective').surjective
```

The Game of Life on this torus is not surjective, so the block returns `False`. Some configurations have no preimage,
and the kind `'garden-of-eden'` finds a pattern among them.

```python continuation
automaton(life, kind='garden-of-eden', random_state=0).pattern is not None
```

The block returns `True`, a pattern without a preimage. A shift moves every cell one site along a ring of eight cells,
through a [`LatticeNeighborhood`](../core_components/spaces.md#latticeneighborhood) with the one offset `(1,)`. It is
reversible, so the kind `'inverse'` finds its inverse, and the kind `'compose'` composes the two.

```python continuation
ring = LatticeSpace(8)
right = LatticeNeighborhood(((1,),))
shift = cellular_automaton(ring, right, TableSpec({(1,): 1}))
back = automaton(shift, kind='inverse').inverse
composition = automaton(shift, kind='compose', second=back).composition.interactions[0]
law = UpdateLaw(var('alive'), composition.rule, neighborhood=composition.neighborhood).prepare()
law.run({'space': ring, 'alive': [1, 0, 1, 1, 0, 0, 0, 1]}).value_.tolist()
```

The composition returns the configuration `[1, 0, 1, 1, 0, 0, 0, 1]` unchanged, so it is the identity. The `run` of a
local rule is one step of the map on configurations. So several steps are a loop over that `run` or a simulation of
the system, and the two give the same configuration.

```python continuation
step = UpdateLaw(var('alive'), TableSpec({(1,): 1}), neighborhood=right).prepare()
cells = [1, 0, 1, 1, 0, 0, 0, 0]
for _ in range(2):
    cells = step.run({'space': ring, 'alive': cells}).value_
started = cellular_automaton(ring, right, TableSpec({(1,): 1}), alive=[1, 0, 1, 1, 0, 0, 0, 0])
frames = started.prepare().simulate(until=2, dt=1).get_values('cells.alive')
cells.tolist() == frames[-1].astype(int).tolist()
```

The block returns `True`. The kinds of `automaton` decide these properties.

- `'injective'` and `'surjective'`, which agree on a finite universe, enumerate the configurations of small universes
  and solve a SAT encoding with PySAT on larger ones.
- `'pre-injective'` asks whether two configurations that differ only inside a `window` can share their image.
- `'garden-of-eden'` finds a pattern without a preimage, by enumeration or by SAT on random patterns.
- `'inverse'` finds the rule of the inverse of a reversible automaton with the smallest memory up to `max_radius`.
- `'compose'` gives the automaton of its composition with a `second` one.

## Linear automata and subshifts

This section decides a linear rule by algebra and restricts configurations to those without a forbidden pattern. A
linear rule `LinearRuleSpec(coefficients, field)` is the element `sum_s a_s s` of the group ring over `'GF(p)'` or
`'QQ'`, so the new value of a cell is a weighted sum of its neighbours modulo `p` or over the rationals. The discrete
Laplacian `(1, -2, 1)` over `GF(2)` runs on a ring of five cells.

```python
from typing import Annotated

from skcomplex.core.analysis import automaton
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import LinearRuleSpec, TableSpec, UpdateLaw
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.specs import SubshiftSpec
from skcomplex.core.systems import System


class Cells(Element):
    def __init__(self, label, *, alive: Annotated[int, {0, 1}] = 0, space=None):
        super().__init__(label, space=space)
        self.alive = alive


def local_rule(label, rule, neighborhood):
    return Interaction(
        label,
        roles={'target': 'cells'},
        laws=lambda s: [UpdateLaw(s.target.alive, rule, neighborhood=neighborhood)],
    )


three = LatticeNeighborhood(((-1,), (0,), (1,)))
laplacian = System(
    elements=[Cells('cells', space=LatticeSpace(5))],
    interactions=[local_rule('laplacian', LinearRuleSpec((1, -2, 1), 'GF(2)'), three)],
)
automaton(laplacian, kind='injective').injective
```

The rule is not injective on the ring, so the block returns `False`. On the infinite line `Z`, the option
`infinite=True` reads the Laurent polynomial of the rule in place of its matrix.

```python continuation
automaton(laplacian, kind='surjective', infinite=True).surjective
```

The block returns `True`, since the polynomial is not zero. The algebra decides each case.

- On a finite universe the rule is a matrix, whose rank decides injectivity, surjectivity, and pre-injectivity
  together.
- On `Z^d`, with `infinite=True`, its Laurent polynomial decides them. The rule is invertible when the polynomial is a
  monomial, and surjective when it is not zero.
- The kinds `'matrix'` and `'polynomial'` of `automaton` return them.

A subshift is the set of configurations that hold no forbidden pattern. `SubshiftSpec(neighborhood, forbidden)`, given
as the allowed values of a value in `Annotated`, allows only the configurations that hold none of the `forbidden`
patterns, read through the `neighborhood`. The golden mean shift forbids two neighbouring cells that are both 1, and
the kind `'count'` counts its configurations on windows of one to seven cells.

```python continuation
GOLDEN_MEAN = SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),))


class GoldenCells(Element):
    def __init__(self, label, *, alive: Annotated[int, GOLDEN_MEAN] = 0, space=None):
        super().__init__(label, space=space)
        self.alive = alive


def golden(space):
    return System(elements=[GoldenCells('cells', space=space)])


[automaton(golden(LatticeSpace(length, boundary='fixed')), kind='count').count for length in range(1, 8)]
```

The counts are 2, 3, 5, 8, 13, 21, and 34, the Fibonacci numbers. The kind `'entropy'` gives `log2 |X_F| / |F|` along
windows `F`, the number of bits per cell that the subshift allows.

```python continuation
windows = [LatticeSpace(24, boundary='fixed')]
system = golden(windows[0])
automaton(system, kind='entropy', windows=windows).entropy.round(3)
```

The entropy on a window of 24 cells is 0.704 bits per cell, which approaches the base 2 logarithm of the golden ratio,
0.694, as the window grows. Admissibility is an analysis kind, and a run does not enforce it. A run checks only that
every cell holds a value of the alphabet, so it runs from a configuration with a forbidden pattern.

```python continuation
forbidden = [1, 1, 0, 0, 0]
keep = local_rule('keep', TableSpec({(1,): 1}), LatticeNeighborhood(((0,),)))
held = System(elements=[GoldenCells('cells', alive=forbidden, space=LatticeSpace(5))], interactions=[keep])
automaton(held, kind='admissible', configuration=forbidden).admissible
```

The configuration `[1, 1, 0, 0, 0]` is not admissible, so the block returns `False`. A run that keeps every cell still
starts from it.

```python continuation
held.prepare().simulate(until=2, dt=1).get_values('cells.alive')[-1].astype(int).tolist()
```

The run ends at `[1, 1, 0, 0, 0]`, the forbidden pattern included. The other kinds read configurations and runs.

- `'admissible'` and `'violations'` check a configuration.
- `'count'` counts the admissible configurations of a window, and `'sample'` samples them by a Markov chain of
  single-cell moves.
- `'difference'` gives the cells where two runs differ after one cell changes.
- `entropy(frames, kind='rate')` of the frames of a run as columns gives the spatial entropy rate of every frame, which
  never increases under an automaton.

The library has `FreeGroupAutomatonSystem`, `LamplighterAutomatonSystem`, `HeisenbergAutomatonSystem`,
`BaumslagSolitarAutomatonSystem`, `GrigorchukAutomatonSystem`, the Laplacian `LinearRingAutomatonSystem`, and the golden
mean shift `SubshiftsSystem`, in `skcomplex.systems.computer_science.automata`. Their groups are written in the forms
above.
