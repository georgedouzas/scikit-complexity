# Analyse

Derive the equations of a system, and measure its runs, its data, and its structure.

## Analysing symbolically

Read the equations a system stands for, and solve them exactly where a closed form exists. The analysis works on the
symbolic laws, so it needs no run.

```python
from typing import Annotated

from skcomplex.core.analysis import symbolic
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.specs import StateSpec


class AtomElement(Element):
    def __init__(
        self, label, *, x: Annotated[object, StateSpec(vector=True, order=2, role='position')] = 0.0, x_d1: object = 0.0
    ):
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1


oscillator = CompositeSystem(
    elements=[AtomElement('p', x=1.0)],
    interactions=[RateInteraction('trap', target='p', rate=lambda s: -4.0 * s.target.x)],
    space=ContinuousSpace(n_dim=1),
)
symbolic(oscillator).equations['p.x[0]']
symbolic(oscillator, kind='solve', use_initial_state=True).solutions['p.x[0]']
```

- `symbolic(x, kind='equations')`, the default kind, returns the equations of a system as Sage relations.
- `symbolic(x, kind='solve')` solves an ODE system exactly where Maxima can. It uses `desolve` for one equation and
  `desolve_system` for linear systems with constant coefficients. With `use_initial_state=True`, the initial state of
  the system gives the initial conditions at `t = 0`. Without it, the solution keeps its integration constants.

## Analysing dynamics and data

Measure what a system does: its stability and chaos, its fluctuations and information, its patterns, agents, and
networks. `skcomplex.core.analysis` measures complex dynamics in the same way for every field.

Every analysis is a function, as the metrics of scikit-learn are. It takes the source first, a system, a trajectory, or
plain arrays of measured data, then its settings as keyword arguments, and returns a frozen result whose fields hold
what it measured. Below, `x` stands for the source, so `distribution(x, kind='gini')` is the Gini coefficient of the
values of `x`. An analysis checks the tags of a system first, so `lyapunov` of a stochastic system raises a `TagError`
that names the tag it needs.

**Stability and chaos.** Find the fixed points of a system, how chaotic it is, and how its long-term behaviour changes
with a parameter.

```python
from skcomplex.core.analysis import lyapunov, stability
from skcomplex.core.experiments import orbit_diagram
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem, LorenzSystem

stability(LotkaVolterraSystem()).classifications
lyapunov(LorenzSystem(), until=200).exponents.round(2).tolist()
logistic_map = LogisticMapSystem()
orbit_diagram(logistic_map, 'map.r', [2.9, 3.2, 3.5], 'map.x', transient=500, keep=32, dt=1).periods.tolist()
```

- `stability` finds the fixed points, the Jacobian, the stability, the nullclines, and the divergence of the
  symbolic vector field of a flow or a map.
- `lyapunov` integrates the tangent dynamics, compiled with Numba.
- The experiments `orbit_diagram` and `basin_stability` run a system over parameters or initial
  conditions.

**Sections.** Reduce a flow to a map by sampling it at chosen moments.

```python
import math

import numpy as np

from skcomplex.core.analysis import section
from skcomplex.systems.physics.nonlinear_oscillators import DuffingSystem

points = section(DuffingSystem(), kind='stroboscopic', period=2 * math.pi / 1.2, n_points=200)
len(np.unique(np.round(points.states, 4), axis=0))
```

`section(x, kind='poincare')` finds the states where a state crosses a value in a direction, located like an event.
`section(x, kind='stroboscopic')` samples a periodically forced system once per period.

**Distributions and events.** Describe heavy tails, and cut series into avalanches or bursts.

```python
import numpy as np

from skcomplex.core.analysis import distribution, events, fit_law

samples = (1 - np.random.default_rng(0).random(5000)) ** (-1 / 1.5)
fit = fit_law(samples, law='power', bootstrap=20, random_state=0)
gini = distribution(samples, kind='gini').gini
round(fit.exponent, 2), fit.p_value, round(gini, 2)
events([0, 2, 3, 0, 0, 1, 0]).sizes.tolist()
```

- `distribution` describes heavy tails by its `kind`: a `'histogram'` with logarithmic bins, the `'ccdf'`, the
  Hill `'tail-index'`, and the `'gini'` coefficient.
- `events` turns series into avalanches or bursts, with sizes, durations, and waiting times.
- `fit_law(x, law='power')` adapts the `powerlaw` library. It chooses the lower cutoff by the Kolmogorov-Smirnov
  criterion of Clauset, Shalizi, and Newman, and accepts an upper cutoff. It uses the exact discrete likelihood for
  integers, and gives a bootstrap p-value. It compares the power law with the exponential, log-normal, stretched
  exponential, and truncated power-law tails of `compare_with`.

**Time series and information.** Measure the memory, diffusion, entropy, and information flow of series.

```python
import numpy as np

from skcomplex.core.analysis import entropy, information, scaling

noise = np.random.default_rng(1).standard_normal(20_000)
scaling(np.column_stack([noise, np.cumsum(noise)])).exponent.round(2).tolist()
periodic = [0, 0, 1, 1] * 200
[entropy(periodic, kind=kind, dimension=4).entropy.round(3).tolist() for kind in ('rate', 'excess')]
driver = np.random.default_rng(2).integers(0, 2, 5000)
transfer = information(np.column_stack([driver, np.roll(driver, 1)]), kind='transfer')
round(transfer.information[0, 1], 2)
```

- `correlation`, `spectrum`, `displacement`, `scaling`, and `rolling` describe
  memory and diffusion.
- The mean squared displacement of `displacement` comes from an FFT over all lags of unwrapped positions. Its
  exponent and diffusion coefficient are fitted over a chosen range of lags.
- The fluctuation analyses come from `nolds`. Detrended fluctuation analysis of any order gives `H + 1` for an
  integrated series such as a random walk, as above. The rescaled range is the other kind.
- `rolling` gives the variance, the lag-one autocorrelation, or the skewness in rolling windows, with its trend.
  A rising trend of these is an early warning of tipping.
- `information` measures structure and information flow by its `kind`, such as `'mutual'` and `'transfer'`, as
  in Lindgren's information theory of complex systems.
- `entropy` gives the Shannon, block, rate, excess, permutation, and Lempel-Ziv entropies by its `kind`. Its
  setting `symbols` turns real series into symbols first: bins of equal width, equally populated bins, or ordinal
  patterns. Ordinal patterns are coded by the lexicographic rank of their permutation with `ordpy`, with a delay and of
  any length. The Lempel-Ziv parsing comes from `antropy`.

**Space and networks.** Measure clusters, fractals, and graphs.

```python
import networkx as nx
import numpy as np

from skcomplex.core.analysis import clusters, network_measures, scaling

labelled = clusters(np.random.default_rng(3).random((100, 100)) < 0.6, kind='labels')
labelled.spanning, labelled.sizes.max()
rows, columns = np.indices((512, 512))
round(scaling((rows & columns) == columns, kind='box-counting').exponent, 2)
network_measures(nx.watts_strogatz_graph(200, 6, 0.1, seed=0), kind='summary').diameter
```

`clusters` labels the clusters of a grid, with periodic edges. `correlation` gives radially averaged
correlations, and `scaling` measures fractals, such as the Sierpinski triangle above. `network_measures` measures
the degrees, the summary, and the centralities of NetworkX graphs and of the networks of trajectories.

### Kinds of data and signatures

Find which analyses apply to a trajectory or an array, and get a default picture of each kind of data it holds. Every
analysis consumes one kind of data, never a system or a field of science: a series, a jump path, a field, agent
snapshots, a network, or an ensemble of runs.

```python
import numpy as np

from skcomplex.core.analysis import applicable_analyses, spectrum
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.statistical_mechanics import IsingSystem

spins = simulate(IsingSystem(size=32), until=20)
applicable_analyses(spins).kinds
spectrum(spins, kind='structure-factor').wavenumbers[:3].round(3).tolist()
discovery = applicable_analyses(np.zeros((16, 16)))
discovery.readings['series'], len(discovery.analyses['field']) > 0
```

The same `spectrum` serves an Ising lattice and a reaction-diffusion field. The same
`events(x, kind='excursions')` serves a sandpile and an earthquake system. Every analysis takes plain arrays or graphs,
so it works on measured data as well.

An analysis given a trajectory extracts its kind of data itself:

- named series of states, reporters, or observables,
- the jump path of a discrete state,
- the frames of a lattice,
- the agents of a group, with their positions unwrapped across periodic sides,
- the network of every frame.

An ensemble of runs is one row per run and one column per common time, as `ExperimentResult.get_ensemble` returns it.

`applicable_analyses` takes a trajectory, a system, or an array, and returns the kinds of data the object holds in
`kinds`. It lists in `analyses` the analyses of the library that apply under each kind, and in `readings` how
they read it.

`signatures` computes one default analysis per kind of data of a trajectory, with its curves, axis labels, title,
and measured values. The gallery draws it for every example that names no analysis of its own.

```python
from skcomplex.core.analysis import signatures
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

predation = simulate(LotkaVolterraSystem(), until=20.0)
[(item.kind, item.title, item.xscale, list(item.values)) for item in signatures(predation).signatures]
```

The signature of each kind of data is this:

- series: the power spectrum and the autocorrelation time, or the return map of a map,
- jump paths: the waiting times and the dispersion index,
- fields: the structure factor of the last frame, with its correlation length or peak wavelength,
- moving agents: the polar and milling orders, or else their nearest-neighbour distances,
- networks: the degree distribution.

### Complexity and information

Measure how regular, complex, or informative a series is, and how information flows between series.

```python
import math

import numpy as np

from skcomplex.core.analysis import complexity, entropy, information

rng = np.random.default_rng(0)
renyi = entropy(np.full(8, 1 / 8), kind='renyi', order=2, probabilities=True)
renyi.entropy.round(3).tolist()
sample = entropy(rng.standard_normal(2000), kind='sample', dimension=2)
sample.entropy.round(1).tolist()
plane = complexity(rng.standard_normal(20_000), dimension=4)
plane.entropy.round(2).tolist(), plane.complexity.round(2).tolist()
x = rng.standard_normal(5000)
y = np.concatenate([[0.0], 0.8 * x[:-1]]) + 0.6 * rng.standard_normal(5000)
transfer = information(np.column_stack([x, y]), kind='transfer', estimator='knn', base=math.e)
transfer.information[0, 1] > transfer.information[1, 0]
```

`entropy` measures every entropy by its `kind`:

- `'renyi'` and `'tsallis'` generalize the Shannon entropy of a distribution.
- `'sample'`, `'approximate'`, `'multiscale'`, and `'spectral'` come from `antropy`, and measure the regularity of a
  series.
- `'permutation'` reads ordinal patterns of series with a delay, or of two-dimensional fields in a window, in its
  Shannon, Rényi, or Tsallis `measure`. Discrete states such as spins have many equal values, which `ties='random'`
  ranks at random.
- `'ensemble'` follows the entropy of an ensemble and its divergence from a `reference` law frame by frame, as in the
  H-theorem.

`complexity` places data on Rosso's complexity-entropy plane, between its bounding curves, with `ordpy`.

The kind `'active-storage'` of `information` measures the information that the past of a series holds about its
next value. The kind `'transfer'` estimates the flow from every series to every other, by plug-in counts of symbols or
by the nearest-neighbour estimator of `ennemi` on continuous values. Above, `y` follows `x` with a lag, so more
information flows from `x` to `y` than back.

### Higher-order information and emergence

Tell whether many variables share information redundantly or synergistically, and whether a macro feature has a life of
its own. The measures of many variables take one row per sample and one column per variable.

```python continuation
from skcomplex.core.analysis import bind_analysis, partial_information, surrogate_test

bits = rng.integers(0, 2, (5000, 2))
xor = np.column_stack([bits, bits[:, 0] ^ bits[:, 1]])
round(information(xor, kind='o-information').information, 2)
gate = np.column_stack([bits, bits[:, 0] & bits[:, 1]])
parts = partial_information(gate)
round(parts.redundancy, 2), round(parts.synergy, 2)
measure = bind_analysis(information, kind='o-information')
surrogate_test(xor, analysis=measure, attribute='information', n=19, alternative='less', random_state=0).p_value
```

Among the kinds of `information`:

- `'total-correlation'` counts the information shared at any order.
- `'dual-total-correlation'` counts the information shared by two or more variables.
- `'o-information'`, their difference, is positive when the variables are redundant and negative when they are
  synergistic, as for the XOR above.
- `'o-information-gradients'` gives the share of every variable.

They take the `'plugin'` estimator of `dit` for symbols, `'gaussian'` for the closed form of a fitted Gaussian, or
`'knn'` for chains of `ennemi` estimates.

- `partial_information` splits what two sources of symbols tell about a target into redundant, unique, and
  synergistic parts, with a redundancy measure of `dit`.
- `local_information` returns the entropy, the mutual information, or the transfer entropy at every sample. Their
  means equal the global measures.
- `surrogate_test` compares the attribute of any of these analyses with its values on copies of the data whose
  variables are shuffled independently.

```python continuation
from skcomplex.core.analysis import emergence, epsilon_machine

parity = rng.integers(0, 2, (5000, 4))
parity[:, -1] = (parity[:, :-1].sum(axis=1) + np.arange(5000) // 100) % 2
criteria = emergence(parity, kind='criteria', partition=lambda micro: micro.sum(axis=1) % 2)
criteria.psi > 0.5
closure = emergence(rng.integers(0, 4, 5000), kind='closure', partition=[0, 0, 1, 1])
round(closure.gap, 2)
golden = [1]
for _ in range(20_000):
    golden.append(1 if golden[-1] == 0 else int(rng.random() < 0.5))
machine = epsilon_machine(golden, max_length=4)
len(machine.histories), round(machine.statistical_complexity, 2)
```

The kind `'criteria'` of `emergence` computes the criteria of Rosas et al. (2020) for a macro feature of micro
variables, given by `partition`:

- Ψ is positive when the feature predicts its own future beyond what each micro variable predicts.
- Δ is positive when it predicts a micro variable beyond the micro variables.
- Γ is zero for a feature decoupled from every micro variable.

The kind `'closure'` takes a partition of micro states, as a label per state or a function of the micro series. It
compares how well the macro past and the micro past predict the macro future. A gap of zero means that the
coarse-graining is closed.

`epsilon_machine` reconstructs the causal states of a discrete series by causal-state splitting, with their
transitions, statistical complexity, and entropy rate. The golden mean process above never repeats a 0.

### Invariants of dynamics

Reconstruct the attractor of a series and measure its dimension, its chaos, and its recurrences.

```python
import numpy as np

from skcomplex.core.analysis import embedding, ergodicity, lyapunov, recurrence, scaling

points = np.random.default_rng(1).random((3000, 2))
round(scaling(points, kind='correlation-dimension').exponent, 1)
round(lyapunov([0.906, 0.0, -14.572]).kaplan_yorke_dimension, 2)
chain = ergodicity(lambda x: 4 * x * (1 - x), kind='ulam', edges=np.linspace(0, 1, 21), samples=200)
chain.density.round(1)[:3].tolist()
sine = embedding(np.sin(np.arange(1000) * 0.1), delay=15, dimension=2).points
round(recurrence(sine).determinism, 2)
```

- `embedding` builds delay coordinates of a series. It puts the delay at the first minimum of the auto mutual
  information, and the dimension where the false nearest neighbours fall below a threshold.
- `scaling(x, kind='correlation-dimension')` counts close pairs of any point set with a k-d tree, so it serves embedded
  series and agent positions alike.
- The Kaplan-Yorke dimension and the Kolmogorov-Sinai entropy of `lyapunov` follow from a Lyapunov spectrum,
  such as the Lorenz spectrum above. `lyapunov(x, kind='largest')` estimates the largest exponent from one series, with
  Rosenstein's method of `nolds`.
- `ergodicity` of kind `'invariant-density'` or `'ulam'` estimates the invariant density of a map.
- `recurrence` gives the recurrence matrix, rate, determinism, laminarity, line lengths, trapping time, and line
  entropy of embedded series.

### Surrogates fitted to data

Learn a system from data, then run it, analyse it, or forecast with it like a system you wrote by hand.

```python
import numpy as np

from skcomplex.core.analysis import mode_decomposition, reservoir, sparse_identification
from skcomplex.core.engine.runtime import simulate

times = np.arange(0.0, 10.0, 0.01)
spiral = np.exp(-0.1 * times)[:, None] * np.column_stack([np.cos(times), np.sin(times)])
decomposition = mode_decomposition(spiral, variables=['x', 'y'], dt=0.01)
surrogate_run = simulate(decomposition.system, until=1.0, dt=0.1)
np.sort(decomposition.eigenvalues.imag).round(3).tolist(), len(surrogate_run.times)
sparse = sparse_identification(spiral, variables=['x', 'y'], degree=1, threshold=0.05, horizon=50, dt=0.01)
sparse.features, sparse.coefficients.round(2).tolist(), sparse.forecast.shape
echo = reservoir(spiral, variables=['x', 'y'], horizon=50)
type(echo.system.time).__name__, echo.forecast.shape
```

`mode_decomposition`, `sparse_identification`, and `reservoir` take one or more trajectories or arrays sampled at a
uniform step `dt`. Their results hold these fields:

- `system`, a `CompositeSystem` of one surrogate element, which `simulate` runs like a hand-written
  system,
- `training_error` and `validation_error`, the relative errors of the fit on the training samples and on the end of
  every series held out by `validation`,
- `forecast`, the variables over `horizon` samples after the last one.

The three differ in what they fit:

- `mode_decomposition` fits dynamic mode decomposition with PyDMD. It sets the continuous eigenvalues
  `ln(mu) / dt` in `eigenvalues`, the modes, their amplitudes, and a surrogate under continuous time with the matrix
  rate law `dg/dt = L g`. A scikit-learn transformer passed as `dictionary`, such as `PolynomialFeatures(2)`, lifts the
  variables to observables first, which is extended DMD. With `inputs`, the names of the variables that are control
  inputs, it fits dynamic mode decomposition with control, `g(t + dt) = A g(t) + B u(t)`, with the `DMDc` of PyDMD. It
  then sets `operator` and `input_matrix`, and its surrogate takes every input as a bounded adaptive parameter, which
  a `schedule` of `simulate` drives. The system `koopman_model_predictive_control` uses these matrices to predict.
- `sparse_identification` fits sparse identification of nonlinear dynamics with PySINDy. It sets the monomials of
  the variables in `features`, their coefficients in `coefficients`, and a surrogate under continuous time with one
  symbolic rate law per variable.
- `reservoir` fits a next-generation reservoir or an echo state network from ReservoirPy, with a ridge readout
  of the increments over one sample. It works on standardized variables unless `standardize` is false. Its surrogate
  runs under discrete time, one step per sample, holds the memory of the reservoir as states, and starts at the last
  sample. So a run of `system` is the autonomous forecast.

### Stochastic processes

Measure passage times, Markov chains, fluctuations, irreversibility, and rare events.

```python
import numpy as np

from skcomplex.core.analysis import irreversibility, large_deviations, markov

ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
stationary = markov(ring, kind='stationary', matrix='rate').stationary
production = irreversibility(ring).entropy_production
stationary.round(2).tolist(), round(production, 4)
gap = markov([[0.9, 0.1], [0.2, 0.8]], kind='spectral-gap', matrix='transition')
round(gap.spectral_gap, 2)
tilted = large_deviations([[2.0]], weights=[[1.0]], tilts=np.linspace(-3, 3, 601), values=3.0)
round(float(tilted.rate_function[0]), 3)
```

- `passage` measures first passages, absorption, and returns of paths, and the committor and mean first-passage
  time of a transition matrix, by its `kind`. `events(x, kind='laminar-durations')` measures laminar phases.
- `markov` estimates transition matrices from paths, and reads transition and rate matrices with `matrix`. It
  gives their stationary laws, spectral gaps, and relaxation times, and Markov state models and implied timescales with
  `pydtmc`.
- `distribution(x, kind='dispersion-index')`, `scaling(x, kind='variance-mean')`, and `events(x, kind='counts')`
  describe fluctuations. `kramers_moyal` estimates the drift and diffusion of a series with kernels over its
  increments.
- `irreversibility` measures entropy production from rates, jump paths, Langevin paths, or transition counts.
  It tests detailed balance and the fluctuation theorem. It checks the thermodynamic uncertainty relation: the squared
  relative fluctuation of a current times the entropy produced is at least 2. It also gives the response to a field and
  the fluctuation-dissipation ratio. The ring above turns one way twice as often as the other, so it produces entropy.
- `large_deviations` gives the scaled cumulant generating function in three ways: from a tilted generator, from
  blocks of one run, or by cloning a population of trajectories of a system through the steppers of the simulation. The
  last serves systems whose tilted generators are too large to write down. It also gives the Legendre transform at
  `values`, rate functions from histograms at several sizes, and reweighted samples.
- `correlation(x, kind='two-time')` and `data_collapse` measure aging.
- `fit_law` with the laws `'extreme-value'`, `'stable'`, and the named laws such as `'exponential'`, and
  `distribution(x, kind='records')`, describe extremes and limit laws.

Cloning advances every clone of a population by an interval, weights it by `exp(s A)` for the increment `A` of a
time-additive observable, and resamples the population by the weights. The observable is the dotted name of a state
whose increment is counted, or a function of the trajectory of each interval. A warning names a population whose
effective size collapses, a sign that the estimate is biased.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.analysis import irreversibility, large_deviations
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.specs import StateSpec


class CounterElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.n = n

    def _get_laws(self, s):
        return [JumpLaw(1.0, {s.n: 1})]


counter = CompositeSystem(elements=[CounterElement('counter')])
cloning = large_deviations(
    counter, kind='cloning', observable='counter.n', tilts=0.5, population=200, until=10.0, random_state=0
)
cloning.scgf.round(2).tolist(), round(float(np.exp(0.5) - 1), 2)
currents = np.random.default_rng(0).normal(10.0, np.sqrt(30.0), 5000)
irreversibility(currents, kind='uncertainty-relation', entropy=10.0, random_state=0).holds
```

The counter jumps at rate 1, a Poisson process, so the cloning estimate matches the exact value `exp(s) - 1`.

### Emergence and ergodicity

Find the coarse-graining at which a Markov chain is most causally effective, and test whether runs explore the same
states.

```python
import numpy as np

from skcomplex.core.analysis import emergence

noisy = np.zeros((8, 8))
noisy[:7, :7], noisy[7, 7] = 1 / 7, 1.0
round(emergence(noisy, kind='effective-information').information, 3)
found = emergence(noisy, kind='causal')
found.partition.tolist(), round(found.macro_information, 3), round(found.gain, 3)
found.macro_matrix.tolist()
```

`emergence` reads a transition matrix for its kinds `'effective-information'` and `'causal'`.

- The effective information measures how much a transition matrix constrains the next state. It is the mutual
  information, in bits, between a uniform intervention on the current state and the next state. It equals the
  determinism of the matrix minus its degeneracy. Rows with no observed transition are left out, with an
  `UnobservedStatesWarning` that names their states.
- The kind `'causal'` reports the gain in effective information of a partition over the micro states. Without a
  partition it searches for one, greedily by merging the pair of blocks with the largest gain, or exhaustively for at
  most 10 states. The partition is an integer label per state or a function of the array of states. The macro matrix
  sums the transitions within every block, weighting the states of a block uniformly or by the stationary distribution.

In the chain of Hoel, Albantakis, and Tononi above, seven noisy states merge into one deterministic macro state, which
carries more effective information than the micro states.

`ergodicity` of kind `'breaking'`, `'metric'`, or `'distance'` diagnoses whether the runs of an ensemble explore
the same states. The ensemble has one row per run, as `ExperimentResult.get_ensemble` returns it.

```python
import numpy as np

from skcomplex.core.analysis import ergodicity

rng = np.random.default_rng(0)
mixing = rng.standard_normal((100, 5000))
frozen = rng.choice([-1.0, 1.0], size=(100, 1)) + 0.1 * rng.standard_normal((100, 5000))
results = []
for runs in (mixing, frozen):
    breaking = ergodicity(runs, kind='breaking').ergodicity_breaking
    metric = ergodicity(runs, kind='metric').exponent
    distance = ergodicity(runs, kind='distance').ergodicity_distance
    results.append((round(breaking, 2), round(metric, 1), round(distance, 1)))
results
```

- The ergodicity-breaking parameter is the variance of the time averages across runs over the variance of all values.
  It tends to 0 for ergodic runs, and stays near 1 for runs frozen in separate states.
- The fluctuation metric of Thirumalai and Mountain is the variance of the running time averages. It decays as `1 / t`
  for ergodic runs, and levels off for frozen ones.
- The ergodicity distance is the median total-variation distance between the values of one run over time and the values
  of the whole ensemble.

### Fields and lattices

Predict patterns before a run, and measure domains, wavelengths, clusters, and transitions of fields after it.

```python
import numpy as np

from skcomplex.core.analysis import clusters, distribution, domains, stability
from skcomplex.systems.chemistry.pattern_formation import BrusselatorSystem

relation = stability(BrusselatorSystem(shape=(128,), b=1.9), kind='dispersion')
relation.band.min().round(2), relation.band.max().round(2)

stripes = np.cos(2 * np.pi * np.arange(64) / 8)[:, None] * np.ones(64)
domains(stripes, kind='wavelength').wavelength.tolist()
grid = np.random.default_rng(2).random((128, 128)) < 0.7
clusters(grid, kind='percolation').spanning
samples = np.random.default_rng(3).standard_normal(10_000)
distribution(samples, kind='moment-ratio').cumulant.round(1).tolist()
```

- `spectrum(x, kind='structure-factor')` averages the power of a field radially, on a periodic grid of any dimension.
  `correlation(x, kind='length')` fits its Ornstein-Zernike form, or the decay of the correlation function.
- `domains` measures coarsening with the kinds `'size'` and `'growth'`, the wavelength and orientation of
  patterns with `'wavelength'`, and the interfaces between domains with `'interface-density'`.
- `scaling(x, kind='variance-mean')` measures the number fluctuations of density fields in boxes of growing side.
- `clusters` with the kinds `'sizes'`, `'percolation'`, and `'spanning'` describes clusters of grids and
  components of graphs.
- `distribution(x, kind='moment-ratio')`, with `data_collapse` of the kinds `'crossings'` and `'scaling'`, locates
  transitions by finite-size scaling. The variance from `distribution(x, kind='moments', absolute=True)` of an order
  parameter, scaled by the system size, is the susceptibility. Its peak over a temperature sweep of a spin system
  locates the transition too.

`stability(x, kind='dispersion')` predicts patterns from the laws of a field system before running it. It linearises
them around a uniform state, and turns every Laplacian into its eigenvalue on each Fourier mode of the lattice. It
returns the growth rate and frequency of every mode, exact for the discretization of the system. A band of growing
modes is a Turing instability, and a growing mode with a nonzero frequency travels.

### Agent populations

Measure how agents align, how they spread in space, and how groups segregate.

```python
import numpy as np

from skcomplex.core.analysis import alignment, coarse_grain, neighbors
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.collective_behaviour import BoidsSystem

flock = simulate(BoidsSystem(), until=5)
alignment(flock, kind='mean-vector').order.shape
positions = np.random.default_rng(4).random((1000, 2))
ratio = neighbors(positions, kind='clark-evans', box=[1.0, 1.0]).ratio
field = coarse_grain(positions, shape=[8, 8], box=[1.0, 1.0]).field
ratio.round(1).tolist(), float(field.sum())
```

- `alignment` measures the alignment of every frame of agents, of a field of angles or vectors, or of series of
  phases. It uses the length of the mean vector, the mean rotation, the axial order, or the mean phase. The active
  matter systems name them the polar, milling, and nematic orders, and the synchronization systems the Kuramoto order.
- `distribution` and `fit_law` read the speeds of moving agents. So `fit_law(x, law='maxwell')` in three
  dimensions and `fit_law(x, law='rayleigh')` in two compare the speeds of a gas, such as the system `ideal_gas`, with
  the Maxwell-Boltzmann law.
- `neighbors` describes point patterns in open or periodic boxes by their nearest-neighbour `'distances'`, their
  `'clark-evans'` ratio, and their `'radial-distribution'`.
- `coarse_grain` turns agents into density or polarization fields for the field analyses.
- `neighbors(x, kind='like-share')` and `domains(x, kind='interface-density')` measure segregation on grids and
  networks. `distribution(x, kind='total-variation')` gives the distance between the distributions of two groups across
  units, the dissimilarity index of segregation studies.

### Criticality

Extract avalanches, fit their laws, and estimate how close a process is to criticality.

```python
import numpy as np

from skcomplex.core.analysis import events

found = events([0, 1, 3, 2, 0, 0, 4, 1, 0], kind='excursions')
found.sizes.tolist(), found.durations.tolist()
rng = np.random.default_rng(5)
activity = np.zeros(5000)
activity[0] = 50
for t in range(1, activity.size):
    activity[t] = rng.poisson(0.8 * activity[t - 1] + 10)
round(events(activity, kind='branching').branching_ratio, 1)
```

- `events(x, kind='excursions')` extracts avalanches from activity series or relaxation records, and
  `fit_law(x, law='power')` fits their size and duration laws.
- `events(x, kind='branching')` estimates the branching ratio by the multistep regression of `mrestimator`, which stays
  unbiased under subsampling. The process above branches with ratio 0.8.
- The exponents `tau_S` of the sizes and `tau_T` of the durations predict the mean size against the duration,
  `<S>(T) ~ T^((tau_T - 1) / (tau_S - 1))`. This is the crackling-noise relation of avalanches, such as those of the
  system `sandpile`.
- `distribution(x, kind='moments')` gives the order parameter and its variance over a sweep, and
  `scaling(x, kind='power')` fits the order-parameter exponent.
- `correlation(x, kind='overlap')` compares replicas.

### Network analysis

Measure the structure of a network, its communities, its motifs, and how it looks at coarser scales.

```python
import networkx as nx

from skcomplex.core.analysis import motifs, network_measures

club = nx.karate_club_graph()
communities = network_measures(club, kind='communities', random_state=0)
communities.modularity.round(2).tolist(), motifs(club).counts[0]['triangle']
```

`network_measures` measures the network of a graph, or of every frame of a trajectory, by its `kind`:

- the `'degrees'`, with their moments and tail exponent,
- the `'assortativity'`, the `'centrality'`, the `'neighbor-degree'`, and the `'summary'`,
- the `'communities'`, by Louvain modularity maximization or label propagation, with their agreement with a reference
  partition.

`motifs` counts motifs of three and four nodes, and scores them against degree-preserving edge switches or the
configuration model. The system `network_structure_analysis` records a network with these measures.

`renormalization` coarse-grains a graph, or the network of the last frame of a trajectory with its node states
as attributes. It sets `graphs`, `degrees`, `exponents`, and `clustering` per scale.

```python
import networkx as nx

from skcomplex.core.analysis import renormalization

blocks = nx.disjoint_union(nx.complete_graph(20), nx.complete_graph(20))
blocks.add_edge(0, 20)
scale = renormalization(blocks, steps=1).graphs[1]
scale.number_of_nodes(), scale.number_of_edges()
```

- The kind `'laplacian'` joins two nodes where the diffusion operator `exp(-time L)` has an entry at least the smaller
  of their diagonal entries, and every component of these joins is a supernode. It runs at `time`, or at the smallest
  peak of the specific heat that merges nodes. Above, the two linked cliques become two linked supernodes.
- The kind `'geometric'` merges `block` consecutive nodes by their `angle`. A supernode sits at their angles weighted by
  `k^max(1, beta)`, with `k` the degree.

In both kinds, two supernodes link when any of their nodes link.

`autocatalytic_sets` finds the largest autocatalytic set, the largest RAF, of a catalytic reaction network read
from a network snapshot or from chosen `frames` of a trajectory:

```python
import numpy as np

from skcomplex.core.analysis import autocatalytic_sets
from skcomplex.core.specs import NetworkSnapshotSpec

reactions = np.array([[0, 1, 2], [2, 0, 3], [3, 1, 4]])
snapshot = NetworkSnapshotSpec(np.arange(5), np.array([3, 4, 0]), np.array([2, 3, 4]), np.arange(3), reactions)
autocatalytic_sets(snapshot).reactions[0].tolist()
```

- Every hyperedge is a reaction that produces its last member from the others, and the -1 that pad a smaller
  hyperedge are left out.
- An edge from `c` to `p` makes `c` a catalyst of every reaction that produces `p`.
- The food set is `food`, or the nodes present that no reaction produces.
- The algorithm of Hordijk and Steel removes the reactions with a reactant outside the closure of the food set or
  without a catalyst in it, until none is removed. It sets `reactions`, `products`, `sizes`, `fractions`, and
  `exists` per frame. The system `tap_catalysis` grows such a network.

`scaling(x, kind='allometric')` fits the exponent of a quantity against system size across runs. It reads a mapping
from size to samples, such as the results of a `sweep` over a grid, or two columns.

```python
import numpy as np

from skcomplex.core.analysis import scaling

rng = np.random.default_rng(0)
samples = {size: size**1.15 * rng.lognormal(0.0, 0.3, 4) for size in np.geomspace(100, 1e5, 50)}
fit = scaling(samples, kind='allometric')
round(fit.exponent, 1), bool(fit.interval[0] < 1.15 < fit.interval[1])
```

It regresses the logarithms of the pairs and sets `exponent`, with the bootstrap confidence interval `interval` of the
level `confidence` over `resamples` resamples.
