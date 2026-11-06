# Analyse

Derive the equations of a system, and measure its runs, its data, and its structure.

## Analysing symbolically

Read the equations a system stands for, and solve them exactly where a closed form exists. The analysis works on the
symbolic laws, so it needs no run.

```python
from skcomplex.core.analysis import SymbolicAnalysis
from skcomplex.core.components.elements import PointElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System

oscillator = System(
    elements=[PointElement('p', x=1.0)],
    interactions=[RateInteraction('trap', target='p', rate=lambda s: -4.0 * s.target.x)],
    space=ContinuousSpace(n_dim=1),
)
SymbolicAnalysis().prepare(oscillator).run().equations_['p.x[0]']
SymbolicAnalysis('solve', use_initial_state=True).prepare(oscillator).run().solutions_['p.x[0]']
```

- `SymbolicAnalysis('equations')`, the default kind, returns the equations of a system as Sage relations.
- `SymbolicAnalysis('solve')` solves an ODE system exactly where Maxima can. It uses `desolve` for one equation and
  `desolve_system` for linear systems with constant coefficients. With `use_initial_state=True`, the initial state of
  the system gives the initial conditions at `t = 0`. Without it, the solution keeps its integration constants.

## Analysing dynamics and data

Measure what a system does: its stability and chaos, its fluctuations and information, its patterns, agents, and
networks. `skcomplex.core.analysis` measures complex dynamics in the same way for every field. Every measure takes a
system, or plain arrays from trajectories or measured data.

**Stability and chaos.** Find the fixed points of a system, how chaotic it is, and how its long-term behaviour changes
with a parameter.

```python
from skcomplex.core.analysis import LyapunovAnalysis, StabilityAnalysis
from skcomplex.core.experiments import OrbitDiagramExperiment
from skcomplex.systems import create_system, describe

StabilityAnalysis().prepare(create_system('lotka_volterra')).run().classifications_
LyapunovAnalysis(until=200).prepare(create_system('lorenz')).run().exponents_.round(2).tolist()
logistic_map = describe('logistic_map').factory
diagram = OrbitDiagramExperiment(logistic_map, 'map.r', [2.9, 3.2, 3.5], 'map.x', transient=500, keep=32, dt=1)
diagram.prepare().run().periods_.tolist()
```

- `StabilityAnalysis` finds the fixed points, the Jacobian, the stability, the nullclines, and the divergence of the
  symbolic vector field of a flow or a map.
- `LyapunovAnalysis` integrates the tangent dynamics, compiled with Numba.
- The experiments `OrbitDiagramExperiment` and `BasinStabilityExperiment` run a system over parameters or initial
  conditions.

**Sections.** Reduce a flow to a map by sampling it at chosen moments.

```python
import math

import numpy as np

from skcomplex.core.analysis import SectionAnalysis
from skcomplex.systems import create_system

section = SectionAnalysis('stroboscopic', period=2 * math.pi / 1.2, n_points=200)
section = section.prepare(create_system('duffing')).run()
len(np.unique(np.round(section.states_, 4), axis=0))
```

`SectionAnalysis('poincare')` finds the states where a state crosses a value in a direction, located like an event.
`SectionAnalysis('stroboscopic')` samples a periodically forced system once per period.

**Distributions and events.** Describe heavy tails, and cut series into avalanches or bursts.

```python
import numpy as np

from skcomplex.core.analysis import DistributionAnalysis, EventAnalysis, FitAnalysis

samples = (1 - np.random.default_rng(0).random(5000)) ** (-1 / 1.5)
fit = FitAnalysis('power', bootstrap=20, random_state=0).prepare(samples).run()
gini = DistributionAnalysis('gini').prepare(samples).run().gini_
round(fit.exponent_, 2), fit.p_value_, round(gini, 2)
EventAnalysis().prepare([0, 2, 3, 0, 0, 1, 0]).run().sizes_.tolist()
```

- `DistributionAnalysis` describes heavy tails by its `kind`: a `'histogram'` with logarithmic bins, the `'ccdf'`, the
  Hill `'tail-index'`, and the `'gini'` coefficient.
- `EventAnalysis` turns series into avalanches or bursts, with sizes, durations, and waiting times.
- `FitAnalysis(law='power')` adapts the `powerlaw` library. It chooses the lower cutoff by the Kolmogorov-Smirnov
  criterion of Clauset, Shalizi, and Newman, and accepts an upper cutoff. It uses the exact discrete likelihood for
  integers, and gives a bootstrap p-value. It compares the power law with the exponential, log-normal, stretched
  exponential, and truncated power-law tails of `compare_with`.

**Time series and information.** Measure the memory, diffusion, entropy, and information flow of series.

```python
import numpy as np

from skcomplex.core.analysis import EntropyAnalysis, InformationAnalysis, ScalingAnalysis

noise = np.random.default_rng(1).standard_normal(20_000)
ScalingAnalysis().prepare(np.column_stack([noise, np.cumsum(noise)])).run().exponent_.round(2).tolist()
periodic = [0, 0, 1, 1] * 200
[EntropyAnalysis(kind, dimension=4).prepare(periodic).run().entropy_.round(3).tolist() for kind in ('rate', 'excess')]
driver = np.random.default_rng(2).integers(0, 2, 5000)
transfer = InformationAnalysis('transfer').prepare(np.column_stack([driver, np.roll(driver, 1)])).run()
round(transfer.information_[0, 1], 2)
```

- `CorrelationAnalysis`, `SpectrumAnalysis`, `DisplacementAnalysis`, `ScalingAnalysis`, and `RollingAnalysis` describe
  memory and diffusion.
- The mean squared displacement of `DisplacementAnalysis` comes from an FFT over all lags of unwrapped positions. Its
  exponent and diffusion coefficient are fitted over a chosen range of lags.
- The fluctuation analyses come from `nolds`. Detrended fluctuation analysis of any order gives `H + 1` for an
  integrated series such as a random walk, as above. The rescaled range is the other kind.
- `RollingAnalysis` gives the variance, the lag-one autocorrelation, or the skewness in rolling windows, with its trend.
  A rising trend of these is an early warning of tipping.
- `InformationAnalysis` measures structure and information flow by its `kind`, such as `'mutual'` and `'transfer'`, as
  in Lindgren's information theory of complex systems.
- `EntropyAnalysis` gives the Shannon, block, rate, excess, permutation, and Lempel-Ziv entropies by its `kind`. Its
  setting `symbols` turns real series into symbols first: bins of equal width, equally populated bins, or ordinal
  patterns. Ordinal patterns are coded by the lexicographic rank of their permutation with `ordpy`, with a delay and of
  any length. The Lempel-Ziv parsing comes from `antropy`.

**Space and networks.** Measure clusters, fractals, and graphs.

```python
import networkx as nx
import numpy as np

from skcomplex.core.analysis import ClusterAnalysis, NetworkAnalysis, ScalingAnalysis

clusters = ClusterAnalysis('labels').prepare(np.random.default_rng(3).random((100, 100)) < 0.6).run()
clusters.spanning_, clusters.sizes_.max()
rows, columns = np.indices((512, 512))
round(ScalingAnalysis('box-counting').prepare((rows & columns) == columns).run().exponent_, 2)
NetworkAnalysis('summary').prepare(nx.watts_strogatz_graph(200, 6, 0.1, seed=0)).run().diameter_
```

`ClusterAnalysis` labels the clusters of a grid, with periodic edges. `CorrelationAnalysis` gives radially averaged
correlations, and `ScalingAnalysis` measures fractals, such as the Sierpinski triangle above. `NetworkAnalysis` measures
the degrees, the summary, and the centralities of NetworkX graphs and of the networks of trajectories.

### Kinds of data and signatures

Find which analyses apply to a trajectory or an array, and get a default picture of each kind of data it holds. Every
analysis consumes one kind of data, never a system or a field of science: a series, a jump path, a field, agent
snapshots, a network, or an ensemble of runs.

```python
import numpy as np

from skcomplex.core.analysis import DiscoveryAnalysis, SpectrumAnalysis
from skcomplex.systems import create_simulation

spins = create_simulation('ising', size=32).prepare().run(until=20).trajectory_
DiscoveryAnalysis().prepare(spins).kinds_
SpectrumAnalysis('structure-factor').prepare(spins).run().wavenumbers_[:3].round(3).tolist()
discovery = DiscoveryAnalysis().prepare(np.zeros((16, 16))).run()
discovery.readings_['series'], len(discovery.analyses_['field']) > 0
```

The same `SpectrumAnalysis` serves an Ising lattice and a reaction-diffusion field. The same
`EventAnalysis('excursions')` serves a sandpile and an earthquake system. Every analysis takes plain arrays or graphs,
so it works on measured data as well.

An analysis prepared on a trajectory extracts its kind of data itself:

- named series of states, reporters, or observables,
- the jump path of a discrete state,
- the frames of a lattice,
- the agents of a group, with their positions unwrapped across periodic sides,
- the network of every frame.

An ensemble of runs is one row per run and one column per common time, as `ExperimentResult.ensemble` returns it.

`DiscoveryAnalysis` prepares on a trajectory, a system, or an array, and keeps the kinds of data the object holds in
`kinds_`. Its run lists in `analyses_` the analyses of the library that apply under each kind, and in `readings_` how
they read it.

`SignatureAnalysis` computes one default analysis per kind of data of a trajectory, with its curves, axis labels, title,
and measured values. The gallery draws it for every example that names no analysis of its own.

```python
from skcomplex.core.analysis import SignatureAnalysis
from skcomplex.systems import create_simulation

predation = create_simulation('lotka_volterra').prepare().run(until=20.0).trajectory_
signatures = SignatureAnalysis().prepare(predation).run().signatures_
[(item.kind, item.title, item.xscale, list(item.values)) for item in signatures]
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

from skcomplex.core.analysis import ComplexityAnalysis, EntropyAnalysis, InformationAnalysis

rng = np.random.default_rng(0)
renyi = EntropyAnalysis('renyi', order=2, probabilities=True).prepare(np.full(8, 1 / 8)).run()
renyi.entropy_.round(3).tolist()
sample = EntropyAnalysis('sample', dimension=2).prepare(rng.standard_normal(2000)).run()
sample.entropy_.round(1).tolist()
plane = ComplexityAnalysis(dimension=4).prepare(rng.standard_normal(20_000)).run()
plane.entropy_.round(2).tolist(), plane.complexity_.round(2).tolist()
x = rng.standard_normal(5000)
y = np.concatenate([[0.0], 0.8 * x[:-1]]) + 0.6 * rng.standard_normal(5000)
transfer = InformationAnalysis('transfer', estimator='knn', base=math.e).prepare(np.column_stack([x, y])).run()
transfer.information_[0, 1] > transfer.information_[1, 0]
```

`EntropyAnalysis` measures every entropy by its `kind`:

- `'renyi'` and `'tsallis'` generalize the Shannon entropy of a distribution.
- `'sample'`, `'approximate'`, `'multiscale'`, and `'spectral'` come from `antropy`, and measure the regularity of a
  series.
- `'permutation'` reads ordinal patterns of series with a delay, or of two-dimensional fields in a window, in its
  Shannon, Rényi, or Tsallis `measure`. Discrete states such as spins have many equal values, which `ties='random'`
  ranks at random.
- `'ensemble'` follows the entropy of an ensemble and its divergence from a `reference` law frame by frame, as in the
  H-theorem.

`ComplexityAnalysis` places data on Rosso's complexity-entropy plane, between its bounding curves, with `ordpy`.

The kind `'active-storage'` of `InformationAnalysis` measures the information that the past of a series holds about its
next value. The kind `'transfer'` estimates the flow from every series to every other, by plug-in counts of symbols or
by the nearest-neighbour estimator of `ennemi` on continuous values. Above, `y` follows `x` with a lag, so more
information flows from `x` to `y` than back.

### Higher-order information and emergence

Tell whether many variables share information redundantly or synergistically, and whether a macro feature has a life of
its own. The measures of many variables take one row per sample and one column per variable.

```python continuation
from skcomplex.core.analysis import PartialInformationAnalysis, SurrogateTestAnalysis

bits = rng.integers(0, 2, (5000, 2))
xor = np.column_stack([bits, bits[:, 0] ^ bits[:, 1]])
round(InformationAnalysis('o-information').prepare(xor).run().information_, 2)
gate = np.column_stack([bits, bits[:, 0] & bits[:, 1]])
parts = PartialInformationAnalysis().prepare(gate).run()
round(parts.redundancy_, 2), round(parts.synergy_, 2)
measure = InformationAnalysis('o-information')
test = SurrogateTestAnalysis(measure, 'information_', n=19, alternative='less', random_state=0)
test.prepare(xor).run().p_value_
```

Among the kinds of `InformationAnalysis`:

- `'total-correlation'` counts the information shared at any order.
- `'dual-total-correlation'` counts the information shared by two or more variables.
- `'o-information'`, their difference, is positive when the variables are redundant and negative when they are
  synergistic, as for the XOR above.
- `'o-information-gradients'` gives the share of every variable.

They take the `'plugin'` estimator of `dit` for symbols, `'gaussian'` for the closed form of a fitted Gaussian, or
`'knn'` for chains of `ennemi` estimates.

- `PartialInformationAnalysis` splits what two sources of symbols tell about a target into redundant, unique, and
  synergistic parts, with a redundancy measure of `dit`.
- `LocalInformationAnalysis` returns the entropy, the mutual information, or the transfer entropy at every sample. Their
  means equal the global measures.
- `SurrogateTestAnalysis` compares the attribute of any of these analyses with its values on copies of the data whose
  variables are shuffled independently.

```python continuation
from skcomplex.core.analysis import EmergenceAnalysis, EpsilonMachineAnalysis

parity = rng.integers(0, 2, (5000, 4))
parity[:, -1] = (parity[:, :-1].sum(axis=1) + np.arange(5000) // 100) % 2
criteria = EmergenceAnalysis('criteria', partition=lambda micro: micro.sum(axis=1) % 2).prepare(parity).run()
criteria.psi_ > 0.5
closure = EmergenceAnalysis('closure', partition=[0, 0, 1, 1]).prepare(rng.integers(0, 4, 5000)).run()
round(closure.gap_, 2)
golden = [1]
for _ in range(20_000):
    golden.append(1 if golden[-1] == 0 else int(rng.random() < 0.5))
machine = EpsilonMachineAnalysis(max_length=4).prepare(golden).run()
len(machine.histories_), round(machine.statistical_complexity_, 2)
```

The kind `'criteria'` of `EmergenceAnalysis` computes the criteria of Rosas et al. (2020) for a macro feature of micro
variables, given by `partition`:

- Ψ is positive when the feature predicts its own future beyond what each micro variable predicts.
- Δ is positive when it predicts a micro variable beyond the micro variables.
- Γ is zero for a feature decoupled from every micro variable.

The kind `'closure'` takes a partition of micro states, as a label per state or a function of the micro series. It
compares how well the macro past and the micro past predict the macro future. A gap of zero means that the
coarse-graining is closed.

`EpsilonMachineAnalysis` reconstructs the causal states of a discrete series by causal-state splitting, with their
transitions, statistical complexity, and entropy rate. The golden mean process above never repeats a 0.

### Invariants of dynamics

Reconstruct the attractor of a series and measure its dimension, its chaos, and its recurrences.

```python
import numpy as np

from skcomplex.core.analysis import (
    EmbeddingAnalysis,
    ErgodicityAnalysis,
    LyapunovAnalysis,
    RecurrenceAnalysis,
    ScalingAnalysis,
)

points = np.random.default_rng(1).random((3000, 2))
round(ScalingAnalysis('correlation-dimension').prepare(points).run().exponent_, 1)
round(LyapunovAnalysis().prepare([0.906, 0.0, -14.572]).run().kaplan_yorke_dimension_, 2)
chain = ErgodicityAnalysis('ulam', edges=np.linspace(0, 1, 21), samples=200)
chain.prepare(lambda x: 4 * x * (1 - x)).run().density_.round(1)[:3].tolist()
sine = EmbeddingAnalysis(delay=15, dimension=2).prepare(np.sin(np.arange(1000) * 0.1)).run().points_
round(RecurrenceAnalysis().prepare(sine).run().determinism_, 2)
```

- `EmbeddingAnalysis` builds delay coordinates of a series. It puts the delay at the first minimum of the auto mutual
  information, and the dimension where the false nearest neighbours fall below a threshold.
- `ScalingAnalysis('correlation-dimension')` counts close pairs of any point set with a k-d tree, so it serves embedded
  series and agent positions alike.
- The Kaplan-Yorke dimension and the Kolmogorov-Sinai entropy of `LyapunovAnalysis` follow from a Lyapunov spectrum,
  such as the Lorenz spectrum above. `LyapunovAnalysis('largest')` estimates the largest exponent from one series, with
  Rosenstein's method of `nolds`.
- `ErgodicityAnalysis` of kind `'invariant-density'` or `'ulam'` estimates the invariant density of a map.
- `RecurrenceAnalysis` gives the recurrence matrix, rate, determinism, laminarity, line lengths, trapping time, and line
  entropy of embedded series.

### Surrogates fitted to data

Learn a system from data, then run it, analyse it, or forecast with it like a system you wrote by hand.

```python
import numpy as np

from skcomplex.core.analysis import ModeDecompositionAnalysis, ReservoirAnalysis, SparseIdentificationAnalysis
from skcomplex.core.engine.compile import PassCompiler

times = np.arange(0.0, 10.0, 0.01)
spiral = np.exp(-0.1 * times)[:, None] * np.column_stack([np.cos(times), np.sin(times)])
decomposition = ModeDecompositionAnalysis(['x', 'y'], dt=0.01).prepare(spiral).run()
report = PassCompiler().prepare(decomposition.system_).report_
np.sort(decomposition.eigenvalues_.imag).round(3).tolist(), report.ok
sparse = SparseIdentificationAnalysis(['x', 'y'], degree=1, threshold=0.05, horizon=50, dt=0.01)
sparse = sparse.prepare(spiral).run()
sparse.features_, sparse.coefficients_.round(2).tolist(), sparse.forecast_.shape
reservoir = ReservoirAnalysis(['x', 'y'], horizon=50).prepare(spiral).run()
type(reservoir.system_.time).__name__, reservoir.forecast_.shape
```

`ModeDecompositionAnalysis`, `SparseIdentificationAnalysis`, and `ReservoirAnalysis` prepare on one or more
trajectories or arrays sampled at a uniform step `dt`. Their run sets these attributes:

- `system_`, a `System` of one surrogate element, which a compiler checks and `Simulation` runs like a hand-written
  system,
- `training_error_` and `validation_error_`, the relative errors of the fit on the training samples and on the end of
  every series held out by `validation`,
- `forecast_`, the variables over `horizon` samples after the last one.

The three differ in what they fit:

- `ModeDecompositionAnalysis` fits dynamic mode decomposition with PyDMD. It sets the continuous eigenvalues
  `ln(mu) / dt` in `eigenvalues_`, the modes, their amplitudes, and a surrogate under continuous time with the matrix
  rate law `dg/dt = L g`. A scikit-learn transformer passed as `dictionary`, such as `PolynomialFeatures(2)`, lifts the
  variables to observables first, which is extended DMD. With `inputs`, the names of the variables that are control
  inputs, it fits dynamic mode decomposition with control, `g(t + dt) = A g(t) + B u(t)`, with the `DMDc` of PyDMD. It
  then sets `operator_` and `input_matrix_`, and its surrogate takes every input as a bounded adaptive parameter, which
  a `schedule` of `Simulation` drives. The system `koopman_model_predictive_control` uses these matrices to predict.
- `SparseIdentificationAnalysis` fits sparse identification of nonlinear dynamics with PySINDy. It sets the monomials of
  the variables in `features_`, their coefficients in `coefficients_`, and a surrogate under continuous time with one
  symbolic rate law per variable.
- `ReservoirAnalysis` fits a next-generation reservoir or an echo state network from ReservoirPy, with a ridge readout
  of the increments over one sample. It works on standardized variables unless `standardize` is false. Its surrogate
  runs under discrete time, one step per sample, holds the memory of the reservoir as states, and starts at the last
  sample. So a run of `system_` is the autonomous forecast.

### Stochastic processes

Measure passage times, Markov chains, fluctuations, irreversibility, and rare events.

```python
import numpy as np

from skcomplex.core.analysis import IrreversibilityAnalysis, LargeDeviationAnalysis, MarkovAnalysis

ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
stationary = MarkovAnalysis('stationary', matrix='rate').prepare(ring).run().stationary_
production = IrreversibilityAnalysis().prepare(ring).run().entropy_production_
stationary.round(2).tolist(), round(production, 4)
gap = MarkovAnalysis('spectral-gap', matrix='transition').prepare([[0.9, 0.1], [0.2, 0.8]]).run()
round(gap.spectral_gap_, 2)
tilted = LargeDeviationAnalysis(weights=[[1.0]], tilts=np.linspace(-3, 3, 601), values=3.0)
round(float(tilted.prepare([[2.0]]).run().rate_function_[0]), 3)
```

- `PassageAnalysis` measures first passages, absorption, and returns of paths, and the committor and mean first-passage
  time of a transition matrix, by its `kind`. `EventAnalysis('laminar-durations')` measures laminar phases.
- `MarkovAnalysis` estimates transition matrices from paths, and reads transition and rate matrices with `matrix`. It
  gives their stationary laws, spectral gaps, and relaxation times, and Markov state models and implied timescales with
  `pydtmc`.
- `DistributionAnalysis('dispersion-index')`, `ScalingAnalysis('variance-mean')`, and `EventAnalysis('counts')`
  describe fluctuations. `KramersMoyalAnalysis` estimates the drift and diffusion of a series with kernels over its
  increments.
- `IrreversibilityAnalysis` measures entropy production from rates, jump paths, Langevin paths, or transition counts.
  It tests detailed balance and the fluctuation theorem. It checks the thermodynamic uncertainty relation: the squared
  relative fluctuation of a current times the entropy produced is at least 2. It also gives the response to a field and
  the fluctuation-dissipation ratio. The ring above turns one way twice as often as the other, so it produces entropy.
- `LargeDeviationAnalysis` gives the scaled cumulant generating function in three ways: from a tilted generator, from
  blocks of one run, or by cloning a population of trajectories of a system through the steppers of the simulation. The
  last serves systems whose tilted generators are too large to write down. It also gives the Legendre transform at
  `values`, rate functions from histograms at several sizes, and reweighted samples.
- `CorrelationAnalysis('two-time')` and `CollapseAnalysis` measure aging.
- `FitAnalysis` with the laws `'extreme-value'`, `'stable'`, and the named laws such as `'exponential'`, and
  `DistributionAnalysis('records')`, describe extremes and limit laws.

Cloning advances every clone of a population by an interval, weights it by `exp(s A)` for the increment `A` of a
time-additive observable, and resamples the population by the weights. The observable is the dotted name of a state
whose increment is counted, or a function of the trajectory of each interval. A warning names a population whose
effective size collapses, a sign that the estimate is biased.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.analysis import IrreversibilityAnalysis, LargeDeviationAnalysis
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.components.system import System
from skcomplex.core.specs import StateSpec


class CounterElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.n = n

    def _get_laws(self, s):
        return [JumpLaw(1.0, {s.n: 1})]


counter = System(elements=[CounterElement('counter')])
cloning = LargeDeviationAnalysis(
    'cloning', observable='counter.n', tilts=0.5, population=200, until=10.0, random_state=0
)
cloning.prepare(counter).run().scgf_.round(2).tolist(), round(float(np.exp(0.5) - 1), 2)
currents = np.random.default_rng(0).normal(10.0, np.sqrt(30.0), 5000)
relation = IrreversibilityAnalysis('uncertainty-relation', entropy=10.0, random_state=0)
relation.prepare(currents).run().holds_
```

The counter jumps at rate 1, a Poisson process, so the cloning estimate matches the exact value `exp(s) - 1`.

### Emergence and ergodicity

Find the coarse-graining at which a Markov chain is most causally effective, and test whether runs explore the same
states.

```python
import numpy as np

from skcomplex.core.analysis import EmergenceAnalysis

noisy = np.zeros((8, 8))
noisy[:7, :7], noisy[7, 7] = 1 / 7, 1.0
round(EmergenceAnalysis('effective-information').prepare(noisy).run().information_, 3)
found = EmergenceAnalysis('causal').prepare(noisy).run()
found.partition_.tolist(), round(found.macro_information_, 3), round(found.gain_, 3)
found.macro_matrix_.tolist()
```

`EmergenceAnalysis` reads a transition matrix for its kinds `'effective-information'` and `'causal'`.

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

`ErgodicityAnalysis` of kind `'breaking'`, `'metric'`, or `'distance'` diagnoses whether the runs of an ensemble explore
the same states. The ensemble has one row per run, as `ExperimentResult.ensemble` returns it.

```python
import numpy as np

from skcomplex.core.analysis import ErgodicityAnalysis

rng = np.random.default_rng(0)
mixing = rng.standard_normal((100, 5000))
frozen = rng.choice([-1.0, 1.0], size=(100, 1)) + 0.1 * rng.standard_normal((100, 5000))
results = []
for runs in (mixing, frozen):
    breaking = ErgodicityAnalysis('breaking').prepare(runs).run().ergodicity_breaking_
    metric = ErgodicityAnalysis('metric').prepare(runs).run().exponent_
    distance = ErgodicityAnalysis('distance').prepare(runs).run().ergodicity_distance_
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

from skcomplex.core.analysis import ClusterAnalysis, DistributionAnalysis, DomainAnalysis, StabilityAnalysis
from skcomplex.systems import create_system

relation = StabilityAnalysis('dispersion').prepare(create_system('brusselator', shape=(128,), b=1.9)).run()
relation.band_.min().round(2), relation.band_.max().round(2)

stripes = np.cos(2 * np.pi * np.arange(64) / 8)[:, None] * np.ones(64)
DomainAnalysis('wavelength').prepare(stripes).run().wavelength_.tolist()
grid = np.random.default_rng(2).random((128, 128)) < 0.7
ClusterAnalysis('percolation').prepare(grid).run().spanning_
samples = np.random.default_rng(3).standard_normal(10_000)
DistributionAnalysis('moment-ratio').prepare(samples).run().cumulant_.round(1).tolist()
```

- `SpectrumAnalysis('structure-factor')` averages the power of a field radially, on a periodic grid of any dimension.
  `CorrelationAnalysis('length')` fits its Ornstein-Zernike form, or the decay of the correlation function.
- `DomainAnalysis` measures coarsening with the kinds `'size'` and `'growth'`, the wavelength and orientation of
  patterns with `'wavelength'`, and the interfaces between domains with `'interface-density'`.
- `ScalingAnalysis('variance-mean')` measures the number fluctuations of density fields in boxes of growing side.
- `ClusterAnalysis` with the kinds `'sizes'`, `'percolation'`, and `'spanning'` describes clusters of grids and
  components of graphs.
- `DistributionAnalysis('moment-ratio')`, with `CollapseAnalysis` of the kinds `'crossings'` and `'scaling'`, locates
  transitions by finite-size scaling. The variance from `DistributionAnalysis('moments', absolute=True)` of an order
  parameter, scaled by the system size, is the susceptibility. Its peak over a temperature sweep of a spin system
  locates the transition too.

`StabilityAnalysis('dispersion')` predicts patterns from the laws of a field system before running it. It linearises
them around a uniform state, and turns every Laplacian into its eigenvalue on each Fourier mode of the lattice. It
returns the growth rate and frequency of every mode, exact for the discretization of the system. A band of growing
modes is a Turing instability, and a growing mode with a nonzero frequency travels.

### Agents

Measure how agents align, how they spread in space, and how groups segregate.

```python
import numpy as np

from skcomplex.core.analysis import AlignmentAnalysis, CoarseGrainAnalysis, NeighborAnalysis
from skcomplex.systems import create_simulation

flock = create_simulation('boids').prepare().run(until=5).trajectory_
AlignmentAnalysis('mean-vector').prepare(flock).run().order_.shape
positions = np.random.default_rng(4).random((1000, 2))
ratio = NeighborAnalysis('clark-evans', box=[1.0, 1.0]).prepare(positions).run().ratio_
field = CoarseGrainAnalysis([8, 8], box=[1.0, 1.0]).prepare(positions).run().field_
ratio.round(1).tolist(), float(field.sum())
```

- `AlignmentAnalysis` measures the alignment of every frame of agents, of a field of angles or vectors, or of series of
  phases. It uses the length of the mean vector, the mean rotation, the axial order, or the mean phase. The active
  matter systems name them the polar, milling, and nematic orders, and the synchronization systems the Kuramoto order.
- `DistributionAnalysis` and `FitAnalysis` read the speeds of moving agents. So `FitAnalysis(law='maxwell')` in three
  dimensions and `FitAnalysis(law='rayleigh')` in two compare the speeds of a gas, such as the system `ideal_gas`, with
  the Maxwell-Boltzmann law.
- `NeighborAnalysis` describes point patterns in open or periodic boxes by their nearest-neighbour `'distances'`, their
  `'clark-evans'` ratio, and their `'radial-distribution'`.
- `CoarseGrainAnalysis` turns agents into density or polarization fields for the field analyses.
- `NeighborAnalysis('like-share')` and `DomainAnalysis('interface-density')` measure segregation on grids and networks.
  `DistributionAnalysis('total-variation')` gives the distance between the distributions of two groups across units,
  the dissimilarity index of segregation studies.

### Criticality

Extract avalanches, fit their laws, and estimate how close a process is to criticality.

```python
import numpy as np

from skcomplex.core.analysis import EventAnalysis

found = EventAnalysis('excursions').prepare([0, 1, 3, 2, 0, 0, 4, 1, 0]).run()
found.sizes_.tolist(), found.durations_.tolist()
rng = np.random.default_rng(5)
activity = np.zeros(5000)
activity[0] = 50
for t in range(1, activity.size):
    activity[t] = rng.poisson(0.8 * activity[t - 1] + 10)
round(EventAnalysis('branching').prepare(activity).run().branching_ratio_, 1)
```

- `EventAnalysis('excursions')` extracts avalanches from activity series or relaxation records, and
  `FitAnalysis(law='power')` fits their size and duration laws.
- `EventAnalysis('branching')` estimates the branching ratio by the multistep regression of `mrestimator`, which stays
  unbiased under subsampling. The process above branches with ratio 0.8.
- The exponents `tau_S` of the sizes and `tau_T` of the durations predict the mean size against the duration,
  `<S>(T) ~ T^((tau_T - 1) / (tau_S - 1))`. This is the crackling-noise relation of avalanches, such as those of the
  system `sandpile`.
- `DistributionAnalysis('moments')` gives the order parameter and its variance over a sweep, and
  `ScalingAnalysis('power')` fits the order-parameter exponent.
- `CorrelationAnalysis('overlap')` compares replicas.

### Network analysis

Measure the structure of a network, its communities, its motifs, and how it looks at coarser scales.

```python
import networkx as nx

from skcomplex.core.analysis import MotifAnalysis, NetworkAnalysis

club = nx.karate_club_graph()
communities = NetworkAnalysis('communities', random_state=0).prepare(club).run()
communities.modularity_.round(2).tolist(), MotifAnalysis().prepare(club).run().counts_[0]['triangle']
```

`NetworkAnalysis` measures the network of a graph, or of every frame of a trajectory, by its `kind`:

- the `'degrees'`, with their moments and tail exponent,
- the `'assortativity'`, the `'centrality'`, the `'neighbor-degree'`, and the `'summary'`,
- the `'communities'`, by Louvain modularity maximization or label propagation, with their agreement with a reference
  partition.

`MotifAnalysis` counts motifs of three and four nodes, and scores them against degree-preserving edge switches or the
configuration model. The system `network_structure_analysis` records a network with these measures.

`RenormalizationAnalysis` coarse-grains a graph, or the network of the last frame of a trajectory with its node states
as attributes. It sets `graphs_`, `degrees_`, `exponents_`, and `clustering_` per scale.

```python
import networkx as nx

from skcomplex.core.analysis import RenormalizationAnalysis

blocks = nx.disjoint_union(nx.complete_graph(20), nx.complete_graph(20))
blocks.add_edge(0, 20)
scale = RenormalizationAnalysis(steps=1).prepare(blocks).run().graphs_[1]
scale.number_of_nodes(), scale.number_of_edges()
```

- The kind `'laplacian'` joins two nodes where the diffusion operator `exp(-time L)` has an entry at least the smaller
  of their diagonal entries, and every component of these joins is a supernode. It runs at `time`, or at the smallest
  peak of the specific heat that merges nodes. Above, the two linked cliques become two linked supernodes.
- The kind `'geometric'` merges `block` consecutive nodes by their `angle`. A supernode sits at their angles weighted by
  `k^max(1, beta)`, with `k` the degree.

In both kinds, two supernodes link when any of their nodes link.

`ScalingAnalysis('allometric')` fits the exponent of a quantity against system size across runs. It reads a mapping
from size to samples, such as the results of a `GridDesign` sweep, or two columns.

```python
import numpy as np

from skcomplex.core.analysis import ScalingAnalysis

rng = np.random.default_rng(0)
samples = {size: size**1.15 * rng.lognormal(0.0, 0.3, 4) for size in np.geomspace(100, 1e5, 50)}
fit = ScalingAnalysis('allometric').prepare(samples).run()
round(fit.exponent_, 1), bool(fit.interval_[0] < 1.15 < fit.interval_[1])
```

It regresses the logarithms of the pairs and sets `exponent_`, with the bootstrap confidence interval `interval_` of the
level `confidence` over `resamples` resamples.
