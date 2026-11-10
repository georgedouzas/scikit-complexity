# Analyse a system

This tutorial measures the predator and prey model of the [build tutorial](build_a_custom_system/index.md) and the pelt
records it describes. It finds the analyses that apply with `applicable_analyses`, reads the equations, the fixed
points, and the cycle of the model, learns the equations back from data, and explains how to read each result. The
second half covers the analyses that need another kind of [system](glossary.md#system), such as a [chaotic](glossary.md#chaos) flow, a lattice, a flock, or a
network, each with a small example of its own.

## How an analysis works

Every analysis is a function, as the metrics of scikit-learn are. It takes the source first, a system, a [trajectory](glossary.md#trajectory), or
plain arrays of measured data, then its settings as keyword arguments, and returns a frozen result whose fields hold
what it measured. Below, `x` stands for the source, so `distribution(x, kind='gini')` is the Gini coefficient of the
values of `x`.

The running example is the catalog class `LotkaVolterraSystem` with the rates that Carpenter (2018) estimated from the
pelt records of the Hudson's Bay Company, run for a hundred years from the records of 1900.

```python
from skcomplex.core.analysis import applicable_analyses
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
trajectory = simulate(system, until=100, dt=0.1)
found = applicable_analyses(trajectory)
found.kinds, found.readings['series'], len(found.analyses['series']) > 10
```

[applicable_analyses][skcomplex.core.analysis.applicable_analyses] reads the [kinds of data](glossary.md#kind-of-data) a source holds and lists the
analyses that apply to each. The trajectory holds one kind, `series`, read as "a series of values, or the columns of an
array as series", and many analyses apply under it, such as `spectrum`, `recurrence`, and `entropy`. The analyses of
the equations, such as `symbolic`, `stability`, and `lyapunov`, take the system itself. They check its [tags](glossary.md#tag) first, so
they refuse a system they cannot read.

```python continuation
from skcomplex.core.analysis import lyapunov
from skcomplex.core.errors import TagError
from skcomplex.systems.general.reactions import ReactionNetworkSystem

reactions = {
    'births': ({'hares': 1}, {'hares': 2}, 0.55),
    'hunting': ({'hares': 1, 'lynx': 1}, {'lynx': 1}, 0.028),
    'feeding': ({'hares': 1, 'lynx': 1}, {'hares': 1, 'lynx': 2}, 0.024),
    'deaths': ({'lynx': 1}, {}, 0.80),
}
jumps = ReactionNetworkSystem(species={'hares': 30, 'lynx': 4}, reactions=reactions, mode='jump')
try:
    lyapunov(jumps, until=20)
except TagError as error:
    message = str(error)
message
```

The same model as random events, as the tutorial [Simulate](simulate.md#random-runs) runs it, is [stochastic](glossary.md#stochastic). The
message says that `lyapunov` of kind `'spectrum'` needs the randomness `deterministic`, but the system has the
randomness `stochastic`. A [Lyapunov exponent](glossary.md#lyapunov-exponent) measures how fast nearby paths separate, which is defined for a
[deterministic](glossary.md#deterministic) flow. The fix is the deterministic model, or an analysis of the series of a random run.

## The equations

[symbolic][skcomplex.core.analysis.symbolic] reads the equations a system stands for. It works on the symbolic [laws](glossary.md#law),
so it needs no run.

```python continuation
from skcomplex.core.analysis import symbolic

symbolic(system).equations
```

The result holds one equation per [state](glossary.md#state), as Sage relations. The prey change at
`r_growth*x_prey - a_predation*x_predators*x_prey`, and the predators at
`b_predation*x_predators*x_prey - d_death*x_predators`. These are the equations of Lotka and Volterra, with each
parameter named after the part that holds it.

The kind `'solve'` solves the equations exactly where a closed form exists. The Lotka and Volterra equations have none,
so a harmonic oscillator shows it, a particle on a spring whose acceleration is `-4 x`.

```python continuation
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
equation = symbolic(oscillator).equations['p.x[0]']
equation, symbolic(oscillator, kind='solve', use_initial_state=True).solutions['p.x[0]']
```

The equation is `diff(x_p_0(t), t, t) == -4.0*x_p_0(t)`, and its solution from the initial position 1 is
`x_p_0(t) == cos(2*t)`.

- `symbolic(x, kind='equations')`, the default kind, returns the equations of a system as Sage relations.
- `symbolic(x, kind='solve')` solves an ODE system exactly where Maxima can. It uses `desolve` for one equation and
  `desolve_system` for linear systems with constant coefficients. With `use_initial_state=True`, the initial state of
  the system gives the initial conditions at `t = 0`. Without it, the solution keeps its integration constants.

## Fixed points and stability

A [fixed point](glossary.md#fixed-point) is a state where every rate is zero, so a system that starts there stays there. Its stability says
whether nearby states move toward it, away from it, or around it. [stability][skcomplex.core.analysis.stability] finds
the fixed points of the symbolic equations and classifies them.

```python continuation
from skcomplex.core.analysis import stability

fixed = stability(system)
fixed.fixed_points, fixed.classifications, [values.round(3).tolist() for values in fixed.eigenvalues]
```

The model has two fixed points.

- At zero hares and zero lynx it is a saddle. The eigenvalues of its [Jacobian](glossary.md#jacobian), the matrix of the derivatives of the
  rates, are 0.55 and -0.8: a few hares grow away from it, while a few lynx alone die out.
- At `d / b` hares and `r / a` lynx, 33.3 and 19.6 thousand with the fitted rates, it is a centre. Its eigenvalues
  `±0.663i` have no real part, so the populations circle it and neither grow nor shrink on average.

The imaginary part 0.663 is an angular frequency. Small cycles around the centre last `2 pi / 0.663`, about 9.5 years.
The result also holds the Jacobian, the nullclines, where one rate is zero, and the divergence of the vector field.

## Chaos and long-term behaviour

The Lyapunov exponents measure how fast two nearby paths separate. A positive exponent means chaos, a negative one
contraction, and zero a direction along which paths neither separate nor approach.

```python continuation
lyapunov(system, until=200).exponents.round(2).tolist()
```

Both exponents of the pelt model are zero to two decimals, 0.0 and -0.0, as for every closed orbit: the cycles neither
attract nor repel. `lyapunov` integrates the tangent dynamics, compiled with Numba. A chaotic flow and a map that
doubles its period show the other readings.

```python continuation
from skcomplex.core.analysis import lyapunov
from skcomplex.core.experiments import orbit_diagram
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem, LorenzSystem

exponents = lyapunov(LorenzSystem(), until=200).exponents.round(2).tolist()
logistic_map = LogisticMapSystem()
periods = orbit_diagram(logistic_map, 'map.r', [2.9, 3.2, 3.5], 'map.x', transient=500, keep=32, dt=1).periods
exponents, periods.tolist()
```

The Lorenz system has the exponents 0.9, 0.0, and -14.57. The positive one makes it chaotic. The logistic map
`x -> r x (1 - x)` settles on a cycle of period 1, 2, and 4 at `r` of 2.9, 3.2, and 3.5, the start of its cascade of
period doublings. The experiments `orbit_diagram` and `basin_stability` run a system over parameters or initial
conditions, as the tutorial [Run experiments](run_experiments.md) shows.

## Sections

A section reduces a flow to a map by keeping the states at chosen moments. A [Poincaré section](glossary.md#section) keeps the moments when a
state crosses a value in a direction, here the hares crossing the 33.33 thousand of the centre.

```python continuation
from skcomplex.core.analysis import section

crossings = section(system, kind='poincare', state='prey.x', value=33.33, until=100)
gaps = crossings.times[1:] - crossings.times[:-1]
crossings.states.round(2)[0].tolist(), gaps.round(2)[:3].tolist()
```

Every crossing falls on the same point, 33.33 thousand hares and 3.96 thousand lynx, so the orbit is closed. The time
between crossings is the period of the cycle, 10.57 years, longer than the 9.5 years of small cycles because this
cycle is large. A stroboscopic section samples a periodically forced system once per period of the force, as for the
Duffing oscillator, a spring with a cubic force driven at the angular frequency 1.2.

```python continuation
import math

import numpy as np

from skcomplex.core.analysis import section
from skcomplex.systems.physics.nonlinear_oscillators import DuffingSystem

points = section(DuffingSystem(), kind='stroboscopic', period=2 * math.pi / 1.2, n_points=200)
len(np.unique(np.round(points.states, 4), axis=0))
```

The 200 sampled states are all distinct, so the forced oscillator does not settle on a periodic orbit.
`section(x, kind='poincare')` finds the crossings like an event, and `section(x, kind='stroboscopic')` samples once
per period.

## Series of a run and of the records

The series analyses read the columns of a trajectory or of an array of data alike, so the model and the records
answer the same question. Here the question is the length of the cycle.

```python continuation
import numpy as np

from skcomplex.core.analysis import spectrum

hares = np.array(
    [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1]
    + [40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
)
lynx = np.array(
    [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4]
    + [8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]
)
records = np.column_stack([hares, lynx])
model = spectrum(simulate(system, until=200, dt=0.1), dt=0.1)
data = spectrum(records, dt=1.0)
[round(float(1 / result.frequencies[result.power[0].argmax()]), 2) for result in (model, data)]
```

The records are the pelts of 1900 to 1920, in thousands, as distributed with the case study of Carpenter (2018,
*Predator-Prey Population Dynamics: the Lotka-Volterra model in Stan*) in the file
`knitr/lotka-volterra/hudson-bay-lynx-hare.csv` of the Stan example models. That file cites the lecture notes of
Howard (2009) at Texas A&M University, which trace the numbers through the notes of Mahaffy (Math 636,
San Diego State University) to Purves, Sadava, Orians, and Heller, *Life: The Science of Biology*.

`spectrum` returns the power spectral density of every series, the share of its variance at each frequency, by Welch's
method. Its `dt` is the time between samples. The period of the strongest frequency is 10.24 years for the model and
10.5 years for the records. Both are coarse, because a spectrum resolves frequencies no finer than one over the length
of the series, which is twenty years for the records.

Two more series analyses describe the shape of the cycle.

```python continuation
from skcomplex.core.analysis import embedding, entropy, recurrence

prey = simulate(system, until=100, dt=0.1).get_values('prey.x')
delayed = embedding(prey)
order = entropy(prey, kind='permutation', dimension=4).entropy
delayed.delay, delayed.dimension, round(recurrence(delayed.points).determinism, 2), order.round(2).tolist()
```

- `embedding` rebuilds a cycle from one series by delay coordinates, the value now and the values some samples back.
  It chooses the delay of 3 samples and the dimension 2.
- `recurrence` finds the pairs of times at which the rebuilt states lie close. Its determinism, the share of
  [recurrences](glossary.md#recurrence) that form diagonal lines, is 0.99, close to 1 for a periodic series.
- `entropy(x, kind='permutation')` counts the orders of four consecutive values. Its normalized value, 0.28, is low,
  because a smooth cycle repeats few orders. Noise would give a value near 1.

[signatures][skcomplex.core.analysis.signatures] computes one default analysis per kind of data of a trajectory, with
its curves, axis labels, title, and measured values. The gallery draws it for every example that names no analysis of
its own.

```python continuation
from skcomplex.core.analysis import signatures

[(item.kind.value, item.title, item.xscale, list(item.values)) for item in signatures(trajectory).signatures]
```

The model has one signature, the power spectrum on a logarithmic axis, with the autocorrelation time of `prey.x`, the
lag at which the series stops resembling itself. The signature of each kind of data is this:

- series: the power spectrum and the autocorrelation time, or the return map of a map,
- [jump](glossary.md#jump) paths: the waiting times and the dispersion index,
- fields: the structure factor of the last [frame](glossary.md#frame), with its correlation length or peak wavelength,
- moving [agents](glossary.md#agent): the polar and milling orders, or else their nearest-neighbour distances,
- networks: the degree distribution.

## Equations learned from data

A [surrogate](glossary.md#surrogate) is a system learned from data, which runs, forecasts, and is analysed like a system written by hand.
Sparse identification of nonlinear dynamics fits the rates of every variable as a sparse sum of candidate terms, here
the monomials of the hares and the lynx up to degree 2.

```python continuation
from skcomplex.core.analysis import sparse_identification

learned = sparse_identification(trajectory, variables=['prey.x', 'predators.x'], degree=2, threshold=0.01, dt=0.1)
learned.features, learned.coefficients.round(3).tolist()
```

The features are `1`, `prey_x`, `predators_x`, `prey_x^2`, `prey_x predators_x`, and `predators_x^2`. The coefficients
of the hares are 0.549 for `prey_x` and -0.028 for the product, and those of the lynx are -0.798 for `predators_x`
and 0.024 for the product. The fit recovers the four rates of the model, and every other term is zero. The yearly
records are short and noisy, so the same fit is rougher.

```python continuation
rough = sparse_identification(records, variables=['hares', 'lynx'], degree=2, threshold=0.005, dt=1.0)
rough.coefficients.round(3).tolist(), round(rough.training_error, 2)
```

On the records the hares grow at 0.434 and are eaten at -0.019 per lynx, and the lynx die at -1.734 and gain 0.016 per
hare. The signs match the model, but the fit adds constant terms, 1.662 and 1.187, and a square of the lynx, and its
relative training error is 0.27. Twenty yearly points cannot pin down six terms per variable. The tutorial on
[calibration](glossary.md#calibration) fits the model itself to the records instead.

Two more surrogates learn other forms. A decaying spiral shows them, since it is linear.

```python
import numpy as np

from skcomplex.core.analysis import mode_decomposition, reservoir, sparse_identification
from skcomplex.core.engine.runtime import simulate

times = np.arange(0.0, 10.0, 0.01)
spiral = np.exp(-0.1 * times)[:, None] * np.column_stack([np.cos(times), np.sin(times)])
decomposition = mode_decomposition(spiral, variables=['x', 'y'], dt=0.01)
surrogate_run = simulate(decomposition.system, until=1.0, dt=0.1)
sparse = sparse_identification(spiral, variables=['x', 'y'], degree=1, threshold=0.05, horizon=50, dt=0.01)
echo = reservoir(spiral, variables=['x', 'y'], horizon=50)
(
    np.sort(decomposition.eigenvalues.imag).round(3).tolist(),
    len(surrogate_run.times),
    sparse.coefficients.round(2).tolist(),
    type(echo.system.time).__name__,
    echo.forecast.shape,
)
```

The decomposition finds the frequencies -1.0 and 1.0 of the spiral, and its surrogate runs for 11 frames. Sparse
identification finds `dx/dt = -0.1 x - y` and `dy/dt = x - 0.1 y`. The reservoir runs under discrete time and
forecasts 50 samples of both variables.

`mode_decomposition`, `sparse_identification`, and `reservoir` take one or more trajectories or arrays sampled at a
uniform step `dt`. Their results hold these fields:

- `system`, a `CompositeSystem` of one surrogate [element](glossary.md#element), which `simulate` runs like a hand-written system,
- `training_error` and `validation_error`, the relative errors of the fit on the training samples and on the end of
  every series held out by `validation`,
- `forecast`, the variables over `horizon` samples after the last one.

The three differ in what they fit:

- `mode_decomposition` fits dynamic mode decomposition with PyDMD. It sets the continuous eigenvalues
  `ln(mu) / dt` in `eigenvalues`, the modes, their amplitudes, and a surrogate under continuous time with the matrix
  rate law `dg/dt = L g`. A scikit-learn transformer passed as `dictionary`, such as `PolynomialFeatures(2)`, lifts the
  variables to [observables](glossary.md#observable) first, which is extended DMD. With `inputs`, the names of the variables that are control
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

## Kinds of data

Every analysis consumes one kind of data, never a system or a field of science: a series, a jump path, a field, agent
snapshots, a network, or an [ensemble](glossary.md#ensemble) of runs. The pelt model holds series alone, and a lattice of spins shows another
kind.

```python
import numpy as np

from skcomplex.core.analysis import applicable_analyses, spectrum
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.statistical_mechanics import IsingSystem

spins = simulate(IsingSystem(size=32), until=20)
wavenumbers = spectrum(spins, kind='structure-factor').wavenumbers[:3].round(3).tolist()
discovery = applicable_analyses(np.zeros((16, 16)))
applicable_analyses(spins).kinds, wavenumbers, discovery.kinds
```

The Ising lattice holds a `field`, and its structure factor starts at the wavenumbers 0.196, 0.393, and 0.589, steps of
`2 pi / 32`. A square array is read both ways, as one grid, a `field`, and as columns, `series`. The same `spectrum`
serves an Ising lattice and a reaction-diffusion field, and the same `events(x, kind='excursions')` serves a sandpile
and an earthquake system. Every analysis takes plain arrays or graphs, so it works on measured data as well.

An analysis given a trajectory extracts its kind of data itself:

- named series of states, reporters, or observables,
- the jump path of a discrete state,
- the frames of a lattice,
- the agents of a [group](glossary.md#group), with their positions unwrapped across periodic sides,
- the network of every frame.

An ensemble of runs is one row per run and one column per common time, as `ExperimentResult.get_ensemble` returns it.

## Distributions and events

Heavy tails, where large values are far more frequent than under a normal law, and series that come in bursts call for
their own measures.

```python
import numpy as np

from skcomplex.core.analysis import distribution, events, fit_law

samples = (1 - np.random.default_rng(0).random(5000)) ** (-1 / 1.5)
fit = fit_law(samples, law='power', bootstrap=20, random_state=0)
gini = distribution(samples, kind='gini').gini
round(fit.exponent, 2), fit.p_value, round(gini, 2), events([0, 2, 3, 0, 0, 1, 0]).sizes.tolist()
```

The samples follow a power law of density exponent 2.5, which the fit recovers, with a bootstrap p-value of 0.7, so the
power law is not rejected. Their Gini coefficient, 0.48, measures inequality between 0 for equal values and 1 for one
value holding everything. The series `0, 2, 3, 0, 0, 1, 0` holds two bursts, of sizes 5 and 1.

- `distribution` describes heavy tails by its `kind`: a `'histogram'` with logarithmic bins, the `'ccdf'`, the
  Hill `'tail-index'`, and the `'gini'` coefficient.
- `events` turns series into avalanches or bursts, with sizes, durations, and waiting times.
- `fit_law(x, law='power')` adapts the `powerlaw` library. It chooses the lower cutoff by the Kolmogorov-Smirnov
  criterion of Clauset, Shalizi, and Newman, and accepts an upper cutoff. It uses the exact discrete [likelihood](glossary.md#likelihood) for
  integers, and gives a bootstrap p-value. It compares the power law with the exponential, log-normal, stretched
  exponential, and truncated power-law tails of `compare_with`.

## Time series and information

The memory, the diffusion, the entropy, and the flow of information of series each have a measure.

```python
import numpy as np

from skcomplex.core.analysis import entropy, information, scaling

noise = np.random.default_rng(1).standard_normal(20_000)
exponents = scaling(np.column_stack([noise, np.cumsum(noise)])).exponent.round(2).tolist()
periodic = [0, 0, 1, 1] * 200
rates = [entropy(periodic, kind=kind, dimension=4).entropy.round(3).tolist() for kind in ('rate', 'excess')]
driver = np.random.default_rng(2).integers(0, 2, 5000)
transfer = information(np.column_stack([driver, np.roll(driver, 1)]), kind='transfer')
exponents, rates, round(transfer.information[0, 1], 2)
```

Detrended fluctuation analysis gives 0.52 for white noise and 1.52 for its sum, a random walk, close to the exact 0.5
and 1.5. The periodic series has the entropy rate 0 and the excess entropy 2 bits, the information needed to know its
phase. The second series copies the first with a lag of one, so 1.0 bit flows from the first to the second.

- `correlation`, `spectrum`, `displacement`, `scaling`, and `rolling` describe memory and diffusion.
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

## Space and networks

Clusters, fractals, and graphs are measured from grids, point sets, and NetworkX graphs.

```python
import networkx as nx
import numpy as np

from skcomplex.core.analysis import clusters, network_measures, scaling

labelled = clusters(np.random.default_rng(3).random((100, 100)) < 0.6, kind='labels')
rows, columns = np.indices((512, 512))
dimension = round(scaling((rows & columns) == columns, kind='box-counting').exponent, 2)
diameter = network_measures(nx.watts_strogatz_graph(200, 6, 0.1, seed=0), kind='summary').diameter
labelled.spanning, int(labelled.sizes.max()), dimension, diameter.tolist()
```

On a grid with 60 percent of its sites occupied, above the percolation threshold of about 0.593, a cluster of 3072
sites spans the grid. The Sierpinski triangle has the box-counting dimension 1.58, close to the exact `ln 3 / ln 2`. A
small-world network of 200 nodes has the diameter 8.

`clusters` labels the clusters of a grid, with periodic edges. `correlation` gives radially averaged correlations, and
`scaling` measures fractals. `network_measures` measures the degrees, the summary, and the centralities of NetworkX
graphs and of the networks of trajectories.

## Complexity and information

Some measures say how regular, complex, or informative a series is, and how information flows between series.

```python
import math

import numpy as np

from skcomplex.core.analysis import complexity, entropy, information

rng = np.random.default_rng(0)
renyi = entropy(np.full(8, 1 / 8), kind='renyi', order=2, probabilities=True)
sample = entropy(rng.standard_normal(2000), kind='sample', dimension=2)
plane = complexity(rng.standard_normal(20_000), dimension=4)
x = rng.standard_normal(5000)
y = np.concatenate([[0.0], 0.8 * x[:-1]]) + 0.6 * rng.standard_normal(5000)
transfer = information(np.column_stack([x, y]), kind='transfer', estimator='knn', base=math.e)
(
    renyi.entropy.round(3).tolist(),
    sample.entropy.round(1).tolist(),
    plane.entropy.round(2).tolist(),
    plane.complexity.round(2).tolist(),
    bool(transfer.information[0, 1] > transfer.information[1, 0]),
)
```

The Rényi entropy of order 2 of eight equal outcomes is 3 bits. The sample entropy of white noise is 2.2, high because
noise rarely repeats a pattern. On the complexity-entropy plane, white noise sits at the normalized entropy 1.0 and the
complexity 0.0. More information flows from `x` to `y` than back, because `y` follows `x` with a lag.

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
by the nearest-neighbour estimator of `ennemi` on continuous values.

### Higher-order information and emergence

Many variables may share information redundantly, when each tells the same, or synergistically, when only their
combination tells it. The measures of many variables take one row per sample and one column per variable.

```python continuation
from skcomplex.core.analysis import bind_analysis, partial_information, surrogate_test

bits = rng.integers(0, 2, (5000, 2))
xor = np.column_stack([bits, bits[:, 0] ^ bits[:, 1]])
gate = np.column_stack([bits, bits[:, 0] & bits[:, 1]])
parts = partial_information(gate)
measure = bind_analysis(information, kind='o-information')
test = surrogate_test(xor, analysis=measure, attribute='information', n=19, alternative='less', random_state=0)
synergy = information(xor, kind='o-information').information
round(synergy, 2), round(parts.redundancy, 2), round(parts.synergy, 2), test.p_value
```

Two random bits and their XOR have the O-information -1.0 bit, fully synergistic, since no single bit tells anything
about the others. For the AND gate, the inputs tell 0.31 bit about the output redundantly and 0.49 bit synergistically.
The surrogate test gives the p-value 0.05, the smallest that 19 shuffled copies allow, so the synergy is not an
accident of the sample.

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

A macro feature, such as the parity of several bits, may have a life of its own, predicting its own future beyond what
the parts predict.

```python continuation
from skcomplex.core.analysis import emergence, epsilon_machine

parity = rng.integers(0, 2, (5000, 4))
parity[:, -1] = (parity[:, :-1].sum(axis=1) + np.arange(5000) // 100) % 2
criteria = emergence(parity, kind='criteria', partition=lambda micro: micro.sum(axis=1) % 2)
closure = emergence(rng.integers(0, 4, 5000), kind='closure', partition=[0, 0, 1, 1])
golden = [1]
for _ in range(20_000):
    golden.append(1 if golden[-1] == 0 else int(rng.random() < 0.5))
machine = epsilon_machine(golden, max_length=4)
bool(criteria.psi > 0.5), round(closure.gap, 2), len(machine.histories), round(machine.statistical_complexity, 2)
```

The parity of the four bits changes every 100 samples, a slow rhythm no single bit shows, so Ψ exceeds 0.5. The
coarse-graining of four random states into two blocks is closed, with a gap of 0.0. The golden mean process, which
never repeats a 0, has two causal states and a statistical complexity of 0.92 bit.

The kind `'criteria'` of `emergence` computes the criteria of Rosas et al. (2020) for a macro feature of micro
variables, given by `partition`:

- Ψ is positive when the feature predicts its own future beyond what each micro variable predicts.
- Δ is positive when it predicts a micro variable beyond the micro variables.
- Γ is zero for a feature decoupled from every micro variable.

The kind `'closure'` takes a partition of micro states, as a label per state or a function of the micro series. It
compares how well the macro past and the micro past predict the macro future. A gap of zero means that the
coarse-graining is closed.

`epsilon_machine` reconstructs the causal states of a discrete series by causal-state splitting, with their
transitions, statistical complexity, and entropy rate.

## Invariants of dynamics

The [attractor](glossary.md#attractor) of a series, the set its states settle on, has a dimension, a degree of chaos, and a pattern of
recurrences.

```python
import numpy as np

from skcomplex.core.analysis import embedding, ergodicity, lyapunov, recurrence, scaling

points = np.random.default_rng(1).random((3000, 2))
dimension = round(scaling(points, kind='correlation-dimension').exponent, 1)
kaplan_yorke = round(lyapunov([0.906, 0.0, -14.572]).kaplan_yorke_dimension, 2)
chain = ergodicity(lambda x: 4 * x * (1 - x), kind='ulam', edges=np.linspace(0, 1, 21), samples=200)
sine = embedding(np.sin(np.arange(1000) * 0.1), delay=15, dimension=2).points
dimension, kaplan_yorke, chain.density.round(1)[:3].tolist(), round(recurrence(sine).determinism, 2)
```

Points spread over a square have the correlation dimension 2.0. The Lorenz spectrum gives the Kaplan-Yorke dimension
2.06, the fractal dimension of its attractor. The invariant density of the logistic map at `r = 4` starts at 1.1, 1.2,
and 1.2 near zero, where it piles up. An embedded sine has the determinism 1.0, a fully periodic orbit.

- `embedding` builds delay coordinates of a series. It puts the delay at the first minimum of the auto mutual
  information, and the dimension where the false nearest neighbours fall below a threshold.
- `scaling(x, kind='correlation-dimension')` counts close pairs of any point set with a k-d tree, so it serves embedded
  series and agent positions alike.
- The Kaplan-Yorke dimension and the Kolmogorov-Sinai entropy of `lyapunov` follow from a Lyapunov spectrum.
  `lyapunov(x, kind='largest')` estimates the largest exponent from one series, with Rosenstein's method of `nolds`.
- `ergodicity` of kind `'invariant-density'` or `'ulam'` estimates the invariant density of a map.
- `recurrence` gives the recurrence matrix, rate, determinism, laminarity, line lengths, trapping time, and line
  entropy of embedded series.

## Stochastic processes

Passage times, Markov chains, fluctuations, irreversibility, and rare events describe random processes.

```python
import numpy as np

from skcomplex.core.analysis import irreversibility, large_deviations, markov

ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
stationary = markov(ring, kind='stationary', matrix='rate').stationary
production = irreversibility(ring).entropy_production
gap = markov([[0.9, 0.1], [0.2, 0.8]], kind='spectral-gap', matrix='transition')
tilted = large_deviations([[2.0]], weights=[[1.0]], tilts=np.linspace(-3, 3, 601), values=3.0)
stationary.round(2).tolist(), round(production, 4), round(gap.spectral_gap, 2), round(float(tilted.rate_function[0]), 3)
```

A ring of five states that turns one way at rate 2 and the other at rate 1 is uniform at stationarity, 0.2 per state,
but it produces entropy at the rate 0.6931, `ln 2`, because it turns one way more often. The two-state chain has the
spectral gap 0.3, so it forgets its start at that rate per step. The rate function of the tilted generator at the
value 3 is 0.216, how unlikely it is to see a time average of 3.

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
  the fluctuation-dissipation ratio.
- `large_deviations` gives the scaled cumulant generating function in three ways: from a tilted generator, from
  blocks of one run, or by cloning a population of trajectories of a system through the steppers of the simulation. The
  last serves systems whose tilted generators are too large to write down. It also gives the Legendre transform at
  `values`, rate functions from histograms at several sizes, and reweighted samples.
- `correlation(x, kind='two-time')` and `data_collapse` measure aging.
- `fit_law` with the laws `'extreme-value'`, `'stable'`, and the named laws such as `'exponential'`, and
  `distribution(x, kind='records')`, describe extremes and limit laws.

Cloning advances every clone of a population by an interval, weights it by `exp(s A)` for the increment `A` of a
time-additive observable, and resamples the population by the weights. The observable is the [dotted name](glossary.md#dotted-name) of a state
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
currents = np.random.default_rng(0).normal(10.0, np.sqrt(30.0), 5000)
holds = irreversibility(currents, kind='uncertainty-relation', entropy=10.0, random_state=0).holds
cloning.scgf.round(2).tolist(), round(float(np.exp(0.5) - 1), 2), holds
```

The counter jumps at rate 1, a Poisson process, so the cloning estimate 0.66 matches the exact value `exp(s) - 1`,
0.65. The currents of mean 10 and variance 30 with an entropy production of 10 satisfy the uncertainty relation, since
`30 / 100 * 10 = 3` is at least 2.

## Emergence and ergodicity

A Markov chain may be most causally effective at a coarser scale, and the runs of an ensemble may or may not explore
the same states.

```python
import numpy as np

from skcomplex.core.analysis import emergence

noisy = np.zeros((8, 8))
noisy[:7, :7], noisy[7, 7] = 1 / 7, 1.0
micro = emergence(noisy, kind='effective-information').information
found = emergence(noisy, kind='causal')
round(micro, 3), found.partition.tolist(), round(found.macro_information, 3), round(found.gain, 3)
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

In the chain of Hoel, Albantakis, and Tononi above, the micro states carry 0.544 bit. The search merges the seven noisy
states into one block, with the partition `[0, 0, 0, 0, 0, 0, 0, 1]`, and the macro chain carries 1.0 bit, a gain of
0.456.

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

The mixing runs give 0.0, -1.0, and 0.0, and the runs frozen near -1 or 1 give 0.99, -0.0, and 0.5.

- The ergodicity-breaking parameter is the variance of the time averages across runs over the variance of all values.
  It tends to 0 for ergodic runs, and stays near 1 for runs frozen in separate states.
- The fluctuation metric of Thirumalai and Mountain is the variance of the running time averages. It decays as `1 / t`
  for ergodic runs, an exponent of -1, and levels off for frozen ones.
- The ergodicity distance is the median total-variation distance between the values of one run over time and the values
  of the whole ensemble.

## Fields and lattices

Patterns can be predicted before a run, and their domains, wavelengths, clusters, and transitions measured after it.

```python
import numpy as np

from skcomplex.core.analysis import clusters, distribution, domains, stability
from skcomplex.systems.chemistry.pattern_formation import BrusselatorSystem

relation = stability(BrusselatorSystem(shape=(128,), b=1.9), kind='dispersion')
stripes = np.cos(2 * np.pi * np.arange(64) / 8)[:, None] * np.ones(64)
grid = np.random.default_rng(2).random((128, 128)) < 0.7
samples = np.random.default_rng(3).standard_normal(10_000)
(
    float(relation.band.min().round(2)),
    float(relation.band.max().round(2)),
    domains(stripes, kind='wavelength').wavelength.tolist(),
    bool(clusters(grid, kind='percolation').spanning),
    distribution(samples, kind='moment-ratio').cumulant.round(1).tolist(),
)
```

The Brusselator grows modes between the wavenumbers 0.49 and 0.74, a Turing band. The stripes have the wavelength 8.0,
a grid with 70 percent of its sites occupied percolates, and the Binder cumulant of normal samples is 0.0, its value
in a disordered phase.

- `spectrum(x, kind='structure-factor')` averages the power of a field radially, on a periodic grid of any dimension.
  `correlation(x, kind='length')` fits its Ornstein-Zernike form, or the decay of the correlation function.
- `domains` measures coarsening with the kinds `'size'` and `'growth'`, the wavelength and orientation of
  patterns with `'wavelength'`, and the interfaces between domains with `'interface-density'`.
- `scaling(x, kind='variance-mean')` measures the number fluctuations of density fields in boxes of growing side.
- `clusters` with the kinds `'sizes'`, `'percolation'`, and `'spanning'` describes clusters of grids and
  components of graphs.
- `distribution(x, kind='moment-ratio')`, with `data_collapse` of the kinds `'crossings'` and `'scaling'`, locates
  transitions by finite-size scaling. The variance from `distribution(x, kind='moments', absolute=True)` of an order
  parameter, scaled by the system size, is the susceptibility. Its peak over a temperature [sweep](glossary.md#sweep) of a spin system
  locates the transition too.

`stability(x, kind='dispersion')` predicts patterns from the laws of a field system before running it. It linearises
them around a uniform state, and turns every Laplacian into its eigenvalue on each Fourier mode of the lattice. It
returns the growth rate and frequency of every mode, exact for the discretization of the system. A band of growing
modes is a Turing instability, and a growing mode with a nonzero frequency travels.

## Agent populations

Agents align, spread in [space](glossary.md#space), and segregate into groups, and each has a measure.

```python
import numpy as np

from skcomplex.core.analysis import alignment, coarse_grain, neighbors
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.collective_behaviour import BoidsSystem

flock = simulate(BoidsSystem(), until=5)
positions = np.random.default_rng(4).random((1000, 2))
ratio = neighbors(positions, kind='clark-evans', box=[1.0, 1.0]).ratio
field = coarse_grain(positions, shape=[8, 8], box=[1.0, 1.0]).field
alignment(flock, kind='mean-vector').order.shape, ratio.round(1).tolist(), float(field.sum())
```

The flock has one polar order per frame, 6 frames in five time units. Uniform random points have the Clark-Evans ratio
1.0, neither clustered, below 1, nor regular, above 1. Their density field on an 8 by 8 grid sums to the 1000 points.

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

## Criticality

A process near criticality, the edge between dying out and blowing up, comes in avalanches whose sizes follow power
laws.

```python
import numpy as np

from skcomplex.core.analysis import events

found = events([0, 1, 3, 2, 0, 0, 4, 1, 0], kind='excursions')
rng = np.random.default_rng(5)
activity = np.zeros(5000)
activity[0] = 50
for t in range(1, activity.size):
    activity[t] = rng.poisson(0.8 * activity[t - 1] + 10)
found.sizes.tolist(), found.durations.tolist(), round(events(activity, kind='branching').branching_ratio, 1)
```

The series holds two avalanches, of sizes 6 and 5 and durations 3 and 2. The activity branches with ratio 0.8, the
mean number of events each event causes, which the estimate recovers. A ratio of 1 is critical.

- `events(x, kind='excursions')` extracts avalanches from activity series or relaxation records, and
  `fit_law(x, law='power')` fits their size and duration laws.
- `events(x, kind='branching')` estimates the branching ratio by the multistep regression of `mrestimator`, which stays
  unbiased under subsampling.
- The exponents `tau_S` of the sizes and `tau_T` of the durations predict the mean size against the duration,
  `<S>(T) ~ T^((tau_T - 1) / (tau_S - 1))`. This is the crackling-noise relation of avalanches, such as those of the
  system `sandpile`.
- `distribution(x, kind='moments')` gives the order parameter and its variance over a sweep, and
  `scaling(x, kind='power')` fits the order-parameter exponent.
- `correlation(x, kind='overlap')` compares replicas.

## Network analysis

The structure of a network, its communities, its motifs, and its shape at coarser scales each have a measure.

```python
import networkx as nx

from skcomplex.core.analysis import motifs, network_measures

club = nx.karate_club_graph()
communities = network_measures(club, kind='communities', random_state=0)
communities.modularity.round(2).tolist(), motifs(club).counts[0]['triangle']
```

Zachary's karate club splits into communities of modularity 0.44, well above the 0 of a random split, and holds 45
triangles.

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

Two cliques of 20 nodes joined by one edge become 2 supernodes joined by 1 edge.

- The kind `'laplacian'` joins two nodes where the diffusion operator `exp(-time L)` has an entry at least the smaller
  of their diagonal entries, and every component of these joins is a supernode. It runs at `time`, or at the smallest
  peak of the specific heat that merges nodes.
- The kind `'geometric'` merges `block` consecutive nodes by their `angle`. A supernode sits at their angles weighted by
  `k^max(1, beta)`, with `k` the degree.

In both kinds, two supernodes link when any of their nodes link.

`autocatalytic_sets` finds the largest autocatalytic set, the largest RAF, of a catalytic reaction network read
from a network snapshot or from chosen `frames` of a trajectory.

```python
import numpy as np

from skcomplex.core.analysis import autocatalytic_sets
from skcomplex.core.specs import NetworkSnapshotSpec

reactions = np.array([[0, 1, 2], [2, 0, 3], [3, 1, 4]])
snapshot = NetworkSnapshotSpec(np.arange(5), np.array([3, 4, 0]), np.array([2, 3, 4]), np.arange(3), reactions)
autocatalytic_sets(snapshot).reactions[0].tolist()
```

All three reactions, 0, 1, and 2, form the autocatalytic set: each is catalysed by a product of the set and fed from
the food set.

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

The fit gives the exponent 1.1, and its bootstrap interval holds the true 1.15. It regresses the logarithms of the pairs
and sets `exponent`, with the bootstrap confidence interval `interval` of the level `confidence` over `resamples`
resamples.
