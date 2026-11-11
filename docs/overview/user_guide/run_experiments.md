# Run experiments

This tutorial runs the hare and lynx model many times and reads what the runs say together. It [sweeps](glossary.md#sweep) the rates,
repeats a [stochastic](glossary.md#stochastic) version, compares an intervention with the baseline, ranks the rates by their effect, measures how
often the lynx fall to a dangerous level, and finds where the cycles settle. Every experiment is a function of
[`skcomplex.core.experiments`][skcomplex.core.experiments] that takes a [system](glossary.md#system) and returns a frozen result, as the model
selection functions of scikit-learn take an estimator.

## The running example

The model is the predator and prey cycle of the snowshoe hare and the Canada lynx, built in the tutorial
[Build a custom system](build_a_custom_system/index.md). `LotkaVolterraSystem` of `skcomplex.systems.biology.ecology`
holds it with the rates of Carpenter (2018): the growth rate `r` of the hares, the rate `a` at which the lynx remove
hares, the rate `b` at which eating hares feeds the lynx, and the death rate `d` of the lynx, all per year. The runs
start from the pelt records of 1900, 30.0 thousand hares and 4.0 thousand lynx.

```python
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
[interaction.label for interaction in system.prepare().system_.interactions]
```

The system has the populations `prey` and `predators`, each with the [state](glossary.md#state) `x`, and the [interactions](glossary.md#interaction) `growth`,
`predation`, and `death`. An experiment names a value in two ways. A name without a dot, such as `r`, is a model
parameter of the system, set on a copy with `set_params`. A dotted path, such as `growth.r` or `prey.x`, is a parameter
or an initial state of a component, which keeps its value for the whole run.

## Sweeps

A sweep runs the system once per row of a design and collects the outputs of every run in one table. Here
[`sweep`][skcomplex.core.experiments.sweep] runs three growth rates of the hares. An output is a state path, whose
series the run keeps, or a function of a [trajectory](glossary.md#trajectory) that returns a number, named after the function.

```python continuation
from skcomplex.core.experiments import sweep


def hare_peak(trajectory):
    return float(trajectory.get_values('prey.x').max())


swept = sweep(system, {'r': [0.45, 0.55, 0.65]}, until=20, dt=1.0, outputs=['predators.x', hare_peak])
[(row['r'], round(row['hare_peak'], 1)) for row in swept.rows]
```

The highest count of hares over twenty years rises with their growth rate, from 69.8 thousand at `r = 0.45` through
81.2 thousand at the estimate of Carpenter to 85.8 thousand at `r = 0.65`. The result is an
[`ExperimentResult`][skcomplex.core.experiments.ExperimentResult]. Its `rows` hold one mapping per run, with the values
of the row, the [scenario](glossary.md#scenario), the replicate, the seed, and the outputs. A state output appears in a row by its last value,
and `get_series` returns its whole series, one row per run.

```python continuation
swept.get_column('hare_peak').round(1).tolist(), swept.get_series('predators.x').shape
```

`get_column` returns the scalar output of every run, here the three peaks, and `get_series` the lynx of the three runs
at the 21 [frames](glossary.md#frame) of the years 1900 to 1920.

### Designs

A grid runs every combination of the values it is given. For several inputs a sampled design covers the space with
fewer runs. The rates below are dotted paths, and `design='latin_hypercube'` draws 8 rows, one in every eighth of the
range of each rate. `convert_to_frame` returns the rows as a pandas data frame.

```python continuation
bounds = {'growth.r': (0.4, 0.7), 'death.d': (0.6, 1.0)}
sampled = sweep(
    system, bounds, design='latin_hypercube', n=8, until=20, dt=1.0, outputs=[hare_peak], random_state=0
)
frame = sampled.convert_to_frame()
frame[['growth.r', 'death.d', 'hare_peak']].round(2).head(3).values.tolist()
```

The first three rows pair a growth rate and a death rate with the peak of the hares they produce, such as 0.53 and 0.79
with a peak of 77.85 thousand. The designs are these:

- `'grid'`, the default, runs every combination of a sequence of values per name, the last name varying fastest.
- `'random'` draws `n` rows uniformly within a pair `(low, high)`, or from a SciPy distribution given per name.
- `'latin_hypercube'` draws `n` rows with one in every stratum of every name.
- `'sobol'` takes `n` points of a scrambled Sobol sequence, best with `n` a power of 2.

The sweep compiled the model once for all eight rows. Runs that change only dotted paths, initial states, or seeds
reuse one compiled problem, while a [model parameter](glossary.md#model-parameter) such as `r` builds a new system. A study of thousands of rows
should therefore sweep dotted paths.

A sequence of mappings is a design of given rows, which run as they are. The first argument may also be a factory of
systems, such as the [system class](glossary.md#system-class) itself, which takes the values of a row as keyword arguments.

```python continuation
records = {'prey': 30.0, 'predators': 4.0, 'r': 0.55, 'a': 0.028, 'b': 0.024, 'd': 0.80}
given = sweep(LotkaVolterraSystem, [records, {**records, 'predators': 8.0}], until=20, dt=1.0, outputs=[hare_peak])
given.get_column('hare_peak').round(1).tolist()
```

Twice as many lynx in 1900 hold the highest count of the hares down from 81.2 to 59.9 thousand.

### Analyses of every run

An analysis function can run on every trajectory of an experiment. `bind_analysis` fixes its settings, and every field
of its result becomes a column named `<function>.<field>`. Here [`spectrum`][skcomplex.core.analysis.spectrum]
computes the power spectrum of the hares, and the frequency of the highest power gives the period of the cycle. The
setting `dt` gives the spectrum the time between frames, so its frequencies are per year.

```python continuation
from skcomplex.core.analysis import bind_analysis, spectrum

rhythm = bind_analysis(spectrum, variables=['prey.x'], dt=0.5)
cycles = sweep(system, {'r': [0.45, 0.65]}, until=60, dt=0.5, analyses=[rhythm])
[round(float(1 / row['spectrum.frequencies'][row['spectrum.power'][0].argmax()]), 1) for row in cycles.rows]
```

The cycle lasts about 12.1 years at `r = 0.45` and 10.1 years at `r = 0.65`, so faster growth of the hares shortens
it, towards the cycle of about ten years in the records. Sixty years give a coarse spectrum. The times between the
peaks of the same runs are 11.4 and 10.0 years, and a study of the period would run longer or measure those times.

## Ensembles

An [ensemble](glossary.md#ensemble) repeats the runs of a stochastic system with independent random streams, to show the spread of its
outcomes. [`ensemble`][skcomplex.core.experiments.ensemble] refuses the [deterministic](glossary.md#deterministic) model, because its runs repeat one
another.

```python continuation
from skcomplex.core.errors import TagError
from skcomplex.core.experiments import ensemble

try:
    ensemble(system, replicates=5, until=10)
except TagError as error:
    message = str(error)
message
```

The message says that `ensemble` needs the randomness `stochastic` while `LotkaVolterraSystem` is deterministic, and
suggests running it once with `simulate` or giving it a component or a [solver](glossary.md#solver) that draws. The
randomness is one of the capability [tags](glossary.md#tag) of a system, which the page [Check a system](check.md)
describes.

On a small island the populations are counted one animal at a time, and chance decides each birth and death. The
[recipe](glossary.md#recipe) `ReactionNetworkSystem` of `skcomplex.systems.general.reactions` runs the four terms of the model as reactions.
In the mode `'jump'` each reaction is a random event that changes a count by one animal, drawn by the algorithm of
Gillespie. Each row of the table holds the reactants, the products, and the rate constant.

```python continuation
from skcomplex.systems.general.reactions import ReactionNetworkSystem

reactions = {
    'births': ({'prey': 1}, {'prey': 2}, 0.55),
    'hunting': ({'prey': 1, 'predators': 1}, {'predators': 1}, 0.028),
    'feeding': ({'prey': 1, 'predators': 1}, {'prey': 1, 'predators': 2}, 0.024),
    'deaths': ({'predators': 1}, {}, 0.80),
}
island = ReactionNetworkSystem({'prey': 30, 'predators': 4}, reactions, mode='jump')
runs = ensemble(island, replicates=20, until=10, dt=1.0, outputs=['predators.x'], random_state=0)
times, values, seeds = runs.get_ensemble('predators.x')
values.shape, int((values[:, -1] == 0).sum())
```

`get_ensemble` returns the common times, one row of values per run, and the seed of every run. The 20 runs of ten years
hold 11 values each, and in 4 of them the lynx have died out by the tenth year. The deterministic model never reaches
zero, so the extinctions come from chance alone. `get_summary` gives the mean, the standard deviation, and the 5%, 50%,
and 95% quantiles of every output across the replicates.

```python continuation
summary = runs.get_summary()[0]
round(summary['predators.x.mean'], 1), round(summary['predators.x.q0.05'], 1), round(summary['predators.x.q0.95'], 1)
```

After ten years the island holds 6.3 lynx on average, with 0.0 at the 5% quantile and 16.2 at the 95% quantile. A study
of the risk of extinction would run hundreds or thousands of replicates.

Every design row and replicate gets its own random stream, spawned from `random_state` in a fixed order. The results
are therefore the same whether the runs go one after another, on threads, or on processes, which the argument
`executor` chooses with `'serial'`, `'thread'`, or `'process'`. A sweep with `replicates` greater than one repeats every
row in the same way.

## Scenarios

A scenario is a named variant of a system, such as an intervention, compared with the baseline under the same random
draws. A [`ScenarioSpec`][skcomplex.core.components.specs.ScenarioSpec] changes values before the run with
`overrides`, changes parameters during the run with `schedule`, or replaces the system for a change of structure. Here
trappers raise the death rate of the lynx from 0.80 to 1.0 from year 10 on, and
[`compare_scenarios`][skcomplex.core.experiments.compare_scenarios] runs both scenarios.

```python continuation
from skcomplex.core.components.specs import ScenarioSpec
from skcomplex.core.experiments import compare_scenarios

scenarios = [ScenarioSpec('baseline'), ScenarioSpec('trapping', schedule=[(10.0, 'death.d', 1.0)])]
compared = compare_scenarios(system, scenarios, until=60, dt=0.1, outputs=['prey.x'], warmup=10)
[row['scenario'] for row in compared.rows], compared.get_series('prey.x').mean(axis=1).round(1).tolist()
```

`warmup` drops the frames before year 10 from the kept series, so the means cover the years after the trapping starts.
The hares average 34.4 thousand without trapping and 41.6 thousand with it. This is the principle of Volterra: the
average of the prey over a cycle is `d / b`, which is 33.3 at `d = 0.80` and 41.7 at `d = 1.0`, so killing the
predators raises the average of their prey. A scenario with `overrides={'predators.x': 8.0}` would start from more lynx
instead. Every scenario of a replicate shares one random stream, so the scenarios of a stochastic system differ only
through what they change.

## Sensitivity

[Sensitivity analysis](glossary.md#sensitivity-analysis) ranks the inputs of a model by their effect on one output. Here the output is the count of hares
in 1903, the year of the first peak, and each of the four rates varies by a tenth to a fifth around its estimate.
[`sensitivity`][skcomplex.core.experiments.sensitivity] reads the last value of the output, so the runs stop at year 3.

```python continuation
from skcomplex.core.experiments import sensitivity

rates = {'growth.r': (0.45, 0.65), 'predation.a': (0.023, 0.033), 'predation.b': (0.019, 0.029), 'death.d': (0.7, 0.9)}
one_at_a_time = sensitivity(system, rates, output='prey.x', method='oat', until=3, dt=1.0)
one_at_a_time.ranking, {name: round(effect, 1) for name, effect in one_at_a_time.effects.items()}
```

The method `'oat'` moves one input at a time from its lower to its upper bound, with the others at their midpoints, and
gives the change of the output. The feeding rate `b` matters most: raising it lowers the peak by 41.0 thousand hares,
because fed lynx multiply faster and eat the hares sooner. The growth rate `r` raises the peak by 16.1 thousand, the
death rate `d` by 15.9 thousand, and the hunting rate `a` lowers it by 11.4 thousand.

One move per input misses interactions between inputs. The method `'morris'` takes `n` random paths through a grid of
four levels per input and averages the absolute effects of the steps, and `'sobol'` splits the variance of the output
into shares, with SALib.

```python continuation
screening = sensitivity(system, rates, output='prey.x', method='morris', n=10, until=3, dt=1.0, random_state=0)
variance = sensitivity(system, rates, output='prey.x', method='sobol', n=64, until=3, dt=1.0, random_state=0)
screening.ranking, {name: round(index, 2) for name, index in variance.total_order.items()}
```

Both methods agree on the ranking `b`, `r`, `d`, `a`. The total Sobol index of `b` is 0.95, so the feeding rate alone
drives most of the variance of the peak, and the index of `r` is 0.2. The Morris screening ran 50 runs, 10 paths of 5
steps each. The Sobol analysis ran 384 runs, six for each of the 64 base points, and its indices carry confidence
intervals in `first_order_conf` and `total_order_conf`. A study would run at least 1024 base points, the default.

## Basin stability

[Basin stability](glossary.md#basin-stability) draws many initial states and asks which final behaviour each one ends in. The share of initial states
that reach a behaviour measures how stable it is against large disturbances. The cycles of Lotka and Volterra are
neutral: every initial state runs on a cycle of its own and no cycle attracts the others. So the classifier below
labels an outcome rather than an [attractor](glossary.md#attractor), whether the lynx fall below one thousand animals during twenty years, a
level at which a real population risks dying out.
[`basin_stability`][skcomplex.core.experiments.basin_stability] draws the initial states uniformly within the bounds.

```python continuation
from skcomplex.core.experiments import basin_stability


def lynx_fate(trajectory):
    lowest = trajectory.get_values('predators.x').min()
    return 'below one thousand' if lowest < 1.0 else 'above one thousand'


starts = {'prey.x': (5.0, 100.0), 'predators.x': (1.0, 40.0)}
basins = basin_stability(system, starts, lynx_fate, n=100, until=20, dt=0.5, random_state=0)
{label: round(share, 2) for label, share in basins.fractions.items()}, round(basins.errors['below one thousand'], 2)
```

Of 100 initial states, 96 keep the lynx above one thousand and 4 drive them below, with a standard error of 0.02 on
each share. The initial states sit in `rows` and their labels in `labels`. With `vectorized=True` all initial states run
as the [members](glossary.md#member) of one simulation, which is faster for many small runs. A study would draw thousands of initial states.

## Orbit diagrams

An [orbit diagram](glossary.md#orbit-diagram) shows the values a state visits after a transient, for every value of a parameter. A single value
means the run settles at a [steady state](glossary.md#steady-state), a few values a periodic cycle, and many values a cycle that does not repeat
at the sampled times. With a carrying capacity `K` the hares grow logistically, and the cycles of the model damp
towards a steady state. [`orbit_diagram`][skcomplex.core.experiments.orbit_diagram] runs three capacities for 200 years
and keeps the next 20 yearly values of the lynx.

```python continuation
from skcomplex.core.experiments import orbit_diagram

crowded = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80, K=200.0)
diagram = orbit_diagram(
    crowded, 'growth.K', [40.0, 200.0, 2000.0], 'predators.x', transient=200, keep=20, dt=1.0, decimals=1
)
diagram.periods.tolist(), [round(float(points.max() - points.min()), 1) for points in diagram.points]
```

`periods` counts the distinct kept values, rounded to `decimals`, here one decimal, and `points` holds them. A capacity
of 40 or 200 thousand hares settles the lynx at one value, while at 2000 thousand the lynx take 18 distinct values and
still swing over a range of 19.3 thousand after 200 years. A richer habitat damps the cycles more slowly. The same
function shows the period doubling of a map, such as the logistic map in the page [Analyse](analyse.md).

## Calibration and evaluation

Fitting the rates to the pelt records is the subject of the page [Calibrate](calibrate.md), which runs every method of
`calibrate`, from a local least squares fit to [Markov chain Monte Carlo](glossary.md#markov-chain-monte-carlo). The same page samples
[virtual populations and fits every individual](calibrate.md#virtual-populations) of a data set on its
own, with `virtual_population` and `calibrate_individuals`.

Whether a fit can be trusted is the subject of the page [Evaluate](evaluate.md). It compares the fit with the records,
checks the [credible intervals](glossary.md#credible-interval) of a [posterior](glossary.md#posterior) method with `posterior_coverage`, and asks with `identifiability` which
rates the records determine.
