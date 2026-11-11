# An opinion model swept and analysed

This case study takes the bounded confidence model of Deffuant et al. from the catalog and asks how many opinion groups
it forms. It checks the model, runs it once, counts the clusters of opinions, sweeps the confidence bound with an
ensemble of runs per value, compares the counts with the known result `1 / (2 d)`, and measures which parameter the
final consensus depends on.

## The model and the known result

Deffuant, Neau, Amblard, and Weisbuch (2000, "Mixing beliefs among interacting agents", *Advances in Complex Systems* 3,
87) proposed the model. Each agent holds an opinion between 0 and 1. At each encounter two random agents compare their
opinions `x` and `y`. If they differ by less than the confidence bound `d`, both move towards each other by the fraction
`mu` of the difference, the convergence rate:

```text
x <- x + mu (y - x)
y <- y + mu (x - y)
```

Agents whose opinions lie further apart than `d` ignore each other. From uniform initial opinions the population settles
into clusters, groups of agents with the same opinion. Deffuant et al. found by simulation that the number of clusters
is close to the integer part of `1 / (2 d)`, and that `mu` changes how fast the opinions settle but not where they
settle. The study checks both statements.

## Selecting and checking the model

The catalog holds the model as [DeffuantSystem](../../../systems/social/opinion_dynamics/deffuant.md). Its
parameters are the number of agents, the bound `confidence`, the rate `convergence`, and the seed `random_state` of the
initial opinions. The study uses 100 agents.

```python
from skcomplex.systems.social.opinion_dynamics import DeffuantSystem

opinions_model = DeffuantSystem(100, confidence=0.2, convergence=0.5)
tags = opinions_model.get_tags()
[element.label for element in opinions_model.prepare().system_.elements], sorted(tags.time), tags.randomness
```

`prepare` builds the components and checks that they fit together. The model has one element, the group `people` that
holds the agents. Time is discrete, since the model steps through rounds of encounters, and the model is stochastic,
since the pairs meet at random. A run therefore needs a seed, and a study needs several runs per setting.

## One run and its clusters

`simulate` runs the model for its tuned number of steps, 50 rounds of encounters. The agents of a group are read at a
frame with `get_agents`, which returns their states.

```python continuation
import numpy as np

from skcomplex.core.engine.runtime import simulate

trajectory = simulate(opinions_model, random_state=0)
final = trajectory.get_agents('people', len(trajectory.times) - 1).values['opinion']
initial = trajectory.get_agents('people', 0).values['opinion']
round(float(initial.std()), 3), round(float(final.std()), 3)
```

The standard deviation of the opinions falls only from 0.303, close to the value `1 / sqrt(12) = 0.289` of uniform
opinions, to 0.257. The agents have not come to one opinion. They have split into groups far apart.

The analysis function [clusters][skcomplex.core.analysis.clusters] finds the groups. With a `radius`, it reads an array
of points, one row per point, and joins the points closer than twice the radius. Here a point is the opinion of one
agent, and the radius 0.005 joins opinions that differ by less than 0.01.

```python continuation
from skcomplex.core.analysis import clusters

groups = clusters(final[:, None], kind='sizes', radius=0.005)
groups.sizes.tolist()
```

The 100 agents end in two clusters, of 53 and 47 agents. At `d = 0.2` the known result gives `1 / (2 d) = 2.5`, so two
or three clusters.

## Sweeping the confidence bound

[sweep][skcomplex.core.experiments.sweep] runs the model over a grid of values of a model parameter. With `replicates`
it repeats every value with different random encounters, an ensemble of runs per value. An output may be a function of
the trajectory, here the number of clusters of at least five agents, so that a few stragglers between the large clusters
do not count. The initial opinions of every run are those of the seed `random_state=0` of the system.

```python continuation
from skcomplex.core.experiments import sweep


def large_clusters(trajectory):
    opinions = trajectory.get_agents('people', len(trajectory.times) - 1).values['opinion']
    sizes = clusters(opinions[:, None], kind='sizes', radius=0.005).sizes
    return int(np.sum(sizes >= 5))


bounds = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
swept = sweep(
    DeffuantSystem(100), {'confidence': bounds}, replicates=4, outputs=[large_clusters], random_state=0
)
counts = swept.get_column('large_clusters').reshape(len(bounds), 4)
counts.mean(axis=1).tolist(), np.floor(1 / (2 * bounds)).tolist()
```

The rows of the result run over the bounds, with the four replicates of a bound next to each other, so the column
reshapes into one row per bound. The mean count falls from 4.25 clusters at `d = 0.1` to one cluster, a consensus, at
`d = 0.4` and above. The integer part of `1 / (2 d)` gives 5, 3, 2, 2, 1, 1, and 1 clusters. The simulated means lie
within one cluster of these values at every bound, as the known result predicts.

## Sensitivity of the final consensus

The variance of the opinions measures the final consensus. It is zero when every agent holds the same opinion and grows
as the opinions split into clusters far apart. An [ObservableSpec][skcomplex.core.specs.ObservableSpec] computes it at
every frame, as it reduces the variable `opinion` of the group `people` with `'var'`.
[sensitivity][skcomplex.core.experiments.sensitivity] then varies the bound and the convergence rate within their bounds
and measures the effect of each on the last value of the observable. The method `'morris'` runs Morris trajectories,
each of which changes one input at a time. Its index `mu_star` is the mean absolute change of the output per change of
an input, and ranks the inputs.

```python continuation
from skcomplex.core.experiments import sensitivity
from skcomplex.core.specs import ObservableSpec

spread = ObservableSpec('spread', group='people', variable='opinion', reduce='var')
ranges = {'confidence': (0.1, 0.5), 'convergence': (0.1, 0.5)}
effects = sensitivity(
    DeffuantSystem(100), ranges, output='spread', method='morris', n=10, until=100, observables=[spread], random_state=0
)
effects.ranking, {name: round(value, 4) for name, value in effects.mu_star.items()}
```

The bound ranks first, with `mu_star` equal to 0.1109, and the convergence rate has 0.0035, about 30 times smaller. The
runs last 100 rounds, so that slow convergence rates also settle. This matches the finding of Deffuant et al. that the
bound decides the final state and the convergence rate only how fast it is reached.

## Visualising the sweep

The figure draws the mean number of clusters per bound, with the spread of the ensemble, against the known result. It
uses the Agg backend of matplotlib, which draws to memory and opens no window.

```python continuation
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(bounds, counts.mean(axis=1), yerr=counts.std(axis=1), marker='o', capsize=3, label='simulated')
grid = np.linspace(0.08, 0.52, 200)
ax.plot(grid, np.floor(1 / (2 * grid)), color='black', linestyle='--', label='floor(1 / (2 d))')
ax.set_xlabel('confidence bound d')
ax.set_ylabel('clusters of at least five agents')
ax.legend()
plt.close(fig)
[text.get_text() for text in ax.get_legend().get_texts()]
```

The legend names the known result and the simulated counts. A script saves the figure with
`fig.savefig('clusters.png')`. Outside the Agg backend, `plt.show()` opens it in a window.

## What a real study would add

The page keeps its budgets small, so it runs in under a minute. A real study would use 1000 agents or more, a finer grid
of bounds, 20 or more replicates per bound with different initial opinions, and the method `'sobol'` of `sensitivity`
with 1024 base points, which also measures how the two inputs interact. The page
[Run experiments](../run_experiments.md) describes each experiment and its settings.
