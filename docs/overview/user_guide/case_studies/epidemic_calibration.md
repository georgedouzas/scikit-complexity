# An epidemic calibrated to case data

This case study fits the epidemic model of Kermack and McKendrick to the daily counts of an influenza outbreak in an
English boarding school in 1978. It selects the model from the catalog, checks it, fits its two rates, checks the fit
and the posterior, asks whether the data determine the rates, and compares scenarios of earlier isolation.

## The data

In January 1978 influenza spread through a boarding school in England, and 512 of its 763 boys fell ill.
The British Medical Journal reported the outbreak (Anonymous, 1978, "Influenza in a boarding school", *British Medical
Journal* 1(6112), News and Notes, pages 586 to 590). The counts below are the boys confined to bed on each day from 22
January to 4 February, and the boys convalescing after their time in bed. They come from the file
`data/influenza_england_1978_school.RData` of the R package `outbreaks`, in the GitHub repository
`reconverse/outbreaks`. Its documentation, `R/influenza_england_1978_school.R`, cites the report and says the numbers
were read from a figure of it by De Vries et al. (2006, *A Course in Mathematical Biology*, SIAM), so other sources give
slightly different counts.

```python
import numpy as np

days = np.arange(14.0)
in_bed = np.array([3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4], dtype=float)
convalescent = np.array([0, 0, 0, 0, 9, 17, 105, 162, 176, 166, 150, 85, 47, 20], dtype=float)
int(days[in_bed.argmax()]), int(in_bed.max())
```

Day 0 is 22 January. The number in bed peaks on day 5, 27 January, at 298 boys.

## Selecting and checking the model

The SIR model of Kermack and McKendrick (1927, *Proceedings of the Royal Society A* 115, 700) splits a population into
the susceptible `S`, who can catch the disease, the infected `I`, who spread it, and the recovered `R`, who are immune.
A susceptible boy meets an infected one and falls ill at the rate `beta S I`, and an infected boy leaves the infected
compartment at the rate `gamma I`:

```text
dS/dt = -beta S I
dI/dt = beta S I - gamma I
dR/dt = gamma I
```

The catalog holds this model as [SIRSystem](../../../systems/biology/epidemiology/sir.md). The study assumes that
the boys in bed are the infected compartment. The school starts with 3 boys in bed and 760 susceptible boys.

```python continuation
from skcomplex.systems.biology.epidemiology import SIRSystem

school = SIRSystem(susceptible=760.0, infected=3.0, recovered=0.0)
tags = school.get_tags()
[element.label for element in school.prepare().system_.elements], sorted(tags.laws), tags.randomness
```

`prepare` builds the parts and checks that they fit together. The system has the elements `S`, `I`, and `R`, its laws
are rates, and it is deterministic, so one run gives the answer and no ensemble is needed. The tags tell `simulate`
which solver fits.

## A first simulation

The default rates of `SIRSystem` describe a slow epidemic, so the first run shows how far they are from the school.

```python continuation
from skcomplex.core.engine.runtime import simulate

trajectory = simulate(school, until=13, dt=1.0)
trajectory.get_values('I.x').round().tolist()[::4]
```

With `beta = 0.0003` and `gamma = 0.1` per day, the number infected creeps from 3 to 5, 8, and 14 boys on days 4, 8, and
12, while the school had 298 in bed on day 5. The rates have to come from the data.

## Calibrating the rates

[calibrate][skcomplex.core.experiments.calibrate] fits the model parameters to observed series. Each series is a pair of
times and values under the path of a state, here `I.x`, the number infected. The bounds keep each rate in a range that a
school epidemic allows. The default method `'local'` is a least squares fit. The basic reproduction number, the number
of boys one sick boy infects in a school of susceptible boys, follows from the fit as `R0 = beta N / gamma` with `N` the
763 boys.

```python continuation
from skcomplex.core.experiments import calibrate

data = {'I.x': (days, in_bed)}
bounds = {'beta': (0.0005, 0.005), 'gamma': (0.1, 1.0)}
fit = calibrate(school, data, bounds)
reproduction = fit.estimates['beta'] * 763 / fit.estimates['gamma']
estimates = round(fit.estimates['beta'], 5), round(fit.estimates['gamma'], 3), round(reproduction, 1)
estimates, round(fit.r2, 3), round(fit.rmse, 1)
```

The fit gives `beta = 0.00223` per pair of boys and day and `gamma = 0.447` per day, so a boy stays infected for about
`1 / gamma = 2.2` days. Each sick boy infected about 3.8 others at the start of the outbreak. The model explains 97.4%
of the variance of the counts, with a root mean square error of 16.6 boys.

## Analysing the fitted run

The analysis function [events][skcomplex.core.analysis.events] finds the episodes when a series stays above a threshold.
Applied to the fitted run and to the data, it compares when the school had more than 100 boys in bed and for how long.

```python continuation
from skcomplex.core.analysis import events

fitted_run = simulate(fit.system, until=13, dt=1.0)
simulated = events(fitted_run, variables=['I.x'], threshold=100.0)
observed = events(in_bed, threshold=100.0)
(simulated.starts.tolist(), simulated.durations.tolist()), (observed.starts.tolist(), observed.durations.tolist())
```

Both the model and the school pass 100 boys in bed on day 4 and stay above it for 6 days.

## Evaluating the posterior

A least squares fit gives one value per rate. A posterior method gives a distribution over the rates, the posterior,
which says how well the data pin each rate down. The method `'variational'` fits a normal distribution to the posterior,
and the loss `'likelihood'` reads the data as the model plus Gaussian noise of unknown size. The result also holds a
posterior predictive check: the band of runs drawn from the posterior and the share of observed counts inside it. The
block starts again from the counts, so it runs on its own.

```python
import numpy as np

from skcomplex.core.experiments import calibrate
from skcomplex.systems.biology.epidemiology import SIRSystem

days = np.arange(14.0)
in_bed = np.array([3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4], dtype=float)
school = SIRSystem(susceptible=760.0, infected=3.0, recovered=0.0)
data = {'I.x': (days, in_bed)}
bounds = {'beta': (0.0005, 0.005), 'gamma': (0.1, 1.0)}
posterior = calibrate(
    school, data, bounds, method='variational', loss='likelihood', n_samples=200, budget=1000, random_state=0
)
low, high = np.quantile(posterior.samples['gamma'], [0.05, 0.95])
round(posterior.estimates['gamma'], 3), round(float(low), 3), round(float(high), 3), posterior.predictive['coverage']
```

The posterior median of `gamma` is 0.448 per day, and 90% of the posterior lies between 0.421 and 0.478. The band from
the 5% to the 95% quantile of the posterior runs holds every one of the 14 counts, a coverage of 1.0.

A band that holds the data does not prove that the intervals are honest.
[posterior_coverage][skcomplex.core.experiments.posterior_coverage] tests the method itself. It draws true rates from
the prior, simulates data from them with noise, calibrates each data set, and counts how often the credible interval
holds the true value. A calibrated method covers the truth with its 90% interval in about 90% of the data sets. The
prior here is narrower than the bounds of the fit, so that every draw makes an outbreak, and the noise of 15 boys
matches the error of the fit.

```python
import numpy as np

from skcomplex.core.experiments import posterior_coverage
from skcomplex.systems.biology.epidemiology import SIRSystem

school = SIRSystem(susceptible=760.0, infected=3.0, recovered=0.0)
prior = {'beta': (0.0015, 0.003), 'gamma': (0.3, 0.6)}
check = posterior_coverage(
    school,
    prior,
    np.arange(14.0),
    paths=['I.x'],
    method='variational',
    n=3,
    levels=(0.9,),
    noise=15.0,
    n_samples=200,
    budget=1000,
    random_state=0,
)
check.coverage[0.9]
```

The 90% intervals held the true `beta` and the true `gamma` in all 3 simulated data sets, a coverage of 1.0 for each.
Three data sets show that the check runs, not that the method is calibrated. Each data set costs one calibration of a
few seconds. A real study uses 100 data sets or more, and compares the ranks in `check.ranks` with a uniform
distribution, the test of simulation-based calibration (Talts et al., 2018, arXiv:1804.06788).

## Identifiability of the two rates

A calibration returns estimates even when the data cannot determine them.
[identifiability][skcomplex.core.experiments.identifiability] tells whether they do. Its default method computes the
profile likelihood of each rate. It fixes the rate at values on both sides of the best fit, fits the other rate again,
and keeps the values whose likelihood stays within the 95% threshold of a likelihood ratio test. A rate is identifiable
when this interval ends inside its bounds on both sides.

```python
import numpy as np

from skcomplex.core.experiments import identifiability
from skcomplex.systems.biology.epidemiology import SIRSystem

days = np.arange(14.0)
in_bed = np.array([3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4], dtype=float)
school = SIRSystem(susceptible=760.0, infected=3.0, recovered=0.0)
bounds = {'beta': (0.0005, 0.005), 'gamma': (0.1, 1.0)}
profile = identifiability(school, {'I.x': (days, in_bed)}, bounds, n_starts=1, random_state=0)
intervals = {name: tuple(float(f'{value:.3g}') for value in pair) for name, pair in profile.intervals.items()}
profile.verdicts, intervals
```

Both rates are identifiable. The 95% interval of `beta` runs from 0.00215 to 0.00231, and that of `gamma` from 0.419 to
0.478, close to the 90% posterior interval above. The outbreak rises and falls within the data, so the rise fixes `beta`
and the decline fixes `gamma`. A real study takes `n_starts=5` or more, so the best fit does not depend on one start.

## Scenarios of earlier isolation

A scenario changes the system before or during a run. Isolating the sick boys from the others lowers the rate of
infection, which the study models as halving `beta` from a given day. The system takes the fitted rates, rounded.
[compare_scenarios][skcomplex.core.experiments.compare_scenarios] runs every
[ScenarioSpec][skcomplex.core.components.specs.ScenarioSpec] on it. The `schedule` of a scenario changes the parameter
`beta` of the interaction `infection` at the given day. An output is a path, kept as a series and as its last value, or
a function of the trajectory, here the highest number of boys in bed.

```python
from skcomplex.core.components.specs import ScenarioSpec
from skcomplex.core.experiments import compare_scenarios
from skcomplex.systems.biology.epidemiology import SIRSystem


def peak_in_bed(trajectory):
    return float(trajectory.get_values('I.x').max())


fitted = SIRSystem(susceptible=760.0, infected=3.0, recovered=0.0, beta=0.00223, gamma=0.447)
scenarios = [
    ScenarioSpec(),
    ScenarioSpec('isolation from day 4', schedule=[(4.0, 'infection.beta', 0.00223 / 2)]),
    ScenarioSpec('isolation from day 2', schedule=[(2.0, 'infection.beta', 0.00223 / 2)]),
]
outcomes = compare_scenarios(fitted, scenarios, until=30, dt=1.0, outputs=['I.x', 'R.x', peak_in_bed])
[(row['scenario'], round(row['peak_in_bed']), round(row['R.x'])) for row in outcomes.rows]
```

Without isolation the fitted model peaks at 290 boys in bed, and 744 boys catch the virus by day 30. Isolation from day
4 lowers the peak to 212 and the total to 647. Isolation from day 2 lowers them to 117 and 594. Two days earlier cut the
peak by another 95 boys. The model predicts more cases in total than the 512 boys the school counted as ill, because it
fits only the counts in bed and assumes every boy could catch the virus.

## Visualising the scenarios

The figure draws the counts and the run of every scenario. It uses the Agg backend of matplotlib, which draws to memory
and opens no window.

```python continuation
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

in_bed = np.array([3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4], dtype=float)
fig, ax = plt.subplots(figsize=(6, 4))
for row, series in zip(outcomes.rows, outcomes.get_series('I.x')):
    ax.plot(np.arange(series.size), series, label=row['scenario'])
ax.plot(np.arange(in_bed.size), in_bed, 'o', color='black', label='in bed, 1978')
ax.set_xlabel('days from 22 January 1978')
ax.set_ylabel('boys')
ax.legend()
plt.close(fig)
[text.get_text() for text in ax.get_legend().get_texts()]
```

The legend names the three scenarios and the counts of 1978. A script saves the figure with `fig.savefig('school.png')`.
Outside the Agg backend, `plt.show()` opens it in a window.

## What a real study would add

The page keeps its budgets small, so it runs in under a minute. A real study would run the variational fit for a few
thousand steps or use `method='mcmc'` with four chains, check the coverage on 100 or more simulated data sets, and fit
the convalescent counts as a fourth compartment. The pages [Calibrate](../calibrate.md) and
[Run experiments](../run_experiments.md) describe each method and its settings.
