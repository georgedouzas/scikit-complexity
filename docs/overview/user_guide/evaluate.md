# Evaluate

This tutorial checks how well the predator and prey model fits the hare and lynx pelt records. It compares a run with
the records, reads the goodness of fit of a [calibration](glossary.md#calibration), checks a [posterior](glossary.md#posterior) against the data and against simulated
truths, and asks whether the data determine the rates at all. The page [Calibrate](calibrate.md) shows the fits
themselves.

## The data and the model

The records and the model are those of the [build tutorial](build_a_custom_system/index.md). The counts are the pelts
of 1900 to 1920, in thousands, as distributed with the case study of Carpenter (2018, *Predator-Prey Population
Dynamics: the Lotka-Volterra model in Stan*) in the file `knitr/lotka-volterra/hudson-bay-lynx-hare.csv` of the Stan
example models. That file cites the lecture notes of Howard (2009) at Texas A&M University, which trace the numbers
through the notes of Mahaffy (Math 636, San Diego State University) to Purves, Sadava, Orians, and Heller, *Life: The
Science of Biology*.

```python
import numpy as np

from skcomplex.systems.biology.ecology import LotkaVolterraSystem

years = np.arange(1900, 1921)
hares = np.array(
    [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1]
    + [40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
)
lynx = np.array(
    [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4]
    + [8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]
)
system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
times = years - 1900.0
data = {'prey.x': (times, hares), 'predators.x': (times, lynx)}
bounds = {'r': (0.3, 0.9), 'a': (0.01, 0.05), 'b': (0.01, 0.05), 'd': (0.5, 1.2)}
system.r, system.d
```

The [system](glossary.md#system) holds the rates that Carpenter estimated: `r`, the growth rate of the hares, `a`, the rate at which
hunting removes hares, `b`, the rate at which it feeds the lynx, and `d`, the death rate of the lynx. Every run starts
from the records of 1900.

## A run against the records

A run of the model gives the populations at the years of the records. Their differences are the [residuals](glossary.md#residual), and the
root mean square of the residuals of each series measures how far the run lies from it.

```python continuation
from skcomplex.core.engine.runtime import simulate

run = simulate(system, until=20, dt=1.0)
residuals = {'hares': run.get_values('prey.x') - hares, 'lynx': run.get_values('predators.x') - lynx}
errors = {name: round(float(np.sqrt(np.mean(values**2))), 1) for name, values in residuals.items()}
worst = {name: int(years[np.abs(values).argmax()]) for name, values in residuals.items()}
errors, worst
```

The run lies 7.8 thousand pelts from the hares on average and 6.7 thousand from the lynx. The largest residuals show
where the model departs from the records. The run misses the hares most in 1914, in the second cycle, and the lynx in
1903, the year their counts jump from 9.8 to 35.2 thousand. The rates of Carpenter came from a fit that also estimated
the initial populations, so a run that starts from the counts of 1900 drifts out of phase over two cycles.

## The goodness of fit of a calibration

A calibration returns its own measures of fit, computed over all data points at the fitted rates. The local fit of
[calibrate][skcomplex.core.experiments.calibrate] below estimates the four rates by least squares. `fit.simulated`
holds the fitted values at the data times, so the residuals of the fit come without a new run.

```python continuation
from skcomplex.core.experiments import calibrate

fit = calibrate(system, data, bounds)
fitted = fit.simulated['prey.x'] - hares
measures = round(fit.r2, 3), round(fit.rmse, 1), round(fit.mae, 1), round(fit.mape, 1)
measures, round(float(np.sqrt(np.mean(fitted**2))), 1)
```

The [CalibrationResult][skcomplex.core.experiments.CalibrationResult] holds four measures.

- `r2`, the coefficient of determination, is the share of the variance of the counts that the fit explains, here
  0.955.
- `rmse`, the root mean squared error, is 4.2 thousand pelts.
- `mae`, the mean absolute error, is 3.1 thousand pelts. It weighs large misses less than the RMSE does.
- `mape`, the mean absolute percentage error, is 17.0%. It weighs the small counts of the troughs most.

The fit lies 4.7 thousand pelts from the hares, against 7.8 thousand for the first run.

## Posterior predictive checks

A [posterior predictive check](glossary.md#posterior-predictive-check) asks whether data simulated from the posterior look like the observed data. Each
posterior method of `calibrate` runs it on at most 200 posterior draws. With the Gaussian [likelihood](glossary.md#likelihood) every draw also
adds the observation noise that its residuals imply. The result holds the 5%, 50%, and 95% quantiles of the simulated
values at every data time, and the share of observed points inside the band from 5% to 95%.

```python continuation
posterior = calibrate(
    system, data, bounds, method='mcmc', loss='likelihood', n_samples=100, chains=1, random_state=0
)
band = posterior.predictive
low, high = band['q05']['prey.x'], band['q95']['prey.x']
bool(band['coverage'] > 0.75), low.shape, bool(np.all(low <= high))
```

The posterior of a short chain of [Markov chain Monte Carlo](glossary.md#markov-chain-monte-carlo) covers more than three quarters of the 42 records with its
90% band. A band of 90% should hold about 90% of the points. A much lower share shows a model that misses the data, and
a share near 100% a band wider than the noise. `q05`, `q50`, and `q95` map each path to 21 values, one per year.

## Posterior coverage

A [credible interval](glossary.md#credible-interval) of level 90% should hold the true rate in 90% of studies.
[posterior_coverage][skcomplex.core.experiments.posterior_coverage] checks this with simulated truths. It draws true
rates from the [prior](glossary.md#prior), simulates data at each with Gaussian noise, calibrates every data set with a posterior method,
and counts how often the central interval of each level holds the truth. It also gives the rank of every truth among
its samples, the check of simulation-based calibration (Talts et al. 2018). Every calibration costs a full run of the
sampler, so this example checks only the growth rate `r`, with the other rates fixed, on three data sets. It needs the
times of the data but not the records, so its block starts afresh.

```python
import numpy as np

from skcomplex.core.experiments import posterior_coverage
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

check = posterior_coverage(
    LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80),
    {'r': (0.4, 0.7)},
    np.arange(21.0),
    method='mcmc',
    n=3,
    levels=(0.5, 0.9),
    noise=3.0,
    n_samples=50,
    chains=1,
    random_state=0,
)
check.levels, check.ranks['r'].shape, len(check.results)
```

`coverage` maps each level to the share of data sets whose interval holds the truth, by parameter. `ranks` holds the
weighted share of samples below the truth, one per data set, and `truths` the true rates. `results` holds the
[CalibrationResult][skcomplex.core.experiments.CalibrationResult] of every data set. A calibrated method covers the
truth at about the rate of each level, and its ranks spread evenly between 0 and 1. Three data sets cannot show that.
A real check uses 40 data sets, the default, or more, and a histogram of the ranks.

## Practical identifiability

A calibration returns estimates even when the data cannot pin them down. Practical [identifiability](glossary.md#identifiability) asks whether the
data at hand fix each rate within its bounds. [identifiability][skcomplex.core.experiments.identifiability] answers it
in two ways. The method `'refits'` of Liyanage et al. (2025) adds Gaussian noise relative to the fitted values, fits
`n` noisy copies again, and reports the average relative error of every rate at every level of noise. The block
repeats the records, so this section runs on its own.

```python
import numpy as np

from skcomplex.core.experiments import identifiability
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

hares = np.array(
    [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1]
    + [40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
)
lynx = np.array(
    [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4]
    + [8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]
)
system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
times = np.arange(21.0)
data = {'prey.x': (times, hares), 'predators.x': (times, lynx)}
bounds = {'r': (0.3, 0.9), 'a': (0.01, 0.05), 'b': (0.01, 0.05), 'd': (0.5, 1.2)}
refits = identifiability(system, data, bounds, method='refits', n=5, noise=(0.05,), random_state=0)
refits.verdicts, {name: bool(errors[0.05] < 0.05) for name, errors in refits.errors.items()}
```

With all 21 years of both series, every rate has the verdict `'strong'`, because its average relative error under
noise of 5% stays below 5%. The verdict is `'weak'` for an error up to ten times the noise and `'none'` above. Five
copies are a demonstration. A study uses the default of 100 copies and several levels of noise, such as
`(0.01, 0.05, 0.1)`.

The default method, `'profile'`, computes the profile likelihood with pyPESTO. The profile of a rate fixes it at
values on both sides of the best fit, fits the other rates again, and keeps the values whose likelihood ratio stays
above the threshold of a chi-square test at level 0.95. A rate is identifiable when both ends of this interval lie
inside its bounds. A profile costs many fits, so this example profiles two rates. The first three hare counts, from
1900 to 1902, come before the lynx rise.

```python continuation
early = {'prey.x': (times[:3], hares[:3])}
profile = identifiability(system, early, {'b': (0.01, 0.05), 'd': (0.5, 1.2)}, n_starts=1, random_state=0)
profile.verdicts, {name: np.round(ends, 3).tolist() for name, ends in profile.intervals.items()}
```

The early counts fix `b`, the feeding rate of the lynx, between 0.014 and 0.032. They do not fix `d`, the death rate
of the lynx, whose interval is open at both bounds and so reads `nan` on both sides. A death rate anywhere between 0.5
and 1.2 fits three years of hares equally well. Widening the bounds and running again shows whether an open side
closes. `profiles` holds the values and the likelihood ratios along every profile, ready to plot. A study profiles
every rate with the default of five starts.
