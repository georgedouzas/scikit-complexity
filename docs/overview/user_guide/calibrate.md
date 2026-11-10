# Calibrate

This tutorial fits the rates of the predator and prey model to the hare and lynx pelt records with every method of
[calibrate][skcomplex.core.experiments.calibrate]. It shows the [losses](glossary.md#loss) each method takes, the estimates and the
diagnostics each returns, and when to choose each one. It ends with [virtual populations](glossary.md#virtual-population) and the [calibration](glossary.md#calibration) of
individuals.

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
len(times), sorted(data)
```

The data map the path of a [state](glossary.md#state), `prey.x` or `predators.x`, to its observed times and values, here 21 years counted
from 1900. The bounds map each [model parameter](glossary.md#model-parameter) of the [system](glossary.md#system) to a pair `(low, high)`. The rates are `r`, the growth
rate of the hares, `a`, the rate at which hunting removes hares, `b`, the rate at which it feeds the lynx, and `d`, the
death rate of the lynx. Every run starts from the records of 1900, so the fits below estimate the four rates and keep
the initial populations fixed.

The [posterior](glossary.md#posterior) methods run thousands of simulations, so their sections below each start a new block that repeats the
records. Each section then runs on its own.

## The losses

A loss scores how far a candidate set of rates lies from the data, and `loss` picks it. `'squares'` sums the squared
[residuals](glossary.md#residual), the differences between the run and the records. `'likelihood'` is the Gaussian [likelihood](glossary.md#likelihood) of the residuals
with an unknown variance. `'events'` is the log-probability of the random draws of a recorded [stochastic](glossary.md#stochastic) run, which
the page [Simulate](simulate.md) shows. Each method takes some of the losses and raises a `ValidationError` for the
others.

```python continuation
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import calibrate

try:
    calibrate(system, data, bounds, method='mcmc', loss='squares')
except ValidationError as error:
    message = str(error)
message
```

The message reads "The method 'mcmc' does not take 'squares'. The method 'mcmc' takes the losses likelihood, events."
The fix is to pass `loss='likelihood'`. The table at the end of the page lists the losses of every method.

## A local fit

A local fit starts from the middle of the bounds and descends to the nearest minimum of the loss. It is the default
method, `'local'`, and it runs a SciPy optimiser, `least_squares` for the loss `'squares'` and `minimize` for the
others.

```python continuation
fit = calibrate(system, data, bounds)
by_likelihood = calibrate(system, data, bounds, loss='likelihood')
estimates = {name: round(value, 3) for name, value in fit.estimates.items()}
estimates, bool(abs(by_likelihood.estimates['d'] - fit.estimates['d']) < 0.001)
```

The fit gives `r = 0.548`, `a = 0.028`, `b = 0.027`, and `d = 0.843` per year. They lie close to the posterior means
of Carpenter, `0.55`, `0.028`, `0.024`, and `0.80`, who also fitted the initial populations. The Gaussian likelihood
reaches the same rates as least squares on these data.

Every evaluation runs the system with `simulate` to the last data time and reads its states at the data times. The
result is a [CalibrationResult][skcomplex.core.experiments.CalibrationResult]. It holds the fitted system, the
goodness of fit, and the covariance of the estimates, taken from the Jacobian of the residuals.

```python continuation
standard_errors = np.sqrt(np.diag(fit.covariance))
fit.success, round(fit.r2, 3), round(fit.rmse, 1), round(fit.system.r, 3), standard_errors.round(3).tolist()
```

The optimiser converged, and the fit explains 95.5% of the variance of the counts with a root mean square error of 4.2
thousand pelts. `fit.system` is a copy of the system with the fitted rates, ready for `simulate`. The standard errors,
the square roots of the diagonal of the covariance, are 0.023 for `r` and 0.037 for `d`.

A local fit takes a few seconds here. It finds the minimum nearest its start, which `initial` sets, and it can stop in
a poor local minimum when the loss is rough.

## A global search

A global search scores candidates spread over the whole box of the bounds and keeps the best. The method `'global'`
draws them with the tree-structured Parzen estimator of Optuna, which learns from the scores so far where good
candidates lie. `budget` sets the number of evaluations and `random_state` the seed.

```python continuation
searched = calibrate(system, data, bounds, method='global', budget=100, random_state=0)
bool(abs(searched.estimates['r'] - 0.55) < 0.05), bool(abs(searched.estimates['d'] - 0.80) < 0.1), searched.r2 > 0.9
```

A hundred evaluations land within 0.05 of the growth rate 0.55 and within 0.1 of the death rate 0.80, with an R²
above 0.9. The search does not refine its best candidate, so its fit is a little worse than the local one. The default
budget is 200 evaluations, and a study with four rates would use several hundred. A global search suits rough losses,
such as those of stochastic runs, and its best candidate can start a local fit through `initial`.

## Approximate Bayesian computation

The point fits above return one value per rate. The three posterior methods below return samples of the posterior,
the distribution of the rates given the data and the [prior](glossary.md#prior). A bound `(low, high)` is read as a uniform prior, and a
SciPy distribution such as `scipy.stats.norm(0.55, 0.1)` sets the prior itself.

[Approximate Bayesian computation](glossary.md#approximate-bayesian-computation), `'abc_smc'`, needs no likelihood. It draws rates from the prior, runs the system,
and keeps the draws whose summaries lie close to the observed ones. Sequential Monte Carlo repeats this in
generations with a smaller tolerance each time. `summary` maps the series by path to summary statistics. Over a cycle
the mean of the hares of the model equals `d / b` and the mean of the lynx `r / a`, so the two means carry information
on all four rates. Every run here costs about 20 milliseconds, so this example fits only `r` and `d`, with `a` and `b`
fixed at their values in the system.

```python
import numpy as np

from skcomplex.core.experiments import calibrate
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


def means(series):
    return [series['prey.x'].mean(), series['predators.x'].mean()]


rates = {'r': (0.3, 0.9), 'd': (0.5, 1.2)}
approximate = calibrate(
    system, data, rates, method='abc_smc', summary=means, n_samples=40, budget=300, random_state=0
)
estimates = approximate.estimates
diagnostics = approximate.diagnostics
close = bool(abs(estimates['r'] - 0.55) < 0.1), bool(abs(estimates['d'] - 0.80) < 0.1)
close, approximate.samples['r'].shape, len(diagnostics['epsilons']), diagnostics['simulations'] >= 300
```

The posterior medians, which are the estimates of a posterior method, lie within 0.1 of `r = 0.55` and `d = 0.80`.
`samples` holds the 40 samples of every rate and `weights` their normalised weights. The diagnostics hold the
tolerance `epsilons` and the acceptance rate `acceptance_rates` of every generation, here three generations, and the
number of simulations, at least 300. The distance between summaries is rescaled at every generation, so the
tolerances of two generations do not compare directly. The budget stops the sampler after about 300 simulations. The
default is 20000, and a real study would draw thousands of samples. Without `summary`, the method compares the whole
series point by point, which needs a larger budget.

## Markov chain Monte Carlo

[Markov chain Monte Carlo](glossary.md#markov-chain-monte-carlo), `'mcmc'`, draws samples from the posterior of a likelihood with numpyro. For a [deterministic](glossary.md#deterministic)
system with the [tag](glossary.md#tag) `differentiable`, whose runs JAX can differentiate, it runs the No-U-Turn Sampler (NUTS), which
follows the gradient of the likelihood. Other systems run a gradient-free ensemble sampler, which is slower.
`n_samples` sets the number of draws per chain, after as many warm-up draws.

```python
import numpy as np

from skcomplex.core.experiments import calibrate
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
chain = calibrate(system, data, bounds, method='mcmc', loss='likelihood', n_samples=100, chains=1, random_state=0)
estimates = chain.estimates
close = bool(abs(estimates['r'] - 0.55) < 0.05), bool(abs(estimates['d'] - 0.84) < 0.05)
system.get_tags().differentiable, chain.diagnostics['sampler'], close
```

The system holds the tag, so the sampler is NUTS. The medians lie within 0.05 of the rates of the local fit, as they
should with a flat prior. The diagnostics also hold the split R-hat and the effective sample size of every rate.

```python continuation
rhat = chain.diagnostics['rhat']
ess = chain.diagnostics['ess']
chain.success, all(value < 1.05 for value in rhat.values()), all(value > 10 for value in ess.values())
```

The split R-hat compares the halves of every chain and is close to 1 when they agree. `success` holds when every
R-hat lies below 1.05, which it does here. The effective sample size counts the independent draws the correlated
chain is worth, above 10 for every rate out of 100 draws. One chain of 100 draws is a demonstration. A study would run
four chains, the default, of 1000 draws each and check the R-hat across chains.

## Variational inference

[Variational inference](glossary.md#variational-inference), `'variational'`, fits a multivariate normal distribution to the posterior instead of sampling
it. It runs stochastic variational inference with numpyro over `budget` steps of the Adam optimiser and then draws
`n_samples` samples from the fitted distribution. It needs the tag `differentiable`.

```python
import numpy as np

from skcomplex.core.experiments import calibrate
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
approximation = calibrate(
    system, data, bounds, method='variational', loss='likelihood', n_samples=200, budget=500, random_state=0
)
estimates = approximation.estimates
losses = approximation.diagnostics['losses']
close = bool(abs(estimates['r'] - 0.55) < 0.05), bool(abs(estimates['d'] - 0.80) < 0.05)
close, losses.shape, bool(losses[-1] < losses[0])
```

The medians lie within 0.05 of `r = 0.55` and `d = 0.80`. The diagnostics hold the loss of every step, 500 of them,
and the last is lower than the first. A flat end of the loss history shows that the fit has converged. The default
budget is 2000 steps. With the loss `'squares'` the method fits the generalised posterior of the sum of squares.

## Choosing a method

The choice turns on what the study needs from the fit and what the system allows. The table lists what each method
takes and returns.

| Method | Losses | Returns | Diagnostics |
| --- | --- | --- | --- |
| `'local'` | `'squares'`, `'likelihood'`, `'events'` | one estimate and its covariance | `success` of the optimiser |
| `'global'` | `'squares'`, `'likelihood'`, `'events'` | the best of `budget` candidates | none |
| `'abc_smc'` | `'squares'` | weighted samples | `epsilons`, `acceptance_rates`, `simulations` |
| `'mcmc'` | `'likelihood'`, `'events'` | samples of every chain | `rhat`, `ess`, `sampler` |
| `'variational'` | `'likelihood'`, `'squares'` | draws of a fitted normal | `losses` |

- `'local'` gives a point estimate in seconds when a good start is known and the loss is smooth.
- `'global'` gives a point estimate when the start is unknown or the loss is rough, as for a stochastic system.
- `'abc_smc'` gives a posterior when no likelihood is available or the data are best compared through summaries. It
  costs the most runs.
- `'mcmc'` gives samples of the posterior of a likelihood itself, with diagnostics of convergence. It is fast with
  NUTS on a differentiable system.
- `'variational'` gives a normal approximation of the posterior of a differentiable system. Its cost is set by the
  steps of `budget`, not by a number of draws, but its normal shape can miss a skewed or multimodal posterior.

The page [Evaluate](evaluate.md) checks these fits against the records and asks whether the data determine the rates.

## Virtual populations

A virtual population is a set of parameter rows whose runs look like real observations.
[virtual_population][skcomplex.core.experiments.virtual_population] draws `n` rows from the priors, runs the system
once per row, and keeps the rows whose outputs lie in plausible ranges. An output is a state path, whose last value
counts, or a function of a [trajectory](glossary.md#trajectory) named by the function.

```python
from skcomplex.core.experiments import virtual_population
from skcomplex.systems.biology.ecology import LotkaVolterraSystem


def hare_peak(trajectory):
    return float(trajectory.get_values('prey.x').max())


def lynx_peak(trajectory):
    return float(trajectory.get_values('predators.x').max())


population = virtual_population(
    LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80),
    {'r': (0.3, 0.9), 'd': (0.5, 1.2)},
    [hare_peak, lynx_peak],
    {'hare_peak': (60.0, 95.0), 'lynx_peak': (40.0, 70.0)},
    n=60,
    until=20.0,
    random_state=0,
)
round(population.acceptance, 2), population.plausible.columns.tolist()
```

Of 60 pairs of `r` and `d` drawn from the bounds, 18% give peaks of hares and lynx within the ranges of the records
over twenty years. `plausible` is a pandas data frame with one row per plausible pair and one column per parameter and
output. With `target`, the observed values of one output, the function also selects the rows whose outputs follow the
observed distribution (Allen et al. 2016). A study draws thousands of rows, the default being 1000.

## The calibration of individuals

[calibrate_individuals][skcomplex.core.experiments.calibrate_individuals] calls `calibrate` once per individual, with
the known model parameters of each. Here the two decades of the records are the individuals, each started from its
own first counts.

```python
import numpy as np

from skcomplex.core.experiments import calibrate_individuals
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
decade = np.arange(11.0)
decades = {
    '1900s': {'prey.x': (decade, hares[:11]), 'predators.x': (decade, lynx[:11])},
    '1910s': {'prey.x': (decade, hares[10:]), 'predators.x': (decade, lynx[10:])},
}
known = {'1900s': {'prey': 30.0, 'predators': 4.0}, '1910s': {'prey': 27.1, 'predators': 7.4}}
bounds = {'r': (0.3, 0.9), 'a': (0.01, 0.05), 'b': (0.01, 0.05), 'd': (0.5, 1.2)}
twins = calibrate_individuals(system, decades, bounds, known=known)
twins.estimates.round(2).to_dict('index')
```

The first decade gives `r = 0.55` and `d = 0.97`, the second `r = 0.65` and `d = 0.70`. One cycle alone pins the
rates less well than the two together. `estimates` is a data frame with one row per individual, and `calibrations`
holds the [CalibrationResult][skcomplex.core.experiments.CalibrationResult] of every individual.
