# A lattice phase transition

This case study locates the phase transition of the two-dimensional Ising model and estimates its critical exponents
from small lattices. It checks the model from the catalog, sweeps it over temperature on lattices of three sizes, finds
the peak of the susceptibility near the exact critical temperature of Onsager, and estimates the exponents by
finite-size scaling.

## The model and the exact result

The Ising model places a spin `s = +1` or `s = -1` on every site of a square lattice. Neighbouring spins prefer to point
the same way, with the energy `-J s s'` per pair. At low temperature most spins align and the lattice is magnetised. At
high temperature thermal noise breaks the order. The magnetization `m`, the mean spin, measures the order. Onsager
(1944, "Crystal statistics. I. A two-dimensional model with an order-disorder transition", *Physical Review* 65, 117)
solved the model exactly. The order vanishes at the critical temperature:

```text
T_c = 2 J / (k_B ln(1 + sqrt(2))) = 2.269 J / k_B
```

The temperatures below are in units of `J / k_B`. Near `T_c` the order parameter and the susceptibility follow power
laws with exponents known from the exact solution: `beta = 1/8` for the magnetization, `gamma = 7/4` for the
susceptibility, and `nu = 1` for the correlation length, the distance over which spins are correlated.

## Selecting and checking the model

The catalog holds the model as [IsingSystem](../../../systems/physics/statistical_mechanics/ising.md). One unit of
time is one sweep, an attempt to flip every spin once by the Metropolis rule. An
[ObservableSpec][skcomplex.core.specs.ObservableSpec] records the magnetization at every sweep, the mean of the variable
`spin` over the group `spins`.

```python
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ObservableSpec
from skcomplex.systems.physics.statistical_mechanics import IsingSystem

magnetization = ObservableSpec('magnetization', group='spins', variable='spin', reduce='mean')
lattice = IsingSystem(16, temperature=1.8, initial='up')
tags = lattice.get_tags()
order = []
for temperature in (1.8, 3.0):
    lattice.set_params(temperature=temperature)
    run = simulate(lattice, until=500, observables=[magnetization], random_state=0)
    order.append(round(float(np.abs(run.get_values('magnetization')[100:]).mean()), 2))
sorted(tags.space), tags.randomness, order
```

The spins live on a lattice and the updates are random. On a lattice of 16 by 16 spins that starts with every spin up,
the mean of `|m|` after 100 sweeps stays at 0.96 at `T = 1.8`, below `T_c`, and falls to 0.17 at `T = 3.0`, above it.

## Sweeping the temperature

The susceptibility measures how strongly the magnetization fluctuates. For a lattice of `N = L^2` spins it is
`chi = N (<m^2> - <|m|>^2) / T`, where the brackets are means over the sweeps of a run. It peaks at the transition, and
on a finite lattice the peak sits a little above `T_c` and moves towards it as the lattice grows.
[sweep][skcomplex.core.experiments.sweep] runs a grid of the side `size` and the `temperature`, two replicates per pair.
An output function returns the moments of `|m|` of a run after 300 sweeps of warm-up.

```python
import numpy as np

from skcomplex.core.experiments import sweep
from skcomplex.core.specs import ObservableSpec
from skcomplex.systems.physics.statistical_mechanics import IsingSystem

magnetization = ObservableSpec('magnetization', group='spins', variable='spin', reduce='mean')


def moments(trajectory):
    values = np.abs(trajectory.get_values('magnetization')[300:])
    return {'m': values.mean(), 'm2': (values**2).mean(), 'm4': (values**4).mean()}


sizes = np.array([8, 16, 32])
temperatures = np.round(np.arange(1.9, 2.95, 0.1), 2)
swept = sweep(
    IsingSystem(initial='up'),
    {'size': sizes, 'temperature': temperatures},
    replicates=2,
    until=1500,
    observables=[magnetization],
    outputs=[moments],
    random_state=0,
)
shape = (sizes.size, temperatures.size, 2)
m, m2, m4 = (swept.get_column(f'moments.{name}').reshape(shape) for name in ('m', 'm2', 'm4'))
chi_runs = sizes[:, None, None] ** 2 * (m2 - m**2) / temperatures[:, None]
chi = chi_runs.mean(axis=2)
peaks = temperatures[chi.argmax(axis=1)]
tc = 2 / np.log(1 + np.sqrt(2))
peaks.tolist(), round(float(tc), 3), bool(abs(peaks[-1] - tc) < 0.3)
```

The grid runs the last name fastest, so the rows reshape into sizes, temperatures, and replicates. The susceptibility
peaks at `T = 2.5`, 2.4, and 2.3 for the sides 8, 16, and 32. The peak moves towards `T_c = 2.269` as the lattice grows,
and on the lattice of 32 by 32 spins it lies within the tolerance 0.3 of the exact value. The grid steps by 0.1, so the
peak cannot be located more finely than that.

## Critical exponents by finite-size scaling

Finite-size scaling, the theory of how a transition looks on a finite lattice, predicts that the susceptibility curves
of all sizes fall onto one curve when they are rescaled. The curve `chi_L(T) L^(-gamma / nu)` is plotted against
`(T - T_c) L^(1 / nu)`. [data_collapse][skcomplex.core.analysis.data_collapse] fits the exponents that make the rescaled
curves overlap best. With `kind='crossings'` it finds where the curves of consecutive sizes cross, here the Binder
cumulant `U = 1 - <m^4> / (3 <m^2>^2)`, whose curves cross near `T_c` independently of the size.
[scaling][skcomplex.core.analysis.scaling] fits power laws. With `kind='allometric'` it fits the growth of the peak of
the susceptibility with the side, `chi_max ~ L^(gamma / nu)`, and with `kind='power'` the decay of the magnetization
below `T_c`, `m ~ (T_c - T)^beta`.

```python continuation
from skcomplex.core.analysis import data_collapse, scaling

binder = 1 - m4.mean(axis=2) / (3 * m2.mean(axis=2) ** 2)
crossings = data_collapse(dict(zip(sizes, binder)), kind='crossings', parameters=temperatures).crossings
collapse = data_collapse(
    dict(zip(sizes, chi)), parameters=temperatures, critical=tc, nu=1.0, exponent=1.75, fit=('nu', 'exponent')
)
growth = scaling(dict(zip(sizes, chi_runs.max(axis=1))), kind='allometric', resamples=200)
below = temperatures < tc
decay = scaling(m[-1].mean(axis=1)[below], kind='power', parameters=temperatures[below], critical=tc)
(
    crossings.round(2).tolist(),
    round(collapse.nu, 2),
    round(collapse.exponent, 2),
    round(growth.exponent, 2),
    round(float(decay.exponent[0]), 2),
)
```

The Binder cumulants of the sides 8 and 16 cross at 2.28, and those of 16 and 32 at 2.22, on both sides of the exact
2.269. The collapse gives `nu = 0.98` against the exact 1, and `gamma / nu = 1.81` against the exact 7/4 = 1.75. The
growth of the peak of the susceptibility gives 1.79. The magnetization of the lattice of 32 by 32 spins gives
`beta = 0.10` against the exact 1/8 = 0.125. The fit uses the temperatures below `T_c`, where a small lattice still
keeps some magnetization close to the transition, so the estimate of `beta` is the least accurate of the four.

## Visualising the collapse

The figure draws the rescaled susceptibility of the three sizes against the rescaled temperature with the fitted
exponents. It uses the Agg backend of matplotlib, which draws to memory and opens no window.

```python continuation
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
for size, curve in zip(sizes, chi):
    rescaled = (temperatures - tc) * size ** (1 / collapse.nu)
    ax.plot(rescaled, curve * size ** (-collapse.exponent), 'o-', label=f'L = {size}')
ax.set_xlabel('(T - T_c) L^(1 / nu)')
ax.set_ylabel('chi L^(-gamma / nu)')
ax.legend()
plt.close(fig)
[text.get_text() for text in ax.get_legend().get_texts()]
```

The legend names the three sizes. A script saves the figure with `fig.savefig('collapse.png')`. Outside the Agg backend,
`plt.show()` opens it in a window.

## What a real study would add

The page keeps its budgets small, so it runs in under a minute. A real study would use lattices up to 128 by 128 spins
or more, a grid of steps of 0.01 near `T_c`, runs of 10^5 sweeps or more with error bars from blocks of sweeps, and a
cluster algorithm such as that of Wolff (1989, *Physical Review Letters* 62, 361), since local rules such as Metropolis
slow down near `T_c`. The page [Run experiments](../run_experiments.md) describes `sweep`, and the page
[Analyse](../analyse.md) the analysis functions.
