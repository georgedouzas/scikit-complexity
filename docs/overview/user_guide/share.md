# Share a system

This tutorial turns the hare and lynx model into a system class of the catalog, checks it the way the tests of the
catalog do, and writes its inventory entry and its gallery example. It ends with the steps of a contribution to the
repository. The page [Extend the library](extend_the_library.md) explains where a new component goes, and this page
follows its placement rule.

## The system class

A system of the catalog is a subclass of [`BaseSystem`][skcomplex.core.components.systems.BaseSystem] whose name ends
in `System`. It takes its model parameters as arguments, stores each under its own name, and builds its components in
`_build_system`. Four class variables describe it:

- `field` is the field path, a field and a subfield of the catalog, such as `'biology/ecology'`.
- `topics` are the words that `all_systems` filters by, such as `'predator-prey'`.
- `references` are the works the model comes from, each an author and a year.
- `until` and `dt` are the tuned run settings, which `simulate` reads when a call leaves them out.

The tutorial [Build a custom system](build_a_custom_system/index.md) wrote the model as `PeltSystem` from core
components. The class below builds it from the same core components. A class of the catalog would build it from the
components that the systems of ecology share in `skcomplex.systems.biology.common`, a population element and the growth,
predation, and mortality interactions, so that one model has one implementation. Its defaults are the rates of Carpenter
(2018) and the pelt records of 1900, and its docstring follows the style of the catalog: a summary, a paragraph on what
the model shows, the arguments, and an example.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import BaseSystem, CompositeSystem
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec


class HareLynxSystem(BaseSystem):
    """Snowshoe hares and Canada lynx whose populations cycle through predation.

    The defaults are the posterior means that Carpenter (2018) fitted to the pelt records of the Hudson's Bay Company,
    and the initial populations are the records of 1900, in thousands of pelts. The system has the populations `hares`
    and `lynx`.

    Args:
        hares:
            The initial number of hares, in thousands, such as `30.0`.

        lynx:
            The initial number of lynx, in thousands, such as `4.0`.

        alpha:
            The birth rate of the hares, per year.

        beta:
            The rate at which the lynx remove hares, per thousand lynx and year.

        gamma:
            The death rate of the lynx, per year.

        delta:
            The rate at which eating hares feeds the lynx, per thousand hares and year.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> HareLynxSystem().prepare().system_.elements[0].label
        'hares'
    """

    field = 'biology/ecology'
    topics = ('predator-prey', 'ode', 'oscillation', 'data')
    references = ('Hewitt 1921', 'Carpenter 2018')
    until = 20.0
    dt = 1.0

    def __init__(
        self,
        hares=30.0,
        lynx=4.0,
        *,
        alpha=0.55,
        beta=0.028,
        gamma=0.80,
        delta=0.024,
        label='system',
    ):
        super().__init__(label)
        self.hares = hares
        self.lynx = lynx
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _build_system(self):
        hares = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(self.hares, StateSpec())})
        lynx = DeclaredElement('lynx', variables={'x': DeclaredVariableSpec(self.lynx, StateSpec())})
        alpha, beta, gamma, delta = self.alpha, self.beta, self.gamma, self.delta
        interactions = [
            RateInteraction('births', target='hares', state='x', rate=lambda s: alpha * s.target.x),
            RateInteraction(
                'hunting',
                target='hares',
                state='x',
                sources=['lynx'],
                rate=lambda s: -beta * s.target.x * s.sources[0].x,
            ),
            RateInteraction(
                'feeding',
                target='lynx',
                state='x',
                sources=['hares'],
                rate=lambda s: delta * s.target.x * s.sources[0].x,
            ),
            RateInteraction('deaths', target='lynx', state='x', rate=lambda s: -gamma * s.target.x),
        ]
        return CompositeSystem(self.label, elements=[hares, lynx], interactions=interactions)


HareLynxSystem().get_params()
```

`get_params` returns the initial populations, the four rates, and the label by name, as for any estimator of
scikit-learn, so `set_params`, `clone`, and the experiments of the library change them. The constructor only stores its arguments, which is what makes this
work.

## Checking the class

The catalog tests run every system class through one generic check. It builds the class with its default parameters,
simulates it for ten frames while a [`RerunObserver`][skcomplex.core.engine.viz.RerunObserver] saves the run, and
requires a frame logged for every simulated frame. Then it clones the system and compiles the copy. The block below
runs the same check on the new class, saving to a temporary file so that no viewer opens.

```python continuation
import tempfile
from pathlib import Path

from sklearn.base import clone

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver

system = HareLynxSystem()
settings = system.get_run_settings()
observer = RerunObserver('hare_lynx', save_path=Path(tempfile.mkdtemp()) / 'run.rrd')
trajectory = simulate(system, until=10 * settings['dt'], observers=[observer], random_state=0)
copy = clone(system)
hares = trajectory.get_values('hares.x').round(1)[:4].tolist()
observer.logged_frames_ == len(trajectory.times), copy is not system, hares
```

The observer logged all eleven frames, the clone is a new object, and the hares rise from 30.0 to 81.2 thousand in
1903, as in the build tutorial. The tuned `dt` of one year set the length of the run.

A class of the catalog must also run like the same components put together by hand, and it must say what it is through
its capability tags. The block below compares the states of a run with those of `LotkaVolterraSystem`, which holds the
same model with the rates `r`, `a`, `b`, and `d`, and reads the tags.

```python continuation
import numpy as np

from skcomplex.systems.biology.ecology import LotkaVolterraSystem

reference = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
difference = simulate(system, until=20).states - simulate(reference, until=20, dt=1.0).states
tags = system.get_tags()
bool(np.abs(difference).max() < 1e-9), sorted(tags.time), tags.randomness, sorted(tags.laws)
```

The states of the two systems agree to within rounding error, below 1e-9 thousand animals, so the new class adds the
data and the names, not a new model. The tags say that
the system runs in continuous time, is deterministic, and holds rate laws. The catalog test of the tags reads the same
values, and the inventory takes them from the class rather than repeating them.

A class is listed by `all_systems` when its module lives in the package `skcomplex.systems` and its body sets `field`.
The listing reads the sources without importing them, so `field`, `topics`, and `references` must be literal values.
A class defined anywhere else, such as on this page, stays out of the listing.

```python continuation
from skcomplex.systems import all_systems

listed = [name for name, _ in all_systems(field='biology/ecology')]
'LotkaVolterraSystem' in listed, 'HareLynxSystem' in listed
```

`LotkaVolterraSystem` is listed and `HareLynxSystem` is not. The listing raises a `RegistryError` for a class of the
package whose name lacks the suffix `System` or whose field is computed at run time.

## The files of a contribution

A new system adds a module, an export, an inventory entry, and a gallery example. The catalog already holds this model
as `LotkaVolterraSystem`, so the files below show the steps rather than a contribution the catalog needs. A real one
adds a system the catalog lacks.

```text
src/skcomplex/systems/biology/ecology/
    _pelts.py                    the class HareLynxSystem
    __init__.py                  imports HareLynxSystem and lists it in __all__
tests/systems/biology/ecology/
    test_pelts.py                the tests of the class, mirroring the source
docs/overview/inventory.toml     the entry hare_lynx under [biology.ecology]
docs/examples/biology/ecology/
    plot_hare_lynx.py            the gallery example
```

The module holds one system with the components only it uses. A component that several systems of ecology share goes to
`skcomplex.systems.biology.common`, and a component that several fields share goes to `skcomplex.systems.general`, as
the placement rule of [Extend the library](extend_the_library.md#where-a-new-component-goes) says.

### The inventory entry

The inventory `docs/overview/inventory.toml` lists every system surveyed from the reference books and articles, with
its status. The name of an entry is the class name without `System` in snake case, so `HareLynxSystem` is `hare_lynx`.
The entry sits in the table of its field path, and the check of the inventory, which the documentation build runs,
compares it with the class.

```toml
[biology.ecology]
hare_lynx = { title = "Hare and lynx pelts", sources = ["Hewitt 1921", "Carpenter 2018"], primitives = ["rates", "interactions"], system = { components = ["element"], bindings = ["members"] }, priority = "example", status = "implemented" }
```

The fields of the entry are these:

- `title` names the system in the gallery.
- `sources` are keys of the table `sources` with a section, arXiv identifiers, or citations of an author and a year.
- `primitives` are the capabilities of the core the system exercises, from the table `primitives`.
- `system` maps the system to the vocabulary: its components, here single elements, and its bindings, here roles bound
  to elements.
- `priority` is `'core'` when the system exercises a set of primitives that no earlier entry exercises, and
  `'example'` otherwise. The rates and interactions of this model repeat those of `lotka_volterra`.
- `status` is `'implemented'` once the class exists, or `'pending <change>'` while an OpenSpec change builds it.

An implemented entry does not write its time, space, laws, or domains, since the check reads them from `get_tags()` of
the class, as the block above did.

### The gallery example

Every system of the catalog has one example in the gallery, a script under
`docs/examples/<field>/<subfield>/plot_<name>.py`. After the inventory entry exists, the command
`PYTHONPATH=docs python -m gallery_tools scaffold` writes a scaffold for every system without an example. The finished
script sets `SYSTEM` to the gallery name, runs the system with a `RerunObserver` and a blueprint, and draws one figure
of an analysis whose result the caption predicts.

```text
"""
# Hare and lynx pelts

The hares and the lynx of the Hudson's Bay Company records cycle out of phase, and every peak of the hares is followed
by a peak of the lynx a year later.
"""

# %%
# The system starts from the records of 1900 and runs for twenty years with the rates of Carpenter (2018).

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import HareLynxSystem

SYSTEM = 'hare_lynx'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(HareLynxSystem(), observers=[observer])

# %%
# The phase plane
# ---------------
# The run is one closed cycle around the equilibrium (gamma / delta, alpha / beta).

fig, axes = plt.subplots()
axes.plot(trajectory.get_values('hares.x'), trajectory.get_values('lynx.x'))
axes.set(xlabel='hares (thousands)', ylabel='lynx (thousands)', title='The phase plane')
plt.show()
```

The docstring gives the title and a caption of the behaviour the thumbnail must show. The command
`PYTHONPATH=docs python -m gallery_tools run hare_lynx` runs the example and captures its screenshot, which a reviewer
compares with the caption.

## Contributing

A contribution goes to the repository as a pull request. The [contributing guide](../../development/contributing.md)
lists the tasks that format the code, run the checks and the tests, and build the documentation, with the convention of
commit messages and the guidelines of pull requests. Before the pull request, run the tests of the catalog,
`pytest tests/systems/test_catalog.py`, which apply the checks of this page to the new class, and build the
documentation, which runs the check of the inventory and the gallery example.
