# Select a ready-made system

Find a system of the catalog, read what it is, and simulate it with the settings it was tuned for.

## The catalog

Every ready-made system is a class, named after the system with the suffix `System`, such as `SIRSystem`. You import
it from its subfield, the package `skcomplex.systems.<field>.<subfield>`, and `simulate` runs it with its tuned
settings. `all_systems` returns the name and the class of every system.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import SIRSystem

trajectory = simulate(SIRSystem())
trajectory.get_values('I.x').max().round()
```

A system class tells you what it is through plain attributes and methods:

- the first line of its docstring describes it,
- `field`, `topics`, and `references` are class attributes,
- `get_params()` gives the parameters of its model with their defaults, as for any scikit-learn estimator,
- `get_tags()` gives its capability tags, such as its time, its space, the kinds of its laws, and its randomness,
- `get_run_settings()` gives the tuned run settings that `simulate` uses for every argument a call leaves out, such as
  the end time `until` and the time between frames `dt`.

```python continuation
SIRSystem.__doc__.splitlines()[0], SIRSystem.field, SIRSystem.topics, SIRSystem.references
```

```python continuation
system = SIRSystem()
system.get_params(), system.get_tags().laws, system.get_run_settings()['until']
```

`simulate` returns a `Trajectory`. Its `values` method returns the recorded values of a state, an observable, or a
parameter by its dotted path, such as `I.x`, and `names` lists every path.

### Finding a system

`all_systems` filters by a field, such as `biology`, by a field path, such as `biology/ecology`, by topics, and by
capability tags. Listing by field and topics reads the sources of the catalog, so it imports only the modules of the
systems it returns. Filtering by tags builds every selected class with its default parameters.

```python
from skcomplex.systems import all_systems

[name for name, _ in all_systems(field='biology/ecology')][:5]
sorted({cls.field for _, cls in all_systems(field='chemistry')})
[name for name, _ in all_systems(topics='oscillation')][:5]
[name for name, _ in all_systems(field='chemistry', tags={'randomness': 'stochastic'})]
```

The tags describe a system with its default parameters. `SIRSystem` runs as differential equations by default, so it
is deterministic, and `SIRSystem(stochastic=True)` is stochastic.

The catalog belongs to the catalog alone. `simulate`, the analyses, and the experiments take a system instance, or a
class or a factory where a function builds several systems, never a name.

Each system has a field path `<field>/<subfield>`, which `all_systems` takes as a filter, and is imported from the
subfield package of the same path, such as `skcomplex.systems.biology.epidemiology` for `biology/epidemiology`. The
page of each system under [Systems](../../systems/index.md) shows its import.

| Field | Subfields |
| --- | --- |
| `physics` | `mechanics`, `celestial_mechanics`, `nonlinear_oscillators`, `synchronization`, `statistical_mechanics`, `criticality`, `soft_matter`, `active_matter`, `fluids`, `waves`, `pattern_formation`, `heat`, `cosmology`, `climate`, `oceanography`, `seismology` |
| `chemistry` | `kinetics`, `pattern_formation` |
| `biology` | `ecology`, `epidemiology`, `evolution`, `neuroscience`, `systems_biology`, `physiology`, `collective_behaviour`, `oncology` |
| `economics` | `macroeconomics`, `markets`, `finance`, `econophysics`, `resources`, `innovation` |
| `social` | `opinion_dynamics`, `segregation`, `cooperation`, `spreading`, `crowds`, `organizations`, `language` |
| `engineering` | `traffic`, `circuits`, `control`, `power_systems`, `robotics`, `processes`, `mechatronics` |
| `mathematics` | `dynamical_systems`, `stochastic_processes`, `networks`, `fractals`, `fractional_dynamics` |
| `computer_science` | `automata`, `evolutionary_computation`, `neural_networks` |

A system lives where the science that poses it lives, and its topics carry its reach into other fields. The library
keeps the catalog apart from the core by three rules, which the checks of the package layout enforce.

- The core never imports the systems.
- Every system is built only from the public names of the core.
- Every subfield package is self-contained, so no system package imports another. Code that several subfields of one
  field share lives in the `common` package of that field, and no field uses the `common` package of another.

### Optional extras

Most systems run on the default install. A few take a library from an optional extra. Without the extra they stay in
the catalog, and preparing or simulating one raises `MissingExtraError` with the name of the system, the extra, and
the install command.

| Extra | Library | Systems | Install notes |
| --- | --- | --- | --- |
| `structures` | Biopython | `fuzzy_oil_drop`, `early_stage_folding_intermediate` | Biopython 1.88 publishes no wheel for macOS on Intel, so the extra builds it from source there with a C compiler. |
| `rna` | ViennaRNA | `rna_neutral_networks` | The licence of ViennaRNA allows research and commercial use, forbids redistribution for a fee, and is not OSI approved, so the default install never brings it. |
| `trajectories` | MDTraj | none, it reads molecular dynamics trajectories for the systems biology systems | MDTraj is under the LGPL. |

### Systems as patterns

The ready-made systems are examples of the core interface, not a separate one. So each is also the pattern for a system
of your own. The page of every field, under [Systems](../../systems/index.md) in the documentation, shows for each
system:

- its class, with its parameters and tags,
- its element and interaction types, with their states, parameters, and laws as written,
- for differential equations and maps, its equations as `symbolic(system).equations` returns them.

A system of your own follows the same steps. Declare the element and interaction types, and write a class that builds
the system from its parameters, as the page [System classes](system_classes.md) shows. The section [Choosing compatible components](build_a_custom_system/choosing_compatible_components.md) shows how to find
which parts fit together, and the
[contributing guide](../../development/contributing.md#adding-a-system-to-the-catalog) covers adding a system to the
catalog.

### Inventory

The system inventory, `docs/overview/inventory.toml`, records every system surveyed from the reference books and
articles, with its sources and its status: implemented, pending a planned change, or blocked by a missing feature of the
core. A system the core cannot express is recorded as blocked on the OpenSpec change that specifies the missing general
feature. It is never worked around inside the system package. A test checks the inventory against the catalog and the
changes. An entry writes the components and the bindings of its system by hand, while the time, the space, the laws,
and the domains of an implemented system are read from `get_tags()` of its class, so they cannot drift from the code.
Every system class has its own page under Systems in the documentation, by field and subfield.

Systems exist first to exercise the core, and ready-made systems are a by-product. Every entry lists the core primitives
it exercises, such as `rates`, `jumps`, `groups`, `reductions`, `member-offsets`, `lattices`, `fields`, `agents`, or
`stocks-and-flows`, from a vocabulary the inventory defines. A primitive that the core does not have yet names the
change that adds it. An entry's priority is `core` when it exercises a primitive, or a combination of primitives, that
no earlier implemented entry exercises, or when it needs a missing one. It is `example` when it repeats a covered
pattern. So new systems go first where the core is least exercised.
