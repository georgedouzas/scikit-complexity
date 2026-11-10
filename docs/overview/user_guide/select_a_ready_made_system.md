# Select a ready-made system

This page finds a system of the catalog, reads what it models, and runs it with the settings it was tuned for. The
catalog is the set of ready-made systems of `skcomplex.systems`, each a class built from the public parts of the core.

## The catalog

This section runs one system of the catalog and reads its description. Every ready-made system is a class named after
the system with the suffix `System`, such as `SIRSystem`, the epidemic model of Kermack and McKendrick (1927). You
import it from its subfield, the package `skcomplex.systems.<field>.<subfield>`, and `simulate` runs it with its tuned
settings.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import SIRSystem

trajectory = simulate(SIRSystem())
trajectory.get_values('I.x').max().round()
```

The number of infected people peaks at 304. `simulate` returns a [trajectory](glossary.md#trajectory), the recorded
run. Its method `get_values` returns the values of a state, an observable, or a parameter by its dotted path, such as
`I.x`, the state `x` of the element `I`, and its attribute `names` lists every path.

A system class says what it is through plain attributes and methods.

- The first line of its docstring describes it.
- `field`, `topics`, and `references` are class attributes.
- `get_params()` returns the parameters of its model with their defaults, as for any scikit-learn estimator.
- `get_tags()` returns its capability [tags](glossary.md#tag), such as its time, its space, the kinds of its laws, and
  its randomness.
- `get_run_settings()` returns the tuned run settings that `simulate` uses for every argument a call leaves out, such as
  the end time `until` and the time between frames `dt`.

```python continuation
SIRSystem.__doc__.splitlines()[0], SIRSystem.field, SIRSystem.topics, SIRSystem.references
```

The docstring reads "An epidemic that moves individuals from susceptible to infected to recovered." The field path is
`'biology/epidemiology'`, the topics are `'compartments'`, `'ode'`, and `'jump'`, and the reference is Kermack and
McKendrick (1927).

```python continuation
system = SIRSystem()
system.get_params(), system.get_tags().laws, system.get_run_settings()['until']
```

The parameters are 990 susceptible, 10 infected, and 0 recovered people, the infection rate `beta = 0.0003`, the
recovery rate `gamma = 0.1`, `stochastic=False`, and the label. The system holds rate laws, so it runs as differential
equations, and its tuned run ends at time 160.

## Finding a system

This section finds systems by field, by topic, and by what they do. `all_systems` returns the name and the class of
every system that matches its filters. A field path such as `'biology/ecology'` selects one subfield.

```python
from skcomplex.systems import all_systems

[name for name, _ in all_systems(field='biology/ecology')][:5]
```

The first five systems of ecology are `AgentBasedLotkaVolterraSystem`, `CometsSystem`, `CommunityFBASystem`,
`CompetitionSystem`, and `ConsumerResourceSystem`. A field alone, such as `'chemistry'`, selects all its subfields.

```python continuation
sorted({cls.field for _, cls in all_systems(field='chemistry')})
```

The systems of chemistry belong to `'chemistry/kinetics'` and `'chemistry/pattern_formation'`. A topic selects systems
across every field.

```python continuation
[name for name, _ in all_systems(topics='oscillation')][:5]
```

The first five oscillators are `AdlerEquationSystem`, `AgentBasedLotkaVolterraSystem`, `BrusselatorODESystem`,
`BrusselatorSystem`, and `CoupledPendulumsSystem`. Tags select systems by what they do, here the systems of chemistry
that draw random numbers.

```python continuation
[name for name, _ in all_systems(field='chemistry', tags={'randomness': 'stochastic'})]
```

They are `ChemicalLangevinEquationSystem` and `ChemicalLangevinNetworkSystem`. A listing by field and topics reads the
sources of the catalog, so it imports only the modules of the systems it returns. A filter by tags builds every
selected class with its default parameters, and the tags describe that default. `SIRSystem` is deterministic by
default, and `SIRSystem(stochastic=True)` is stochastic.

`simulate`, the analyses, and the experiments take a system instance, never a name. A function that builds several
systems, such as a sweep, takes a class or a factory.

### Fields and subfields

This section lists the field paths of the catalog. Each system has a field path `<field>/<subfield>`, which
`all_systems` takes as a filter, and you import the system from the subfield package of the same path, such as
`skcomplex.systems.biology.epidemiology` for `biology/epidemiology`. The page of each system under
[Systems](../../systems/index.md) shows its import.

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

A system lives in the field whose science poses it, and its topics carry its reach into other fields. Three rules keep
the catalog apart from the core, and the checks of the package layout enforce them.

- The core never imports the systems.
- Every system is built only from the public names of the core.
- No subfield package imports another. Code that several subfields of one field share lives in the `common` package of
  that field, and no field uses the `common` package of another.

### Optional extras

This section says which systems need an optional install. Most systems run on the default install. A few need a
library from an optional extra. Without the extra they stay in the catalog, and preparing or simulating one raises
`MissingExtraError` with the name of the system, the extra, and the install command.

| Extra | Library | Systems | Install notes |
| --- | --- | --- | --- |
| `structures` | Biopython | `fuzzy_oil_drop`, `early_stage_folding_intermediate` | Biopython 1.88 publishes no wheel for macOS on Intel, so the extra builds it from source there with a C compiler. |
| `rna` | ViennaRNA | `rna_neutral_networks` | The licence of ViennaRNA allows research and commercial use, forbids redistribution for a fee, and is not OSI approved, so the default install never brings it. |
| `trajectories` | MDTraj | none, it reads molecular dynamics trajectories for the systems biology systems | MDTraj is under the LGPL. |

### Systems as patterns

This section shows how a ready-made system serves as the pattern of a system of your own. The ready-made systems use
the core interface and no other, so the page of every field under [Systems](../../systems/index.md) shows for each
system what a builder needs to copy:

- its class, with its parameters and tags,
- its element and interaction types, with their states, parameters, and laws as written,
- for differential equations and maps, its equations as `symbolic(system).equations` returns them.

A system of your own follows the same steps. Declare the element and interaction types, and write a class that builds
the system from its parameters, as the page [System classes](system_classes.md) shows. The tutorial
[Check a system](check.md#what-an-interaction-needs) shows how to find which parts fit together, and the
[contributing guide](../../development/contributing.md#adding-a-system-to-the-catalog) covers adding a system to the
catalog.

### Inventory

This section explains the record of every system the library has surveyed. The inventory,
`docs/overview/inventory.toml`, lists the systems found in the reference books and articles, with their sources and
their status: implemented, pending a planned change, or blocked by a missing feature of the core. A system that the
core cannot express is recorded as blocked on the OpenSpec change that specifies the missing general feature. It is
never worked around inside the system package.

An entry writes the components and the bindings of its system by hand. The time, the space, the laws, and the domains
of an implemented system come from `get_tags()` of its class, so they cannot drift from the code. A check that the
documentation build runs compares the inventory with the catalog and the changes.

The systems exist first to exercise the core, and the ready-made systems are a by-product. Every entry lists the core
primitives it exercises, such as `rates`, `jumps`, `groups`, `lattices`, or `agents`, from a vocabulary that the
inventory defines. A primitive that the core lacks names the change that adds it. The priority of an entry is `core`
when it exercises a primitive, or a combination of primitives, that no earlier implemented entry exercises, or when it
needs a missing one. It is `example` when it repeats a covered pattern. So new systems go first where the core is least
exercised.
