# User guide

This guide follows what you do as a modeler, in order. Start with a ready-made system, change it, then build your own,
simulate it, observe it, analyse it, run experiments over many runs, and extend the library with parts of your own.
Each part builds only on the ones before it.

```python
from skcomplex.systems import create_simulation, describe

describe('sir').formalism
create_simulation('sir').prepare().run().trajectory_.values('I.n').max().round()
```

There are two ways in.

- **Select a ready-made system.** `catalog` lists the systems, `describe` tells you what one is, `create_system`
  returns it, and `create_simulation` returns a simulation of it with its tuned settings. You import these four
  functions from `skcomplex.systems` and nothing else.
- **Build a custom system.** You write elements and interactions with `skcomplex.core.components`, use specs from
  `skcomplex.core.specs`, simulate with `skcomplex.core.engine`, and study the results with `skcomplex.core.analysis`.

Every ready-made system is written with the same classes as a custom one, so the page of a system under
[Systems](../../systems/index.md) also works as a pattern. The section
[Choosing compatible components](build_a_custom_system/choosing_compatible_components.md) shows how to find which parts of a custom system fit
together.

Read the parts in order.

1. [Select a ready-made system](select_a_ready_made_system.md)
2. [Change it](change_it.md)
3. [Build a custom system](build_a_custom_system/index.md)
4. [Simulate](simulate.md)
5. [Observe](observe.md)
6. [Analyse](analyse.md)
7. [Run experiments](run_experiments.md)
8. [Extend the library](extend_the_library.md)
