# Design

## Context

The items come from the issue log of the agents that applied `add-systems-biology-models` and
`add-statistical-physics-models`. Each one was reproduced on the current code with `.venv/bin/python`, after
`fix-core-defects` was applied:
- `series(s.t, times, values, 'previous')` with 150 points runs, with 200 points raises `SyntaxError: too many nested
  parentheses`, and with 1000 points raises `RecursionError` in `bind` of `_printing.py`. `series` nests one `where`
  per point, so the expression is as deep as the data is long.
- `Rate(s.cells.x, s.cells.x / count(over=s.cells))` in an interaction raises `ValueError: power::eval(): division by
  zero` from `_outside_reductions`, which replaces every reduction with 0 to find the symbols outside reductions. The
  same law in an element compiles, since `_settle_rows` returns before that call for one group without roles.
- `Element.define('C', ..., visual=VisualSpec('heights')).lattice('c', LatticeSpace((3,)))` has the visual `image`.
  `Element.lattice` replaces a missing visual argument with an image and never reads the type's visual.
- The default blueprint of `ideal_gas(n_dim=1)` holds the time series view `gas/x` with no time axis property, while
  `_series_view` gives scalar, observable, reporter, and parameter views the whole-run axis. `_element_views` passes the
  time series views of points visuals through unchanged.
- A group observable whose member expression is `m.x * total(m.x)` raises "The function total has no NumPy
  equivalent", and with `weights=W` raises a named error about the symbol `_weights_<hash>`. The observable compiler
  prints member expressions directly, without the reduction lowering that laws go through.
- `zero_range_process(n)` compiles in 0.2 s for 100 links, 1.6 s for 400, and 8.5 s for 800 on NumPy, and a run of 200
  links until 5 takes 26 s on NumPy and 78 s on Numba, most of it compiling 200 kernels. A profile at 800 links shows
  `_Declarations._declare` summing the sizes of every earlier entry for each new variable, 2.9 million generator steps,
  and `breakpoints_of`, `_prune_foreign`, and `_with_fields` scanning every block for every spec. A `Jump` on a group
  rejects `shift_member(s.n, 1)` as a target, so each link must be its own interaction.
- A plain role bound to `'sites[3]'` raises "is not an element or group path", while a variadic role bound to
  `['sites[3]']` reads member 3.
- `Simulation(create('nanoparticle_corona')).run(until=10800)` with the default solver is still running after 60 s,
  while `run('nanoparticle_corona')` with its registered `ScipyODE(method='LSODA')` takes 0.18 s. `oregonator` takes
  3.7 s against 0.5 s. With `LSODA`, `lorenz` until 100 takes 1.2 s against 1.7 s with `RK45`, and `lotka_volterra`
  0.2 s against 0.3 s.
- The real-time benchmarks were run while other agents loaded the machine, so their timings are not evidence either
  way.

## Goals / Non-Goals

**Goals:**

- Fix each confirmed defect at its source, with a regression test that fails before the fix.
- Make conserved hopping on rings and lines one group law, using the offset reads of `extend-interaction-roles`.
- Decide how a system runs well with the default solver when it is stiff.

**Non-Goals:**

- Offsets along a tree or a network. `dissipative_zero_range` keeps one interaction per link, and benefits only from
  the linear compile time. Jumps along graph edges belong with `add-link-fields`.
- Changing the automatic backend choice. It counts group members, so a ring of 1000 sites written with offset jumps
  picks Numba. Counting one-link interactions instead would pick Numba for per-link models, whose cost is compiling one
  kernel per link, which is slower on Numba than on NumPy.
- Items of the first round, which `fix-core-defects` covers, such as the serial kernels of event loops and the
  persistent SciPy integrator.

## Decisions

### Series as a balanced tree of conditions

`series` builds its conditions as a balanced binary search over the times: the condition at the middle time splits the
points, and each half recurses. The depth is `ceil(log2 n)`, 10 for 1000 points. The branches, the values, and the
breakpoints are those of the nested form, so `breakpoints_of` finds every data time as before.

An opaque interpolation function lowered to `np.interp` was rejected. Every backend would need its own lowering, Lepton
has none, and the breakpoints and derivatives that conditions give for free would need code of their own.

### Symbols outside reductions

`_outside_reductions` walks the expression tree and does not descend into reduction calls, instead of substituting
them. Substituting 1 was rejected, since `1 / (count - 1)` fails the same way, and any constant can be a pole.

### Reductions in observables

The observable compiler sends member expressions through the reduction lowering of laws, with the observable's group
as the reading group, so `total`, `mean`, `count`, and their weights have the same meaning in an observable and in a
law. A reduction over another group in a member expression raises the named error that laws raise.

### Members bound to plain roles

Role resolution accepts a member reference `group[i]` for a plain role through the same path that resolves the entries
of variadic roles, and binds the role to one single-element view of that member. The member keeps its states in the
group's columns, so trajectories and parameters keep their paths.

### Jumps at member offsets

A `Jump` of a group law may name `shift_member(s.n, k)` as a target, with the same `boundary` choices as reads. The
compiler gives the jump one channel per member and fills the existing target table of the Gillespie loops: the row of
member `i` holds the column of `i` for `s.n` and the column of the shifted member for the offset target. A fixed end
writes the empty target `-1`, so a particle hopping past the last site leaves the system, and a reflecting end maps the
target back into the group, as reads do. The Numba loop needs no change, since it already skips `-1`. The NumPy loop
and the stoichiometry matrix of the problem skip `-1` targets too, since they scattered every target before.

`zero_range_process` and `asep` become one group of sites with one hop law each, and the reservoirs of `asep` stay
interactions on its end members. The registered default backend stays `numba`. The hop laws are the group
interactions `ZeroRangeHops`, bound to the sites of a ring, and `ExclusionHops`, bound to the sites of a line, which
replaces the per-link `ExclusionHop` that nothing else uses. `ZeroRangeHop` stays for the tree of
`dissipative_zero_range`. Each group interaction holds a forward and a backward hop law, so a model compiles one jump
kernel with one channel set per law, two channels per site.

The survey for this decision found no maintained Python library for lattice exclusion and zero-range processes that the
core could wrap. The existing Gillespie table already holds the general case, so the change is an adapter of offsets
to that table.

### Linear compile time

`_Declarations` keeps the running offset of its states and of its parameters, so declaring a variable costs the same
whatever the number of earlier ones. The passes that scan every block for every kernel spec, `_prune_foreign` and
`_with_fields`, read an index of blocks by path built once per compilation. `breakpoints_of` visits each distinct
subexpression once.

### Default ODE solver detects stiffness

The default ODE stepper is `ScipyODE()`, whose method `'auto'` is `LSODA` for every system whose dense matrix fits,
as below. ODEPACK's `LSODA`, which SciPy wraps, starts with Adams methods
and switches to BDF when it detects stiffness, so a stiff system with the default solver no longer stalls. Its dense
output, events, and the persistent integrator of `fix-core-defects` work through the same `OdeSolver` interface as
`RK45`. `LSODA` takes real states only. `add-complex-states` compiles every complex state into its real and imaginary
parts, so the state vector a stepper receives is always real and systems with complex states also run with `LSODA`.

Two alternatives were weighed:
- **Systems carry the run settings of their factory.** `create` would attach the registered `solver` and `dt` to the
  system, and `Simulation` would read them when its own arguments are left at their defaults. It fixes only systems
  built through the catalog, not a stiff system a user writes. It makes a run depend on where its system came from,
  which the constitution forbids as hidden state, and settings tuned for the default size of a model, such as an
  `RK4` step, would follow a system rebuilt at another size. The catalog's `run` keeps applying the registered
  settings, and `describe(name).defaults` gives them to users who build a `Simulation` themselves.
- **Stiffness detection of our own**, such as a stiffness ratio from the Jacobian at the start, was rejected. It would
  reimplement what `LSODA` does continuously during the run, and a system that becomes stiff later would still stall.

Registered models that set `ScipyODE(method='LSODA')` with their own tolerances keep them.

Two defects of SciPy's `LSODA` showed up when it became the default, and the stepper guards against both:
- With an infinite `t_bound`, its first step returns a NaN state. The stepper bounds every integrator by the largest
  finite time instead of infinity, which keeps the persistent integrator of `fix-core-defects`.
- On a solution that blows up in finite time it keeps returning steps that do not advance time, where `RK45` reports
  a failure. The stepper raises the named solver error when a step makes no progress or the state is not finite.

Running the whole catalog and the documentation on the new default found two more:
- When the drift vanishes at the start, as in `catalyst_pellet` at `c = 0` or a forced oscillator at rest, LSODA
  guesses its first step from the distance to the end and fails before its first step. The stepper then selects the
  first step from the drift's change over a trial step, as SciPy's other solvers do, and keeps LSODA's own guess
  otherwise, since a trial step across a discontinuous drift, such as the stick-slip friction of `burridge_knopoff`,
  gives needlessly small steps at every restart.
- LSODA stores a dense matrix of the size of the state squared, so the 40 000 states of a gas of 10 000 particles
  ask for 12 GB. The default method is therefore `'auto'`: `LSODA` while its matrix fits, for at most 4096 states or a
  Jacobian with a narrow band. Larger systems get `BDF` with the sparsity pattern of the Jacobian when fields limit
  the stable step by diffusion, the stiffness signal the stepper already warns on, and `RK45` otherwise, since `BDF`
  runs a gas of 2000 particles 9 times slower than `RK45`. `ScipyODE(method='LSODA')` asked for more states raises the named solver error instead of exhausting
  memory. This is not stiffness detection of our own: it chooses by storage, and LSODA keeps detecting stiffness.

## Risks / Trade-offs

- [Trajectories of default ODE runs change within tolerance] → Tests that compare against analytic solutions keep
  their tolerances. Tests that pin values from an `RK45` run move to tolerances, and the changelog lists the change as
  breaking.
- [`LSODA` does not take a sparse Jacobian pattern] → `adopt-scipy-sparse-jacobians` picks a stepper for large sparse
  systems itself, and its choice wins over the default.
- [Offset jumps follow the flat member order] → On a one-dimensional `LatticeSpace` it is the lattice order, which
  offset reads already rely on. On a lattice of more dimensions an offset does not follow lattice neighbours, and the
  docstring of `Jump` says so. Hops between lattice neighbours of more dimensions stay with lattice Monte Carlo.
- [The balanced series is longer to print than a lookup table] → It grows as `n log n` at worst, and a 1000-point
  series prints in well under a second.
