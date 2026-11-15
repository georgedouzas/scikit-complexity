# Design

## Context

See `proposal.md` for the motivation and the audit. This design was reached by first listing what twelve fields need,
mechanics, chemistry, epidemiology, ecology, circuits, heat and fluids, lattice models, neuroscience, quantum systems,
economics, machine learning, and formal languages, and then keeping only what all of them share.

Every one of them is states plus rules for how the states change, and each rule is one of a few mathematical kinds:

| Kind | Mathematics | Fields |
| --- | --- | --- |
| Rate | `dx/dt += f(x, t)` | mechanics, kinetics, ecology, heat, quantum, neurons |
| Noise | `dx += g(x) dW` | Langevin dynamics, finance, noisy neurons |
| Jump | at the rate `a(x)`, `x += ν` | reactions, epidemics, births and deaths |
| Update | `x_next = F(x)` | maps, cellular automata, agent rules, rewriting, training steps |
| Constraint | `0 = g(x)` | rods, Kirchhoff's laws, budgets, conservation |
| Event | when `h(x)` crosses zero, `x = R(x)` | collisions, spikes, thresholds |
| Objective | a scalar `U(x)` to minimise or to sample | equilibria, linear programs, Ising models, losses |

Two more facts hold in every field. Contributions add up: forces add into an acceleration, reactions into a rate,
currents into the balance of a node that must be zero. And some choices belong to the numerics, not to the model: the
same reaction network is solved as ODEs or by Gillespie's method, the same Ising energy is sampled by Metropolis or
Wolff, the same mechanics is integrated by Runge-Kutta or Verlet.

## Goals / Non-Goals

**Goals:**

- One concept per job and one way to do each thing, from the falling particle to a network epidemic.
- The scikit-learn mental model: parts with parameters, one estimator that is prepared once and simulated many times,
  functions for experiments.
- A core that holds only the components general systems of several fields are built from, with every concrete system
  in the systems packages as a consumer of the core.
- Less code: fewer classes, short constructors, no hidden conventions, no private hooks for users, and specs only where
  users write them.

**Non-Goals:**

- No new numerical method. Solvers, backends, and kernels keep their algorithms and change only where they are keyed.
- No change to the analysis functions, which read trajectories.
- No change to the package layout beyond moving field-specific components out of the core into the systems packages.
  Shortening import paths is a separate decision.
- No aliases for removed names. The release that carries this change is breaking.

## Decisions

### 1. The pipeline

```
System  ──prepare──▶  Problem  ──Solver──▶  Trajectory  ──▶  analyses
(what)                (maths)    (how)
                                   │
                                Backend (where the numbers are computed)
```

| Layer | Job | Chosen by |
| --- | --- | --- |
| System | Describes the model: elements, interactions, space, nested systems | the user |
| Problem | The mathematics the laws add up to: ODE, SDE, jump process, map, DAE, hybrid, optimisation, sampling | `prepare`, from the kinds of law present |
| Solver | The algorithm for one kind of problem: RK45, Verlet, Euler-Maruyama, Gillespie, IDA, Metropolis, an LP solver, Mesa, LAMMPS | a default per problem, or the system's `solver` setting |
| Backend | The array library and hardware: NumPy, Numba, JAX, and executors for many runs | a default, or the system's `backend` setting |

The parts never name a solver. The system holds the solver and the backend as settings, as `LogisticRegression` holds
`solver` beside `C`, so a sweep can compare solvers like any parameter.

| Laws present | Problem | Default solver | Alternatives |
| --- | --- | --- | --- |
| Rate | ODE | adaptive Runge-Kutta | Verlet, implicit for stiff problems |
| Rate and Noise | SDE | Euler-Maruyama | Milstein |
| Jump | jump process | Gillespie | tau-leaping, the ODE limit of many units with an ODE solver |
| Rate and Constraint | DAE | the ODE solver, with the algebraic states solved at every step | |
| Update | map | iteration | Mesa for agents |
| Event with any of the above | hybrid | the same solver with root finding | |
| Objective | optimisation | `MinimizeSolver` on `scipy.optimize.minimize` | an LP solver for linear objectives |
| Objective with `MetropolisSolver(temperature=T)` | sampling | Metropolis | Wolff, heat bath |
| Objective with `GradientFlowSolver(dt=h)` | gradient flow | gradient descent | |

### 2. Elements are data

An element holds its states, which change, and its own parameters, which describe that element alone, such as a mass,
a charge, or a capacity. It returns no laws.

```python
class Particle(Element):
    def __init__(self, label, x=0.0, m=1.0):
        super().__init__(label)
        self.x = x
        self.m = m
```

- An element is a plain Python class with plain arguments, stored unmodified. It declares no states and no
  parameters.
- States and parameters follow from the laws: a value that some law changes is a state, and a value that no law
  changes is a parameter, a state with zero rate. The compiler finds which is which.
- The kind of a value comes from its Python type hint or its value: `float` is real, `int` integer, `bool` boolean,
  `str` a category, and a list or an array a vector or a matrix of its shape.
- A constraint on the values is written only where it is needed, as the allowed values in `Annotated`, such as
  `n: Annotated[float, RealSet.closed_open(0, oo)] = 40.0` or `status: Annotated[str, {'S', 'I', 'R'}] = 'S'`.
- Derivatives, lags, and delays are written in Sage, not declared: an interaction receives every state as a function of
  the time `t`, and writes `x.diff(t, 2)`, `y(t=t - 1)`, and `x(t=t - tau)`. The library defines no functions of its own
  for them, and the normalisation of laws translates these Sage forms into the internal markers of the engine. The
  order of a state is the highest derivative or lag its laws use, and the compiler reduces every state to first order, as the
  orders pass already does. A velocity is the derivative `x'`, never a declared state, so no element and no interaction
  states the rule of motion. Derivatives are named with a prime and past values with an index, such as `"ball.x'"` and
  `"economy.y[-1]"`, in initial states and trajectories, and they default to zero. `order`, `lags`, `initial`, periods,
  and the other flags of `StateSpec` go.
- `size=N` makes N members of one element class, and `space=` places one member at every site of a space. Both replace
  `GroupElement` and `FieldElement`. Per-member values are arrays whose first dimension is the number of members.
- `Element(label, values={...})` covers elements without a class of their own, such as
  `Element('ball', values={'x': 10.0, 'm': 1.0})` or a model whose names come from a file, so `DeclaredElement` goes. A subclass adds named values as constructor arguments, and both kinds of value are
  inferred alike.
- `visual=` leaves every element. An element class gives its default drawing in its class attribute, such as
  `visual = VisualSpec('phase_ring', {'phase': 'theta'})`, and a group or field draws like its member class. The
  visualisation observer draws chosen elements otherwise by path, such as
  `RerunObserver(visuals={'oscillators': VisualSpec('sites', {'states': ['theta']})})`.

### 3. Interactions are behaviour

An interaction binds roles to elements, owns the parameters of the relation, such as `g`, `k`, or `beta`, and learns
its laws when it is prepared. Every component has the one public method `prepare(context)`, which calls the hook
`_prepare` that a subclass implements, as `fit` calls `_fit`. An interaction is prepared against the namespace of the
elements its roles bind and sets `laws_`, an element is prepared against its space and checks its values, and a system
nested in another is prepared like any component. The engine prepares a clone of an interaction for every binding, as
`cross_val_score` fits clones, so the user's object is unchanged and compiling in parallel is safe. Influences from
outside the system, such as gravity, are interactions too.

```python
class Gravity(Interaction):
    def __init__(self, label, *, target: Role, g=9.81):
        super().__init__(label)
        self.target = target
        self.g = g

    def _prepare(self, context):
        super()._prepare(context)
        x, t = context.target.x, context.t
        self.laws_ = [RateLaw(x.diff(t, 2), -context.g)]
```

- A process of one element is an interaction with one role: growth, decay, recovery, the membrane dynamics of a neuron.
  This is the only place a rule can live.
- Where an interaction applies is a binding argument of the base class, not a class:

  | Binding | Example |
  | --- | --- |
  | named roles | `Spring('spring', a='p1', b='p2', k=10.0)` |
  | every pair within a neighbourhood | `LennardJones('lj', among='gas', within=RadiusNeighborhood(2.5))` |
  | every edge or hyperedge of a network | `Infection('infect', among='people', along='contacts', beta=0.3)` |

  `among` names the group at the ends of the edges, and `along` is `True` for the edges of the network the group lives
  on or the name of a layer of a network with several, such as `'contacts'`. Laws read the two ends as `s.source` and
  `s.target`. A subclass lists `among` and `along` in its own `__init__` and passes them on. When the laws request
  changes of the network, such as `UpdateLaw(TopologyRequestSpec('rewire'), p)`, the members on the network and its
  edges become variable without any setting.

- A one-off rule needs no class: `Interaction('gravity', roles={'target': 'ball'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), -9.81)])`.
- The core keeps only `BaseInteraction` and `Interaction`. The engine names no other interaction, and laws written in
  Sage make the generated ones one line each: a gradient flow is `RateLaw(x, -U.diff(x))`, Hamilton's equations are
  `RateLaw(q, H.diff(p))` and `RateLaw(p, -H.diff(q))`. `HamiltonianInteraction` and `EulerLagrangeInteraction` move to
  `skcomplex.systems.physics.common.mechanics`, with the parts of the invariants analysis that read their energies, and
  `GradientInteraction` moves to `skcomplex.systems.general.gradients`, since physics, social, and engineering use it.
- The interactions that only wrap one law go, since `Interaction` with a law does the same: `RateInteraction`,
  `JumpInteraction`, `MapInteraction`, `CrossingInteraction`, `TransitionInteraction`, `LocalRuleInteraction`,
  `TransferInteraction`, and `SamplingInteraction`. So do the classes that only fix a binding: `NeighborInteraction`,
  `PairInteraction`, `HyperedgeInteraction`, `GroupingInteraction`, and `CouplingInteraction`.
- Spatial operators, such as `laplacian`, `divergence`, `gradient`, and reductions over neighbours, are functions laws
  call, not interactions. `ConservationInteraction` becomes a rate with `divergence`.
- A change of the network itself is an `UpdateLaw` on the edges of the space, so `TopologyInteraction` becomes an
  `Interaction` with such laws.
- Ports go. A circuit is node elements with potentials and component interactions whose constraints add into the
  balance of each node, so `ConnectionInteraction` and `ConnectorSpec` go.

### 4. Laws are seven kinds of value

`RateLaw`, `NoiseLaw`, `JumpLaw`, `UpdateLaw`, `ConstraintLaw`, `EventLaw`, and `ObjectiveLaw` are frozen values that interactions set in `laws_`.
A law is not a component: it has no label, no parameters of its own, and no `prepare` or `run`. It is the mathematics an
interaction states, so the components are elements, interactions, spaces, and systems. The kinds stay as values rather
than bare Sage equations, since several laws add into one derivative where two equations would read as a contradiction,
and Sage has no form for noise, jumps, events, or objectives. Contributions of several laws to one state add up, for
every kind where adding means something: rates, noises, jump channels, and constraint balances.

| Current law | New form |
| --- | --- |
| `RateLaw`, `GeneratorLaw` | `RateLaw`. The linear generator of quantum evolution is a `RateLaw` written with operators in the quantum package |
| `NoiseLaw` | `NoiseLaw` |
| `JumpLaw`, `TransferLaw`, `Source`, `Sink` | `JumpLaw` with a change mapping, where an empty side is a source or a sink |
| `UpdateLaw`, `ArrayUpdateLaw`, `RuleLaw`, `LocalRuleLaw`, `TransitionLaw` | `UpdateLaw`, with a random draw for transitions and a neighbourhood binding for local rules |
| `EquationLaw`, `BoundaryLaw` | `ConstraintLaw`, a boundary being a constraint applied after each step |
| `CrossingLaw` | `EventLaw` |
| `LinearProgramLaw` | `ObjectiveLaw` with `ConstraintLaw`, solved as a linear program when both are linear |
| `SampleLaw` | `ObjectiveLaw` with a sampling solver |
| `FluxLaw` | `RateLaw` with `divergence`, and the finite-volume scheme as a solver setting of the fluids package |
| `DissipatorLaw` | the quantum package, with its solver |
| `CascadeLaw`, `ExtremalLaw` | the criticality package, with its solvers |
| `ProductionLaw`, `SubstitutionLaw` | the package of rewriting systems, with its solvers |
| `BilinearLaw` | the package of kinetic models, with its solver |
| `MatchLaw` | `skcomplex.systems.general`, as an `UpdateLaw` |
| `GradientStepLaw` | the learning package, as an `UpdateLaw` |

A field law that needs engine support of its own moves with that support: the field package brings a solver built on
the public engine bases. `prepare` chooses the compiler by itself, so `compiler` is no longer a setting users give. The engine core keys only on the seven
kinds and never names a class of the systems packages.

### 5. The system is the estimator

```python
system = System(
    elements=[Particle('ball', x=10.0)],
    interactions=[Gravity('gravity', target='ball')],
    solver='auto',
    backend='auto',
).prepare()

system.states_        # ("ball.x", "ball.x'")
system.parameters_    # ('ball.m', 'gravity.g')
system.equations_     # {"ball.x": "ball.x'", "ball.x'": "-g"}
system.form_          # 'ordinary differential equations'
system.solver_        # ScipyODESolver(...)

earth = system.simulate(until=1.4)
pushed = system.simulate(until=2.0, initial_state={"ball.x'": 3.0})
moon = system.set_params(gravity__g=1.62).prepare().simulate(until=1.4)
```

| Learned attribute | What it is |
| --- | --- |
| `states_` | the names of the values the laws change, with their derivatives and lags |
| `parameters_` | the names of the values no law changes |
| `equations_` | one readable equation per state, summed over the laws and reduced to first order |
| `form_` | the kind of problem: ODE, SDE, jump process, map, DAE, static, and the others |
| `solver_`, `backend_` | the resolved runners |

The compiled problem, the vector function `f(t, y, p)` and its layout that the solver calls thousands of times, is
private. Compiling it is the expensive step, so `prepare` does it once and every `simulate` reuses it.

| scikit-learn | scikit-complexity |
| --- | --- |
| `LogisticRegression(C=1.0, solver='lbfgs')` | `System(elements=..., interactions=..., solver=..., backend=...)` |
| `fit(X, y)` returns `self` | `prepare()` checks every component, compiles the problem, picks the solver and backend, sets the learned attributes, and returns `self` |
| `predict(X)` returns the predictions | `simulate(until, initial_state=None, random_state=None, observables=(), observers=())` returns the `Trajectory` |
| nested parameters of a `Pipeline` | `get_params` and `set_params` with `<label>__<parameter>`, and `<system>__<label>__<parameter>` for nested systems |
| `cross_val_score`, `GridSearchCV` | `ensemble`, `sweep`, `calibrate`, `sensitivity`, `basin_stability`, as functions |
| `fit` calling `_fit` | every component's `prepare` calling its `_prepare`, which sets `laws_` on an interaction and `system_` on a ready-made system |

- `CompositeSystem` becomes `System`. A system holds a label, elements, interactions, nested systems, a space, a
  solver, and a backend, and nothing else. Time is inferred from the law kinds: rates flow in continuous time, updates
  step in discrete time, and objectives and constraints alone are static. An activation schedule of agents is a setting
  of the agent solver.
- Parameter names use `__`, as scikit-learn requires. Paths of values in trajectories and initial states use `.`, such
  as `'ball.x'`.
- `simulate` on a system that was not prepared, or whose parameters changed since it was prepared, raises
  `NotPreparedError`. Preparing again compiles again. The experiments keep their own cache of compiled problems by
  structure within one call.
- `prepare` returns `self`, and `simulate` returns the trajectory, so one prepared system runs many times without
  overwriting a stored result.
- The free function `simulate` is removed, so there is one way to simulate. Its other run arguments, `invariants`,
  `schedule`, `clock`, `checked`, `record_draws`, and `replay`, become arguments of `System.simulate`.
- Elements, interactions, spaces, and neighbourhoods derive from `BaseComponent`, which gives parameters, cloning,
  tags, and `prepare`, and no `run`. Solvers, backends, observers, compilers, and kernels keep `prepare` and `run`,
  which the system calls. Every family has `prepare` and its own verb, if any, as scikit-learn estimators share `fit`
  and differ in `predict` or `transform`.

### 6. Ready-made systems

A ready-made system is a class deriving from `BaseSystem` with catalog metadata, `field`, `topics`, `references`, and
tuned run settings, and a constructor that stores its model parameters. Its hook `_prepare` sets `system_`, the
`System` of its components, as `fit` calls `_fit`, and the public `prepare` compiles it. A ready-made class earns its place when it reparametrizes, such as
`ProjectileSystem(speed, angle)`, which turns an angle into velocity components. A user who only wants to tune the
parameters of parts builds a `System` and uses nested parameters, with no class.

### 7. What the core holds

The core holds only the components that general systems of several fields are built from:

- elements: `BaseElement`, `Element`,
- interactions: `BaseInteraction`, `Interaction`,
- laws: `RateLaw`, `NoiseLaw`, `JumpLaw`, `UpdateLaw`, `ConstraintLaw`, `EventLaw`, `ObjectiveLaw`,
- operators: `laplacian`, `divergence`, `gradient`, and the reductions over neighbours,
- systems: `BaseSystem`, `System`,
- spaces: `BaseSpace`, `LatticeSpace`, `NetworkSpace`, `ContinuousSpace`, `PhaseSpace`, and `CellComplexSpace`, into which
  `MeshSpace` merges since a mesh is a 2-complex,
- neighbourhoods: `BaseNeighborhood`, `LatticeNeighborhood`, `AdjacencyNeighborhood`, `RadiusNeighborhood`, into which
  `BallNeighborhood` merges as a distance in the metric of the space, `NearestNeighborhood`, `DelaunayNeighborhood`,
  and `CofaceNeighborhood`.

  A mesh is `CellComplexSpace(points=p, cells=t)`, beside `CellComplexSpace(LatticeSpace(...))` and
  `CellComplexSpace(NetworkSpace(...))`, with exactly one source given. `RadiusNeighborhood(radius, metric=...)` takes
  the offsets within the radius under the metric on a lattice, such as `metric='chebyshev'` for the Moore neighbourhood
  and `metric='manhattan'` for von Neumann, and the members within the distance in a continuous space. Its dimension
  comes from the space, and `center` is a boolean.

`SimplexSpace` moves to `skcomplex.systems.social.common`, and `CayleySpace`, `MemoryNeighborhood`, and
`StateTranslationNeighborhood` move to `skcomplex.systems.computer_science.common.automata` with their engine support.
`BaseTime`, `ContinuousTime`, `DiscreteTime`, and `StaticTime` leave the public API. The laws of single fields move as the
table of section 4 says: `DissipatorLaw` to the quantum package, `CascadeLaw` to criticality, `ExtremalLaw` to the
evolution package of biology, `BilinearLaw` to `skcomplex.systems.general.kinetics`, `MatchLaw` to
`skcomplex.systems.general.matching`, and `ProductionLaw` and `SubstitutionLaw` to `skcomplex.systems.general.rewriting`.
The core keeps 25 public classes in these families, down from 79.

Everything else that is field-specific moves to `skcomplex.systems.<field>.common.<family>` when one field uses it, or
to `skcomplex.systems.general.<concept>` when several fields share it as a recipe. Moving code never deletes meaningful
specialised code: it keeps working in its new place and serves as an example there. The concrete systems of each field
are the consumers of the core and of their field's common package.

### 8. Specs

| Kind | Fate |
| --- | --- |
| `StateSpec`, `ParameterSpec`, `VariableSpec`, `DeclaredVariableSpec` | removed. Elements are plain classes, and the laws decide what is a state |
| `RoleSpec` | public in `skcomplex.core.components.interactions`, with the alias `Role`, since an interaction says which arguments name elements |
| `VisualSpec`, `ConnectorSpec` | `VisualSpec` moves to the visualisation observer, and `ConnectorSpec` goes with ports |
| Value objects users pass or read: `ObservableSpec`, `WienerProcessSpec`, `ScheduleSpec`, `Tags`, and a few more | public, beside the class that uses each, and kept only if a system or a document uses them. `ObservableSpec` becomes `Observable` |
| Field values, such as `HilbertSpec` and `PartnerSpec` | move to the field package that uses them |
| Engine records and compile specs, such as `LoweredFunctionsSpec`, `DrawLogSpec`, `EventRecordSpec`, and the eleven specs of `skcomplex.core.components.specs` | private frozen dataclasses beside the code that uses them |

Both specs packages, `skcomplex.core.specs` and `skcomplex.core.components.specs`, are removed. Every value lives
beside the class that uses it: `RoleSpec` and `Role` with the interactions, `Tags` with the components, `Observable`
with the runtime, the noise values with the noise law, schedules with the solvers that read them, and field values in
their fields. Engine records become private frozen dataclasses beside the code that uses them, and only they keep the
`Spec` suffix.

### 9. Docstrings

Every docstring of the core follows the scikit-learn layout: one summary line, one paragraph in plain words with the
formula when there is one and one sentence of uses from several fields, a line that links the section of the user
guide, the arguments, a See Also section of one line per neighbouring class, and an Examples section of three to six
lines that uses real parts and defines no class only to have something to work on. The docstrings are written last,
against the new API.

## Risks / Trade-offs

- [A generic `UpdateLaw` is slower than the specialised kernels of sandpiles or rewriting] → The field package that owns
  such a law brings its own solver on the public engine bases, so its kernel survives the move.
- [An agent's behaviour as an interaction reads less like Mesa, where behaviour is a method of the agent] → One place
  for every rule is worth it, and the agent guide shows the pattern once.
- [Inferring time from law kinds hides a choice] → `prepare` records the inferred problem in `form_`, the tags show
  the time, and a system mixing kinds that do not combine raises a named error listing them.
- [The rewrite touches every system, test, and page] → The new API is built beside the old one inside the core, the
  fields migrate in waves, and the old classes are deleted only when no system uses them.
- [Inferring states from laws needs the laws before the values can be laid out] → `prepare` already collects every
  law before it compiles, so the split of states and parameters is one more step of that pass.
- [Systems no longer reject `solver=` as they used to] → The solver and backend are the only execution settings a
  system takes. Observers, observables, and initial states stay arguments of `simulate`.

## Migration Plan

1. Build `System`, `prepare`, `simulate`, the new elements, interactions, and laws in the core, beside the old
   classes, lowering the seven kinds onto the existing compilers. The parts keep their engine-facing `prepare` and
   `run` until the old classes are deleted, and only then lose them.
2. Move the field-specific components out of the core into the systems packages, with the solvers they need.
3. Migrate the systems field by field, in waves of at most four parallel agents, each wave rewriting its systems,
   tests, and gallery examples on the new API.
4. Delete the old classes, the free function `simulate`, `CompositeSystem`, the times, the wrapper interactions, and
   the old specs, and update the convention check.
5. Rewrite the docstrings, the README, the user guide, and the scenarios of the main specs that call `simulate(`.

Rollback is by reverting the commits of a wave. The release that carries the change is breaking.
