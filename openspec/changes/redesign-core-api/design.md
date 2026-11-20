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
- `on='edges'` or `on='hyperedges'` puts one member on every link or hyperedge of a network space instead of every
  site, such as `Element('links', values={'sign': 1}, space=network, on='edges')` for the signed links of triadic
  percolation. Interactions read them through the `along=` layer binding.
- A group has no capacity. When rules or topology add members, the engine enlarges the arrays of the group as a Python
  list grows, so a network may start empty and grow without a setting.
- How a field is discretized belongs to its space, which already owns its cells, volumes, and operators. A space takes
  the scheme, the boundary of each state, and the symmetry, such as
  `LatticeSpace(64, discretization='spectral', boundary={'T': ('fixed', 0.0)})`, and every element on it is read
  that way. `Element` keeps only label, values, size, space, and on.
- What quantity a value is, such as an energy, a mass, or a charge, is written with the same `Annotated` as its
  allowed values, such as `E: Annotated[float, Quantity('energy')] = 0.0`, and conservation monitors read it. A plain
  `Element(values=...)` has no quantity.
- A value that wraps around, such as an angle or a phase, says so in `Annotated` too, such as
  `theta: Annotated[float, Period(2 * pi)] = 0.0`. It wraps after every step, and the analyses read a periodic value
  as a phase, or as a heading when the element moves in a space. No role names are written.
- The generic ready-made bases that build one element from a mapping of states, such as `DynamicalSystem` and
  `StockFlowSystem`, take the allowed values and quantities of those states as their own arguments, such as
  `allowed={'n': RealSet.closed_open(0, oo)}`, and keep them on the element they build. `Element` keeps its arguments.
- A value that holds the index of a member of another group, such as the employer of a worker among the firms, says
  so in `Annotated`: `employer: Annotated[int, Member('firms')] = -1`. The value -1 means none, and removing a member
  of that group clears the links to it.
- A network space names its one axis `node`, so an agent on a network starts at the node its value `node` gives and
  moves by changing it, as `x` and `y` place agents on other spaces.
- Values keep the order in which they are given, so states and parameters come out in that order and results do not
  depend on alphabetical order.
- A value named after an axis of the element's space is a position, with no flag. In
  `Element('walker', values={'x': 0.0}, space=ContinuousSpace(...))` the value `x` is the position of the walker, which
  walks, particle analyses, and drawings read. `role` and the other flags of `StateSpec` go.
- A value of a group whose members are added or removed is a state even when no law changes it, since it grows and
  shrinks with the group. The degree of each node of a growing network is recorded at every step for this reason.
- A value given as an `int` is an integer, also when it comes through `values=`, and a complex value is a complex state
  that a law may target like a real one, such as the `z` of a Julia set.
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
| `RateLaw`, `GeneratorLaw` | `RateLaw`, and `GeneratorLaw` with `HilbertSpec` and a `QuantumCompiler` in the quantum package |
| `NoiseLaw` | `NoiseLaw` |
| `JumpLaw`, `TransferLaw`, `Source`, `Sink` | `JumpLaw` with a change mapping, where an empty side is a source or a sink |
| `UpdateLaw`, `ArrayUpdateLaw`, `RuleLaw`, `LocalRuleLaw`, `TransitionLaw` | `UpdateLaw`, with a random draw for transitions, a neighbourhood binding for local rules, and a NumPy function of whole arrays, `UpdateLaw(state, f, arrays=True)`, for numerics that need them, such as SU(N) links or sequence sampling |
| `EquationLaw`, `BoundaryLaw` | `ConstraintLaw`, a boundary being a constraint applied after each step |
| `CrossingLaw` | `EventLaw` |
| `LinearProgramLaw` | `ObjectiveLaw` with `ConstraintLaw`, solved as a linear program when both are linear |
| `SampleLaw` | `ObjectiveLaw` with a sampling solver |
| `FluxLaw` | `RateLaw(u, -divergence(flux))` on a finite-volume field, whose `FiniteVolumeSpec` holds the limiter and the Riemann solver. The scheme stays in the core |
| `DissipatorLaw` | the quantum package, with its solver |
| `CascadeLaw`, `ExtremalLaw` | `UpdateLaw(state, relax(rules, stable, drive=...))` with the core operator `relax`, and Bak-Sneppen as an `UpdateLaw` on the site where `argmin` falls |
| `ProductionLaw`, `SubstitutionLaw` | the package of rewriting systems, with its solvers |
| `BilinearLaw` | `RateLaw(target, bilinear(rate, outcomes, first, second))` with the core operator `bilinear` |
| `MatchLaw` | `pairs` of `laws.utils` inside an agent rule |
| `GradientStepLaw` | an `UpdateLaw` recipe of the training interaction |

Two kinds of model need forms that the table above does not name yet.

- Agents. An agent is a member of a group element, and its behaviour is an interaction on that group whose `_prepare`
  sets an `UpdateLaw` with a rule, a Python function that reads and changes one agent, such as
  `UpdateLaw(s.target, move)` with `def move(agent, system): agent.x += system.rng.normal()`. Births and deaths are
  rules that add or remove members of the group, as a branching process does. The agent solver runs the rules, so
  `AgentView` and `form='agent'` go, and the activation schedule is a setting of that solver. `BirthDeathInteraction`
  stays as a shared recipe of `skcomplex.systems.general.agents`, an interaction that sets these `UpdateLaw` rules.
- Updates of whole arrays. A law that needs the numerics of a whole array, such as the SU(N) links of a gauge field
  or the rows of a sequence sampler, says so with `UpdateLaw(state, f, arrays=True)`, and its function receives the
  array of the state, shaped like the lattice or with one row per member. Without `arrays`, a function value is
  `f(t, values)` of one state. The form is never read from the signature of the function.
- Rules of a whole system. A rule whose target is the namespace of the system itself, `UpdateLaw(s, rule)` with
  `def rule(system): ...`, runs once per step after the rules of the agents, such as the matching of riders and
  drivers or a braking event drawn for the whole road.
- Positions of agents. The values of an agent named after the axes of its space are its location, which the agent
  solver places it at and moves it to, on a continuous space and on the cells of a lattice alike.
- Local rules. A function of a cell and its neighbours is an `UpdateLaw` with a neighbourhood binding, such as
  `UpdateLaw(s.target.alive, life, neighbors=MooreNeighborhood())`, lowered to the same lookup tables and fused lattice
  kernels as `LocalRuleLaw`, which then goes.
- Fractional orders. A derivative of order `alpha` between 0 and 1 is the core operator `fractional(x, alpha)`, used as
  the target of a `RateLaw`, such as `RateLaw(fractional(s.target.x, 0.5), -s.k * s.target.x)` for anomalous
  relaxation. Sage has no such derivative, so the operator carries the order and the Caputo solver reads it.

- Flows between stocks. A flow is one labelled `RateLaw` whose target maps each stock to the sign of its change, as a
  `JumpLaw` change map does: `RateLaw({s.alive: -1, s.dead: +1}, f, label='deaths')`. The amount `f` stays as written,
  a flow from outside the system has only its sink, `RateLaw({s.water: +1}, f, label='tap')`, and a flow out of it only
  its source. The loops analysis and the flow matrix read these maps, and the trajectory records each flow by its label.
  No law kind is added and no sign is guessed.
- Controls. An `ObjectiveLaw` names the values it chooses over time, `ObjectiveLaw(cost, controls=[s.plant.u])`, and
  their bounds come from the allowed values in `Annotated` on the element, as optimal control needs. A cost paid at
  the end of the run goes in `terminal=`, `ObjectiveLaw(running, controls=[s.u], terminal=final)`, the Bolza form.

- Boundaries and corrections after a step. A value corrected after each continuous step, such as a wall reflecting
  `x` to `abs(x)` or a bound clipping a stock, is an `UpdateLaw(x, abs(x))` beside the rate laws, applied once per step.
  No law kind is added for it.
- Roles that bind several elements, such as the reactants of a reaction, are typed `tuple[Role, ...]`, and the states
  an interaction acts on are a plain `tuple[str, ...]` argument. Both follow from the type hint, so `RoleSpec` goes.

- Random groups. An interaction bound to random groups of a group each step is written
  `Interaction(label, among='citizens', along=RandomPartition(3))`, as `along='hyperedges'` binds hyperedges, and its
  laws read `s.member` and `s.members`. `GroupingInteraction` goes.
- Starts in a region. A continuous space may hold named regions, `ContinuousSpace(..., regions={'tract_1': polygon})`,
  and an element value naming a region, `Element('residents', values={'region': 'tract_1'}, size=100)`, starts each
  member at a point the run draws inside that region from its own generator.

- Trained parameters. A training interaction names the parameters it trains by path,
  `TrainingInteraction(..., trained=['recurrence.A'])`, and the learning compiler makes exactly those paths
  adaptive when it compiles the inner system. No marker goes on the value.

- Newborn agents. A member that a rule creates takes, for every value the rule does not give, the default of that
  argument of its element class. A plain `Element(values=...)` has no class default, so its rule gives every value its
  group holds per member, or creation raises an error naming the missing ones.
- A field or group bound to rules keeps as states only the values the rules change, found by running the rule once as
  for agents. Its other values stay parameters, as for every other element.

- Sampling. A sampled system is an `ObjectiveLaw` run by a sampling solver such as `MetropolisSolver`, which says only
  how states move. The temperature is a value of the system that the objective reads, such as
  `ObjectiveLaw(energy / s.target.T)`, so sweeps and annealing schedules change it like any other value.

- Public type names. The names users annotate with that are not element specs, such as `Role`, live in one public module
  of the core, `from skcomplex.core.types import Role`. The element specs `Quantity`, `Period`, `Member`, `Tensor`, and
  `SubshiftSpec` stay with the elements.
- Linear programs keep their duals. `ObjectiveLaw(..., duals=True)` records the dual value of each constraint, the
  shadow prices of flux balance, and a variable of a program that no constraint bounds is a `prepare` issue.

- Recipes with an agent walk. A recipe such as `ChoiceInteraction` takes no `form` argument. Without a `schedule` it sets
  symbolic laws, and with one, such as `'synchronous'` or `'random-sequential'`, it sets one rule of the whole system,
  `UpdateLaw(s, rule)`, that walks the agents in that order with its own draws.

A field law first becomes a recipe of the seven kinds and the core reductions where it can, such as Bak-Sneppen as an
`UpdateLaw` on the site where `argmin` falls, so it needs no engine code. A law that truly needs numerics of its own,
such as the Lindblad equation, finite volumes, Cayley graphs, or rewriting, moves with that support: the field package
brings a solver built on the public engine bases. A space answers what the engine asks through methods of `BaseSpace`,
such as its cell volumes and the operators it supports. `prepare` chooses the compiler by itself, so `compiler` is no longer a setting users give. The engine core keys only on the seven
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

- Each nested `System` keeps the time its own laws imply. A discrete child inside a continuous parent steps once per
  unit of its time, as co-simulation does, so a controller that updates every second can drive a continuous plant, and
  a training loop in epochs can hold a network that runs in steps. No time argument comes back.
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

A ready-made system takes `solver=` and `backend=` like a `System` and passes them to `system_` when it prepares, such as
`LorenzSystem(backend=NumbaBackend()).prepare().simulate(until=50)`. They are settings of the object, as scikit-learn
keeps settings on the estimator, so `simulate` takes no solver.

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
`BaseTime`, `ContinuousTime`, `DiscreteTime`, and `StaticTime` leave the public API. The laws of single fields go as the
table of section 4 says: the quantum generator and dissipator with `HilbertSpec` and their compiler to the quantum
package, `ProductionLaw` and `SubstitutionLaw` with their compiler to `skcomplex.systems.general.rewriting`, `CascadeLaw`
to the core operator `relax`, `BilinearLaw` to the core operator `bilinear`, `ExtremalLaw` to an `UpdateLaw` recipe,
`MatchLaw` to `pairs`, and `FluxLaw` to a `RateLaw` of a divergence on a finite-volume space.

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

`BaseSpec`, the base of the public value objects, moves to `skcomplex.core.base` beside the other bases.

The old model that the engine still works on, `CompositeSystem`, `GroupElement`, `FieldElement`, `DeclaredElement`,
`StateSpec` and its family, and the lowered laws such as `BoundaryLaw`, `SampleLaw`, `RuleLaw`, and `LocalRuleLaw`,
becomes private records of the compiler. No public name of them is left, and the compiler keeps lowering the new laws
onto them.

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

### Per-entry parameters

A value typed `tuple[Role, ...]` binds one role per entry, but the engine does not check that a parameter read
beside it has one entry per role. Catalog systems pair such roles with sequences of other lengths, and a list of
floats looks the same as a vector along the axes, so a mismatch fails where the law reads the entry.

### Primed values of agents

An agent rule reads and writes a plain value as an attribute, such as `agent.x`, and a primed value through
`agent.get_value("x'")` and `agent.set_value("x'", v)`, which every agent view gives. Engine names such as `x_d1`
never reach a rule.

### Derivatives in member reductions

An observable that reduces over the members of a group may read their derivatives, such as the mean of `x.diff(t)`
over the agents, as an observable of the whole system does. The engine reads them in every reduction path.

### Flattening once

The compiler flattens a system once. Flattening is structural, and the declarations that the probe infers, the
orders, lags, and agents, apply to the flat tree afterwards.

### Pair energies

A pair interaction with `within=` may give its energy as `ObjectiveLaw(V(r))` in place of its forces. The engine
derives the forces by differentiating the energy, and the LAMMPS solver reads the energy for its Lepton pair style.
This replaces the pair energy of `PairInteraction`, which phase 6.1 removes.

The engine tells a pair energy apart from an objective by where the law comes from, not by a flag on the law. An
`ObjectiveLaw` set by an interaction with `within=` is the energy `V` of each pair it binds, and every other
`ObjectiveLaw` is an objective to minimise, sample, or steer, as the law table says. When the interaction states its
laws for a pair, the engine replaces its energy by the rate laws `RateLaw(s.first.x.diff(s.t, 2), -dV/dx_first / m)`
and `RateLaw(s.second.x.diff(s.t, 2), -dV/dx_second / m)`, so the problem holds forces only and keeps the time the rates
give. A pair energy names no targets and no controls, since it acts on the positions of both members. The force is
divided by the mass `m` of each member when its element has one, as LAMMPS does, and is used as it is otherwise. LAMMPS evaluates the same energy with the first member at the distance `r` from the second and prints
it as a Lepton expression of `r`. With `among`, both members of a pair come from that group.

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
