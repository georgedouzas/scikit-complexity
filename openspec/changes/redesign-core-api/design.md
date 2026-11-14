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
| Jump | jump process | Gillespie | tau-leaping, the ODE limit of many units |
| Rate and Constraint | DAE | IDA | |
| Update | map | iteration | Mesa for agents |
| Event with any of the above | hybrid | the same solver with root finding | |
| Objective | optimisation | an LP or NLP solver | |
| Objective with a temperature | sampling | Metropolis | Wolff, heat bath |
| Objective with a gradient solver | gradient flow | gradient descent | |

### 2. Elements are data

An element holds its states, which change, and its own parameters, which describe that element alone, such as a mass,
a charge, or a capacity. It returns no laws.

```python
class Particle(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec(rate='v')] = 0.0,
        v: State[float] = 0.0,
        m: Parameter[float] = 1.0,
        size=None,
        space=None,
    ): ...
```

- Every state is declared. `StateSpec(rate='v')` says that `v` is the time derivative of `x`. It replaces `order`, the
  hidden companion states `x_d1`, and `derivative_of`. A particle with no interactions moves in a straight line.
- `State[T]`, `Parameter[T]`, and `Role` are the short spellings of `Annotated[T, StateSpec()]`,
  `Annotated[T, ParameterSpec()]`, and `Annotated[str, RoleSpec()]` for the plain case. Options are written with the
  spec.
- `StateSpec` keeps `domain`, `shape`, and `rate`. Delays become a function laws call on a state. Fractional orders,
  Lie groups, tensors, and agent quantities move to the fields that use them. A state is algebraic when a constraint
  fixes it, which the compiler infers.
- `size=N` makes N members of one element class, and `space=` places one member at every site of a space. Both replace
  `GroupElement` and `FieldElement`. Per-member values are arrays whose first dimension is the number of members.
- `DeclaredElement(label, states={...}, parameters={...})` covers elements known only as data, such as imported models.
- `visual=` leaves every element. How an element is drawn is a setting of the visualisation observer, keyed by path.

### 3. Interactions are behaviour

An interaction binds roles to elements, owns the parameters of the relation, such as `g`, `k`, or `beta`, and returns
laws from the hook `_get_laws(s)`, where `s` gives the bound states and parameters by name. The base defines the
public `get_laws(s)` once, which checks the namespace and the returned laws and calls the hook, as `BaseSearchCV.fit`
calls the `_run_search` that each search implements in scikit-learn.

```python
class Gravity(Interaction):
    def __init__(self, label, *, target: Role, g: Parameter[float] = 9.81): ...

    def _get_laws(self, s):
        return [Rate(s.target.v, -s.g)]
```

- A process of one element is an interaction with one role: growth, decay, recovery, the membrane dynamics of a neuron.
  This is the only place a rule can live.
- Where an interaction applies is a binding argument of the base class, not a class:

  | Binding | Example |
  | --- | --- |
  | named roles | `Spring('spring', a='p1', b='p2', k=10.0)` |
  | every pair within a neighbourhood | `LennardJones('lj', among='gas', within=RadiusNeighborhood(2.5))` |
  | every edge or hyperedge of a network | `Infection('infect', among='people', along='contacts', beta=0.3)` |

- A one-off rule needs no class: `Interaction('gravity', roles={'target': 'ball'}, laws=lambda s: [Rate(s.target.v, -9.81)])`.
- The core keeps the interactions the engine generates from one function: `GradientInteraction` from a potential,
  `HamiltonianInteraction` from a Hamiltonian, and `EulerLagrangeInteraction` from a Lagrangian.
- The interactions that only wrap one law go, since `Interaction` with a law does the same: `RateInteraction`,
  `JumpInteraction`, `MapInteraction`, `CrossingInteraction`, `TransitionInteraction`, `LocalRuleInteraction`,
  `TransferInteraction`, and `SamplingInteraction`. So do the classes that only fix a binding: `NeighborInteraction`,
  `PairInteraction`, `HyperedgeInteraction`, `GroupingInteraction`, and `CouplingInteraction`.
- Spatial operators, such as `laplacian`, `divergence`, `gradient`, and reductions over neighbours, are functions laws
  call, not interactions. `ConservationInteraction` becomes a rate with `divergence`.
- A change of the network itself is an `Update` on the edges of the space, so `TopologyInteraction` becomes an
  `Interaction` with such laws.
- Ports go. A circuit is node elements with potentials and component interactions whose constraints add into the
  balance of each node, so `ConnectionInteraction` and `ConnectorSpec` go.

### 4. Laws are seven kinds of value

`Rate`, `Noise`, `Jump`, `Update`, `Constraint`, `Event`, and `Objective` are frozen values that `_get_laws` returns.
They have no parameters of their own and no `prepare` or `run`. Contributions of several laws to one state add up, for
every kind where adding means something: rates, noises, jump channels, and constraint balances.

| Current law | New form |
| --- | --- |
| `RateLaw`, `GeneratorLaw` | `Rate`. The linear generator of quantum evolution is a `Rate` written with operators in the quantum package |
| `NoiseLaw` | `Noise` |
| `JumpLaw`, `TransferLaw`, `Source`, `Sink` | `Jump` with a change mapping, where an empty side is a source or a sink |
| `UpdateLaw`, `ArrayUpdateLaw`, `RuleLaw`, `LocalRuleLaw`, `TransitionLaw` | `Update`, with a random draw for transitions and a neighbourhood binding for local rules |
| `EquationLaw`, `BoundaryLaw` | `Constraint` |
| `CrossingLaw` | `Event` |
| `LinearProgramLaw` | `Objective` with `Constraint` |
| `SampleLaw` | `Objective` with a sampling solver |
| `FluxLaw` | `Rate` with `divergence`, and the finite-volume scheme as a solver setting of the fluids package |
| `DissipatorLaw` | the quantum package, with its solver |
| `CascadeLaw`, `ExtremalLaw` | the criticality package, with its solvers |
| `ProductionLaw`, `SubstitutionLaw` | the package of rewriting systems, with its solvers |
| `BilinearLaw` | the package of kinetic models, with its solver |
| `MatchLaw` | `skcomplex.systems.general`, as an `Update` |
| `GradientStepLaw` | the learning package, as an `Update` |

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

earth = system.simulate(until=1.4)
higher = system.simulate(until=2.0, initial_state={'ball.x': 20.0})
moon = system.set_params(gravity__g=1.62).prepare().simulate(until=1.4)
```

| scikit-learn | scikit-complexity |
| --- | --- |
| `LogisticRegression(C=1.0, solver='lbfgs')` | `System(elements=..., interactions=..., solver=..., backend=...)` |
| `fit(X, y)` returns `self` | `prepare()` checks every part, compiles the problem, picks the solver and backend, sets `problem_`, `solver_`, `backend_`, and returns `self` |
| `predict(X)` returns the predictions | `simulate(until, initial_state=None, random_state=None, observables=(), observers=())` returns the `Trajectory` |
| nested parameters of a `Pipeline` | `get_params` and `set_params` with `<label>__<parameter>`, and `<system>__<label>__<parameter>` for nested systems |
| `cross_val_score`, `GridSearchCV` | `ensemble`, `sweep`, `calibrate`, `sensitivity`, `basin_stability`, as functions |
| kernels of a Gaussian process | elements, interactions, spaces: parameters, no `fit` |

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
- Elements, interactions, spaces, and neighbourhoods derive from `BaseComponent`, which gives parameters, cloning, and
  tags, and no `prepare` or `run`. Solvers, backends, observers, compilers, and kernels keep `prepare` and `run`, which
  the system calls. Each family has `prepare` and its own verb, as scikit-learn estimators share `fit` and differ in
  `predict` or `transform`.

### 6. Ready-made systems

A ready-made system is a class deriving from `BaseSystem` with catalog metadata, `field`, `topics`, `references`, and
tuned run settings, and a constructor that stores its model parameters. Its hook `_prepare` sets `system_`, the
`System` of its components, as `fit` calls `_fit`, and the public `prepare` compiles it. A ready-made class earns its place when it reparametrizes, such as
`ProjectileSystem(speed, angle)`, which turns an angle into velocity components. A user who only wants to tune the
parameters of parts builds a `System` and uses nested parameters, with no class.

### 7. What the core holds

The core holds only the components that general systems of several fields are built from:

- elements: `BaseElement`, `Element`, `DeclaredElement`,
- interactions: `BaseInteraction`, `Interaction`, `GradientInteraction`, `HamiltonianInteraction`,
  `EulerLagrangeInteraction`,
- laws: `Rate`, `Noise`, `Jump`, `Update`, `Constraint`, `Event`, `Objective`,
- operators: `laplacian`, `divergence`, `gradient`, and the reductions over neighbours,
- systems: `BaseSystem`, `System`,
- spaces and neighbourhoods that systems of two or more fields use, decided by the audit task with the placement rule.
  The candidates to leave are `CayleySpace`, `PhaseSpace`, `SimplexSpace`, `CofaceNeighborhood`,
  `MemoryNeighborhood`, and `StateTranslationNeighborhood`. `BaseTime`, `ContinuousTime`, `DiscreteTime`, and
  `StaticTime` leave the public API.

Everything else that is field-specific moves to `skcomplex.systems.<field>.common.<family>` when one field uses it, or
to `skcomplex.systems.general.<concept>` when several fields share it as a recipe. Moving code never deletes meaningful
specialised code: it keeps working in its new place and serves as an example there. The concrete systems of each field
are the consumers of the core and of their field's common package.

### 8. Specs

| Kind | Fate |
| --- | --- |
| Declarations users write: `StateSpec`, `ParameterSpec`, `RoleSpec` | public, with the aliases `State`, `Parameter`, `Role`. `VisualSpec` moves to the visualisation observer. `ConnectorSpec` goes with ports |
| Value objects users pass or read: `ObservableSpec`, `WienerProcessSpec`, `ScheduleSpec`, `Tags`, and a few more | public, kept only if a system or a document uses them |
| Field values, such as `HilbertSpec` and `PartnerSpec` | move to the field package that uses them |
| Engine records and compile specs, such as `LoweredFunctionsSpec`, `DrawLogSpec`, `EventRecordSpec`, and the eleven specs of `skcomplex.core.components.specs` | private frozen dataclasses beside the code that uses them |

The rule that every dataclass is a public `BaseSpec` in a specs package goes. A spec is public only when users build
it or read it from a result. The target is about fifteen public specs.

### 9. Docstrings

Every docstring of the core follows the scikit-learn layout: one summary line, one paragraph in plain words with the
formula when there is one and one sentence of uses from several fields, a line that links the section of the user
guide, the arguments, a See Also section of one line per neighbouring class, and an Examples section of three to six
lines that uses real parts and defines no class only to have something to work on. The docstrings are written last,
against the new API.

## Risks / Trade-offs

- [A generic `Update` is slower than the specialised kernels of sandpiles or rewriting] → The field package that owns
  such a law brings its own solver on the public engine bases, so its kernel survives the move.
- [An agent's behaviour as an interaction reads less like Mesa, where behaviour is a method of the agent] → One place
  for every rule is worth it, and the agent guide shows the pattern once.
- [Inferring time from law kinds hides a choice] → `prepare` records the inferred problem in `problem_`, the tags show
  the time, and a system mixing kinds that do not combine raises a named error listing them.
- [The rewrite touches every system, test, and page] → The new API is built beside the old one inside the core, the
  fields migrate in waves, and the old classes are deleted only when no system uses them.
- [Two spellings for declarations, `State[float]` and `StateSpec(...)`] → They serve two cases, the plain one and the
  one with options, and the guide shows both once.
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
