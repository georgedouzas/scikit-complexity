# Design

## Context

The four existing forms keep state in one flat vector of fixed size. Agent-based models break both assumptions: rules
are arbitrary Python, and populations change. A probe of Mesa 3.5.1 confirmed the API this design relies on:
`Model(rng=...)`, `Agent(model)`, `Agent.create_agents`, `model.agents.shuffle_do('step')`, `model.time`,
`run_until`, `schedule_event`, the `discrete_space` grids (`OrthogonalMooreGrid`, `OrthogonalVonNeumannGrid`,
`HexGrid`, `Network`, `VoronoiGrid`), and `experimental.continuous_space`. Mesa 3.5 requires Python 3.12 or later and
imports `networkx` at top level without declaring it.

## Goals / Non-Goals

**Goals:**

- Agent rules described in the framework's vocabulary and executed by Mesa.
- Any existing Mesa model runnable with the framework's runtime and Rerun visualization.

**Non-Goals:**

- Coupling an agent-based subsystem to an ODE subsystem in one system. It needs the co-simulation scheduler, which is a
  separate change.
- Mesa's own Solara visualization. Rerun is the framework's visualization.
- Compiling rules with Numba. Rules are Python, and agent counts that need compiled rules can use element groups with
  update laws instead.

## Decisions

### The agent-based form keeps its state in Mesa

`Form.AGENT` problems hold no flat state vector for agents. The Mesa backend owns a `mesa.Model`, and each frame takes a
snapshot: per agent group, the agent ids and one array per declared variable, plus model reporters. `Frame` gains an
`agents` mapping for these snapshots, and `Trajectory` stores them per frame. Parameters stay in the flat parameter
vector, so live updates work as for other forms and are pushed into the model before each step.

- Alternative considered: a fixed-capacity state vector with an alive mask. Rejected because capacity must be guessed
  and Mesa already owns agent identity, spaces, and scheduling.

### Rules

`Rule(function)` is a law whose function receives an `AgentView` with attribute access to the agent's declared
variables, `neighbors()`, `move_to(cell)`, `create(type, **values)`, and `remove()`, plus the model view with parameters
and random generator. `AgentType` subclasses `Element` and declares variables with `state` and `parameter` as today.
The compiler generates one `mesa.Agent` subclass per agent type whose `step` calls the rules in declaration order.

### Spaces

`GridSpace(width, height, neighborhood='moore' | 'von_neumann', torus=False)`, `NetworkSpace(graph)`, and
`ContinuousSpace(bounds, torus=False)` are `Space` subclasses that the Mesa backend maps to
`OrthogonalMooreGrid`, `OrthogonalVonNeumannGrid`, `Network`, and the experimental continuous space.

### Wrapped Mesa models

`MesaSystem(model_class, parameters, model_reporters, agent_reporters)` holds a Mesa model class and the
constructor parameters, and declares reporters as names mapped to functions of the model or of an agent. It compiles to
an agent-based problem whose parameters are the declared ones. The backend constructs the model with the seed from
`Simulation.random_state` and sets parameter attributes on live updates.

### Visuals

Agent groups on grids and in continuous spaces log one `Points2D` per group with agent ids as hidden labels, and
networks log `GraphNodes` and `GraphEdges`. A `color_by` option reuses the group visual's color scale.

### Dependencies

`mesa>=3.5` and `networkx` become runtime dependencies. The Project Profile records why `networkx` is explicit.

### Implementation notes

- The Mesa runner is `MesaStepper` in `skcomplex.solvers`, registered as the default solver of the agent-based form,
  because the runtime resolves default solvers without importing `skcomplex.backends`, and Mesa needs nothing from the
  model library.
- A rule of an agent type runs for every agent as `function(agent, model)`, and a rule of an interaction runs once per
  step after the agents as `function(model)`.
- `AgentType` has a plain `location` field, one location or one per agent of a group, for initial placement. Without
  it agents go to random cells, empty ones under a capacity, and network agents fill the nodes in node order.
- Group names in `create` and `agents_here` resolve as paths, or as labels relative to the acting agent's subsystem,
  so model libraries keep working when nested.
- Agent-based models share the discrete map's integer frame duration, which defaults to 1.
- Every group reports its size as `<group>.count`, and wrapped models report their model reporters as
  `<label>.<reporter>`. Trajectories return reporters by name and agents with `agents(group, frame)`.

## Risks / Trade-offs

- [Mesa's continuous space is experimental] → The continuous space adapter is isolated in one module and its tests pin
  the behaviour it relies on.
- [Snapshots copy agent variables every frame] → Snapshots copy only declared variables, and the Rerun observer's
  `interval` thins logging for large models.
- [Rules cannot be analysed symbolically] → `equations_of` returns no equations for the agent-based form, and
  `solve_symbolically` raises its analysis error because the form is not an ODE.
