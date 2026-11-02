## Purpose

Lets users extend every general core class for their own field without hidden restrictions: each fixed decision has a
hook, each numeric path takes a callable, and each remaining limit raises a clear error and is recorded with its reason.

## ADDED Requirements

### Requirement: Subclasses rename roles in every form
A subclass of a general interaction SHALL be able to rename any role it binds, and the renamed role SHALL work in the
symbolic form as well as the agent form.

#### Scenario: Renamed role in a symbolic rate
- **WHEN** a subclass of the rate interaction binds its target as `stock` and runs symbolically
- **THEN** the system compiles and the trajectory equals that of the plain rate interaction on the same element

#### Scenario: Renamed role in an agent transfer
- **WHEN** a subclass of the transfer interaction names its source `pasture` and its target `cows` and runs as agents
- **THEN** cows take amounts from the pasture of their cell as the plain transfer would

### Requirement: Subclasses name their own parameters
A subclass of a general interaction SHALL be able to name each parameter the general class reads, or fix its value
without declaring it as a parameter.

#### Scenario: Renamed probability
- **WHEN** a movement subclass calls its move probability `boldness`
- **THEN** the movement reads `boldness`, and the subclass need not declare `probability`

#### Scenario: Fixed value
- **WHEN** a movement subclass fixes its probability to 1 without declaring a parameter
- **THEN** every agent moves at every step

### Requirement: Variants accept specs and callables
Every variant argument of a general interaction SHALL let a subclass add a variant without bypassing the checks of its
parent. A pairing, rule, grouping, selection, replacement, action, or mode SHALL accept a spec besides its named values,
and a pairing SHALL also accept a callable. A step, placement, or move SHALL accept its named values or a callable, and a
pick its named values; each of these SHALL refuse a spec with a clear error, since a callable says what a spec would say.

#### Scenario: Custom pairing
- **WHEN** a transfer receives a callable that pairs each agent with the agent of largest wealth in its cell
- **THEN** the transfer runs with those pairs, and the other checks of the transfer still apply

#### Scenario: Spec given to a step
- **WHEN** a movement receives a spec as its step
- **THEN** the check raises an error listing the named steps and saying that a function is accepted

#### Scenario: Unknown name
- **WHEN** a variant receives a name that is neither a named value, a spec, nor a callable
- **THEN** the check raises an error listing the named values and saying that a spec or callable is accepted

### Requirement: Agent forms take a schedule and a tie rule
Every agent form SHALL take a schedule (synchronous, member, or ordered) and every rule that picks a best option SHALL
take a tie rule (random, first, or nearest), with defaults that keep today's behaviour.

#### Scenario: Member schedule of imitation
- **WHEN** agent imitation runs with the member schedule
- **THEN** each agent imitates in its own activation, reading the choices of agents that acted before it

#### Scenario: Random ties
- **WHEN** a best-step movement with vision runs with random ties
- **THEN** among equally good cells at equal distance, each is chosen with equal frequency over many runs

### Requirement: Callable laws on every numeric path
Every law kind whose value the engine evaluates numerically SHALL accept a callable: rates, noise, jumps and their
changes, updates, boundaries, crossing guards and resets, flux wave speeds, transition matrices, renewal waiting times,
reductions, and observables. An emission matrix and a reset of a renewal stay symbolic, because each shares its draws
with another law, and each says so when given a callable.

#### Scenario: Callable crossing
- **WHEN** a crossing law has a callable guard and a callable reset
- **THEN** the event fires where the guard changes sign and applies the reset, as the symbolic crossing would

#### Scenario: Learned observable
- **WHEN** an observable of a non-agent system is a callable of time and values that calls a trained model
- **THEN** the trajectory records its value at every frame

### Requirement: Callables read their component and the random stream
A callable law SHALL be able to read values keyed by its own component, receive the simulation's random stream, and run
once over all members of a group.

#### Scenario: Portable element law
- **WHEN** an element class with a callable rate is used under two labels and inside a nested system
- **THEN** the same callable reads its own states in each case without naming a global path

#### Scenario: Reproducible stochastic policy
- **WHEN** a callable update draws from the stream it receives and the run is repeated with one seed
- **THEN** both runs give the same trajectory

#### Scenario: Batch evaluation
- **WHEN** a callable rate over a group of 1000 members is declared to run over the whole group
- **THEN** it is called once per evaluation with arrays of all members

### Requirement: Symbolic-only laws say so
A law kind that needs symbols by nature SHALL raise, when given a callable, an error naming the law kind and the
callable alternative.

#### Scenario: Callable equation
- **WHEN** an equation law receives a callable
- **THEN** the error says equations are solved symbolically and points to a callable rate law of the highest derivative

### Requirement: Limits raise errors naming the missing hook
A combination the core cannot run SHALL raise an error that names the operation, the reason, and the hook or argument
that would lift it, and SHALL never give a result that silently differs from the documented one.

#### Scenario: Callable Stratonovich amplitude
- **WHEN** a noise law with a callable amplitude asks for the Stratonovich convention without a derivative
- **THEN** the check raises an error saying the drift correction needs a symbolic amplitude or a supplied derivative

#### Scenario: Grid operation in continuous space
- **WHEN** a transfer paired by cell runs on agents in a continuous space
- **THEN** the error says cell pairing needs a grid or network and names the radius pairing

#### Scenario: Ignored phase activation
- **WHEN** a phase of update laws asks for an activation other than the default
- **THEN** the check raises an error saying update phases take only period and offset

### Requirement: Restrictions are recorded with reasons
Every restriction the core keeps SHALL be stated in a specification or design with its reason and the change that
would lift it.

#### Scenario: Event ties
- **WHEN** a user reads why two crossings in one step apply in declaration order
- **THEN** the specification states the rule, its reason, and the option that would treat them as simultaneous

### Requirement: Tied crossings apply in declaration order
The crossings of one step SHALL apply earliest first, and crossings at one time within the minimum separation SHALL all
fire in declaration order, then by member, each reset reading the state the one before it left.

#### Scenario: Two crossings at one time
- **WHEN** two crossings of one system change sign at one time
- **THEN** both fire, and swapping their declaration swaps the order of their resets
