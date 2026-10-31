# Design

## Context

See proposal.md for the motivation. Six audits, in `inventory/`, map every element and interaction class of
`skcomplex.models`, about 510 elements and 180 interactions, to the operations they perform. They show three facts:
- Almost no model uses a core interaction other than the binding bases. The models write the same operations inline
  across fields, and several fields write them twice.
- The first set of mechanisms of this change named field quantities and field models: forces, damping, foraging,
  regrowth, alignment, growth forms, named oscillators, and price-named protocols. A user of another field cannot reuse
  them without reading them as physics, ecology, or economics.
- The engine already has the general mechanisms the operations need: states of any order, rate, noise, jump, update,
  equation, crossing, transition, local-rule, sample, flux, cascade, encounter, transfer, and rule laws, neighbour and
  pair bindings, reductions, delays, and agent rules.

## Goals / Non-Goals

**Goals:**
- The core gives one general class per mathematical operation, configured by the states it binds and by functions the
  user passes. Together the classes cover every current model.
- A field case is a thin subclass of a general class, even a pure renaming. It lives with its one model, or in the
  `common` package of its field when several packages of the field use it. These subclasses double as examples of how a
  user builds a part.
- Every model class is rebuilt on the general classes and keeps only what is its own.

**Non-Goals:**
- New models, or changes to what a model computes.
- Laws per hyperedge, which `add-higher-order-interactions` adds. Group mechanisms run here as agent groupings.
- Gradient learning, which `add-learning-dynamics` adds.

## Decisions

### A core class names an operation, never a quantity
A core class names what it does mathematically: a rate, a gradient, a map, a coupling, a jump, a transition, a choice,
a movement, a transfer, a grouping. It never names what a state means. There is no force, position, velocity, mass,
price, infection, energy, or harvest in a core name or argument. A force is a rate on a state of order 2, whose laws give
its second derivative. Its velocity is the first derivative of that state. Its mass is a divisor.

Alternative considered: one core class per recurring mechanism, such as `ConstantForceInteraction` or
`ForagingInteraction`. Rejected, because each is a special case of a more general operation and binds the core to a
field.

### Arguments: roles, state names, functions, divisor, variants
- Roles bind groups: `target` is the role whose states the laws change, `sources` are roles they only read.
- State arguments name the states read and written, so one class serves any meaning.
- Functions are callables of the namespace that return symbolic expressions. In the agent form they are callables of
  the agent and the model view.
- `divisor` divides what a rate, gradient, or pair coupling adds to the highest derivative, such as a mass, a friction,
  or a capacitance. It is a named argument, not part of the user's function, so backends such as LAMMPS can read it.
- A variant is a string argument checked in `_check` against a module constant, or a configurable class when it carries
  its own algorithm, such as a matching rule.

### Two lowerings, chosen by an argument
A class whose operation agent models use takes `form='symbolic'` or `form='agent'`. In the agent form it lowers to an
agent `RuleLaw` in its `phase`, and its functions take the agent and the model view, so they can read the model's
history and storage and the agent's neighbours and cell. The form is an argument because the engine probes laws with
empty namespaces and cannot tell an agent group from a symbolic one, and a rule found at probe time would turn a
symbolic group into agents.

### Several discrete-time updates of one state
A state takes one `UpdateLaw`, unless each of its update laws declares how it combines: an `increment` gives the state
plus a change, and a law with a `source` moves only a state that holds that value. Such laws combine in one step as
`x + sum_k (f_k - x)`, all read from the start of the step, so increments add and moves from distinct source values
compose. Two laws with the same source, or a law that declares neither, stay an error. An `UpdateLaw` with an `order`
updates the members one after the other, in increasing order of that expression, each reading the values set before
it, which gives the sequential and random-order schedules of maps.

### The catalog of general interactions

Continuous time:

| Class | Operation | Lowers to |
| --- | --- | --- |
| `RateInteraction(target, state, rate, sources=(), noise=None, divisor=None)` | adds `rate(ns)`, and `noise(ns) dW`, to the highest derivative of a state of any order | `RateLaw`, `NoiseLaw`, `FractionalNoiseLaw` |
| `RelaxationInteraction(RateInteraction)(target, state, goal, time, noise=None)` | relaxes a state towards a goal expression, `(goal - x) / time`, with optional noise (Ornstein-Uhlenbeck) | `RateLaw`, `NoiseLaw` |
| `GradientInteraction(target, states, potential, sources=(), divisor=None, variational=False, conserved=False)` | moves states down the gradient of a potential: gradient flow on order 1, Newtonian on order 2, variational and conserved on fields | `RateLaw` |
| `EulerLagrangeInteraction(target, states, lagrangian, dissipation=None, forcing=None)` | equations of motion of a Lagrangian, with Rayleigh dissipation and generalized forcing | `EquationLaw` |
| `ConservationInteraction(target, state, flux, source=None)` | a conserved state of a field with a flux and a source | `FluxLaw` |
| `DiffusionInteraction(ConservationInteraction)(target, state, coefficient)` | flux minus a coefficient times the gradient | Laplacian of the field |

Discrete time:

| Class | Operation | Lowers to |
| --- | --- | --- |
| `MapInteraction(target, state, function, sources=(), mode='replace', schedule='synchronous', form='symbolic', phase=None)` | `x(t+1) = f(ns)`, or `x + f(ns)` in the `'increment'` mode, updated synchronously, sequentially, or in random order (`'random-sequential'`) | `UpdateLaw` with an `order`, or an agent `RuleLaw` |
| `LocalRuleInteraction(target, state, rule, neighborhood)` | cellular automata from a rule table, a totalistic rule, or a function of the ordered neighbourhood | `LocalRuleLaw` |
| `CascadeInteraction(target, state, threshold, mode, drive=None, neighborhood=None)` | drives, then relaxes every member at or above a threshold by reset, redistribution, or a random share, recording cascades as events | `CascadeLaw`, `DriveLaw`, `LocalRuleLaw` |

Coupling, over a structure:

| Class | Operation | Binding |
| --- | --- | --- |
| `CouplingInteraction(target, source=None, read, write, function, reduce='sum', mode='edges', weights=None, include_self=False, lag=0.0, delay=0.0, form='rate', combine=None)` | member `i` gets a reduction over `j` of `function(ns)`, where `ns` reads `i`, `j`, the edge, the parameters, and the time. It works over the edges of a lattice or network, over the whole group or every member of a separate `source` group, or over the next member of an ordered group. The reductions are sum, mean, max, min, normalized mean, and circular mean; the weights are an edge state, a matrix, or a callable of `ns` (indicators, member states). It adds to a rate (`form='rate'`) or sets the next value with `combine(ns, reduction)`, `ns` reading the member, the parameters, and the time (`form='map'`) | `NeighborInteraction`; a Laplacian on fields |
| `PairCouplingInteraction(first, second, cutoff, read, write, function=None, potential=None, divisor=None, period=None)` | every pair within a cutoff adds `function(ns)` or minus the gradient of `potential(ns, r)` to both members; with several states `write` the function returns one value per state, so a pair couples them jointly | `PairInteraction` |

Events and discrete states:

| Class | Operation | Lowers to |
| --- | --- | --- |
| `JumpInteraction(target, changes, rate=None, probability=None, waiting=None, reset=None)` | stochastic events that change states by signed amounts, at a rate, with a probability per step, or after drawn waiting times | `JumpLaw` |
| `ReactionInteraction(JumpInteraction)(reactants, products, k, propensity=None, modifiers=(), mode='rate')` | mass action between species given as role and state, with real stoichiometry and modifiers that the propensity reads but the reaction does not consume, as rates, jumps, or Langevin noise | `RateLaw`, `JumpLaw`, `NoiseLaw` |
| `TransitionInteraction(target, state, source, destination, rate=None, probability=None, condition=None, matrix=None, emission=None, observed=None, over='member', neighborhood=None, after=None, form='symbolic', phase=None)` | a discrete state moves between values at a rate, with a probability per step (combined over neighbours as `1 - prod(1 - p)`), when a condition holds, by a transition matrix, or by an emission matrix given another state, after a residence time (`residence(x)`) | `JumpLaw`, `UpdateLaw` with a `source`, `TransitionLaw`, `EmissionLaw`, or an agent `RuleLaw` |
| `SpreadingInteraction(TransitionInteraction)` | the active value of a state spreads along edges, with recovery and return | edge and member laws |
| `EncounterInteraction(candidate, field, kernel, outcome)` | bilinear encounters over a density on activity classes, conservative, destructive, or proliferative | `EncounterLaw`, `DestructionLaw`, `ProliferationLaw` |
| `CrossingInteraction(target, guard, reset=None, direction=0, terminal=False)` | an event when a guard crosses zero, which resets states or ends the run, once per member of a group whose guard reads the members | `CrossingLaw` |
| `ImpulseInteraction(target, times, reset)` | resets states at given or periodic times | `CrossingLaw` on time |
| `SamplingInteraction(target, states, energy, move, temperature)` | Monte Carlo sweeps by moves accepted from the change of an energy, or Wolff cluster flips (`move='cluster'`) | `SampleLaw` |

Choice and learning:

| Class | Operation |
| --- | --- |
| `ChoiceInteraction(target, state, options, utility, rule='logit', intensity=1.0, schedule='synchronous', activity=1.0)` | draws an option by the logit, Metropolis, Fermi, or best rule, symbolic or agent |
| `ImitationInteraction(target, state, payoff, options, rate, rule)` | copies a neighbour's state at random, by majority, from the best, or by the Fermi rule |
| `ReinforcementInteraction(target, scores, payoff, rate)` | moves the score of every option towards its realized payoff |
| `EvolutionInteraction(target, strategy, fitness, selection, replacement, crossover, mutation)` | selection by tournament, roulette, or rank, replacement of the generation, by Moran, or death-birth, crossover, and mutation over an alphabet, Gaussian, or a function |

Agents, groups, and exchange:

| Class | Operation |
| --- | --- |
| `MovementInteraction(target, step, probability=1.0, rate=None, condition=None, accept=None, exclusive=False)` | moves agents by neighbour, Gaussian, Lévy, best-cell, or a user step |
| `TransferInteraction(source, target, amount, pairing, legs=(), guard=None)` | moves amounts of states between members, cells, or a source and a sink, as a rate or as agent rules |
| `LifeCycleInteraction(target, birth, death, offspring=1, placement='here', inherit=(), mutate=None)` | births by splitting or by condition, and deaths by condition |
| `GroupingInteraction(target, groups, act, size=2, capacity=None)` | draws pairs, a random partition, neighbourhoods, or groups by a chosen state each step, then acts on every group |
| `MatchingInteraction(requesters, providers, rule, limit, quantity, budget, transfers, on_match)` | pairs requesters with providers by the algorithm its rule spec names, then transfers |
| `FeedbackInteraction(rule)` | reads observables once per round and sets parameters |
| `TopologyInteraction(target, action, rate=None, probability=None, condition=None, toward=None)` | adds or removes nodes and edges, and rewires, preferentially towards a weight |

The matching rules are frozen specs of `skcomplex.specs` that hold only the settings of their algorithm:
`RandomSearchSpec(tries, clearing)`, the default, `PartnerSwitchSpec(sample, intensity, partner)`, `QueueSpec(order)`,
`IterativeClearingSpec(adjustment, iterations, tolerance)`, `ContinuousClearingSpec(arrival)`, `BatchClearingSpec()`,
`StableMatchingSpec(requester_preference, provider_preference, clearing)`, and
`GreedyMatchingSpec(score, candidates, capacity, clearing)`. `MatchingInteraction` runs each algorithm as its own
private method, chosen by the type of its `rule`, on the offers it reads directly, so no protocol class or round spec
exists.

### Elements, laws, and utilities
- Elements: `Element`, `GroupElement`, `FieldElement`, `LatticeElement`, `CompartmentElement`, whose state `n` a model renames by a
  thin subclass, since states are declared per class, `DensityElement`, and `PointElement`, which replaces `ParticleElement`: a member with a vector state of order 2
  in a continuous space. LAMMPS accepts an element by its declared states, and reads the velocity as the first
  derivative of the position.
- Laws: `RelaxationLaw` becomes `CascadeLaw`, so relaxation means relaxation towards a goal. `DriveLaw`, `FluxLaw`,
  `EncounterLaw`, `DestructionLaw`, and `ProliferationLaw` keep their names and lose their field wording.
  `StateSpec.velocity_of` is removed.
- Utilities of the law namespace: `hill`, `logistic`, `pulse_train` with a count, `clip`, `minimum`, `maximum`, `sign`,
  the stock-and-flow delays `delay1`, `delay3`, and `smooth`, sums, means, and counts keyed by a state (`by=`), and a
  circular mean.

### Coupling extensions
- A second group of nodes: `NeighborInteraction` may bind a second group of nodes instead of an edge group. The laws run
  once per member of the first group and their reductions over the members of the second, with no edges. Roles may be
  declared `RoleSpec(optional=True)` and bound to `None`; `CouplingInteraction.source` is such a role, used in the mode
  `'group'` only (two populations of the Kuramoto-Sakaguchi chimera).
- The extremes over the own group need `include_self` or a source group, since a pairwise maximum cannot leave the
  member out; the coupling to the next member reduces by sum or mean only.
- `combine` takes `(namespace, reduction)`, so a map reads the member's parameters, the interaction's parameters, and
  the time (coupled logistic maps, Vicsek noise `eta` read from the member).
- Field realizations built on these: `ConfidenceCouplingInteraction` (social, Hegselmann-Krause),
  `LogisticCouplingInteraction` and `SwarmalatorCouplingInteraction` (physics synchronization),
  `HeadingAlignmentInteraction` (physics active matter, Vicsek and Ginelli nematics), and the chimera on
  `KuramotoCouplingInteraction(source=...)`; `MeanFieldCouplingInteraction` is removed.
- Still open: coupling a list of separate elements (an element set) and a single hub element as the target, the
  Laplacian lowering of a coupling on fields, coupling two separate elements, and weighted group coupling whose term
  reads the member under a matrix (pairwise reductions take no matrix weights).

### What leaves the core
- `PotentialInteraction` and `PairPotentialInteraction` become `GradientInteraction` and `PairCouplingInteraction`.
- `PairForceInteraction`, `ConstantForceInteraction`, `DampingInteraction`, and `OrnsteinUhlenbeckInteraction` become
  physics subclasses of `PairCouplingInteraction`, `RateInteraction`, and `RelaxationInteraction`.
- `GrowthInteraction` becomes growth subclasses of `RateInteraction` in the fields that use them.
- `ForagingInteraction` and `RegrowthInteraction` become compositions of `TransferInteraction`, `LifeCycleInteraction`,
  and `MapInteraction` in the fields that use them.
- `AlignmentInteraction` becomes a `CouplingInteraction` in the map form in physics.
- `RandomWalkInteraction` becomes the steps of `MovementInteraction`.
- `ThresholdInteraction` becomes `CascadeInteraction`.

### What stays in the models
A mechanism that one field alone uses keeps its own laws on the general laws, with no core class: quantum generators
and dissipators, rewriting systems, references from one agent to another, queues and set-valued states, barter swaps,
and algebraic feedback of control loops. A field-shared realization goes to `<field>/common/<family>`, as the
inventory lists, such as `physics/common/oscillators`, `biology/common/pharmacokinetics`, and
`economics/common/forecasting`.

### Hyperedges
The engine compiles the laws of a `HyperedgeInteraction` once per member and ignores the hyperedges. Laws run once per
hyperedge belong to `add-higher-order-interactions`. Until it lands, the group mechanisms of Galam, public goods,
Kolkata, and the minority game run as agent groupings with `GroupingInteraction`.

## Risks / Trade-offs

- **[Risk] Rewritten models change their numbers.** Stochastic models draw in a different order once a general class
  draws. → Seeds stay. A model whose test tolerance fails is compared over 20 seeds against its previous trajectory,
  and only a statistically equal model is accepted.
- **[Risk] General classes grow many arguments.** → Each argument has a default that gives the plain operation. A
  model that needs more subclasses the class and overrides one private hook.
- **[Risk] Field-term creep.** → The field-term test runs on every public name of the core.
- **[Risk] LAMMPS loses its class lookup.** → LAMMPS reads the declared states and the `divisor`, and the physics
  subclasses keep their pair styles and fixes.
- **[Trade-off] Field users bind state names.** A physics model writes `state='x'` and `divisor='m'` where a force
  class took them for granted. In exchange, every field uses the same classes.

## Migration Plan

Work happens on `development`. The core is rebuilt first by parallel agents with disjoint modules, then the models
are rewritten by field, each agent on a worktree branch cut from `development` that merges back as soon as it reports
and its tests pass. Moved public names keep no aliases. Rollback is reverting the merges.

## Open Questions

- None. The hyperedge gap is answered above.
