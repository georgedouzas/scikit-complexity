# Tasks

## 1. Done before the redesign

- [x] 1.1 Work on `development`, cutting one worktree branch per parallel agent from it, and verify `git status` is clean before the first merge.
- [x] 1.2 Extend the edge kernel of `NeighborInteraction` with per-member laws of either end, the degrees of both ends, `delayed` reads of both ends, and cascade, drive, and local-rule laws; verify each on a network and a lattice.
- [x] 1.3 Let a reduction read the reading member apart from the reduced one; verify the mean of `sin(other.x - x)` against NumPy.
- [x] 1.4 Add a reduction that returns the state of the neighbour at which an expression is largest; verify it on a ring and a network.
- [x] 1.5 Guard the public names of the core with the field-term test; verify it rejects a name with a field term.
- [x] 1.6 Move matching and feedback to the core as `MatchingInteraction`, seven matching rules (now specs), and `FeedbackInteraction`, and port their users; verify the nine ported models give identical trajectories.
- [x] 1.7 Find whether hyperedge laws compile per hyperedge; verify by a run, and record in design.md that they do not.

## 2. Core rebuild

Each task adds its classes, one per private module, exported from their package, with tests mirroring the source and a runnable doctest, and removes the classes design.md moves out of the core together with their tests, moving each to the physics, biology, or economics family the inventory names.

- [x] 2.1 Continuous time: add `RateInteraction`, `RelaxationInteraction`, `GradientInteraction`, `EulerLagrangeInteraction`, `ConservationInteraction`, and `DiffusionInteraction`, with `divisor` and noise options (process, convention, fractional); remove `PotentialInteraction`, `OrnsteinUhlenbeckInteraction`, `ConstantForceInteraction`, `DampingInteraction`, and `GrowthInteraction` from the core; verify the scenarios of the rates, relaxation, gradients, Lagrangians, conservation, and the force-as-rate scenario.
- [x] 2.2 Coupling and points: generalize `CouplingInteraction` (read and write states, function of both members and the edge, weights, `include_self`, ordered next member, map form with `combine`), add `PairCouplingInteraction` replacing `PairPotentialInteraction` and `PairForceInteraction`, remove `AlignmentInteraction`, replace `ParticleElement` by `PointElement` and remove `StateSpec.velocity_of`, let LAMMPS accept elements by their states and read the divisor, and rebuild `physics/common/particles` on them; verify the Kuramoto, Vicsek, Lennard-Jones, thermostat, and LAMMPS agreement tests.
- [x] 2.3 Events and discrete states: add `MapInteraction`, `LocalRuleInteraction`, `JumpInteraction`, `TransitionInteraction`, `EncounterInteraction`, `CrossingInteraction`, `ImpulseInteraction`, and `SamplingInteraction`; rename `ThresholdInteraction` to `CascadeInteraction` and `RelaxationLaw` to `CascadeLaw`; make `ReactionInteraction` a `JumpInteraction` with species as role and state, real stoichiometry, and modifiers, and `SpreadingInteraction` a `TransitionInteraction`; verify the map, sandpile, reaction, epidemic-threshold, integrate-and-fire, and Ising scenarios.
- [x] 2.4 Choice and learning: give `ChoiceInteraction` the agent form, the `'best'` rule, a schedule, and an activity probability; give `ImitationInteraction` the Fermi rule and discrete time; add `ReinforcementInteraction`; give `EvolutionInteraction` rank selection, Moran and death-birth replacement, and function mutation; port the users of `logit_choice` and rename `economics/common/learning` to `forecasting` with only the forecasting heuristics; verify the Glauber, voter, choice-reading-history, and evolution scenarios.
- [x] 2.5 Agents, groups, and exchange: add `MovementInteraction` absorbing `RandomWalkInteraction`, `TransferInteraction` with its symbolic and agent forms, `LifeCycleInteraction`, `GroupingInteraction`, and `TopologyInteraction`; remove `ForagingInteraction` and `RegrowthInteraction`; turn the matching protocols into the rule specs `RandomSearchSpec`, `PartnerSwitchSpec`, `QueueSpec`, `IterativeClearingSpec`, `ContinuousClearingSpec`, `BatchClearingSpec`, `StableMatchingSpec`, and `GreedyMatchingSpec` that `MatchingInteraction` runs as its private methods, removing `MatchingRoundSpec`; verify the walker-diffusion, matching, Galam-grouping, and transfer-conservation scenarios.
- [x] 2.6 Laws, utilities, and elements: add `hill`, `logistic`, `pulse_train` with a count, `clip`, `minimum`, `maximum`, `sign`, `delay1`, `delay3`, `smooth`, keyed reductions, and a circular mean to the law namespace; remove the field wording of `DriveLaw`, `FluxLaw`, `EncounterLaw`, `DestructionLaw`, and `ProliferationLaw`; keep `CompartmentElement` with its state `n`, renamed by thin subclasses; verify each utility against NumPy and the field-term test.

- [x] 2.7 Engine extensions the discrete classes need: the agent form (`form='agent'`) of `MapInteraction` and `TransitionInteraction`; sequential and random-order schedules of `MapInteraction`; the residence time `after` of `TransitionInteraction`; cluster moves of `SamplingInteraction`; several discrete-time updates of one state combined in one step; and crossings and impulses on the members of a group; verify a discrete SIR with two transitions on one state, a Wolff run of the Ising model against its Metropolis magnetization, an integrate-and-fire group, and a sequential Hopfield map.

- [x] 2.8 Coupling extensions: a separate source group, maximum, minimum, normalized, and circular means, callable weights, `combine(namespace, reduction)`, and pair coupling of several states; rebuild the chimera, swarmalators, coupled map lattices, Hegselmann-Krause, Vicsek, and Ginelli nematics; verify identical trajectories. Separate elements, hubs, and the contour cell move to `complete-general-model-rewrites`, kernels to `add-nonlocal-operators`, and matrix weights to `add-learning-dynamics`.
- [x] 2.9 Per-member reactions: `ReactionInteraction` on a group or field with species as states and subclasses with renamed roles; merge the reaction-diffusion duplicates into `chemistry/common/kinetics` and rebuild Gray-Scott, the Brusselators, Schnakenberg, cubic autocatalysis, and the compartment transfers; verify trajectories equal to round-off. Parametric coefficients (Oregonator) move to `complete-general-model-rewrites`.
- [x] 2.10 Agent ecology: member-by-member scheduling of agent interactions, transitions acting on the agents of a cell, movement with vision, transfers that eat a partner whole or deposit into a field site; rebuild the epidemiology walkers and travellers, wolf-sheep, and the Sugarscape foragers; verify each over 20 seeds. The meadow, ants, trails, angiogenesis, and boids move to `complete-general-model-rewrites`.
- [x] 2.11 Life cycle variants: births into another group, drawn values, random-cut splits, replacement, renewal in place, culling kin, Poisson arrivals, and node growth from outside a network; rebuild Barabasi-Albert, Holme-Kim, vertex copying, Galton-Watson, fragmentation, and ridesharing arrivals; verify identical networks or equality over 20 seeds. The commons herders and tangled nature move to `complete-general-model-rewrites`.
- [x] 2.12 Learning and evolution: proportional and truncation selection, best-mutant replacement, mutation choices, a period, fitness functions, cumulative and vector reinforcement, and agent imitation of a better neighbour; rebuild Moran, adaptive climbing, neutral speciation, bit-string reproduction, the tournament, the minority games, El Farol, and order-book imitation; verify each over 20 seeds. The perishables market and observable history move to `complete-general-model-rewrites`.
- [x] 2.13 Transfers and pairing: named flows, disjoint neighbour pairing, and a fallback group; rebuild eight stock-flow models and both barter models; verify identical reporters and equality over 20 seeds. Edge transfers and composite transfers move to `complete-general-model-rewrites`.
- [x] 2.14 Agent state kinds: reference states, referenced transfer legs, group entries of variadic roles, transitions from a set of values, and sign adjustment; rebuild credit, labour, shops, commons owners, Metaphoria, and banking; verify exact or 20-seed equality. Queues and set-valued states stay in their models, as design.md records.
- [x] 2.15 Fields and mechanics: coupled conservation states with per-cell parameters, a transport class, vector resets in jumps, a divisor given as an expression, and Lagrangians summed across interactions; then rebuild shallow water, Payne, Kerner-Konhauser, KS, vorticity, Boussinesq, the walkers, and the spring pendulums; verify each against its previous trajectory.

## 3. Model rewrites

Each task rebuilt the classes of its inventory file that the general classes cover: pure classes became a general class or a thin subclass, composite classes bind general classes for their part, shared realizations moved to `<field>/common/<family>`, and duplicates within the field merged. Each verified the tests and doctests of every touched package, with stochastic models compared over 20 seeds against their previous trajectories. The classes still on their own laws are recorded, with what blocks them, in `complete-general-model-rewrites`.

- [x] 3.1 Rewrite biology collective behaviour, ecology, epidemiology, and evolution (`inventory/biology_1.md`).
- [x] 3.2 Rewrite biology neuroscience, oncology, physiology, and systems biology (`inventory/biology_2.md`).
- [x] 3.3 Rewrite physics (`inventory/physics.md`), including `common/oscillators`, `common/active`, and `common/fields`.
- [x] 3.4 Rewrite economics (`inventory/economics.md`).
- [x] 3.5 Rewrite social and engineering (`inventory/social_engineering.md`), with the group mechanisms on `GroupingInteraction`.
- [x] 3.6 Rewrite mathematics, computer science, chemistry, and earth (`inventory/math_cs_chem_earth.md`).

## 4. Docs and integration

- [ ] 4.1 Update the user guide's sections on elements, interactions, and laws to the general classes, with one example per class, and the constitution's extension points; verify `pytest docs/overview` passes.
- [ ] 4.2 Update the model inventory and the examples that import moved names; verify `grep` finds no removed name in `src`, `tests`, or `docs`.
- [ ] 4.3 Run the checks session and the full suite with doctests, and verify only the known timing test fails.
- [ ] 4.4 Merge every agent branch into `development` as it reports and delete its worktree, and verify `git worktree list` and `git branch` list none of them.

## Workflow follow-up

- Archive the change with `openspec archive general-elements-and-interactions -y` once merged.
