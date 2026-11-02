# Audit: models/social and models/engineering against the general core of BRIEF.md

Scope: every element and interaction class in `src/skcomplex/models/social/**` and `src/skcomplex/models/engineering/**`
(views and model factories excluded; factories checked for which core classes they assemble). 96 classes.
Abbreviations: soc=social, eng=engineering; `RateI`=RateInteraction, `GradI`=GradientInteraction, `CoupI`=CouplingInteraction,
`TransI`=TransitionInteraction, `ChoiceI`, `ReacI`=ReactionInteraction, `MoveI`=MovementInteraction, `XferI`=TransferInteraction,
`LifeI`=LifeCycleInteraction, `MatchI`, `FbI`=FeedbackInteraction, `EvoI`=EvolutionInteraction. "holder" = element that only
declares states (no laws); it is a plain `Element` with state specs and needs nothing else.

## 1. Table (grouped by mechanism)

### A. Explicit ODE vector field on one element (order-1 / order-2 rates)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| CircuitElement | eng/circuits/_circuits | EquationLaw | linear 2nd-order forced oscillator L q''+R q'+q/C=V cos wt | RateI(state=q order 2, rate=(V cos wt - R q' - q/C)/L) or DampingI+GradI(q^2/2C)+RateI(drive) | nothing (pure composition) |
| ChuaCircuitElement | eng/circuits/_circuits | RateLaw x3 | 3D autonomous ODE | RateI x3 on x,y,z | Chua piecewise-linear vector field |
| MultiscrollChuaCircuitElement | eng/circuits/_circuits | RateLaw x3 | 3D ODE | RateI x3 | sine-scroll nonlinearity `_scroll_nonlinearity` |
| MixedModeChuaCircuitElement | eng/circuits/_circuits | RateLaw x3 | slow-fast 3D ODE | RateI x3 | vector field |
| MemristorCircuitElement | eng/circuits/_circuits | RateLaw x3 | 3D ODE | RateI x3 | memristor vector field |
| TunnelDiodeCircuitElement | eng/circuits/_circuits | RateLaw x2 | van der Pol-type 2D ODE | RateI x2 | cubic current law |
| PiezoHarvesterElement | eng/circuits/_harvesters | RateLaw x2 (order 2) | two masses joined by spring+damper, base excitation | RateI(order 2) x2; inner spring-damper = CoupI(potential=k(y1-y2)^2/2)+relative damping | piezo voltage observable |
| RegenerativeCarElement | eng/power_systems/_electromechanics | RateLaw x3 | electromechanical ODE with dry friction | RateI x3 | EM coupling A*I, dry friction `_dry_friction` |
| FaradayDiskElement | eng/power_systems/_electromechanics | RateLaw x3 | homopolar generator ODE | RateI x3 | EM coupling, dry friction, cubic load |
| RailSegmentElement | eng/power_systems/_electromechanics | RateLaw x2 (order 2) | two-mass spring-damper + pulse-train load | RateI(order 2) x2, pulse_train | cubic stiffness, wheel load |
| GridNodeElement | eng/power_systems/_grids | RateLaw (order 2) | swing equation phi''=P-gamma phi' | ConstantForceI(P)+DampingI(gamma) = RateI(order 2) | nothing |
| DrivenOscillatorElement | eng/control/_regulation | RateLaw (order 2) | damped harmonic oscillator + adaptive input `force` | GradI(omega^2 x^2/2)+DampingI(2 zeta omega)+RateI(force) | nothing |
| SatelliteElement | eng/control/_control | RateLaw (order 2) + RateLaw | radial Kepler + bang-bang thrust | GradI(effective potential)+RateI(thrust) | thrust law (relay on energy) -> see GAP G9 |
| SwingingPendulumElement | eng/control/_control | RateLaw (order 2) x2 | pendulum + bang-bang energy swing-up | GradI(-omega^2 cos theta)+RateI(control) | swing-up relay law; `f` state integrates the control twice (odd) |
| CarElement | eng/traffic/_car_following | RateLaw | kinematics x'=v (v is a separate order-1 state) | RateI(x, v) — or declare x as order 2 | nothing |
| LeadCarElement | eng/traffic/_car_following | RateLaw x2 | kinematics + sinusoidal acceleration | RateI(x,v), RateI(v, forcing(t)) | nothing |
| StageElement | eng/processes/_processes | RateLaw | constant feed | RateI(m, feed) (= XferI(Source -> m)) | nothing |

### B. Implicit (mass-matrix) mechanics

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| CartPendulumElement | eng/control/_regulation | EquationLaw x2 | coupled implicit 2nd-order equations (Lagrangian), adaptive input | NO brief class (see CORRECTION C1): EquationLaw directly | equations of motion |
| KickedCircuitElement | eng/circuits/_circuits | EquationLaw, RateLaw(clock,1), CrossingLaw | linear oscillator + periodic impulsive kick via a clock state | RateI(order 2) + periodic event (GAP G8) | kick size A/LC |

### C. Relaxation toward a target, first-order lag (x' = (target - x)/tau)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| RoomElement | eng/control/_control | RateLaw | saturated heating toward setting + Newton leak toward oscillating outside | RateI x2 = Relaxation(T, outside, 1/leak) + saturated P-control (GAP G1, G9) | daily swing |
| ControlledProcessElement | eng/control/_control | RateLaw x2 | first-order plant with inline PID (derivative eliminated algebraically into tau+gain Kd) | RateI(plant) + PID controller (duplicate of PIDInteraction) | algebraic elimination of Kd |
| MotivationInteraction | soc/crowds/_pedestrians | RateLaw (vector) | v' = (speed*e - v)/tau | Relaxation(v, speed*direction, tau) ⊂ RateI | nothing |
| OptimalVelocityInteraction | eng/traffic/_car_following | RateLaw | v' = a (V(headway) - v); headway via `shift_member` | Relaxation(v, V(h), 1/a) + ordered-neighbour read (GAP G6) | OV function V(h) |
| IntelligentDriverInteraction | eng/traffic/_car_following | RateLaw | IDM acceleration of x, v, leader v, headway | RateI with ordered-neighbour read (CoupI over 'next member', needs both states x, v, CORRECTION C3) | IDM formula |
| DelayedFollowingInteraction | eng/traffic/_car_following | RateLaw with `delayed` | GM follower: lambda (v_l - v_f)(t-T)/(x_l - x_f)^m (t-T) | CoupI(function of BOTH x and v of both ends, tau=T) on a 2-node directed edge (C3) | GM sensitivity |
| LangevinDrivingInteraction | eng/traffic/_langevin | RateLaw, NoiseLaw | OU toward density-dependent speed, multiplicative noise | OrnsteinUhlenbeckI with target and sigma as expressions (C4) + ordered-neighbour headway | target speed V(rho) |

### D. Continuum fields (conservation laws / PDE)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| LighthillWhithamRichardsElement | eng/traffic/_continuum | FluxLaw(godunov) | scalar conservation law | NO brief class (GAP G5: ConservationInteraction) | Greenshields flux |
| PayneElement | eng/traffic/_continuum | FluxLaw x2, RateLaw | 2-equation hyperbolic system + relaxation source | G5 + Relaxation(q, Q_e(rho), tau) | flux, equilibrium |
| KernerKonhauserElement | eng/traffic/_continuum | FluxLaw x2, RateLaw with laplacian | as Payne + viscosity mu Δ(q/rho) | G5 + Relaxation + diffusion term in RateI | KK equilibrium speed |
| TrafficMomentsElement | eng/traffic/_langevin | RateLaw with divergence/laplacian | moment PDEs (Fokker-Planck closure) | RateI on FieldElement with differential operators | the closure |

### E. Stock-and-flow transfers between stocks (symbolic)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| TubElement | eng/processes/_processes | TransferLaw x2 (Source/Sink) | inflow/outflow | XferI symbolic form (CORRECTION C2) | step start |
| PassageInteraction | eng/processes/_processes | RateLaw x2 (-r m_s, +r m_s) | linear transfer between two elements written as two rates | XferI(source.m -> target.m, rate*source.m) — should be ONE TransferLaw (conservation + flow record) | nothing |
| RiceMillElement | eng/processes/_systems | TransferLaw x8, RateLaw | mill stocks, delay3, smooth, min | XferI x8 + Relaxation(capacity, smooth(indicated), expansion_time) | flow expressions, shares |
| WasteSystemElement | eng/processes/_systems | TransferLaw x8, RateLaw | waste chain with lookup acceptance | XferI x8 + Relaxation(capacity, generation, capacity_time) | flows, lookup table |
| ReliefPipelineElement | eng/processes/_systems | TransferLaw x3 | supply line with stock/pipeline correction, delay3 | XferI x3 | anchoring-and-adjustment ordering rule |
| IrrigationSchemeElement | eng/processes/_systems | TransferLaw x9 | reservoir/soil/groundwater | XferI x9 | demand/release/pumping rules |
| ContestElement | soc/organizations/_traps | TransferLaw x4 | success-to-the-successful, lookup allocation | XferI x4 (Source->a,b; a,b->Sink as decay) | allocation table |
| BurdenElement | soc/organizations/_traps | TransferLaw x3, distributed_delay | shifting the burden | XferI x3 | symptom/erosion rule |
| ArmsRaceElement | soc/organizations/_traps | TransferLaw x2, distributed_delay | escalation with perceived delay | XferI x2 (= one-sided Relaxation toward (1+m) delayed other) | margin |

### F. Count-level reactions / kinetic (Boltzmann) encounters on distributions

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| ListenersElement | soc/spreading/_rumours | — (holder) | count compartment | CompartmentElement | — |
| RumourContactsInteraction | soc/spreading/_rumours | JumpLaw x2 | Maki-Thompson: I+S->2S, S+(S|R)->R+.. with custom propensity | ReacI(mode='jump', propensity=...) x2 | propensities |
| OpinionLevelElement / CitizensElement / SecurityAgentsElement / PedestriansElement | soc/opinion_dynamics/_kinetic_opinion; soc/crowds/_criminality, _evacuation | — (holders `f`) | density over an activity grid | DensityElement | — |
| OpinionEncountersInteraction | soc/opinion_dynamics/_kinetic_opinion | EncounterLaw | bilinear h+k -> i+k with table A (step toward/away within confidence) | NO brief class; = ReacI with a reaction tensor (GAP G4) | `persuasion_table` |
| CriminalContagionInteraction | soc/crowds/_criminality | EncounterLaw | bilinear table (step up if field higher, down if lower) | G4 | `contagion_table` |
| DeterrenceInteraction | soc/crowds/_criminality | EncounterLaw | candidate x external field group | G4 (field role = other group) | `deterrence_table` |
| RecruitmentInteraction | soc/crowds/_criminality | RateLaw + moment | security density driven by mean criminality, decay | RateI(security.f, recruitment*moment(...) - retirement f) | nothing |
| WalkingInteraction | soc/crowds/_evacuation | EncounterLaw | transport on room graph x speed lattice as table | G4 with constant field (actually a linear Markov transfer: TransI on density) | `walking_table` (shortest-path routing) |
| CrowdingInteraction | soc/crowds/_evacuation | EncounterLaw x2 | speed down-shift by crowd in room, recovery | G4 | `crowding_table` |

### G. Pair forces / potentials in continuous space

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| SocialRepulsionInteraction | soc/crowds/_pedestrians | PairPotentialInteraction subclass | exponential pair potential | CoupI(over='radius', potential=A B exp((2R-r)/B)) | the potential |
| WallsInteraction | soc/crowds/_pedestrians | RateLaw (vector) | soft exponential walls on y | GradI(potential U(y)) | wall potential |
| TransmissionLineInteraction | eng/power_systems/_grids | RateLaw x2 | K sin(phi_j - phi_i) both ends, per explicit pair | CoupI(function=sin, edges, order-2 state, edge weight K) — needs edge weights (C3) | nothing (the factory should build a network) |

### H. Self-propulsion / movement of agents

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| LightSensitiveRobotElement | eng/robotics/_robots | RateLaw x2, NoiseLaw, BoundaryLaw x2, delayed | active Brownian particle: speed(light(t-d)) along heading, rotational diffusion, confinement to disk | RateI (vector rate s e(phi)), RateI(phi, 0, noise), confinement BoundaryLaw (GAP G10) | light field, speed law |
| SwarmRobotElement | eng/robotics/_robots | RuleLaw | count neighbours in radius, turn rules, avoid nearest, step along heading, clamp to arena | MoveI(step=heading*speed, clamp) + neighbour count (CoupI agent form, reduce='count', over radius) | turn/shadow rules |
| SupervisorElement | soc/organizations/_sabotage | RuleLaw(_wander) | random walk to neighbour cell | MoveI(step='neighbour cell') | nothing |
| ContenderElement | soc/organizations/_hierarchies | RuleLaw | decay of strength, random step; if cell occupied: logistic fight, ±1 transfer | MoveI(neighbour cell, on_blocked=encounter) + discrete decay RateI(update) + pairwise rule (G2) + XferI(1 unit) | win probability |
| ResidentElement | soc/segregation/_schelling | RuleLaw | if fraction similar < threshold move to random empty cell | MoveI(step='random empty cell', condition=fraction-of-neighbours < theta) (C5) | nothing |
| BlockMovesInteraction / DwellerElement(holder) | soc/segregation/_blocks | RuleLaw | random-sequential proposal of random empty cell, accepted by Fermi on utility change from block counts | MoveI(step=random empty cell, accept='fermi'(Delta u)) (C5) + group-by count (GAP G3) | block utility, altruism mix |
| MotoristElement | eng/traffic/_drivers | RuleLaw | brake if gap to rank+1 < safe, else accelerate, clip | ordered-neighbour read (G6) + agent update | rule |
| CarelessBrakingInteraction | eng/traffic/_drivers | RuleLaw | random brake of one car, then move with no-overtaking clamp | MoveI(step=v, exclusion/order-preserving) (G6) | brake event |
| DriverElement | eng/traffic/_drivers | RuleLaw | NaSch on continuous ring: accelerate, noise, stop if > gap, move periodic | MoveI + ordered neighbour (G6) | rule |
| RoadElement | eng/traffic/_traffic | LocalRuleLaw x2 (stochastic) | NaSch cellular automaton (gap, slowdown, move) | LocalRuleLaw (CA) — exclusion process on lattice | NaSch rule |
| AdoptionInteraction (move part) | soc/opinion_dynamics/_diffusion | RuleLaw | with prob `shifting` unit step in random direction, periodic | MoveI(step='gaussian/unit random direction', probability) | — |

### I. Agent scheduling / service / life cycle / exchange

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| TaskElement (holder) + ExecutionInteraction | soc/organizations/_queues | RuleLaw | Barabasi priority queue: serve argmax priority w.p. p else random; age others; reset served; report wait | NO brief class: GAP G7 (SelectionInteraction / service policy) | priority redraw |
| RiderElement | eng/traffic/_ridesharing | RuleLaw(phase) | waited += 1 per step | discrete clock: RateI(update form, +1) | — |
| CabElement | eng/traffic/_ridesharing | RuleLaw(phase) | Manhattan step toward target, task state machine, fare transfer, remove rider | MoveI(step='toward target') + XferI(money) + LifeI(death of rider) | task machine, fare |
| ArrivalInteraction | eng/traffic/_ridesharing | RuleLaw(phase) | Poisson arrivals creating riders with random attributes, report | LifeI(birth ~ Poisson(demand), initial values) | spawn attributes |
| (factory) MatchingInteraction+StableMatchingProtocol | eng/traffic/_ridesharing | core | rider-cab stable matching | MatchI (covered) | `_nearness`, `_patience`, `_dispatch` |
| EaterElement / RestaurantElement(holder) | soc/organizations/_obesity | RuleLaw | move home->work->home, buy meal from restaurant in radius, weekly weight update | MoveI(step='go to anchor') + XferI(meal: restaurant.sold, agent.calories) + periodic schedule (GAP G8) | BMR formula, meal draw |
| EmployeeElement | soc/organizations/_sabotage | RuleLaw | random walk; expectation as weighted means; stress exponential smoothing toward lambda conditional on supervisor nearby; random rehire (reset) | MoveI + discrete Relaxation(stress, lambda_cond, alpha) + TransI/LifeI(reset w.p. turnover) | feeling/expectation model, risk sum |
| PremisesElement, TravellerElement (holders) + DailyMovementsInteraction | soc/organizations/_trafficking | RuleLaw | random relocation of one free agent, monthly moves of victims between brothels, count per cell, logistic bills, quarterly reset | MoveI x2 + group-by count (G3) + periodic schedule (G8) | city split, bills |
| GrowingWordElement(holder) + WritingInteraction | soc/spreading/_growth | RuleLaw | Simon/Yule urn: with prob mu create new member else increment one chosen ∝ count | LifeI(birth w.p. mu) + proportional selection (GAP G7) | — |

### J. Discrete-state imitation / voting / threshold adoption (agent or lattice)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| VoterElement | soc/opinion_dynamics/_voters | RuleLaw | copy random neighbour, discrete time | ImitationI(rule='random') but discrete-time agent form (C6) | — |
| MajorityVoterElement | soc/opinion_dynamics/_voters | LocalRuleLaw(TotalisticSpec) | majority (>=5 of Moore+self) CA | TransI(condition form, sum over closed nbhd >= 5) or LocalRuleLaw | — |
| VariantVoterElement | soc/opinion_dynamics/_voters | RuleLaw | random copy or threshold flip, contrarian inverts | ImitationI(random) / TransI(condition fraction disagreeing > theta) + output map 1-x if contrarian | contrarian map |
| ThresholdNodeElement | soc/spreading/_cascades | RuleLaw | Watts: inactive -> active if active fraction >= theta_i, irreversible | TransI(from 0 to 1, condition fraction >= theta) (C6) | — |
| AdoptionInteraction / AdopterElement(holder) | soc/opinion_dynamics/_diffusion | RuleLaw | count adopters within radius (all-pairs, periodic); infection: any; influence: percent > threshold | MoveI + TransI(condition on count/fraction over radius) (C6) | percent rule |
| CoevolvingVoterElement | soc/opinion_dynamics/_coevolution | UpdateLaw x2 + neighbors + uniform | degree; flip w.p. activity(1-rewiring) * discordant/degree | TransI(probability form, rate from neighbour fraction) + degree = neighbour count | — |
| ClubMemberElement | soc/opinion_dynamics/_coevolution | UpdateLaw x2 + neighbors + uniform | logit choice w.p. `deciding` on neighbour field + appeal | ChoiceI(rule='logit', utility=neighbour sum, activity=deciding) (C7) | — |
| SeparationInteraction / TieElement(holder) | soc/opinion_dynamics/_coevolution | UpdateLaw on TopologyRequestSpec('rewire') x4 | adaptive network: discordant edge rewired toward same/opposite | NO brief class: GAP G11 RewiringInteraction | toward rule |
| PrisonerElement | soc/cooperation/_spatial_games | LocalRuleLaw (stochastic, 5x5) | Nowak-May PD payoff over Moore+self, then imitate best or Fermi | CoupI(update, reduce='sum' -> payoff) + ImitationI(rule='best'|'fermi') — two stages, 2-hop (C6, G2) | payoff matrix |
| ContributorElement | soc/cooperation/_spatial_games | LocalRuleLaw (stochastic, diamond r=3) | spatial public goods: payoff summed over the 5 overlapping von-Neumann groups, Fermi imitation | group payoff = GAP G2 (groups = closed neighbourhoods) + ImitationI('fermi') | synergy r |
| StandardizationInteraction / ComplexAgentElement(holder) | soc/opinion_dynamics/_standardization | RuleLaw | random-sequential Glauber sweep, proposals among H+2 configurations, global sums updated incrementally | ChoiceI(rule='fermi'/heat-bath, update='random-sequential', utility from global aggregates) (C7) | K/J energy, configuration->spin map |
| SocialImpactInteraction / InfluenceeElement(holder) | soc/opinion_dynamics/_persuasion | RuleLaw | all-pairs distance^-a weighted persuasion/support impact; flip if impact + uniform noise > 0 | CoupI(over='all', function of both namespaces and distance) -> TransI(condition) (C3) | impact formula |

### K. Pairwise random encounters (agent)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| DiscussantElement(holder) + EncountersInteraction | soc/opinion_dynamics/_bounded_confidence | RuleLaw | Deffuant: N random pairs, compromise if |dx| < eps | NO brief class: GAP G2 PairingInteraction(sample='random pairs') ; core `random_pairing` exists but is unused | compromise rule |
| ConfidentElement | soc/opinion_dynamics/_bounded_confidence | UpdateLaw + neighbors(normalized-mean, weight, center) | Hegselmann-Krause | CoupI(form='update', reduce='normalized-mean', weight=indicator, include_self) (C3) | nothing |
| ThinkerElement(holder) + ConversationsInteraction | soc/opinion_dynamics/_cognition | RuleLaw | random speaker -> random neighbour hearer; average one component; hearer settles internal network to fixed point | G2(sample='neighbour', directed) | `settle` internal coherence dynamics |
| NamerElement(holder) + NamingGamesInteraction | soc/opinion_dynamics/_conventions | RuleLaw | naming game, random ordered pairs | G2(sample='random pairs', ordered) | naming rule |
| VillageElement | soc/opinion_dynamics/_axelrod | RuleLaw | Axelrod: random neighbour, interact w.p. similarity, copy one differing trait | G2(sample='neighbour') | trait encoding, rule |
| CultureCensusInteraction | soc/opinion_dynamics/_axelrod | RuleLaw (reports only) | measurement (largest region, number of cultures) | not an interaction: ObservableSpec/reporter | `largest_region` |
| BelieverElement | soc/opinion_dynamics/_diffusion | RuleLaw | belief histogram += learning*influence_j at each neighbour's modal level (random tie) | CoupI(form='update', reduce='sum', vector state, f = onehot(argmax x_j)*influence_j) (C3) | argmax caring |
| StrategistElement(holder) + TournamentInteraction | soc/cooperation/_tournaments | RuleLaw | ring melee of IPD games -> fitness; death w.p. d replaced by fitness-proportional offspring + bit mutation | G2(pairs = cyclic permutation) for fitness + EvoI(replacement='death-birth') (C8) | `play` (IPD), genome encoding |

### L. Group mechanisms (agent)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| CitizenElement(holder) + DiscussionsInteraction | soc/opinion_dynamics/_persuasion | RuleLaw | Galam: random partition into groups of size r; each group adopts majority (ties -> 0) | NO brief class: GAP G2 GroupingInteraction(partition='random', size=r, reduce='majority', broadcast) | tie bias |
| PersuaderElement | soc/opinion_dynamics/_persuasion | RuleLaw | Sznajd: agent + random neighbour agree -> union of both neighbourhoods adopts | G2 (pair -> neighbourhood union broadcast) | — |
| WorkplaceElement | soc/organizations/_employment | LocalRuleLaw x2 (reads 2 states) | 2-sublattice CA: workers follow employer, employer follows workers' count | LocalRuleLaw (CA) | rule |

### M. Control (symbolic feedback)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| PIDInteraction | eng/control/_pid | RateLaw(integrator), BoundaryLaw(control input) | PID writes adaptive parameter `force` after each step, saturated | no brief class for SYMBOLIC feedback (C9): RateI(integral)+ assignment of input | gains |
| StateFeedbackInteraction | eng/control/_state_feedback | BoundaryLaw | u = -K x, saturated | C9 | gains (from LQR) |
| IntegratorElement | eng/control/_integrator | — (holder) | integral state for PID | absorbed by the feedback class (it owns its integral state) | — |

## 2. GAPS — mechanisms no general class of the brief covers

G1. **Relaxation toward a target (first-order lag).** `x' = (target(ns) - x)/tau`, discrete form `x += alpha (target - x)`.
Used by: MotivationInteraction, OptimalVelocityInteraction, LangevinDrivingInteraction (noisy), Payne/KK sources,
RiceMill/Waste capacity, RoomElement leak, EmployeeElement stress, ArmsRace build (one-sided). Brief only has it implicitly
as RateI or OU with constant mean. Proposal: `RelaxationInteraction(state, target, time, noise=None, form='rate'|'update')`
as a thin RateI subclass; OU becomes its noisy special case. Requires renaming the current core `RelaxationLaw`
(sandpile cascades) to e.g. `CascadeLaw` to free the name.

G2. **Pairing and grouping of members (random or structured), with a rule applied jointly to the pair/group.**
- Pairs: EncountersInteraction (random pairs), NamingGamesInteraction (random ordered pairs), ConversationsInteraction
  (random member + random network neighbour), VillageElement (neighbour), TournamentInteraction (cyclic permutation),
  ContenderElement (contact on move), PersuaderElement (pair then broadcast). Every one re-implements pair drawing; the core
  `random_pairing` helper is unused.
  Proposal: `PairingInteraction(members, rule, sample='random'|'neighbour'|'permutation'|'contact', count=N, ordered=False,
  form='sequential'|'synchronous')` lowering to agent RuleLaw; `rule(first, second, model)`.
- Groups: DiscussionsInteraction (Galam: random partition into size-r groups, majority broadcast), ContributorElement (public
  goods: every closed von-Neumann neighbourhood is a group, member payoff sums over all groups it belongs to),
  PersuaderElement (neighbourhood union). Out of scope here but same shape: Kolkata (group by chosen restaurant),
  minority game (group by chosen side). Proposal: `GroupingInteraction(members, groups='random-partition'|'neighborhood'|
  'hyperedge'|'by-state', size, reduce, act)`: form groups, reduce a state inside each group (sum, mean, count, majority),
  and write back to every member (broadcast) or accumulate per member over all groups it is in (public goods). This is
  what `HyperedgeInteraction` should become once hyperedges can be dynamic (random partition per step) and derived
  (closed neighbourhoods, equal values of a state).

G3. **Group-by reductions (count/sum of members sharing a key).** BlockMovesInteraction (counts per block),
DailyMovementsInteraction (counts per cell), AdoptionInteraction (count within radius), plus Kolkata/minority outside scope.
Core reductions (`count`, `total`, `mean`) reduce over a whole group or neighbourhood, never keyed. Proposal: a `by=`
argument on the reductions (`count(over=group, by=state_or_cell)`), used by GroupingInteraction(groups='by-state').

G4. **Bilinear (binary-encounter) transitions on a discrete distribution.** OpinionEncounters, CriminalContagion,
Deterrence, Walking, Crowding all use `EncounterLaw(candidate, field, eta, A)`: `df_i = sum eta_hk A_hki f_h g_k -
f_i sum eta_ik g_k`. None of TransI/ReacI in the brief takes a tensor of outcomes over a density. Mathematically it is mass
action of reactions `h + k -> i + k`, so propose either `ReactionInteraction(..., table=A, rate=eta)` (tensor form on a
DensityElement) or a dedicated `BilinearTransitionInteraction(candidate, field, rate, table, test=None)` lowering to the
renamed law (see flags). Walking is the linear case (field constant) = a Markov generator on the density, i.e. TransI on
a density.

G5. **Conservation law with a flux (finite volume).** LWR, Payne, KK use FluxLaw (Godunov / Riemann). The brief lists no
general class. Propose `ConservationInteraction(state, flux, riemann=None)` (`u_t + div F(u) = 0`), sources added by
RateI/RelaxationInteraction, viscosity by a diffusion RateI.

G6. **Ordered-neighbour (follow-the-leader) coupling and exclusion movement.** OV/IDM/Langevin use
`shift_member(x, 1)` with periodic modulo; DelayedFollowing pairs leader/follower explicitly; Motorist/Driver look up
`rank+1` by scanning all agents; CarelessBraking clamps to keep order; RoadElement is a lattice exclusion CA. Proposals:
(a) CoupI must accept `over='next'` (directed chain/ring by order, with period) and a function of both namespaces (see C3);
(b) MoveI needs `exclusion=True` (order-preserving / hard-core steps). The agent `rank` lookup should be a view helper
(`agent.next()`), not O(N) scans in every model.

G7. **Selection of one or few members by a policy (service / proportional choice).** ExecutionInteraction (argmax
priority w.p. p, else uniform; served member reset; others age), WritingInteraction (choose ∝ count, Simon urn),
TournamentInteraction (roulette ∝ fitness). Queueing in the core exists only as a matching protocol (QueueProtocol for
requester/provider rationing), not as a service of a single queue. Proposal: `SelectionInteraction(members, policy=
'max'|'random'|'proportional'|'mixture', key, count=1, act)` — select k members per step by a key and apply `act`
(serve/reset, increment, replicate). Simon urn = SelectionInteraction(proportional, act=+1) + LifeI(birth w.p. mu).

G8. **Time-triggered (periodic) events and schedules.** KickedCircuitElement (clock state with rate 1 + CrossingLaw reset
every T), EaterElement (`day % 7`), DailyMovementsInteraction (`day % 30`, `% 90`), plus `pulse_train` loads. Each model
keeps its own day counter state. Proposal: (a) `EventInteraction(condition, assignments, direction)` wrapping CrossingLaw
for impulses/resets, with a `period=` shortcut that creates the clock; (b) an `every=k` (and offset) on agent RuleLaw /
phases so periodic agent rules need no `day` state.

G9. **Symbolic feedback control (continuous) with saturation and integral state.** PIDInteraction, StateFeedbackInteraction,
inline PID in ControlledProcessElement, saturated proportional heating in RoomElement, bang-bang laws in Satellite and
SwingingPendulum. Brief's FeedbackInteraction is only the agent once-per-round actor. Proposal: symbolic
`FeedbackInteraction(input, law, limits=None, integral=None)`: an algebraic control law `u = law(ns)` (clipped), substituted
into the plant's rates, with an optional integral state it owns (removes IntegratorElement). PID, state feedback, relay
(bang-bang = sign law) are thin subclasses. Currently the input is written by `BoundaryLaw` after each step onto an
adaptive parameter: semantically a boundary, numerically a one-step lag, and the derivative term `derivative(measured,1)`
inside a boundary value. Needs a core *algebraic/assignment law* (auxiliary variable) — none exists (EquationLaw requires
highest derivatives).

G10. **Confinement/boundaries of a region.** LightSensitiveRobot clamps radially to a disk with two BoundaryLaws;
SwarmRobot clamps to the arena in Python; pedestrians use soft walls. Brief has no boundary class. Proposal:
`ConfinementInteraction(state, region, mode='clamp'|'reflect'|'periodic')` lowering to BoundaryLaw (soft walls stay GradI).

G11. **Adaptive topology (rewiring / edge creation-deletion).** SeparationInteraction (UpdateLaw on TopologyRequestSpec
'rewire', keep end, toward predicate). No brief class. Proposal: `RewiringInteraction(edges, probability, keep='source'|
'target', toward=predicate)` on NeighborInteraction.

G12. **Pointwise min/max/clip/sign symbolic helpers.** `_smaller`, `_positive`, `_limited`, `_sign`, `_dry_friction` are
re-written with `where` in processes/_systems, control/_utils, control/_control, power_systems/_electromechanics. Core
`maximum`/`minimum` are member reductions, not pointwise. Add pointwise `clip`, `sign`, `positive_part` (or document
Sage `min_symbolic/max_symbolic`) in core law utils.

## 3. BRIEF CORRECTIONS

C1. **RateInteraction cannot express mass-matrix coupled mechanics.** CartPendulumElement couples x'' and theta'' implicitly;
CircuitElement/KickedCircuit write `L q'' + R q' + q/C = V` as EquationLaw. Keep `EquationLaw` usable as an element's own law
(it is a general law) and say so in the brief, or add `form='implicit'` to RateI.

C2. **TransferInteraction must have a symbolic form, not only agent RuleLaw.** All stock-and-flow models (Tub, RiceMill, Waste,
Relief, Irrigation, Contest, Burden, ArmsRace) use symbolic `TransferLaw(Source|state, state|Sink, rate)`; PassageInteraction
writes the same flow as two RateLaws (losing conservation and the flow label). XferI(source, target, amount) must lower to
`TransferLaw` when states are symbolic and to `transfer()` RuleLaw for agents (ridesharing fare, Contender ±1, Eater meals).

C3. **CouplingInteraction's `function(x_j, x_i)` of ONE state is too narrow.** Needed: (a) function of both members'
namespaces (IDM/GM read x and v of both ends; SocialImpact reads persuasiveness_j, supportiveness_j; Believer reads
influence_j and argmax of a vector state); (b) a separate `weight` and `include_self` (HK `normalized-mean` with indicator
weight, center=True); (c) distance `d_ij` as an argument (SocialImpact d^-a, pair potentials); (d) edge weights (doubled
transmission line K); (e) `over='next'` ordered topology with periodic length (car following); (f) for order-2 states add
to the 2nd derivative (swing-equation Kuramoto); (g) agent form (SwarmRobot neighbour count, Believer) besides symbolic.

C4. **OrnsteinUhlenbeck / RateInteraction noise must accept expressions**: Langevin traffic has target V(rho(headway)) and
multiplicative sigma sqrt(2 D rho). Brief OU has constant mean/sigma.

C5. **MovementInteraction(step) lacks condition and acceptance.** Schelling moves only if fraction-similar < theta; block
segregation accepts a random-empty-cell proposal with Fermi probability on a utility change; adoption moves w.p. `shifting`;
Contender needs `on_blocked`; car-following/NaSch need exclusion. Add `condition=`, `accept=` (rule + intensity),
`probability=`, `exclusion=`, and step kinds 'random empty cell', 'toward target', 'go to anchor'.

C6. **TransitionInteraction(state, from_, to, rate) is continuous-time only; Imitation lacks Fermi and discrete time.**
Watts cascades, Adoption, VariantVoter, MajorityVoter, CoevolvingVoter, SocialImpact flip are discrete-time and either
deterministic on a condition (fraction >= theta_i, per-member threshold state) or w.p. per step. Add
`form='rate'|'probability'|'condition'`, reads of neighbour fractions/counts over network, lattice or radius, and
per-member thresholds. ImitationInteraction needs `rule='fermi'` (spatial games) and a discrete-time/agent form (Voter).
Spatial games also need a two-stage step (payoff by CoupI(update) over 2-hop, then imitation) in one time step.

C7. **ChoiceInteraction needs schedule and activity.** Brief/core: "every member draws at once". Standardization is a
random-sequential Glauber sweep with incrementally updated global sums; ClubMember revises only w.p. `deciding`; Block
moves are sequential. Add `update='synchronous'|'random-sequential'`, `activity=` (revision probability), and utilities that
read group aggregates (mean field) so sequential updates see the latest values; allow many options mapping to fewer
values (H+2 configurations -> spins).

C8. **EvolutionInteraction is generational GA only.** Tournament is a Moran-like death-birth process (die w.p. d, replaced by
∝-fitness offspring with one-bit flip) with fitness from a pairwise game. Add `replacement='generational'|'death-birth'`
and let fitness come from a PairingInteraction stage.

C9. **FeedbackInteraction exists only in agent form.** See G9: add the symbolic form (algebraic input law with saturation
and owned integral state). The current PID/state-feedback realization via BoundaryLaw on an adaptive parameter is a
misuse of BoundaryLaw's meaning.

C10. **LifeCycleInteraction(birth, death) needs Poisson batch births with initial-value samplers** (ArrivalInteraction
creates Poisson(demand) riders with random location/destination) and removal triggered by another rule (rider removed
at pickup).

C11. **ReactionInteraction propensity must be a free expression of counts** (Maki-Thompson stifling uses
`spreaders*(spreaders-1+stiflers)/N`, not mass action) — brief says "any propensity": confirm it covers this.

C12. **ParticleElement**: soc/crowds social_force uses `ParticleElement` for pedestrians — evidence the brief is right that
"point particle with position, velocity, mass" is physics-named; it is a generic order-2 vector state holder
(`Element` with `x: StateSpec(order=2)`).

## 4. FIELD-SHARED (`<field>/common/<family>`)

engineering:
- `engineering/common/control`: PIDInteraction, StateFeedbackInteraction (now in control only, but RoomElement and
  ControlledProcessElement re-implement P/PID inline, Satellite/Pendulum re-implement relay control) — thin subclasses of
  symbolic FeedbackInteraction (G9); plus `_limited`/`_sign` move to core utils (G12).
- `engineering/common/mechanics` (or `vibration`): two-mass spring-damper coupling shared by PiezoHarvesterElement
  (circuits) and RailSegmentElement (power_systems): `SpringDamperInteraction(first, second, k, b)` = CoupI(potential=k r^2/2)
  + relative damping; dry friction `_dry_friction` shared by RegenerativeCar and FaradayDisk (power_systems) -> one
  `DryFrictionInteraction` (RateI with sign) or core `sign`.
- `engineering/common/stock_flow`: `delay1/delay3/smooth/smooth3` (processes/_delays.py, module functions — against the API
  goal) are used by processes/_systems and the same distributed_delay calls appear in social/organizations/_traps:
  cross-field, so they belong in core law utils (`distributed_delay` already there) or are dropped as aliases; the
  `_smaller/_positive/_season` helpers likewise.
- `engineering/common/traffic` is not needed (single package), but `_headway` (shift_member mod L) is shared by
  _car_following and _langevin -> becomes CoupI(over='next', period) (C3e).

social:
- `social/common/kinetic`: grid builders and table builders shared by crowds (_criminality, _evacuation) and opinion_dynamics
  (_kinetic_opinion): `contagion_table` and `persuasion_table` are the same "step one level toward (or away from) the field
  level with probability p (within confidence)" tensor; `mean_*`/`*_fraction`/`*_variance` observables are all
  `moment(f, k, grid)`; holders `OpinionLevelElement/CitizensElement/SecurityAgentsElement/PedestriansElement` are the same
  DensityElement. One `NearestLevelEncounterInteraction` (thin subclass of G4 class) + one density holder.
- `social/common/imitation`: `cooperation/_imitation.imitate` (public module function, against the API goal) duplicates
  `_imitate_best` in _spatial_games and VoterElement/VariantVoterElement random copy; Fermi imitation `_imitate_fermi` is
  shared by PrisonerElement and ContributorElement -> ImitationI rules (C6); `social/cooperation/types.Score` goes with it.
- `social/common/encounters`: random pairing loops of EncountersInteraction, NamingGamesInteraction, ConversationsInteraction,
  VillageElement, TournamentInteraction (opinion_dynamics + cooperation) -> PairingInteraction (G2) subclasses.
- `social/common/threshold`: fraction-of-neighbours threshold rules in ThresholdNodeElement (spreading), AdoptionInteraction
  and VariantVoterElement (opinion_dynamics), ResidentElement (segregation, as move condition) -> TransI(condition) (C6).
- `social/common/movement`: `_wander` (random neighbour-cell step) in organizations/_sabotage, the same in
  _hierarchies.ContenderElement, `random_empty_cell` moves in segregation (_schelling, _blocks) -> MoveI realizations.
- `_shares`/`_random_cells`/`_random_strategies` (Bernoulli initial arrays) repeated in _persuasion, _voters, _spatial_games.

## 5. Flags: field-flavoured core names or meanings

- `EncounterLaw` ("conservative encounter of active particles... candidate/field/test particle", "Bellomo's gain and
  loss"), `DestructionLaw`, `ProliferationLaw`: kinetic-theory vocabulary; operation is a bilinear transition on a discrete
  distribution -> e.g. `BilinearTransitionLaw` with roles `source`, `field`, `target`.
- `FluxLaw` ("physical flux"): operation is a conservation law -> `ConservationLaw` / docstring without "physical".
- `RelaxationLaw` / `DriveLaw` ("dropping a grain"): sandpile vocabulary; RelaxationLaw also blocks the natural name of G1 ->
  `CascadeLaw`, and DriveLaw docstring example only.
- `ThresholdInteraction` (core) means sandpile firing/redistribution, while social "threshold" models (Watts, Granovetter)
  mean fraction-of-neighbours adoption: name clash; core class is a cascade/toppling interaction.
- `ParticleElement` (physics meaning; used for pedestrians).
- `PotentialInteraction(position='x', velocity='v', mass='m')`: argument names are physical quantities.
- `ConstantForceInteraction`, `DampingInteraction`: force/damping names (brief already lists them as RateI special cases).
- `PriceAdjustmentProtocol`, `PriceTimePriorityProtocol`, `UniformPriceProtocol` (+ `_Offer.budget`): economics-named
  matching protocols in core (operation: tatonnement / priority queue by limit / uniform clearing value).
- `ForagingInteraction`, `RegrowthInteraction`, `SpreadingInteraction` (active/removed = epidemics), `ImitationInteraction`
  (`payoff`) carry field meanings; `MatchingInteraction` docs use requester/provider (acceptable).
- `StateSpec(quantity='water'|'grain')` in processes is fine (user binding), but `TransferLaw` labels like `Source('farms')`
  are model text, fine.
- Model-side: `CultureCensusInteraction` is a measurement, not an interaction (should be an ObservableSpec/reporter);
  `IntegratorElement` exists only to hold the PID integral; `cooperation.imitate` and `processes.delay*` are isolated public
  functions outside classes.
