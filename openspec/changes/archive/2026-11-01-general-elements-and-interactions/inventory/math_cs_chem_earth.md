# Audit: mathematics, computer_science, chemistry, earth

Scope: every element / interaction / law class under `src/skcomplex/models/{mathematics,computer_science,chemistry,earth}`
(views, specs, kernels, compilers, visuals listed only where they carry a mechanism). Paths below are relative to
`src/skcomplex/models/`. `s.` = the namespace of the element.

Headline facts:

- **No model in these four fields uses any core interaction.** Every mechanism is an element's own `_get_laws`
  (RateLaw / UpdateLaw / JumpLaw / NoiseLaw / LocalRuleLaw / RuleLaw / TransitionLaw / TopologyRequest updates), or a
  model-local interaction (`BindingInteraction`, `DissociationInteraction`, `RewiringInteraction`,
  `ReproductionInteraction`, a second `EvolutionInteraction`). Several of these duplicate an existing core class.
- **Name clash:** `computer_science/evolutionary_computation/_genetic_algorithms.py::EvolutionInteraction` shadows
  `core.interactions.EvolutionInteraction` and reimplements it (rank selection, one-point crossover, per-gene mutation).
- **Exact duplicates of core:** `FaultElement` (OFC) re-implements `core ThresholdInteraction(mode='reset')`;
  `SpeciesElement` == `CompartmentElement` (state `n`); `Binding/DissociationInteraction` + `_reaction_laws` ==
  `ReactionInteraction(mode='rate'|'jump'|'langevin')`; `DiffusionCellElement` == discrete diffusion
  (`CouplingInteraction` update form); `NodeContentElement` == network diffusion `CouplingInteraction`.
- **A whole problem form lives in a model package:** `mathematics/fractals` defines its own laws (`ProductionLaw`,
  `SubstitutionLaw`), compiler (`RewritingCompiler`), solver (`RewritingSolver`), kernels and structure specs.

Legend for "built from": `Rate(state, rate, noise)` = RateInteraction; `Coupling(...)` = CouplingInteraction;
`Reaction(...)` = ReactionInteraction; `Transition(...)` = TransitionInteraction; `Choice(...)`; `LifeCycle(...)`;
`Movement(...)`; `Transfer(...)`; `Evolution(...)`; `Threshold(...)` = existing core ThresholdInteraction. A class
marked **NEW:** is a proposed class from section GAPS.

## 1. Continuous vector fields (element's own RateLaws, order-1 states, deterministic)

These are the canonical "element carrying its own laws" case: the vector field IS the model. Building them out of
`Rate(...)` interactions adds nothing except where a term is a reusable mechanism (growth, decay, reaction, coupling).

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| LorenzElement | mathematics/dynamical_systems/_flows.py | RateLaw x3 | polynomial ODE | Element + own RateLaws | whole Lorenz field |
| RosslerElement | mathematics/dynamical_systems/_flows.py | RateLaw x3 | polynomial ODE | Element + own RateLaws | whole field |
| BlowUpElement | mathematics/dynamical_systems/_flows.py | RateLaw | power growth `a x^p` | GrowthInteraction(form='power', r=a, p) on state x | nothing (pure realization) |
| SlowManifoldElement | mathematics/dynamical_systems/_koopman.py | RateLaw x2 | linear + relaxation to `x1^2` | Rate(x1, mu x1) (exponential growth); Rate(x2, lam (x2 - x1^2)) | the x1^2 manifold term |
| SaddleNodeElement | mathematics/dynamical_systems/_normal_forms.py | RateLaw | normal form `mu - x^2` | Rate(x, polynomial) | polynomial |
| HopfElement | mathematics/dynamical_systems/_normal_forms.py | RateLaw x2 | normal form (Cartesian) | Rate on (x,y) | normal form |
| TakensBogdanovElement | mathematics/dynamical_systems/_normal_forms.py | RateLaw x2 | normal form, order-2 in disguise | Rate(x, order=2 state) possible: x'' = b1 + b2 x + x^2 + s x x' | normal form |
| CuspElement | mathematics/dynamical_systems/_normal_forms.py | RateLaw | cubic `b1 + b2 x - x^3` | Rate(x, cubic) | cubic (shared with earth folds, see FIELD-SHARED) |
| SubcriticalHopfElement | mathematics/dynamical_systems/_subcritical_forms.py | RateLaw x2 (helper `_subcritical_hopf_rates`) | quintic Hopf normal form | Rate on (x,y) | normal form; helper should be a private method (memory: helpers as methods) |
| JerkElement | mathematics/dynamical_systems/_flows.py | RateLaw on `StateSpec(order=3)` reading `derivative(x,1..2)` | 3rd-order ODE | Rate(x, ...) on order-3 state | the jerk function |
| relaxation_type / oscillator_type / van_der_pol_type / lorenz_type (`FractionalElement` x4, factory-generated classes) | mathematics/fractional_dynamics/_fractional.py | RateLaw on `StateSpec(order=q)` non-integer | Caputo fractional ODEs | same RateLaws as integer versions, `StateSpec(order=q)` | nothing: should be the integer elements with a fractional `order` argument; class factories (`@cache`, `_published`) are machinery to remove |
| BrusselatorReactionElement | chemistry/kinetics/_oscillators.py | RateLaw x2 | mass-action scheme A->X, B+X->Y+D, 2X+Y->3X, X->E | Reaction x4 (mode='rate') on states u, v of ONE member (needs correction C1) | nothing |
| OregonatorElement | chemistry/kinetics/_oscillators.py | RateLaw x3 | scaled mass-action (FKN) | Reaction x5 with time-scale divisors in propensity | scaling eps, delta |
| CubicAutocatalysisElement | chemistry/kinetics/_oscillators.py | RateLaw x2 | A+2B->3B, B->C, CSTR feed/outflow | Reaction(A+2B->3B), Reaction(B->0), Rate(a, F(1-a)) feed, Rate(b, -F b) outflow | nothing (== Gray-Scott kinetics) |
| CatalystPelletElement | chemistry/kinetics/_reactors.py | RateLaw | feed + Langmuir-Hinshelwood consumption | Rate(c, h(1-c)) + Reaction(propensity=k c/(1+K c)^2) | LH propensity |
| PulsedElectrodeElement | chemistry/kinetics/_reactors.py | RateLaw x2, `pulse_train(t)` | Butler-Volmer + pulsed forcing | Rate(eta, (I(t) - faradaic)/C), Rate(c, relaxation - kappa faradaic) | Butler-Volmer current |
| IceAlbedoClimateElement | earth/climate/_energy_balance.py | RateLaw | 0-D energy balance | Rate(T, absorbed - emitted) | albedo tanh, T^4 |
| DaisyworldElement | earth/climate/_energy_balance.py | RateLaw x2 (helper `_daisyworld_laws`) | competitive logistic growth on shared area + death | GrowthInteraction-like Rate(white, w (bare g(T_w) - gamma)) per species; competition through `bare` | temperature-dependent growth g |
| SaltzmanMaaschElement | earth/climate/_ice_ages.py | RateLaw x3 (helper) | forced polynomial ODE | Rate x3, forcing reads t | field + orbital forcing |
| VanDerPolIceAgesElement | earth/climate/_ice_ages.py | RateLaw x2 (helper) | forced relaxation oscillator | Rate x2 | VdP field (shared with fractional VdP) |
| CarbonCycleElement | earth/climate/_ice_ages.py | RateLaw x2 (helper) | Hill-saturation fluxes | Rate x2; could be Transfer(continuous) between c and w | Hill functions |
| Lorenz84Element | earth/climate/_low_order.py | RateLaw x3 | polynomial ODE | own RateLaws | field |
| Lorenz96Element | earth/climate/_low_order.py | RateLaw on vector state (helper) | cyclic shift coupling `(x_{i+1}-x_{i-2}) x_{i-1} - x_i + F` | Coupling on a ring with ORDERED neighbours (needs NEW LocalMap / shift_member) + Rate(x, -x + F) | advection term |
| GeodynamoElement | earth/climate/_low_order.py | RateLaw x3 | polynomial ODE | own | field |
| StommelBoxesElement | earth/oceanography/_thermohaline.py | RateLaw x2 | two-box exchange with \|T-S\| flow | Rate x2 (relaxation + flow); could be Transfer between boxes | flow law |
| EnergySectorElement | earth/climate/_net_energy.py | TransferLaw x4 (Source/Sink), RateLaw | stock-flow with EROI | Transfer(continuous, source/sink env) x4 + Rate(capacity, reinvest - depreciation) | EROI curve (needs correction C5) |
| SpringBlockElement (on FieldElement) | earth/seismology/_spring_blocks.py | RateLaw x2 + `laplacian` (helper) | Burridge-Knopoff stick-slip chain | Coupling(x, edges, f=x_j-x_i, K=k_c) into the v rate + Rate(v, plate pull - friction) | velocity-weakening friction and stick condition (`where(stuck, 0, ...)`) |

## 2. Stochastic and delay differential equations

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| DrivenRosslerElement | mathematics/dynamical_systems/_driven_flows.py | RateLaw x4, NoiseLaw | Rossler driven by OU colored noise eta | Rossler RateLaws + OrnsteinUhlenbeckInteraction(eta, theta=1/tau, sigma=rho/tau) | Rossler field |
| IntermittentVariableElement | mathematics/stochastic_processes/_intermittency.py | RateLaw, NoiseLaw(convention='stratonovich') | multiplicative noise `sqrt(2D) x dW` | Rate(x, eps x - x^3, noise=sqrt(2D) x) — needs `convention` (C2) | none |
| WeakeningParticleElement | mathematics/stochastic_processes/_walks.py | RateLaw, NoiseLaw | Langevin in potential with weakening restoring force | Rate(x, -k x/(1+(x/l)^2), noise=sqrt(2D)) or Gradient(x, U) | force law |
| ClimateAnomalyElement | earth/climate/_variability.py | RateLaw, NoiseLaw | OU process | OrnsteinUhlenbeckInteraction(T, lam, sigma) | nothing (pure realization) |
| StochasticEnergyBalanceElement | earth/climate/_tipping_types.py | RateLaw x3, NoiseLaw | SDE with control parameters as drifting/logistic states | Rate(T, quartic, noise), Rate(mu, -drift), GrowthInteraction(lam, logistic) | quartic balance |
| RampedFoldElement | earth/climate/_tipping_types.py | RateLaw, NoiseLaw, `ramp(t)` | rate-induced tipping of cubic fold | Rate(x, cubic + ramp(t), noise=sigma) | ramp control |
| DelayedOscillatorElement | earth/climate/_variability.py | RateLaw with `delayed(T, delta)` | delay DE (ENSO) | Rate(T, T - T^3 - alpha delayed(T, delta)) — rate fn must read `delayed` | nothing |
| SchloglElement | chemistry/kinetics/_reactions.py | RateLaw + NoiseLaw x4 (helper `_schlogl_laws`) | chemical Langevin of 4 reactions, buffered A, B | Reaction x4 (mode='langevin', propensity=c1/2 A x(x-1) etc.) on the same state x of one element | nothing; buffered species as parameters need correction C1 |
| TippingElementElement (FieldElement on network) | earth/climate/_tipping.py | RateLaw + `laplacian` + degree | cascading folds on a network | Rate(x, -x^3 + x + FOLD c) + Coupling(x, edges, f=(x_j+1), K=d) [laplacian(x)+deg(x+1) == sum_j (x_j+1)] | fold |

## 3. PDE / field laws (FieldElement + RateLaw with spatial operators)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| NodeContentElement | mathematics/networks/_processes.py | RateLaw D laplacian(u) | network diffusion | Coupling(u, edges, f=x_j-x_i, K=D) | nothing (pure realization; delete) |
| GrayScottElement | chemistry/pattern_formation/_reaction_diffusion.py | RateLaw x2 + laplacian | reaction-diffusion | Coupling(u, Du) + Coupling(v, Dv) + cubic-autocatalysis Reaction set (== CubicAutocatalysisElement) | nothing |
| BrusselatorElement | chemistry/pattern_formation/_reaction_diffusion.py | RateLaw x2 + laplacian | reaction-diffusion | Coupling x2 + Brusselator Reaction set (== BrusselatorReactionElement) | nothing |
| SchnakenbergElement | chemistry/pattern_formation/_reaction_diffusion.py | RateLaw x2 + laplacian | reaction-diffusion | Coupling x2 + Reaction(0->u, u->0, 2u+v->3u, 0->v) | nothing |
| LatitudeBandElement | earth/climate/_energy_balance.py | RateLaw + laplacian + gradient (helper) | diffusion on the sphere `d/dx((1-x^2) dT/dx)` | Coupling(T, edges, f=x_j-x_i, weight depending on edge midpoint coordinate (1-x^2)) — needs C3 + Rate(T, (Q(x)(1-a(T)) - A - BT)/C) | insolation, albedo |

## 4. Deterministic discrete-time maps (UpdateLaw, element's own)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| LogisticMapElement | mathematics/dynamical_systems/_maps.py | UpdateLaw | 1-D map | NEW Map(x, r x (1-x)) | map |
| HenonMapElement | ".../_maps.py" | UpdateLaw x2 | 2-D map (simultaneous) | NEW Map((x,y), ...) | map |
| StandardMapElement | ".../_maps.py" | UpdateLaw x2 | area-preserving, theta uses NEW p (sequential, recomputed inline) | NEW Map((p,theta)) | map; shows UpdateLaws are simultaneous so sequential maps inline the composition |
| CircleMapElement | ".../_maps.py" | UpdateLaw, `mod` | circle map | NEW Map on periodic state | map |
| TentMapElement | ".../_maps.py" | UpdateLaw, `where` | piecewise-linear | NEW Map | map |
| BakerMapElement | ".../_maps.py" | UpdateLaw x2 | piecewise map | NEW Map | map |
| TowelMapElement | ".../_maps.py" | UpdateLaw x3 | 3-D map | NEW Map | map |
| IntermittencyMapElement | ".../_maps.py" | UpdateLaw | Pomeau-Manneville | NEW Map | map |
| NewtonMapElement | ".../_maps.py" | UpdateLaw x2 | Newton iteration of z^3-1 | NEW Map | map |
| CatMapElement | ".../_maps.py" | UpdateLaw x2 | toral automorphism | NEW Map (linear on torus) | matrix |
| CubicWalkElement | mathematics/dynamical_systems/_chaotic_ensembles.py | UpdateLaw x2 | chaotic map + deterministic walk `y += 2x-1` | NEW Map(x) + NEW Map(y, increment) | map |
| SkewTentMapElement | ".../_chaotic_ensembles.py" | UpdateLaw | piecewise map | NEW Map | map (generalizes TentMap) |
| DoublingMapElement | ".../_chaotic_ensembles.py" | UpdateLaw | discretized doubling map | NEW Map | map |
| QuadraticMapElement (FieldElement over parameter grid c) | mathematics/fractals/_quadratic.py | UpdateLaw x2 | z -> z^2+c with escape count | NEW Map(z) + NEW Map(n, increment while bounded) | escape test |
| SoilTemperatureElement | earth/climate/_soil.py | UpdateLaw x2 | coupled logistic map | NEW Map | map (logistic map reused) |
| NodeRankElement | mathematics/networks/_processes.py | UpdateLaw + laplacian(rank/degree) | PageRank power iteration `(1-d)/N + d sum_j r_j/k_j` | Coupling(rank, edges, form='update', f=x_j/deg_j) — needs C3 (endpoint degree) + teleport term | damping |
| ReservoirUnitElement | computer_science/neural_networks/_echo_state.py | UpdateLaw + `total(r, weights=W)` + sin(t) | leaky tanh recurrent map | Coupling(r, form='update', weights=W matrix, reduce=sum, then tanh) — needs C3 (weight matrix not graph) | leak + tanh + input |

## 5. Random maps and per-member random draws in discrete time

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| ChaosPointElement | mathematics/fractals/_chaos_game.py | UpdateLaw + `categorical` | random iterated function system | NEW Map(p, random=categorical(weights)) | the affine maps |
| PersistentWalkerElement | mathematics/stochastic_processes/_walks.py | UpdateLaw x2 + `bernoulli('turn')` shared by name | correlated random walk | NEW Map((d,x)) with one named draw shared by both laws | nothing |
| MemoryWalkerElement | ".../_walks.py" | UpdateLaw x2 + `normal` | AR(1) velocity walk | NEW Map(v, alpha v + noise) + NEW Map(x, x+v) | nothing |
| DilutedNeuronElement | computer_science/neural_networks/_diluted.py | UpdateLaw + `total(spin, weights=J)` + `uniform` | parallel Glauber dynamics | Choice(spin, options=(-1,1), utility=sum_j J_ij s_j * s, rule='logit', intensity=beta) — needs weighted neighbour sum in utility (C3/C4) | sparse couplings J |
| BernoulliTrialElement | mathematics/stochastic_processes/_counting.py | RuleLaw | iid Bernoulli draws + success count | NEW Map(outcome, bernoulli(p)); Map(successes, increment) — no agent needed | nothing |
| ReducingBallElement | mathematics/stochastic_processes/_reducing.py | RuleLaw | sample-space-reducing process (draw uniform below current, restart) | NEW Map(level, random integer with state-dependent support) | restart rule |
| AgeingWalkerElement | ".../_walks.py" | RuleLaw | walk with up-probability `p/(1+ups)^nu` | NEW Map((x, ups), bernoulli(state-dependent p)) | ageing probability |

## 6. Continuous-time Markov jumps (JumpLaw) and discrete-time chains (TransitionLaw)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| PoissonCounterElement | mathematics/stochastic_processes/_counting.py | JumpLaw | Poisson process | Reaction(reactants=[], products=[counter], k=rate, mode='jump') | nothing |
| QueueElement | mathematics/stochastic_processes/_queues.py | JumpLaw x2 | M/M/1 birth-death | Reaction(0->n, k=lambda, jump) + Reaction(n->0, propensity=mu*[n>0], jump) | nothing |
| RandomWalkerElement | ".../_walks.py" | JumpLaw x2 | CTRW on Z (+1/-1) | needs NEW `JumpInteraction(state, changes, rate)` — Reaction forbids negative counts (x on Z) | nothing |
| PolyaUrnElement | mathematics/stochastic_processes/_urns.py | JumpLaw x2 | reinforcement draw: +added of drawn colour with prob share | Reaction(red->(1+added) red, propensity=red/total) — needs integer stoichiometry (C1) | nothing |
| MarkovSourceElement | mathematics/stochastic_processes/_sources.py | TransitionLaw | discrete-time Markov chain by matrix | Transition(state, matrix=P, time='discrete') — brief's signature is rate-based (C4) | nothing |
| HiddenMarkovSourceElement | ".../_sources.py" | TransitionLaw + EmissionLaw | HMM | Transition(hidden, P) + NEW `EmissionInteraction`/conditional draw (symbol | hidden, E) | nothing |

## 7. Agent rules (RuleLaw) — birth/death, movement, no-op

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| IndividualElement | mathematics/stochastic_processes/_branching.py | RuleLaw (create Poisson(m) children with generation+1, remove self) | Galton-Watson branching | LifeCycle(birth=offspring count ~ Poisson(m), inherit={generation: +1}, death=always) — needs C6 | nothing |
| FragmentElement | mathematics/stochastic_processes/_reducing.py | RuleLaw (split length at random cut, child gets remainder) | fragmentation | LifeCycle(split=fn(parent)->(parent_state, child_state)) — needs C6 | cut law |
| NetworkWalkerElement | mathematics/networks/_processes.py | RuleLaw (move to random neighbour cell) | random walk on graph | Movement(step='neighbour') | step counter |
| NeuronElement | computer_science/neural_networks/_hopfield.py | RuleLaw (spin = sign(sum_j w_j s_j) over graph neighbours, agents run sequentially) | asynchronous Hopfield | Coupling(spin, edges, form='update', reduce=sum, weight=edge weight, then sign, schedule='sequential') — needs C3 + C4 | Hebbian weights |
| NetworkNodeElement | mathematics/networks/_structure.py | RuleLaw `stay` (no-op) | none: a no-op rule only to get the agent trait / drawing | Element without laws | FLAG: no-op law is a workaround |
| NetworkMeasuresElement | mathematics/networks/_structure.py | none | static parameter holder | Element | measures belong in an analysis, not an element |
| LinkElement | mathematics/networks/_growth.py | none | edge group placeholder | Element | — |

## 8. Network topology change (UpdateLaw on `TopologyRequestSpec`)

No brief class covers this.

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| FailingElement | mathematics/networks/_ensembles.py | UpdateLaw(degree, neighbors(1)), UpdateLaw(TopologyRequest('remove-node'), prob) with hub targeting `maximum(degree)` | random failure / targeted attack | NEW Topology(op='remove-node', probability=(1-targeted) rate + [is max degree]) | targeting criterion |
| EnsembleNodeElement | mathematics/networks/_ensembles.py | UpdateLaw(degree), UpdateLaw(TopologyRequest('add-edge'), rate) | Erdos-Renyi edge by edge | NEW Topology(op='add-edge', probability=rate, toward=uniform) | nothing |
| RewiringInteraction (NeighborInteraction) | mathematics/networks/_ensembles.py | UpdateLaw(TopologyRequest('rewire', keep='source'), rate) | Watts-Strogatz rewiring | NEW Topology(op='rewire', keep='source', probability=rate) | nothing (model-local interaction = should be core) |
| VertexElement | mathematics/networks/_growth.py | UpdateLaw(age, age+1), UpdateLaw(degree, neighbors(1)) | bookkeeping | NEW Map(age, increment); degree is an observable, not a law | — |
| TriadVertexElement | mathematics/networks/_growth.py | + UpdateLaw(near, neighbors(...)), UpdateLaw(TopologyRequest('add-edge', toward=weight)) | Holme-Kim triad closure | NEW Topology(op='add-edge', edges=m, toward=triad/PA mixture) | weight function |
| CopyingVertexElement | mathematics/networks/_growth.py | + UpdateLaw(inverse, 1/degree) | cached 1/degree for copying | — | — |
| AttachmentElement | mathematics/networks/_growth.py | UpdateLaw(TopologyRequest('add-node', edges, toward=degree), 1) | Barabasi-Albert | NEW Topology(op='add-node', edges=m, toward=degree) | nothing |
| CopyingElement | mathematics/networks/_growth.py | UpdateLaw(TopologyRequest('add-node', toward=copy weight), 1) | vertex copying | NEW Topology(op='add-node', toward=r + (1-r) neighbors(inverse)) | weight |

## 9. Cellular automata (LocalRuleLaw on LatticeElement / group spaces) and relaxation cascades

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| ElementaryCellElement | computer_science/automata/_automata.py | LocalRuleLaw(Box, Wolfram table) | ordered 3-cell table | NEW LocalMap(alive, Box, TableSpec) | Wolfram code -> table |
| LifeCellElement | computer_science/automata/_automata.py | LocalRuleLaw(Moore, life(born, survive)) | outer totalistic | NEW LocalMap(alive, Moore, OuterTotalisticSpec) or Coupling(form='update', reduce=count) + centre | B/S sets |
| AntCellElement | computer_science/automata/_automata.py | LocalRuleLaw((color, ant), VonNeumann(center), _ant_rule, reads=...) | Langton ant encoded as two cell states, ORDERED neighbours | NEW LocalMap with multiple targets and ordered neighbours | turn rule (an agent-on-lattice encoded as CA; Movement would be the agent version) |
| MajorityCellElement | computer_science/automata/_group_automata.py | LocalRuleLaw(memory set on group, _majority with `degree`) | majority vote on Cayley graph | Coupling(form='update', reduce=mean, f=x_j, then threshold 1/2, ties keep) or NEW LocalMap(TotalisticSpec) | tie rule |
| ParityCellElement | computer_science/automata/_group_automata.py | LocalRuleLaw(TotalisticSpec mod 2) | linear over GF(2) | NEW LocalMap(TotalisticSpec) | — |
| LaplacianCellElement | computer_science/automata/_group_automata.py | LocalRuleLaw(Box, (l + r - 2c) mod p) | discrete Laplacian mod p | Coupling(form='update', f=x_j-x_i) + modulus — needs `mod` post-map | modulus |
| GoldenMeanCellElement | computer_science/automata/_group_automata.py | LocalRuleLaw(Box, schedule='random-sequential', stochastic) | Metropolis-like chain on golden-mean shift | Choice(alive, (0,1), utility=-inf if blocked, rule='metropolis', schedule='random-sequential') — needs C4 schedule | constraint |
| NoisyElementaryCellElement | computer_science/automata/_variants.py | LocalRuleLaw(Box, stochastic) | probabilistic CA | NEW LocalMap(TableSpec, noise flips) | reimplements Wolfram bits (should reuse `Wolfram.table()`) |
| DiffusionCellElement | computer_science/automata/_variants.py | LocalRuleLaw(VonNeumann(center)) | explicit discrete diffusion | Coupling(level, form='update', f=x_j-x_i, K=r) | nothing (pure realization) |
| PascalCellElement | computer_science/automata/_variants.py | LocalRuleLaw(PREDECESSORS, left+center) | additive one-sided CA | NEW LocalMap or Coupling(form='update', over=left neighbour, f=x_j) + self | nothing |
| CyclicCellElement | chemistry/kinetics/_cyclic.py | LocalRuleLaw(VonNeumann(center), _advance_cyclically) | Griffeath cyclic CA (count of successor level >= threshold) | Coupling(form='update', reduce=count where x_j == x_i+1 mod n, then threshold) / NEW LocalMap | cyclic successor |
| HodgepodgeCellElement | chemistry/kinetics/_hodgepodge.py | LocalRuleLaw(Moore(center), rule) | hodgepodge machine (counts and sums of neighbours) | NEW LocalMap (outer totalistic with two counts and a sum) | rule |
| FaultElement | earth/seismology/_faults.py | RelaxationLaw(drive=DriveLaw(load to max), relax=LocalRuleLaw(slip), stable, activity) | OFC: drive to threshold, cascade with reset-and-share | core **ThresholdInteraction(state='force', threshold, share=alpha, mode='reset', drive=load)** exactly | nothing — duplicate of core |
| Automaton / LinearAutomaton / Subshift (plain classes, no base) | computer_science/automata/_automaton.py, _linear_automaton.py, _subshift.py | none (own `step`, `run`, `matrix`, `count`, `sample`) | CA global map, linear CA over K[G], SFT counting/sampling outside the simulation stack | — | FLAG: a second way to run a CA ("as LocalRuleLaws do"), off the class hierarchy; `Subshift.sample` is a constrained Gibbs sampler (== SampleLaw with hard constraint) |

## 10. Reaction networks (chemistry)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| SpeciesElement | chemistry/kinetics/_reactions.py | none | amount `n` | **CompartmentElement** (identical state `n`) — thin subclass or delete | — |
| BindingInteraction | chemistry/kinetics/_reactions.py | RateLaw / JumpLaw / Rate+NoiseLaw(process='channel') via `_reaction_laws` | A + B -> C | Reaction(reactants=[first, second], products=[product], k, mode) | nothing (duplicate of core) |
| DissociationInteraction | chemistry/kinetics/_reactions.py | same | C -> A + B | Reaction(reactants=[reactant], products=[first, second], k, mode) | nothing (duplicate) |

## 11. Rewriting systems (fractals) — own law types, own compiler and solver

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| ProductionLaw (BaseLaw) | mathematics/fractals/_production_law.py | — | parallel string rewriting production (context, condition, probability, parameters) | NEW core RewriteLaw family (see GAPS G7) | L-system grammar syntax |
| SubstitutionLaw (BaseLaw) | mathematics/fractals/_substitution_law.py | — | tile substitution by affine pieces | NEW core RewriteLaw family | affine placements |
| WordElement | mathematics/fractals/_word_element.py | ProductionLaw (subclasses) | word state (counts per symbol) rewritten each generation | NEW core structure-valued element | turtle geometry (GeometrySpec) |
| KochElement | mathematics/fractals/_substitution.py | ProductionLaw('F','F+F--F+F') | L-system | WordElement subclass | production |
| PatchElement | mathematics/fractals/_patch_element.py | SubstitutionLaw (subclasses) | tiling patch rewritten each generation | NEW core structure-valued element | prototiles |
| CantorElement | mathematics/fractals/_substitution.py | SubstitutionLaw | Cantor set | PatchElement subclass | ratio |
| SierpinskiElement | mathematics/fractals/_substitution.py | SubstitutionLaw | Sierpinski triangle/carpet | PatchElement subclass | pieces |
| RewritingCompiler / RewritingSolver / WordKernel / PatchKernel / *StructureSpec | mathematics/fractals/_rewriting_*.py, _word_kernel.py, _patch_kernel.py | — | a full problem form (compiler + solver) inside a model package | — | FLAG: compiler/solver are core machinery, not a realization |

## 12. Evolutionary computation

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| StrategyElement | computer_science/evolutionary_computation/_genetic_algorithms.py | none (agent state actions, score) | strategy holder | Element | — |
| EvolutionInteraction (model) | ".../_genetic_algorithms.py" | RuleLaw(breed) | generational GA: RANK selection, one-point crossover, per-gene uniform mutation, then fitness by simulation | core Evolution(members, strategy='actions', fitness='score', selection='rank', alphabet=7) + a score step — needs C7 | Robby room simulation `_evaluate` |
| BitStringElement | computer_science/evolutionary_computation/_landscapes.py | none | genome as integer + fitness | Element | — |
| ReproductionInteraction | computer_science/evolutionary_computation/_landscapes.py | RuleLaw(reproduce) | overlapping generations: death at rate, fitness-proportional parent, single bit flip; fitness recomputed from genome | core Evolution(replacement='death-birth', death, selection='roulette', mutation, fitness=callable) — needs C7 | fitness landscape `_fitness_of` |

## GAPS: mechanisms no general class of the brief covers

G1. **Discrete-time map, `MapInteraction(state, function, mode='replace'|'increment')`** (alias name per brief:
`UpdateInteraction`). Needed alongside `RateInteraction`: 14 map elements (section 4) plus 7 random-map/agent classes
(section 5) are `x_{t+1} = f(x_t)`, and the brief has only `CouplingInteraction(form='update')`, which is relational.
Semantics must be stated: RateLaws on one state add, UpdateLaws do not (one next value; simultaneous across states,
see StandardMap inlining the sequential composition). So `mode='increment'` (x += g) composes additively with other
increment maps and with Coupling(form='update'); `mode='replace'` is exclusive. `function(ns)` must read named random
draws (`bernoulli`, `normal`, `uniform`, `categorical`) shared by name within a step (PersistentWalker uses one
`'turn'` draw in two laws). Subclasses by operation: `AffineMapInteraction` (chaos game maps, cat map, AR(1)),
`RandomMapInteraction` = MapInteraction with a draw. Most named maps (logistic, Henon, ...) stay as elements with own
UpdateLaws; MapInteraction is for maps that act on a state of another element.

G2. **Ordered-neighbourhood local map, `LocalMapInteraction(states, over, rule, reads, schedule, stochastic)`**
wrapping `LocalRuleLaw`. Coupling's `reduce over j of f(x_j, x_i)` is order-blind; Elementary/NoisyElementary (Wolfram
table), Pascal (left only), Ant (heading + direction), GoldenMean (left/right), Lorenz96 (i+1, i-1, i-2) need the
ORDERED tuple. Rule kinds already exist as root specs (`TableSpec`, `TotalisticSpec`, `OuterTotalisticSpec`), so
this is one class with rule polymorphism, not one class per CA family.

G3. **Discrete-time Markov chain and conditional emission.** `TransitionLaw` (matrix, discrete time) and
`EmissionLaw` have no interaction. Proposal: `TransitionInteraction(state, matrix=None, rate=None)` with `matrix` for
discrete time and `rate` for continuous time, and `EmissionInteraction(target, given, matrix)` ("draw a state from the
row a second state selects"). The brief's `TransitionInteraction(state, from_, to, rate)` is continuous-time only.

G4. **Signed count jumps, `JumpInteraction(changes, rate)`** (the general form behind Reaction's 'jump' mode):
RandomWalker moves on Z (x may be negative; Reaction's units are non-negative counts), Poisson counter, M/M/1 queue
(propensity with guard `n > 0`), Polya urn (increment `added`). Reaction should be a subclass that derives `changes`
from stoichiometry.

G5. **Topology change, `TopologyInteraction(operation, probability, edges, toward, keep)`** with operation in
`add-node | add-edge | remove-node | rewire` (graph rewriting). 7 network classes use UpdateLaw on
`TopologyRequestSpec`, and the model already wrote one interaction for it (`RewiringInteraction`). Subclasses by
operation: `AttachmentInteraction(toward=...)` (BA, copying, Holme-Kim), `RewiringInteraction`, `RemovalInteraction`.

G6. **Asynchronous / sequential update schedule** for any discrete-time update interaction (Hopfield RuleLaw runs
members sequentially; GoldenMean uses `schedule='random-sequential'`; Diluted is parallel). Needs a `schedule`
argument on Map/LocalMap/Coupling(update)/Choice, not a new class.

G7. **Rewriting of structure-valued states** (strings, tile patches): `ProductionLaw`, `SubstitutionLaw`, a compiler,
a solver, kernels and structure specs live in `mathematics/fractals`. Parallel rewriting is a mathematical operation
(also behind graph growth G5 and branching G8). Proposal: core `RewriteLaw` base + structure-state element
(`StructureElement`) + the rewriting form in core compile/solve; fractals keep only `WordElement`/`PatchElement`
realizations, L-system parsing, turtle geometry and visuals.

G8. **Branching / splitting with drawn multiplicity and inheritance** (Galton-Watson, fragmentation): the brief's
`LifeCycleInteraction(birth, death)` "by conditions" cannot express "n ~ Poisson(m) children with generation+1" or
"split a state between parent and child". See C6.

G9. **Constrained sampling / subshift**: `Subshift.sample` is a Gibbs sampler with hard forbidden patterns and
`Subshift.count` a transfer-matrix count; nothing general covers it outside the model. `SampleLaw` with a constraint
(energy = +inf for forbidden patterns) would cover sampling; counting is analysis.

G10. **Fractional-order states** are already supported (`StateSpec(order=q)`) — no gap, but `fractional_dynamics`
generates classes at run time with `@cache` factories; with `order` as a constructor argument of an ordinary element
the factories disappear.

Not gaps (covered): forcing signals (`ramp`, `pulse_train`, `cos(t)`) and delays (`delayed`) are read inside
`rate(ns)`; non-smooth stick-slip via `where` inside rate; threshold cascades (`ThresholdInteraction`).

## BRIEF CORRECTIONS (signature insufficient, with evidence)

C1. **`ReactionInteraction(reactants, products, k, mode, propensity)` binds element paths; it must also bind STATES
of one member** (and of every member of a field/group). Evidence: Brusselator/Oregonator/CubicAutocatalysis/
Gray-Scott/Schnakenberg/Schlogl hold all species as states (`u`, `v`, `x`, `y`, `z`) of one element or one field site;
the reaction-diffusion fields need the reaction per site. Proposal: reactants/products as `(role, state)` pairs or
state names of the bound member, plus integer **stoichiometric coefficients** (Polya urn `added`, Schlogl `2X`, `3X`)
instead of repeating paths, plus **buffered species as parameters** (Schlogl A, B). Also `'langevin'` mode must allow
one Wiener process per reaction channel (`process='channel'` in `_langevin_laws`; Schlogl one noise per reaction).

C2. **`RateInteraction(state, rate, noise)`** needs `convention='ito'|'stratonovich'` (IntermittentVariable uses
Stratonovich multiplicative noise) and an optional shared `process` name (chemical Langevin channels). Must also
state that `rate(ns)` may read `delayed(...)`, `t`, and derivatives of a higher-order state (Jerk, DelayedOscillator).

C3. **`CouplingInteraction(state, function, reduce, over, ...)`: `function(x_j, x_i)` is too narrow.** Evidence:
PageRank needs the degree of j (`rank_j / degree_j`); Budyko-Sellers needs a coordinate-dependent edge weight
`(1 - x^2)` at the link; Hopfield, Diluted, Echo-state need a **weight per pair from a matrix** (`total(s, weights=J)`,
not a network edge attribute: dense/sparse matrix, asymmetric); Tipping needs `x_j + 1` (fine). Proposal:
`function(ns_j, ns_i, edge)` receiving full namespaces of both ends (states, parameters, degree, coordinates) and the
edge member, and `weights=` accepting a matrix. The update form also needs a post-map of the reduction
(`sign`, `tanh`, threshold, `mod p`): Hopfield `sign(sum)`, Echo `tanh(sum + input)`, Majority, LaplacianCell mod p.
I.e. update form = `x_i <- g(x_i, reduce_j f(...))`, not `x_i + K reduce`.

C4. **Schedules and time form.** `CouplingInteraction(form='update')`, `ChoiceInteraction`, Map, LocalMap need
`schedule='synchronous'|'sequential'|'random-sequential'` (Hopfield vs Diluted; GoldenMean). `ChoiceInteraction`'s
`utility(namespace, option)` must be able to read a weighted neighbour sum (Diluted = logit choice with
`u = s * sum_j J_ij s_j`), so its namespace needs neighbour reductions with matrix weights.
`TransitionInteraction(state, from_, to, rate)` needs a discrete-time `matrix` form (Markov sources).

C5. **`TransferInteraction(source, target, amount)` lowers to agent RuleLaws in the brief; it needs a continuous
form** (TransferLaw at a rate between states, with `Source`/`Sink` environments). Evidence: EnergySectorElement uses
4 continuous `TransferLaw`s with `Source('births')`, `Sink('deaths')`; CarbonCycle and Stommel boxes are continuous
two-stock exchanges.

C6. **`LifeCycleInteraction(birth, death)`** needs `offspring=` a drawn count (Galton-Watson Poisson(m)),
`inherit=` a mapping from child state to a function of the parent (`generation + 1`), and `split=` a function
returning (parent states, child states) for conservative fission (FragmentElement). "By conditions" alone covers
neither.

C7. **`EvolutionInteraction`**: needs `selection='rank'` (Robby GA `_offspring` uses rank weights; core has
tournament/roulette), `replacement='generational'|'death-birth'` with a death rate (ReproductionInteraction: Moran-like
overlapping generations), and `fitness=` a callable evaluated on the new strategies (both models recompute fitness
of children: simulated sessions; landscape function of the genome), not only a stored state. Genome as an integer
bitstring (BitStringElement) needs a bit-flip mutation, i.e. `mutate=` callable.

C8. **`GrowthInteraction`** is field-flavoured ("population `n`", "harvest") but is the right primitive for BlowUp
(`power`), SlowManifold x1 (`exponential`), Stochastic energy balance `lam` (`logistic`), Daisyworld (logistic on
shared area with a state-dependent rate). Daisyworld needs `r` as a function of states (temperature) and the
carrying capacity as an expression (`bare = p - white - black`), i.e. `r`, `K` callables, not parameters only.

## FIELD-SHARED realizations (`<field>/common/<family>`)

mathematics:
- `mathematics/common/flows`: Lorenz field (dynamical_systems `LorenzElement` == fractional `lorenz_type`), Van der Pol
  (fractional `van_der_pol_type`; earth `VanDerPolIceAges` is a forced VdP), damped oscillator (fractional
  `oscillator_type`). Fractional versions become the same elements with `StateSpec(order=q)` via an `order` argument.
- `mathematics/common/maps`: logistic (dynamical_systems LogisticMap, fractals QuadraticMap as complex logistic,
  earth SoilTemperature reuses logistic), tent family (TentMap, SkewTentMap), doubling/circle (DoublingMap,
  CircleMap, IntermittencyMap mod 1).
- `mathematics/common/walks`: random walks used by stochastic_processes (RandomWalker, PersistentWalker, MemoryWalker,
  AgeingWalker, WeakeningParticle) and networks (NetworkWalker) — thin subclasses of Map/Jump/Movement.
- `mathematics/common/normal_forms`: cubic fold `b1 + b2 x - x^3` (Cusp, IntermittentVariable drift); used
  cross-field by earth (Tipping, RampedFold, DelayedOscillator) — candidate for a core `PolynomialRateInteraction`
  only if a third field needs it.
- networks only: `VertexElement` degree/age bookkeeping already a base (single package, no common needed).

computer_science:
- `computer_science/common/threshold_units`: binary/sign units with weighted input — Hopfield `NeuronElement`
  (sequential sign), `DilutedNeuronElement` (Glauber), automata `MajorityCellElement` (majority = sign of centred
  sum), echo-state `ReservoirUnitElement` (tanh unit) — as subclasses of Coupling(form='update') with post-maps.
- `computer_science/common/rules`: Wolfram code (`_codes.Wolfram`) reused by `ElementaryCellElement` and
  `NoisyElementaryCellElement` (the latter re-derives the bits by hand).
- evolutionary_computation: both GA interactions in one package; fold into core `EvolutionInteraction` (no common).

chemistry:
- `chemistry/common/kinetics`: Brusselator (kinetics `BrusselatorReactionElement` == pattern_formation
  `BrusselatorElement` minus diffusion), cubic autocatalysis (kinetics `CubicAutocatalysisElement` == Gray-Scott
  reaction terms), Schnakenberg — each a set of Reaction(..., mode='rate') subclasses usable both in a well-mixed
  element and per site of a FieldElement with Coupling diffusion. `SpeciesElement` -> CompartmentElement there.
- `chemistry/common/automata` is NOT needed: Cyclic and Hodgepodge are CAs; they should be thin LocalMap realizations
  (shared mathematically with computer_science/automata, cross-field).

earth:
- `earth/common/folds` (only if more earth packages use it; today all are in climate): cubic fold with forcing
  (Tipping, RampedFold, DelayedOscillator, StochasticEnergyBalance quartic). Across earth packages (climate,
  oceanography, seismology) there is no shared realization today; seismology's FaultElement should become core
  `ThresholdInteraction`, SpringBlock a Coupling + friction rate.

## Field-flavoured names / meanings in core (flags)

Laws: `FluxLaw` ("physical flux", conservation law - name by operation `ConservationLaw`/`DivergenceLaw`);
`DriveLaw` ("dropping a grain" - whole-lattice map, `LatticeMapLaw`); `EncounterLaw`/`DestructionLaw`/
`ProliferationLaw` ("particles", "candidate", "field", "test", "activity" - kinetic-theory vocabulary of Bellomo;
general = bilinear kernel operators on densities: gain-loss / removal / creation kernels); `RelaxationLaw` ("toppling");
`SampleLaw` ("energy" - fine as Monte Carlo); `EmissionLaw` (HMM vocabulary - general "conditional draw");
`GeneratorLaw`/`DissipatorLaw` (quantum-only by construction, Hamiltonian, Lindblad - acceptable as operator names);
`Source` doc "a bank that creates money" (example only).
Elements: `ParticleElement` (position/velocity/mass - should be Element with an order-2 vector state; mass a divisor
in the rate function); `DensityElement` ("activity class" - is a CompartmentElement with state `f`; merge);
`CompartmentElement` OK.
Interactions: `ConstantForceInteraction`, `DampingInteraction`, `PairForceInteraction` (physics: Rate on order-2
states); `GrowthInteraction` ("population n", "harvest"); `ForagingInteraction`, `RegrowthInteraction` ("resource",
"reserve" - Transfer + logistic Map); `PriceAdjustmentProtocol`, `PriceTimePriorityProtocol`,
`UniformPriceProtocol` ("price" - market clearing protocols); `SpreadingInteraction` (epidemic recovery/removed -
Transition subclasses); `AlignmentInteraction` ("in the plane" - Vicsek, Coupling update form).
Model-level duplicates of core names: `computer_science/.../EvolutionInteraction` shadows core
`EvolutionInteraction`.
