# Audit biology_1: collective_behaviour, ecology, epidemiology, evolution

Scope: every Element / Interaction class in `src/skcomplex/models/biology/{collective_behaviour,ecology,epidemiology,evolution}`
(views `*View(AgentView)` are attribute shells, not parts; excluded). Read-only audit, commit f391e686.

Starting fact: none of these 4 packages uses any core interaction today except the bases `Interaction`,
`NeighborInteraction`, `PairPotentialInteraction` and the elements `GroupElement`, `LatticeElement`, `FieldElement`,
`ParticleElement`. Core `GrowthInteraction`, `ReactionInteraction`, `SpreadingInteraction`, `RandomWalkInteraction`,
`ForagingInteraction`, `RegrowthInteraction`, `CouplingInteraction`, `AlignmentInteraction`, `ChoiceInteraction`,
`EvolutionInteraction` all already exist and are re-implemented inline by the models below.

Notation: `ns` = namespace; `RI` = RateInteraction (brief), `CI` = CouplingInteraction (brief), `TI` = TransitionInteraction,
`RxI` = ReactionInteraction, `MvI` = MovementInteraction, `LcI` = LifeCycleInteraction, `TrI` = TransferInteraction,
`ChI` = ChoiceInteraction, `EvI` = EvolutionInteraction. NEW:* = proposed in GAPS.

## Table (grouped by mechanism)

### M1. Pure state holders (no laws)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| RatsElement | cb/_decisions | none | count holder `n` | `CompartmentElement` (pure rename or drop) | nothing |
| PopulationElement | eco/_populations | none | count holder `n` | `CompartmentElement` | nothing |
| CompartmentElement | epi/_compartments | none | count holder `n` | **duplicate of core `CompartmentElement` (same name, same state)**: delete, import core | nothing |
| StandElement | eco/_succession | none | holder `volume`, param `mature` | `Element` / `CompartmentElement` with state `volume` | nothing |
| MicrobeElement | eco/_microbial_communities | none | holder `x`, `b` | `Element` in `GroupElement` | nothing |
| StrategyElement | evo/_games | none | holder `x` | `Element` in `GroupElement` | nothing |
| MoleculeElement | evo/_prebiotic | none | holder `x`, `growth` | `Element` in `GroupElement` | nothing |
| GenotypeClassElement | evo/_prebiotic | none | holder `x` | `CompartmentElement`(state x) | nothing |
| EpidemicCenterElement | epi/_recurrent | none | holder `x,a,recovery` | `Element` | nothing |
| LocusElement | evo/_substitutions | none | holder spin `s`, field `h` | `Element` in `GroupElement` | nothing |
| IndividualElement | evo/_finite_populations | none (agent) | agent with `trait,fitness` | `Element` (agent group) | nothing |
| OrganismElement | eco/_speciation | `[]` (agent) | agent with `species` | `Element` (agent group) | nothing |
| CowElement | eco/_meadow | none (agent) | agent with `bite` | `Element` (agent group) | nothing |
| InfluenceElement | eco/_sequential | none | edge state `a`, flag `fresh` | `Element` as edge of `GroupElement` network | `fresh` flag is a workaround (see GAP G4) |
| CouplingElement | evo/_networks | none | edge state `J`, flag `fresh` | idem | idem |
| ContactElement | epi/_adaptive | none | stateless edge | `Element` as edge | nothing |
| CatalyticLinkElement | evo/_networks | none | stateless edge | `Element` as edge | nothing |

### M2. Single-member continuous rate (ODE on own / bound states)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| ExponentialGrowthInteraction | eco/_populations | RateLaw | `n' = r n` | core `GrowthInteraction(form='exponential')` / `RI(state='n', rate=r n)` | nothing (duplicates core) |
| LogisticGrowthInteraction | eco/_populations | RateLaw | `r n (1-n/K)` | `GrowthInteraction(form='logistic')` | nothing (duplicate) |
| SeasonalLogisticGrowthInteraction | eco/_populations | RateLaw | logistic with `K(t)=K(1+A sin 2πt/T)` | `RI(n, rate=logistic with K(t))`; core `GrowthInteraction` cannot (K is a scalar parameter) | the seasonal K(t) |
| DelayedLogisticGrowthInteraction | eco/_populations | RateLaw + `delayed` | Hutchinson `r n (1-n(t-τ)/K)` | `GrowthInteraction(form='logistic', tau=τ)` | nothing (duplicate) |
| MortalityInteraction | eco/_populations | RateLaw | `-d n` | `RI(n, -d n)` (decay subclass) or `RxI(reactants=[t], products=[])` | nothing |
| BranchElement | cb/_decisions | RateLaw / JumpLaw | decay of marks `n`, deterministic or ±1 jumps | `RxI(reactants=[branch], k=evaporation, mode='rate'|'jump')` | nothing |
| SelfPropulsionInteraction | cb/_flocks | RateLaw (vector) | Rayleigh `v'=(α-β|v|²)v/m` | `RI(state='x' order 2, rate=(α-β v·v) v / m)` | Rayleigh rate function |
| QuotaPhytoplanktonElement | eco/_cell_quota | RateLaw ×2 | Droop quota: MM uptake − dilution; growth `r∞(1-q/Q)` | `RI(Q, ...)`, `RI(P, ...)` | Droop growth/uptake formulas |
| CompetitionInteraction | eco/_communities | RateLaw ×2 | 2-species LV competition | `RI` ×2, or `CI(over whole group, weights=α matrix, multiplicative)` (= GLV) | nothing beyond GLV |
| MutualismInteraction | eco/_communities | RateLaw ×2 | saturating mutualism `a b n1 n2/(1+a T n_j)` | `RI` ×2 on bound roles | saturating mutualism function |
| RosenzweigMacArthurInteraction | eco/_predation | RateLaw ×2, `where`, `t` | Holling II predation, time-ramped crowding, time-scale κ | `RI` ×2 | RM rates + ramp |
| PioneerGrowthInteraction | eco/_succession | RateLaw | MM regrowth suppressed by other stands | `RI(volume, ...)` reading roles | succession formula |
| SuccessionalGrowthInteraction | eco/_succession | RateLaw | logistic with cross crowding + seedling input | `RI(volume, ...)` | crowding formula |
| VaccinatedPopulationElement | epi/_vaccination | RateLaw ×3 | SIR + control `u S` inside one element | `RxI` on species S,I,R of ONE member (needs correction C2) or `RI` ×3 | control `u` (optimal-control model) |
| GroundElement | cb/_trails | RateLaw ×2, `laplacian` | `G'=-G/T`; `V'=D ΔV + (G-V)/τ` | `RI(G, -G/T)`; `CI` diffusion on field for V + `RI(V,(G-V)/τ)` | nothing |
| PheromoneElement | cb/_foraging | RateLaw, `laplacian` | diffusion + evaporation on field | `CI` diffusion (field Laplacian lowering, C3) + `RI(level, -e level)` | nothing |
| EmotionMarksElement | eco/_stigmergy | RateLaw ×4 | decay of 4 field states | `RI` decay ×4 (or one `RI` over a state list) | nothing |
| FisherKPPElement | eco/_fronts | RateLaw, `laplacian` | `D Δu + r u(1-u)` | `GrowthInteraction(logistic, K=1)` + `CI` diffusion | nothing |

### M3. Count-level transitions (mass action / propensity, rate or jump)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| InfectionInteraction | epi/_compartments | RateLaw/JumpLaw via `_transfer` | `S+I→2I`, `β S I` | `RxI(reactants=[S,I], products=[I,I], k=β, mode='rate'|'jump')` | nothing (thin subclass) |
| RecoveryInteraction | epi/_compartments | idem | `I→R`, `γ I` | `RxI(reactants=[I], products=[R])` | nothing |
| ImmunityLossInteraction | epi/_compartments | idem | `R→S`, `ξ R` | `RxI(reactants=[R], products=[S])` | nothing |
| NervousnessInteraction | cb/_decisions | idem ×2 | calm→nervous at `(s + c N/(C+N)) C`; nervous→calm `γ N` | `RxI` ×2 with `propensity=` (frequency-dependent) | the alarm propensity |
| TrailChoiceInteraction | cb/_decisions | RateLaw/JumpLaw | inflow `∅→chosen` at Deneubourg choice prob. reading `other` | `RxI(reactants=[], products=[chosen], propensity=...)` — needs read-only role `other` (C2) | Deneubourg choice function |
| PredationInteraction | eco/_populations | RateLaw ×2 | LV: prey `-a xy`, predator `+b xy` | `RxI` with yields (C2: non-integer stoichiometry) or `RI` ×2 | nothing |
| PredatorPreyEventsInteraction | eco/_predation | RateLaw/JumpLaw ×5 via `_change` | birth, crowding, Holling II kill, conversion, death | 5 × `RxI(..., propensity=, mode)` | Holling II propensity |
| HabitatPatchElement | eco/_clustering | JumpLaw ×2 | linear birth/death `α n` | `RxI(n→2n)`, `RxI(n→∅)`, mode='jump' | nothing |
| NeutralSpeciesElement | eco/_neutral | JumpLaw ×2, `where` | immigration+birth `b0+b1 n`; death `d0[n>0]+d1 n` | `RxI` (∅→n, n→2n, n→∅) with propensities | nothing |
| HoppingInteraction | eco/_clustering | JumpLaw per link (hand-written `links` list, indexes `patches[i]`) | hop of one count i→j at `β n_i` along edges | `RxI` per edge (C2: edge binding) or `CI(mode='jump')` on counts | nothing |
| SubpopulationElement | epi/_networks | RateLaw ×3, `laplacian` | metapopulation SIR: `β S I/N` + graph diffusion | `RxI` on states S,I,R of one member (C2) + `CI` diffusion over edges per state | nothing |

### M4. Network / mean-field coupling (continuous)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| CarrierElement | epi/_adaptive | RateLaw + `neighbors(other.p)` | QMF SIS `p'=-p+λ(1-p)Σ_j p_j` | `RI(p,-p)` + `CI(state p, function=λ(1-p_i)p_j, reduce=sum, over edges)` | nothing |
| DegreeClassElement | epi/_networks | RateLaw + `total` | HMF SIS, θ = Σ kPρ / Σ kP | `RI(rho,-rho)` + `CI(over whole group, reduce=weighted mean with weights k P)` (C3) | θ weighting by degree |
| ImmunizingDegreeClassElement | epi/_outbreaks | RateLaw ×3 + `total` | HMF SIR, θ with excess degree (k-1) | `RxI` on s,i,r (C2) with propensity using `CI` reduction | excess-degree θ |
| GeneralizedLotkaVolterraInteraction | eco/_microbial_communities | RateLaw + `total(weights=A)` | `x_i(b_i+Σ a_ij x_j)` | `CI(over whole group, weights=A matrix, function=a_ij x_i x_j)` + `RI(x, b x)` (C3) | nothing |
| ReplicationInteraction | evo/_games | RateLaw + `total(weights=A)` | replicator `x_i((Ax)_i − x·Ax)` | `CI` (matrix) + NEW:ProjectionInteraction (G3) | nothing (field-shared, F2) |
| CatalysisInteraction | evo/_prebiotic | RateLaw + `total(weights=topology)` | hypercycle: catalysed growth − dilution keeping Σx=1 | `CI` (matrix) + NEW:ProjectionInteraction | nothing (F2) |
| DependencyFieldInteraction | evo/_networks | UpdateLaw `h = neighbors(1, weight=J)` | field = weighted in-degree from edge state J | `CI(form='update', function=1, weight=edge.J, reduce=sum)` (C3: edge-state weights) | nothing |
| LinearGrowthInteraction | eco/_sequential | UpdateLaw + `neighbors(weight=a)` + `total` | `x + r Σ a_ij x_j`, clipped ≥0, normalised | `CI(form='update', weight=edge.a)` + clip + NEW:ProjectionInteraction(update) | nothing |
| SubstitutionInteraction | evo/_substitutions | JumpLaw ×2 + `total(weights=J)` | spin flips `s→-s` at Kimura fixation rate of fitness change | `TI(state s, from -1 to 1 / 1 to -1, rate=...)` reading `CI` matrix reduction | Kimura origin-fixation rate |
| MorsePotentialInteraction | cb/_flocks | (PairPotential) RateLaw ×2 | Morse pair potential on order-2 positions | `CI(potential=Morse, over radius)` / today `PairPotentialInteraction` (needs `velocity=`,`mass=` physics args) | Morse energy |

### M5. Discrete-time maps of continuous states (UpdateLaw)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| RabbitsElement | eco/_factories | UpdateLaw + `lag` | Fibonacci `x_{t+1}=x_t+x_{t-1}` | NEW:MapInteraction(state, map) (G1) | nothing |
| ParasitismInteraction | eco/_predation | UpdateLaw ×2 | Nicholson-Bailey map | NEW:MapInteraction ×2 | NB map |
| UnstablePopulationsElement | eco/_instability | UpdateLaw ×2, `normal`, `bernoulli`, `where`, clip | stochastic map with demographic noise, mutation jumps, clip ≥0 | NEW:MapInteraction(map, noise) + clip (G1) | Jensen eqs (13.3–13.4) |
| VisitsInteraction | epi/_recurrent | UpdateLaw + many `lag`s | SIRS map with refractory memory of `recovery` steps | NEW:MapInteraction with lags | refractory sum |
| MasterSequenceElement | evo/_prebiotic | UpdateLaw | Eigen sharp-peak: `x0(1-u)/φ` | NEW:MapInteraction + NEW:ProjectionInteraction(update) (division by mean fitness = normalisation) | sharp-peak landscape; element also serves as parameter store for `u,σ` read by others (smell) |
| MutationInteraction | evo/_prebiotic | UpdateLaw | gain from class k-1, keep own, / φ | NEW:MapInteraction (linear mutation matrix) + Projection | mutation matrix of the error-class chain |
| CouplingDriftInteraction | evo/_networks | UpdateLaw ×2, `normal`, `uniform`, `where fresh` | edge state random walk `J += σ N(0,1)` + first draw of new edges | NEW:MapInteraction on edges (or `RI(J, 0, noise=σ)` Euler, dt=1) + init-on-create (G4) | nothing |
| InfluenceDrawInteraction | eco/_sequential | UpdateLaw ×2, `uniform`, `where fresh` | first draw of the state of a new edge | init-on-create (G4) | nothing |

### M6. Discrete states of lattice / network members (jumps, Bernoulli steps)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| HostElement | epi/_adaptive | JumpLaw ×2 + `neighbors` | SIS contact process on network | core `SpreadingInteraction(beta, gamma)` = `TI(i, 0→1, β Σ_j i_j)` + `TI(i, 1→0, γ)` | nothing |
| ActiveHostElement | epi/_adaptive | UpdateLaw, `bernoulli`, `(1-λ)^Σ`, topology add-edge | discrete-time SIS (prob per step) + activity-driven edges | `TI` in per-step probability form (C4) + NEW:TopologyInteraction(add-edge, prob a, m) | nothing |
| CyclicSpeciesElement | evo/_games | LocalRuleLaw (random-sequential, stochastic), `_invade` | cyclic dominance invasion (RPS) on lattice | `TI(state, from k to k+1 mod 3, rate = #neighbours in k+1)` per edge, or `ChI`/Imitation with conditional copy | dominance relation |
| SpeciesElement | evo/_coevolution | ExtremalLaw | Bak-Sneppen: renew min site + neighbours | NEW:ExtremalInteraction wrapping core `ExtremalLaw` (G5) | `_renew` (uniform redraw) |

### M7. Topology change of networks

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| RewiringInteraction | epi/_adaptive | JumpLaw on `TopologyRequestSpec('rewire', keep, toward)` | S–I edge rewired to a susceptible at rate w | NEW:TopologyInteraction(action='rewire', rate, toward) (G2) | the S–I condition |
| ExpiryInteraction | epi/_adaptive | UpdateLaw on `TopologyRequestSpec('remove-edge')` | every edge removed after one step | NEW:TopologyInteraction(action='remove-edge', probability=1) | nothing |
| AbundanceElement | eco/_sequential | UpdateLaw age+1, add-node/remove-node when x<thr | extinct node replaced by new node with k edges | NEW:MapInteraction(age) + NEW:TopologyInteraction(action='replace-node', condition, edges=k) | extinction threshold (F3) |
| TaxonElement | evo/_networks | idem, condition h<0 | idem | idem | condition h<0 (F3) |
| ChemicalElement | evo/_networks | UpdateLaw ×6: map+normalise, `uniform` tie, periodic `mod(t)`, argmin via `minimum`, add/remove node, add-edge toward fresh | Jain-Krishna: catalysed map, periodic replacement of least-fit, links to newcomer with prob p | `CI(update)` + NEW:ProjectionInteraction + NEW:ExtremalInteraction(network, period) + NEW:TopologyInteraction | Jain-Krishna schedule |

### M8. Agent movement and flocking (RuleLaw on agents)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| BoidElement | cb/_flocks | RuleLaw `flock` | velocity += cohesion·mean offset − separation·Σ close offsets + alignment·(mean v − v); clamp speed; periodic move | `CI(form='update', over radius)` three terms (C3: composable increments, several radii) + clamp (G6) + `MvI(step=ballistic)` | Reynolds weights |
| ShoalMemberElement | cb/_leadership | RuleLaw `turn` (phase) + `swim` | Couzin zones (repulsion priority, else attraction+alignment), informed goal ω, max turn, angular noise; constant-speed move | core `AlignmentInteraction(repulsion, attraction, noise)` is the symbolic analogue; + max-turn clamp (G6) + preferred direction term + `MvI(ballistic, speed)` | priority rule, goal weighting ω |
| AntElement | cb/_foraging | RuleLaw `forage`, `model.deposit`, `model.field` | carrying state machine; homing step + deposit; weighted step by pheromone × outward bias | `MvI(step='weighted', weight(agent,cell,model))` (C5) + NEW:DepositInteraction (G7) + condition-triggered `TI` in agent form (carry flag on reaching food/nest) | nest/food logic, weight function |
| WalkerElement (trails) | cb/_trails | RuleLaw `walk`, deposit, field | goal choice, logit step on goal heading + trail gradient, saturating deposit | `MvI(step='weighted')` + NEW:DepositInteraction(amount = I(1-G/Gmax)) | goal choice, step weights |

### M9. Agent life cycle, feeding, predation (RuleLaw on agents)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| PreyElement | eco/_agent_predation | RuleLaw `live` | neighbour step; birth with prob if no other prey in cell | `RandomWalkInteraction(step='neighbor')` + `LcI(birth=prob, condition)` | "alone in cell" condition |
| PredatorElement | eco/_agent_predation | RuleLaw `hunt` | step; kill one co-located prey (other group); breed with prob after a meal; die with prob | `RandomWalkInteraction` + NEW:EncounterInteraction (G8, remove co-located agent of another group) + `LcI(birth after encounter, death prob)` | nothing |
| GrassElement | eco/_wolf_sheep | RuleLaw `regrow` | countdown regrowth | core `RegrowthInteraction(form='delay', resource='grown', clock='countdown', delay=regrowth)` — exact match | nothing (duplicate) |
| SheepElement | eco/_wolf_sheep | RuleLaw `graze` | step paying 1 energy, eat grown grass (cell resource → energy), split energy with offspring at prob, die <0 | core `RandomWalkInteraction(cost=1, budget='energy')` + core `ForagingInteraction(resource='grown', reserve='energy', gain, probability)` | nothing (duplicate of core) |
| WolfElement | eco/_wolf_sheep | RuleLaw `hunt` | step with cost, eat co-located SHEEP AGENT (remove) gaining energy, split, die | `RandomWalkInteraction(cost)` + NEW:EncounterInteraction(remove prey agent, gain to reserve) + `LcI` split | nothing |
| PastureElement | eco/_meadow | RuleLaw `regrow` | logistic map `G+gG(1-G)` per agent | core `RegrowthInteraction(form='logistic', capacity=1)` | nothing |
| GrazingInteraction | eco/_meadow | RuleLaw `graze` | remove `bite` from co-located pasture, clipped at 0, to nowhere | `TrI(source=co-located pasture.grass, target=sink, amount=bite, clip)` (C6) | nothing |
| HerdingInteraction | eco/_meadow | RuleLaw `herd` | random-order move to NEAREST FREE patch with grass ≥ thr (exclusive occupancy), else death | `MvI(step='best', score=-distance, admissible=..., exclusive=True)` + `LcI(death when no target)` (C5) | admissibility criterion |
| PlantElement | eco/_stigmergy | RuleLaw `spread` | age counter; offspring into empty neighbour cell at maturity, age reset | `LcI(birth when age≥maturity, placement='empty-neighbour', reset)` + age `MapInteraction` | nothing |
| ForagerElement | eco/_stigmergy | RuleLaw `live` | hunger counter, starvation death, greedy move to best cell (food + marks), eat target agent, smoothed emotions, deposit marks, birth with emotion-modulated prob | `LcI` + `MvI(step='best', score)` + NEW:EncounterInteraction + NEW:DepositInteraction + exponential smoothing (`RI`/Map) | emotion appraisal, preference score |
| StigmergicPreyElement | eco/_stigmergy | (inherits) `sense` | distress = #predators around | thin subclass of ForagerElement | `sense` |
| StigmergicPredatorElement | eco/_stigmergy | (inherits) | distress = hunger/starvation | thin subclass | nothing |

### M10. Agent epidemics (state machines with timers on agents)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| PersonElement | epi/_agents | RuleLaw `live` | walk with prob (lockdown window), infect co-located S with prob β, E→I after `incubation` steps (clock), I→R/D competing probs, R→S waning | `RandomWalkInteraction(probability=f(t))` (C7) + `TI` agent form: co-located contact, per-step prob, residence-time `after=`, competing exits (C4) | nothing (fully general) |
| WalkerElement (epi) | epi/_agents | RuleLaw `walk` | walk with prob; infect co-located; countdown → deceased | `RandomWalkInteraction` + `TI` agent form with `after=lifetime` | nothing |
| TravellerElement | epi/_agents | RuleLaw `live` | infect co-located residents with prob τ; recover after `days`; hop on network | `RandomWalkInteraction(step='neighbor', probability=m)` + `TI` agent form | nothing |

### M11. Population genetics / evolutionary algorithms (RuleLaw)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| MoranStepInteraction | evo/_finite_populations | RuleLaw `generation` | N birth-death events, fitness-proportional parent `(1+s)^trait`, ±1 mutation | `EvI(selection='moran', fitness=..., mutation=±1)` (C8) | fitness map `(1+s)^trait` |
| NeutralReplacementInteraction | eco/_speciation | RuleLaw `replace` | neutral Moran replacement; mutation to a NEW label at ν | `EvI(selection='moran', fitness=const, mutation='innovation')` (C8) | nothing |
| TangledNatureStepInteraction | evo/_finite_populations | RuleLaw | frequency-dependent reproduction prob (logistic of `k/N J·counts − μN`), split into 2 mutants, random death `p_kill` | `LcI(birth prob = f(composition), offspring=2 mutated, death prob)` (C9) | TaNa weight, couplings `_couplings` |
| AdaptiveClimbingInteraction | evo/_finite_populations | RuleLaw | SSWM: whole population jumps to best of one-bit mutants | `EvI(selection='best-mutant', population='monomorphic')` or `ChI(propose=bit-flips, rule='best')` (C8) | random landscape |
| ClimberElement | evo/_finite_populations | RuleLaw `climb` | greedy one-bit-flip hill climb on NK | `ChI(state genome, propose=random bit flip, utility=NK fitness, rule='metropolis', intensity=∞)` (C8) | NK fitness tables |

## 1. GAPS (mechanisms no general class of the brief covers)

- **G1 MapInteraction(state, map, noise=None, lags)** — discrete-time assignment `x_{t+1} = f(ns)` of continuous
  states (with `lag`, random draws, clipping). Users: RabbitsElement, ParasitismInteraction, UnstablePopulationsElement,
  VisitsInteraction, MasterSequenceElement, MutationInteraction, CouplingDriftInteraction, age counters in
  Abundance/Taxon/Chemical/Plant. `RateInteraction` is ODE-only; `CouplingInteraction(form='update')` only covers
  neighbour reductions. Lowers to `UpdateLaw`. Name by operation (map/difference equation), not "generation".
- **G2 TopologyInteraction(action, rate | probability | condition, edges=, keep=, toward=)** — add/remove/rewire/
  replace node or edge. Users: RewiringInteraction, ExpiryInteraction, ActiveHostElement (add-edge), AbundanceElement,
  TaxonElement, ChemicalElement. Today written as `UpdateLaw/JumpLaw` on `TopologyRequestSpec` keys, a second way
  hidden in laws. Lowers to those laws.
- **G3 ProjectionInteraction(state, over=group, total=1, form='rate'|'update')** — keep Σx constant: rate form
  `g_i − x_i Σ_j g_j` (replicator/dilution flux), update form `x/Σx` (normalisation, also clip ≥0). Users:
  ReplicationInteraction, CatalysisInteraction, LinearGrowthInteraction, ChemicalElement, Master/MutationInteraction
  (division by mean fitness φ is exactly normalisation). Not a coupling (needs the total of the other laws' rates).
- **G4 Initial value on creation** — draw of the state of a newly created edge/member (`fresh` flag + `where`).
  Users: InfluenceDrawInteraction, CouplingDriftInteraction, ChemicalElement, and offspring states in LifeCycle.
  Proposal: an `initial=` (value or callable(rng)) argument on `TopologyInteraction`/`LifeCycleInteraction`/
  `GroupElement` for created members, removing the `fresh` state.
- **G5 ExtremalInteraction(state, neighborhood, rule, extremum='min', repeat, period)** — wraps core `ExtremalLaw`;
  needs a network variant (ChemicalElement: argmin over nodes with ties, every `period` steps). Users: SpeciesElement,
  ChemicalElement.
- **G6 Bounded update (clamp)** — clamp norm of a vector (boid min/max speed) and clamp turning angle (Couzin
  max_turn), clip ≥0 (Instability, LinearGrowth). Proposal: `ClampInteraction(state, low, high, norm=False,
  relative_to=previous)` lowering to an `UpdateLaw` applied after the others (or a `bounds=` argument on update-form
  classes). Not covered by BoundaryLaw (spatial walls).
- **G7 DepositInteraction(source agents, field, state, amount)** — agent writes into the field cell it stands on
  (`model.deposit`). Users: AntElement, WalkerElement (trails), ForagerElement (marks). The brief's
  `TransferInteraction(source,target,amount)` moves between members; a field cell is not a member and the source
  loses nothing. Either extend TransferInteraction (C6) or add this class.
- **G8 EncounterInteraction(first, second, where='cell'|radius, effect)** — agent-level co-location binding: for each
  agent of group A, the agents of group B in its cell (agent analogue of `PairInteraction`). Effects: remove B
  (predation: PredatorElement, WolfElement, ForagerElement), gain to A's reserve, transition B's state (agent
  epidemics: PersonElement, WalkerElement, TravellerElement infect co-located). Core `ForagingInteraction` only takes
  from a resource member per cell, never removes/affects agents.

## 2. BRIEF CORRECTIONS

- **C1 RateInteraction lacks a count/jump mode.** BranchElement, NervousnessInteraction, PredatorPreyEvents,
  InfectionInteraction etc. all toggle `stochastic` between `RateLaw` and `JumpLaw ±1`. Either RateInteraction gets
  `mode='rate'|'jump'` or the brief states that every "population ODE" with a stochastic twin must be a
  `ReactionInteraction` (preferred: one way).
- **C2 ReactionInteraction signature.** (a) Species are roles with one shared `state='n'`; SubpopulationElement,
  VaccinatedPopulationElement, ImmunizingDegreeClassElement hold S,I,R as several states of ONE member -> need species
  addressable as `(role, state)` or state names on a single bound member. (b) No stoichiometric coefficients /
  yields: PredationInteraction (`-a`, `+b`), PredatorPreyEvents conversion `e` -> products/reactants as
  `{species: coefficient}`. (c) Propensity may read roles that are neither reactants nor products (TrailChoice reads
  `other`; Nervousness reads totals) -> `modifiers=` read-only roles. (d) No edge binding: HoppingInteraction and the
  metapopulation hop are reactions per edge (reaction-diffusion master equation) -> ReactionInteraction must accept
  `over='edges'` like NeighborInteraction, else CouplingInteraction needs `mode='jump'` for counts.
- **C3 CouplingInteraction signature.** Current core has `function(x_j, x_i)`, `K`, `alpha`, `tau`,
  `mode='edges'|'mean-field'`. Missing: (a) `weight=` from an EDGE state (DependencyField: J, LinearGrowth: a) or a
  dense matrix over a group (GLV, Replicator, Catalysis, Substitution `total(weights=...)`); (b) weighted mean with
  member-state weights (HMF θ = Σ kPρ/Σ kP) — `reduce` must accept weights from member states; (c) update form must
  compose: boids add three terms with different radii into one velocity, so `form='update'` should mean
  `x ← x + Σ terms` (increment) or `combine=` — an `UpdateLaw` assignment per interaction cannot be summed across
  interactions; (d) field lowering: on a `FieldElement`/lattice, `function=x_j−x_i, reduce=sum` must lower to the
  symbolic `laplacian` (Pheromone, Ground V, FisherKPP, Subpopulation) with the field's boundary, not per-edge laws;
  (e) agent form: boids/Couzin are agent groups using `agent.neighbors(radius)`, so the radius coupling needs an
  agent (RuleLaw) lowering or those models must move to point groups with `x,v` states.
- **C4 TransitionInteraction needs discrete-time and timer forms.** ActiveHostElement uses per-step Bernoulli with
  `1-(1-λ)^{#infected}`; agent epidemics use per-step probabilities, deterministic residence times (E→I after
  `incubation`, death after `lifetime`, recovery after `days`) and competing exits (I→R or I→D from one draw). Add
  `probability=` (per step) beside `rate=`, `after=` (age-in-state, replaces hand-kept `clock` states) and a list of
  `to` with branch probabilities. Agent form must read co-located agents (via G8).
- **C5 MovementInteraction(step) is too narrow.** Needs `step='weighted'` with `weight(agent, cell, model)` (Ant,
  trail Walker: logit over neighbour cells reading fields), `step='best'` with `score`, admissibility and
  `exclusive=True` occupancy + random sequential order (HerdingInteraction, ForagerElement), `step='ballistic'`
  (x += speed·heading, periodic wrap) for Boid/Shoal, and a failure hook (death when no admissible cell). Core
  `RandomWalkInteraction` (neighbor/gaussian/levy, cost/budget) already covers Prey/Predator/Sheep/Wolf/epi walkers.
- **C6 TransferInteraction.** Needs co-location binding (source = the member of another group in the agent's cell),
  a sink/source "nothing" end (Grazing removes bite to nowhere; deposit creates from nothing), clipping at the
  available amount, and an amount callable reading the target (trail deposit `I(1−G/Gmax)`).
- **C7 Time-dependent parameters.** SeasonalLogistic K(t), RosenzweigMacArthur ramp φ(t), PersonElement lockdown
  window d(t): brief arguments are scalars; either every rate/probability argument accepts an expression of `t`, or
  ParameterSpec gets schedules. Otherwise SeasonalLogistic cannot be a thin GrowthInteraction subclass.
- **C8 EvolutionInteraction is a GA only (tournament/crossover/bit mutation on a strategy).** Biology needs:
  `selection='moran'` (N sequential birth-death events, fitness-proportional), neutral (constant fitness),
  `mutation='innovation'` (new label, infinite alleles: NeutralReplacement), ±1 step mutation (Moran),
  `'best-mutant'`/SSWM monomorphic update (AdaptiveClimbing). Hill climbing (ClimberElement) is better a
  `ChoiceInteraction` with a `propose=` callable (state-dependent options; brief's `options` is a fixed list) and
  `rule='metropolis'` at infinite intensity (greedy).
- **C9 LifeCycleInteraction(birth, death).** Needs: offspring count (TaNa: parent replaced by 2), offspring state
  (energy halving, mutated genome), placement (`same cell` vs `empty neighbour cell`: Plant/Forager), triggers by
  condition (age ≥ maturity, after an encounter) and by probability, death by condition (energy<0, hunger>starvation,
  no admissible move). Core `ForagingInteraction` hard-codes one variant (threshold + probability, halve reserve).
- **C10 PairPotentialInteraction/`potential=` binding.** MorsePotential inherits `velocity='v', mass='m'` physics
  arguments; brief already says order-2 state with inertia inside the rate function — confirmed needed.

## 3. FIELD-SHARED (`biology/common/<family>`)

- **F1 `biology/common/compartments`**: the `_transfer(source, destination, rate, stochastic)` helper is duplicated in
  `collective_behaviour/_decisions.py` and `epidemiology/_compartments.py`, and `_change` in `ecology/_predation.py` is
  its one-sided twin. All of InfectionInteraction, RecoveryInteraction, ImmunityLossInteraction,
  NervousnessInteraction, BranchElement decay, HabitatPatch, NeutralSpecies, PredatorPreyEvents become thin
  `ReactionInteraction` subclasses; Infection/Recovery/ImmunityLoss (and the `sir/sis/sirs` factories) are the
  shared "compartmental transfers" of epidemiology + collective behaviour + ecology. RatsElement/PopulationElement/
  epi CompartmentElement -> core `CompartmentElement`.
- **F2 `biology/common/replicators`**: ReplicationInteraction (evolution/games), CatalysisInteraction
  (evolution/prebiotic), Master/MutationInteraction (prebiotic) and LinearGrowth/Chemical (ecology/sequential,
  evolution/networks) share "growth by matrix coupling + projection to constant total": a thin
  `ReplicatorInteraction(state, matrix)` = CouplingInteraction(matrix) + ProjectionInteraction, used by evolution and
  ecology (GLV is its un-projected sibling).
- **F3 `biology/common/networks`**: AbundanceElement (ecology/_sequential) and TaxonElement (evolution/_networks) are
  the same "age counter + replace extinct node by a node with k edges" law with a different condition;
  InfluenceDrawInteraction (ecology) and the fresh branch of CouplingDriftInteraction (evolution) are the same
  "first draw of a new edge"; InfluenceElement ≈ CouplingElement (edge with weight + fresh). One shared
  `ExtinctionReplacement` element/interaction (thin over TopologyInteraction + MapInteraction) and one weighted
  edge element.
- **F4 `biology/common/agents`**: `_wander`/`_step` helpers duplicated in ecology/_agent_predation,
  ecology/_wolf_sheep, epidemiology/_agents -> all are core `RandomWalkInteraction(step='neighbor', probability,
  cost)`. `_live` (energy split / death) in wolf_sheep and `_spawn` in stigmergy -> LifeCycle realizations. Predator
  eating co-located prey (agent_predation, wolf_sheep, stigmergy) -> one shared predation encounter (over G8).
- **F5 `biology/common/stigmergy`**: deposit-into-field + field-weighted step + field evaporation/diffusion is shared
  by collective_behaviour/_foraging (ants), collective_behaviour/_trails (walkers), ecology/_stigmergy (emotion
  marks): a thin realization over DepositInteraction + MovementInteraction(weighted) + Rate/Coupling on the field.
- **F6 `biology/common/growth`**: ecology's Exponential/Logistic/DelayedLogistic/Mortality duplicate core
  `GrowthInteraction`; keep at most as renames (Seasonal needs C7). Also used by oncology growth (outside scope).

## Field-flavoured core names these models rely on (or should rely on)

- `ForagingInteraction` (resource, reserve, uptake, metabolism, foragers) — ecology-named; operation is "take from
  co-located member into own state, pay cost, split/die": should decompose into TransferInteraction + LifeCycle; it
  is the class SheepElement should use.
- `RegrowthInteraction` (resource, capacity, clock) — the operation is a bounded growth map / countdown reset; covered
  by MapInteraction + TransitionInteraction(after=). GrassElement/PastureElement would rely on it.
- `SpreadingInteraction` (beta, gamma, omega, active/removed) — epidemic-named; operation is a contact-process
  transition = TransitionInteraction reading neighbours. HostElement would rely on it.
- `GrowthInteraction` (population `n`, harvest `h`, target) — ecology-named.
- `CompartmentElement` — epidemiology/pharmacology word for "amount holder"; also clashes with a model class of the
  same name in epidemiology/_compartments.py.
- `ParticleElement` + `PairPotentialInteraction(velocity, mass)` — physics-named, used by `bird_swarm`/Morse.
- `AlignmentInteraction(heading, velocity, position, repulsion, attraction)` — Vicsek/Couzin named; operation is a
  radius coupling in update form (brief CI).
- `ExtremalLaw` itself is general; the `'extinction'` event label is model-side (fine).
- `TopologyRequestSpec` keys (`'add-node'`, `'rewire'`) used as law targets: general, but only reachable through
  raw laws (G2).
- `DensityElement` ("activity class") and `ImitationInteraction(payoff)` — social flavour; not used here.
