# Audit biology_2: neuroscience, oncology, physiology, systems_biology

Scope: every `Element` / `Interaction` class in `src/skcomplex/models/biology/{neuroscience,oncology,physiology,systems_biology}`
(104 classes; the 3 `*View` agent views in oncology are listed with their elements). Laws read from `_get_laws` / rule
methods. Notation for the brief classes: `Rate(state, rate)`, `Gradient`, `Coupling(state, function, reduce, over, lag,
delay, form)`, `Transition`, `Choice`, `Reaction(reactants, products, k, mode, propensity)`, `Movement`, `Transfer`,
`LifeCycle`, `Matching`, `Feedback`, `Evolution`. New classes proposed in GAPS are marked with `*`.
`ns` = namespace of the bound roles. Core law utilities used by these models: `laplacian`, `delayed`, `pulse`,
`pulse_train`, `step`, `where`, `mod`, `maximum`, `mean`, `total`, `shift_member`, `categorical`, `moment`.

Shared observation for every "monolithic" element below: the element carries its states AND all its laws. Under the
brief the states belong to an `Element` (or `CompartmentElement` per population) and the laws decompose into the listed
interactions; what "stays own" is the genuinely model-specific nonlinear term. Where a row says "stays own: whole RHS",
splitting would not produce a reusable piece (only a renaming of `Rate`).

## 1. Scalar growth / decay of one state (Rate subclasses: growth forms, decay)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| TumourElement | oncology/_growth | none (state `n`) | population holder | `CompartmentElement` (state `n`); pure rename | nothing |
| VasculatureElement | oncology/_growth | none (state `K`) | holder of a capacity state | `CompartmentElement` with state renamed `K` (needs `state=` name arg on CompartmentElement) | nothing |
| ExponentialGrowthInteraction | oncology/_growth | RateLaw | `n' = r n` | `Rate('target.n', r*n)` = core `GrowthInteraction(form='exponential')` (duplicate of ecology `ExponentialGrowthInteraction`) | nothing |
| LossInteraction | oncology/_growth | RateLaw | `n' = -d n` | `Rate` decay subclass (dup. of ecology `MortalityInteraction`) | nothing |
| PowerLawGrowthInteraction | oncology/_growth | RateLaw | `n' = r n^p` | `Rate` = `GrowthInteraction(form='power')` | nothing |
| GeneralizedLogisticGrowthInteraction | oncology/_growth | RateLaw | Richards `r n (1-(n/K)^p)` | `Rate` = `GrowthInteraction(form='logistic', p)` (dup. ecology `LogisticGrowthInteraction`) | nothing |
| GompertzGrowthInteraction | oncology/_growth | RateLaw | `r n log(K/n)` | `Rate` = `GrowthInteraction(form='gompertz')` | nothing |
| ExponentialLinearGrowthInteraction | oncology/_growth | RateLaw+`where` | `min(r0 n, r)` | `Rate('target.n', min(r0 n, r))` | the min form (not a core growth form) |
| VascularSupportInteraction | oncology/_growth | RateLaw x2 | Gompertz with capacity = state of another role; `K' = d n^(2/3)` | `Rate(tumour.n, r n log(K/n))` + `Rate(vasculature.K, d n^(2/3))`; core GrowthInteraction cannot (K is a parameter) | Hahnfeldt capacity law `d n^(2/3)` |
| LogKillInteraction | oncology/_growth | RateLaw, `pulse_train` | `-a dose pulse_train(t) n` | `Rate` decay with time-dependent coefficient | nothing (schedule = core util) |
| CascadeStageElement | systems_biology/_cascades | RateLaw | `x' = -beta x` | `Element` + `Rate` decay | nothing |
| GoodwinStageElement | systems_biology/_networks | RateLaw | `x' = -b x` | `Element` + `Rate` decay (same as CascadeStageElement) | nothing |
| InhibitorElement | neuroscience/_networks | RateLaw | `z' = -phi z` | `Element` + `Rate` decay | nothing |
| RateUnitElement | neuroscience/_populations | RateLaw | `x' = x(I-x)/tau` (logistic, capacity I) | `Rate` = logistic growth with r=I/tau, K=I | nothing |

## 2. One role's state drives another role's rate (directed regulation / synapse)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| ActivationInteraction | systems_biology/_cascades | RateLaw | `down.x' += alpha up.<state>` | `Rate(down.x, alpha*up.s)` with 2 roles, or `Coupling(over=edge(up->down), f=x_j, read state != write state)` | nothing |
| SynthesisInteraction | systems_biology/_networks | RateLaw | `target.x' += k source.x` | same as ActivationInteraction (duplicate) | nothing |
| HillRepressionInteraction | systems_biology/_networks | RateLaw | `target.x' += a/(1+(y/K)^n)` | `Rate(target.x, hill_rep(repressor.x))` 2 roles | Hill function -> core util `hill` |
| RepressionInteraction | systems_biology/_genes | RateLaw or JumpLaw | `alpha0 + alpha/(1+(p/K)^n)` as rate or birth jump of `m` | `Reaction(reactants=[], products=[target.m], propensity=hill_rep(repressor.p), mode='rate'|'jump')` | nothing |
| DelayedSynapseInteraction | neuroscience/_networks | RateLaw+`delayed` | `target.v' += J source.s(t-D)` | `Coupling(state read 's', write 'v', f=x_j, delay=D, over=directed edges, weight J)`; currently N^2 separate Interactions over separate Elements | nothing |

## 3. Reaction networks (mass action, Michaelis-Menten, power law, Hill)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| MetaboliteElement | systems_biology/_networks | none (state `x>0`) | species holder | `CompartmentElement` (state `x`, open domain) | nothing |
| GeneralizedMassActionInteraction | systems_biology/_networks | RateLaw | `x' = N (gamma prod x^F)` | one `Reaction` per flux: real stoichiometry column, `propensity = gamma prod x^f` with modifiers not among reactants | matrix bookkeeping only -> thin subclass |
| SSystemInteraction | systems_biology/_networks | RateLaw | `x_i' = a prod x^g - b prod x^h` | two `Rate` per species (production / degradation power laws) or 2 `Reaction`s | nothing |
| GeneElement | systems_biology/_genes | RateLaw or JumpLaw | mRNA decay, translation, protein decay | `Reaction(m->0, delta)`, `Reaction(m -> m+p, b)` (catalysed), `Reaction(p->0, beta)`, `mode='rate'|'jump'` | nothing |
| PerClockElement | systems_biology/_clocks | RateLaw x5 | Goldbeter PER: Hill repression, MM phospho/dephospho, transport | `Reaction` with MM propensities `V x/(K+x)` for P0<->P1<->P2, `Reaction(P2<->PN)`, Hill-repressed synthesis of M | nothing beyond parameter choice (decomposes fully) |
| PhosphorylationLoopElement | systems_biology/_networks | RateLaw x3 | MM (de)phosphorylation with conserved totals, catalysed by other species | `Reaction` with MM propensity, catalyst = other state | conserved-total form `(A_T - A)` |
| GProteinElement | systems_biology/_cascades | RateLaw, `pulse` | `g' = k_on H(t)(1-g) - k_gtp g` | `Rate` with pulse forcing + `Rate` decay | nothing |
| CoagulationElement | physiology/_regulation | RateLaw x4, `pulse` | Hill(2) positive feedback, catalysed inhibition, chain | `Reaction`s with Hill / mass-action propensities + pulse source | nothing |
| GlucoseControlElement | physiology/_regulation | RateLaw x3, `pulse` | glucose with insulin (Hill) and glucagon (1-Hill) feedback | `Rate`s + `Reaction`(catalysed removal `k_i I G`) + pulse forcing | meal/exercise forcing composition |
| MinSkeletonElement | systems_biology/_min_proteins | RateLaw x5, `laplacian` | Min reaction-diffusion: attach/recruit/release/exchange | `FieldElement` + 4 `Reaction`s (mass action, autocatalytic attach `(k_d + k_dd md) cdt`) + diffusion `Coupling(over=field)` per state | nothing |

## 4. Conservative transfer between compartments (linear chains, flows)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| ChamberElement | physiology/_circulation | none (state `V`) | elastic compartment | `CompartmentElement(state V)` + parameters; pressure is a function of the element | time-varying elastance `_pressure` |
| TransferInteraction (physiology) | physiology/_circulation | RateLaw x2 | flow `Q=(P_up-P_down)/R` (valve: Q>=0), -Q / +Q | `Reaction([up.V],[down.V], propensity=Q, mode='rate')` or symbolic `Transfer`; NAME CLASH with brief `TransferInteraction` | pressure law |
| TwoCompartmentElement | physiology/_pharmacokinetics | RateLaw x2 | `Uc<->Up` (k12,k21), elimination k10 | `Reaction(Uc->Up,k12)`, `Reaction(Up->Uc,k21)`, `Reaction(Uc->0,k10)` | nothing |
| CellCycleElement | oncology/_cell_kinetics | RateLaw x3 | phase chain A1->A2->A3 -> 2 A1, blocked/killed fractions u1,u2 | `Reaction(A1->A2,l1)`, `Reaction(A2->A3,(1-u2)l2)`, `Reaction(A3->2 A1, (1-u1) l3)` (+ loss) | nothing |
| DormantCellCycleElement | oncology/_cell_kinetics | RateLaw x3 | dormant recruitment + cycle with split daughters | `Reaction`s with fractional products (b0,b1 split of 2 daughters) | nothing |
| TamoxifenTumourElement | oncology/_hormone_therapy | RateLaw x8 | Gompertz phase cycle (room factor), linear PK chain X0->X1->X2->X3, kills | PK chain = `Reaction`s; dosing `Rate`; kills = catalysed `Reaction`s | Gompertz room factor multiplying transitions |

## 5. Single-cell excitable / oscillator ODEs (continuous)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| FitzHughNagumoNeuronElement | neuroscience/_conductances | RateLaw x2 | cubic fast + linear slow | `Element` + `Rate` x2 | whole RHS (normal form) |
| HodgkinHuxleyElement | neuroscience/_conductances | RateLaw x4, CrossingLaw (record only) | conductance currents + 2-state gates `x'=a(v)(1-x)-b(v)x` | gates = `Transition`/`Reaction`(closed<->open, voltage-dependent rates) in fraction form; current sum `Rate`; spike record = `Crossing*` | ionic current law, rate functions |
| WilsonCowanElement | neuroscience/_populations | RateLaw x2 | relaxation to sigmoid response of E/I input | `Rate` x2 | shifted sigmoid response |
| ThetaNeuronElement | neuroscience/_populations | RateLaw (periodic state) | theta normal form | `Element(state period 2pi)` + `Rate` | whole RHS |
| PolyhomeostaticNeuronElement | neuroscience/_populations | RateLaw x3, `sin(t)` | driven leaky x, intrinsic plasticity of gain/threshold | `Rate` x3 (gradient of a KL objective in principle -> `Gradient` not usable: objective is stochastic expectation) | adaptation rules |
| MackeyGlassElement | physiology/_physiology | RateLaw, `delayed` | delayed Hill production - decay | `Rate` with `delayed` + `Rate` decay | nothing (Hill w/ delay) |
| AdaptiveResponderElement | physiology/_adaptivity | RateLaw x7, `pulse_train`, `where` | stimulus, effort, capacity, 4 cost integrators | `Rate` x7 (integrators of flows) | the cost model |
| DrugToxicityElement | physiology/_toxicity | RateLaw x2, `pulse_train`, `step` | finite infusion train, clearance, toxicity integrator | `Rate` infusion (finite pulse train) + decay; `Rate(T, c - eta T)` | nothing |

## 6. Threshold crossing with reset (hybrid / event dynamics)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| LeakyIntegrateAndFireElement | neuroscience/_neurons | RateLaw, CrossingLaw | leak + spike reset | `Rate` (OU-like relaxation without noise) + `Crossing*(v-th, {v: reset}, +1)` | nothing |
| IzhikevichElement | neuroscience/_neurons | RateLaw x2, CrossingLaw | quadratic v, recovery u, reset v and increment u | `Rate` x2 + `Crossing*` (reset of 2 states) | quadratic RHS |
| SpikingNeuronElement | neuroscience/_networks | RateLaw x2, CrossingLaw | LIF + exponential trace incremented at spike | `Rate` x2 (decays) + `Crossing*` | nothing |
| IntermittentSuppressionElement | oncology/_hormone_therapy | RateLaw x3, CrossingLaw x2 | hysteresis switch u on/off at PSA thresholds h1/h0 | `Rate`s + 2 `Crossing*` on observable `psa` (relay with hysteresis) | growth/death Michaelis forms |
| PiecewiseLinearProstateElement | oncology/_hormone_therapy | RateLaw x3, CrossingLaw x2 | switched linear system `x' = (u W_on + (1-u) W_off) x` + hysteresis | `Rate`s (linear) + 2 `Crossing*` | the two matrices |
| IrradiatedTumourElement | oncology/_radiotherapy | RateLaw x7, CrossingLaw (guard on t with counter state k) | scheduled impulses: LQ survival `exp(-damage)` multiplicative jump, transfers to damaged chain | `Rate`s/`Reaction` chain A5->A6->A2->A3->A4 + `Impulse*(times=start+k*period, reset)`; counter state `k` is a workaround | LQ damage expression |

## 7. Reaction-diffusion fields (FieldElement + laplacian)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| AlievPanfilovElement | physiology/_cardiac | RateLaw x2, `laplacian` | excitable medium + diffusion of u | `FieldElement` + diffusion `Coupling(over=field, f=x_j-x_i)` + `Rate` x2 | AP kinetics |
| FitzHughNagumoElement | physiology/_physiology | RateLaw x2, `laplacian` | FHN + diffusion (dup. of neuroscience FHN with D) | same as FitzHughNagumoNeuronElement + diffusion `Coupling` | nothing beyond FHN |
| TissueElement | oncology/_angiogenesis | RateLaw x2, `laplacian` | diffusion + decay of two fields | diffusion `Coupling(over=field)` + `Rate` decay | nothing |
| ExchangingCellElement | physiology/_exchange | RateLaw, `laplacian`, `degree` | MM consumption + network Laplacian + `xi*degree*x` | `Rate` (MM) + `Coupling(over=network edges, f=x_j-x_i, K=g+xi)` + `Rate(xi*degree*x)` (needs `degree` binding) | nothing |
| TermanWangElement | neuroscience/_networks | RateLaw x2, `laplacian` of a function | relaxation oscillator + lattice sum of neighbours' sigmoid activity `W(lap(S)+4S)` = `W sum_j S(x_j)` | `Rate` x2 + `Coupling(over=lattice edges, f(x_j,x_i)=W m S(x_j), reduce=sum)` | oscillator RHS |

## 8. Group-level (global / mean-field / bipartite) coupling

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| PulseCouplingInteraction | neuroscience/_populations | RateLaw, `mean` | `theta_i' += (1+cos theta_i) kappa mean_j P_n(theta_j)` | `Coupling(over=group, reduce=mean, include_self=True, f=(1+cos x_i) kappa P(x_j))` | pulse shape `P_n` (normalised raised cosine) |
| MutualInhibitionInteraction | neuroscience/_populations | RateLaw, `total` | `x_i' -= beta x_i sum_{j!=i} x_j / tau_i` over a LIST of separate elements | `Coupling(over=list of elements, reduce=sum, include_self=False, f=-beta x_i x_j / tau_i)` | nothing |
| GlobalInhibitionInteraction | neuroscience/_networks | RateLaw x2, `maximum` | hub z driven by max over lattice of S(x); each oscillator inhibited by S(z) | two `Coupling`s between a hub element and a group: group->hub `reduce=max`, hub->group broadcast | nothing |

## 9. Discrete-time maps, logical networks, stochastic automata

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| BooleanNodeElement | systems_biology/_boolean | none (state x in {0,1}) | binary holder | `Element` (or `CompartmentElement` with discrete domain) | nothing |
| BooleanFunctionInteraction | systems_biology/_boolean | UpdateLaw | synchronous truth-table update | `Map*(target.x, f(inputs))` with f = truth table polynomial | truth-table encoding (`_pattern`) |
| ThresholdFunctionInteraction | systems_biology/_boolean | UpdateLaw, `where` | sign of weighted sum, hold or decay at 0 | `Map*(target.x, threshold(sum w x))`; ~ core `ThresholdInteraction` but synchronous and weighted | nothing |
| ProbabilisticNodeElement | systems_biology/_probabilistic | none (x, function index) | binary + record of chosen function | `Element` | nothing |
| ProbabilisticFunctionInteraction | systems_biology/_probabilistic | UpdateLaw x2, `categorical` | each step draw a function with fixed probabilities, apply, record index | `Map*` with a categorical draw; not `Choice` (no utilities, fixed probabilities) | nothing |
| ContourCellElement | physiology/_exchange | UpdateLaw, `shift_member` | coupled map ring `x_i' = (1-c) r x(1-x) + c x_{i+1}^p` | `Map*` local logistic + update-form `Coupling(over=ring next, f=x_j^p)` combined convexly -> cannot be two separate UpdateLaws | nothing |
| AdaptrodeElement | neuroscience/_synapses | UpdateLaw x3, `where`, `mod` | 3-level cascade of leaky integrators driven by finite pulse train | `Map*` x3 (Euler map of relaxation cascade) | the cascade |
| ExcitableElementElement | neuroscience/_excitable | LocalRuleLaw (stochastic) | Kinouchi-Copelli cyclic CA: Q->E with prob 1-exp(-r)(1-p)^n, E->R1->..->Q deterministic | `Transition(0->1, prob 1-exp(-r)prod_j(1-p)^{[x_j=1]})` + deterministic cyclic `Map*`; brief Transition is rate-based | nothing |

## 10. Kinetic theory of active particles (activity-class densities)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| CellsElement | oncology/_immune_competition | none (state `f`) | density on activity class | IDENTICAL to core `DensityElement` -> delete or pure rename | nothing |
| TumourGrowthInteraction | oncology/_immune_competition | ProliferationLaw, EncounterLaw | proliferative + conservative encounters with host (bilinear tables) | `Bilinear*(candidate=tumour.f, field=host.f, rate, outcomes, mode='create'|'conserve')` | `step_table` outcome tensor, `activity_grid` |
| ImmuneCompetitionInteraction | oncology/_immune_competition | DestructionLaw, EncounterLaw | destructive (kill ∝ activity) + conservative down-shift | `Bilinear*(mode='destroy')` + `Bilinear*(mode='conserve')` | kill table from activity grid |

## 11. Agents on lattice fields (agent RuleLaws)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| TumourCellElement (+TumourCellView) | oncology/_angiogenesis | RuleLaw | consume field oxygen, set hypoxic flag, die below necrosis level, secrete VEGF if hypoxic, divide into a free neighbour cell with prob | `Transfer(source=field@location, target=Sink, amount=min(c, field))`, `Transfer(Source -> field VEGF)` conditioned, `LifeCycle(death: field<necrosis; birth: prob, placement=free neighbour)`, state update `hypoxic = field<h` | the ordering of the decision tree |
| VesselElement (+VesselView) | oncology/_angiogenesis | RuleLaw | supply oxygen to its cell each step | `Transfer(Source -> field@location, supply)` | nothing |
| SproutElement (+SproutView) | oncology/_angiogenesis | RuleLaw | activate above VEGF level; softmax chemotaxis step over neighbour cells; leave vessel agent behind; branch with prob | `Movement(step = Choice over neighbour cells with logit utility chemotaxis*field/level)` + `LifeCycle(birth of VESSELS at old location)` + `LifeCycle(birth of self, prob branching)` | activation rule |

## 12. Monolithic multi-population therapy ODEs (decomposition)

Common pieces: logistic growth `r X (1 - b X)` (=growth form), first-order clearance `-k U` (=decay), infusion `u`, pulse
train dose (=Rate with schedule), catalysed kill `-c E A` (=Reaction A+E->E), saturating kill `-c E A/(g+A)` (Reaction,
MM propensity), recruitment `rho E A/(alpha+A)` (Reaction products E, catalyst A), `1-exp(-U)` drug kill, delays.

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| TumourImmuneChemotherapyElement | oncology/_chemotherapy | RateLaw x4 | de Pillis-Radunskaya E/A/N/U | logistic `Rate` x2, `Reaction` kills, recruitment `Reaction` (MM), drug `Rate` infusion + decay | kill `1-exp(-U)` |
| MultidrugResistanceElement | oncology/_chemotherapy | RateLaw x14 | 8 resistance classes on a cube, mutation at division, threshold drug kill, 3 drugs + toxicity | mutation = `Reaction(S -> R1, r m1)` with parent growth; kills `Rate`; drug & toxicity `Rate` | hypercube mutation topology, `_excess` threshold kill |
| ResistanceSwitchingElement | oncology/_chemotherapy | RateLaw x3 | shared-capacity logistic, mutation/switch transitions, drug-induced switching | `Rate` (shared room), `Reaction(S->Rm)`, `Reaction(S<->Rp)` | shared room factor |
| RestingHuntingChemotherapyElement | oncology/_chemotherapy | RateLaw x4, `delayed` | delayed recruitment Q->E, saturating drug kill, drug consumption | `Reaction` with delayed propensity, logistic `Rate`, MM kill `Rate` | drug consumption by cells |
| MetastasisChemotherapyElement | oncology/_chemotherapy | RateLaw x6, `delayed` | two sites, delayed migration Ap -> As | 2x competition `Rate`s, migration `Transfer`/`Reaction` with delay (loss now, arrival after tau, fraction epsilon) | nothing beyond composition |
| OptimalChemotherapyElement | oncology/_optimal_therapy | inherits | same with bounded control input `u` (ParameterSpec(adaptive, bounds)) | as TumourImmuneChemotherapyElement; control = engine optimiser, not an interaction | control objective (factory observables) |
| OptimalImmunotherapyElement | oncology/_optimal_therapy | RateLaw x3 | Kirschner-Panetta with two controls | as KirschnerPanetta | control objective |
| AngiogenicTumourElement | oncology/_antiangiogenic | RateLaw x3, `pulse_train` | Gompertz to vascular capacity (state), Hahnfeldt K law, inhibitor | `Rate`(Gompertz with state capacity), `Rate`s, dosing `Rate` | Hahnfeldt stimulation/inhibition |
| VascularizedSpheroidElement | oncology/_antiangiogenic | RateLaw x7, `pulse_train` | geometric spheroid model with sigmoid switches | `Rate` x7 | whole RHS (geometry) |
| KirschnerPanettaTumourElement | oncology/_immunotherapy | RateLaw x3 | effector/tumour/IL-2 with MM terms | `Rate`s + MM `Reaction`s | nothing |
| VaccinatedTumourElement | oncology/_immunotherapy | RateLaw x5, finite pulse train | DC vaccination, logistic-modulated proliferation | `Rate`s, dosing `Rate` | product-logistic proliferation |
| CheckpointTumourElement | oncology/_immunotherapy | RateLaw x3, `where` | PD-1/PD-L1 complex inhibition, drug binding | `Rate`s | complex expression |
| DelayedVaccinatedTumourElement | oncology/_immunotherapy | RateLaw x9, `delayed` (incl. 3-arg with max delay) | 9 populations with multiple discrete delays | `Rate`s with `delayed` | whole RHS |
| SuppressedTumourElement | oncology/_immunotherapy | RateLaw x12 | 12 populations, cytokine suppression | `Rate`s, decays | whole RHS |
| TcrGeneTherapyElement | oncology/_gene_viral_therapy | RateLaw x2 | effector source + MM self-proliferation, logistic tumour, MM kill | `Rate`s + `Reaction` | nothing |
| SirnaTgfBetaElement | oncology/_gene_viral_therapy | RateLaw x5 | cytokine-modulated growth, siRNA silencing | `Rate`s | silencing factor |
| OncolyticVirusElement | oncology/_gene_viral_therapy | RateLaw x2 | shared room growth + mass-action infection | `Rate`(shared room) + `Reaction(AU+AV -> 2AV, alpha)` | nothing |
| OncolyticVirusDendriticElement | oncology/_gene_viral_therapy | RateLaw x6 | frequency-dependent infection/kill (÷ total) | `Reaction` with frequency-dependent propensity | nothing |
| CellQuotaProstateElement | oncology/_hormone_therapy | RateLaw x6, `pulse_train` | Droop cell quota growth, Hill mutation both ways, PSA | `Rate`s; Droop form shared with ecology `QuotaPhytoplanktonElement` | Droop uptake (`_quota_rate`) |
| AndrogenSuppressionElement | oncology/_hormone_therapy | RateLaw x3 | competition + deprivation-driven mutation AD->AI, U relaxes to U0(1-a) | `Rate`s + `Reaction(AD->AI, m1 (1-U/U0))` + OU-like relaxation `Rate` | nothing |
| AndrogenReceptorProstateElement | oncology/_hormone_therapy | RateLaw x5 | receptor-driven growth/death, PSA leak chain | `Rate`s + `Reaction(UIP->UP)` | receptor growth/death functions, age volume |
| ErythropoiesisElement | oncology/_cell_kinetics | RateLaw x4, `delayed` | delayed loss after fixed lifespan, MCV balance divided by L | `Rate`s with `delayed` | MCV balance |
| RadiationCompetitionElement | oncology/_radiotherapy | RateLaw x2, `pulse_train` | Lotka-Volterra competition + pulsed irradiation kill | competition `Coupling`/`Rate` + decay with schedule | nothing |
| StemCellDifferentiationElement | oncology/_stem_cells | RateLaw x8, `pulse` | two 4-stage lineages, feedback renewal, saturating returns, therapy modes | `Reaction` chain (k12,k23,k34...), logistic `Rate`s, `Reaction(NPr->APr, m1)` | renewal feedback, therapy switch |
| BoneMarrowBloodElement | oncology/_stem_cells | RateLaw x5, `delayed` | Gompertz-like marrow, delayed release with survival factor | `Rate`s; delayed transport = `Transfer` with delay + survival | survival factors |
| LeukaemiaStemCellsElement | oncology/_stem_cells | RateLaw x5 | niche exp(-beta(NS+AS)), progeny feedback | `Rate`s | niche feedback |
| ChemoimmunotherapyTumourElement | oncology/_combination_therapy | RateLaw x6, finite pulse sum | de Pillis 2009, ratio-dependent kill `D` | `Rate`s, dosing `Rate`s | `D` term |
| ClonalTumourElement | oncology/_combination_therapy | RateLaw x3 | two competing clones + immune | `Rate`s (competition) | nothing |
| ChemovirotherapyTumourElement | oncology/_combination_therapy | RateLaw x4 | infection, burst, drug with 3 infusion shapes | `Reaction(AU+V->AV)`, `Rate`s | infusion shape (param switch raising ParameterError in `_get_laws`) |
| GliomaBrainElement | oncology/_combination_therapy | RateLaw x6, `where` | glia/neuron/glioma + virus + TMZ | `Rate`s, `Reaction` infection | neuron loss `where(glia<0)` |
| Her2TumourElement | oncology/_combination_therapy | RateLaw x3, `where` on t | time-switched drug onset, binding, uptake | `Rate`s with `step(t)` | binding kinetics |
| AngiogenesisImmunotherapyTumourElement | oncology/_combination_therapy | RateLaw x9 (two drugs via `_drug_laws`) | Hahnfeldt + two 2-compartment PK drugs + resistance integrator + Hill effect | PK = `Reaction(c<->q)`, `Reaction(c->0)`, dosing `Rate`; Gompertz `Rate` with state capacity | resistance-shifted Emax |

## GAPS (mechanisms no brief class covers)

1. **Threshold events with reset** (`CrossingLaw`): LeakyIntegrateAndFire, Izhikevich, SpikingNeuron, HodgkinHuxley
   (record only), IntermittentSuppression, PiecewiseLinearProstate (hysteresis relays), IrradiatedTumour. Law shape:
   `CrossingLaw(guard(ns), reset={state: expr(ns)}, direction, terminal, label)`. Proposed
   **`CrossingInteraction(guard, reset, direction=0, terminal=False)`** on any roles (reset may write several states,
   of several roles). Relays/hysteresis = two crossings on one switch state; a thin subclass
   `HysteresisInteraction(signal, state, on, off)` could live in core as a special case (Schmitt trigger is pure math).
2. **Scheduled impulses**: IrradiatedTumour fakes `t = start + k period` with a counter state `k` and a crossing.
   Proposed **`ImpulseInteraction(times | (start, period, count), reset)`**: instantaneous resets at given times
   (also covers bolus dosing; the oncology pulse-train "infusions" are its smoothed counterpart).
3. **Synchronous discrete-time map** (`UpdateLaw` of a state from roles): BooleanFunction, ThresholdFunction,
   ProbabilisticFunction, ContourCell, Adaptrode. The brief has update form only inside `Coupling`. Proposed
   **`MapInteraction(state, function, inputs=roles)`** -> `UpdateLaw(state, f(ns))`; function may contain draws
   (`categorical`), so probabilistic Boolean networks are `MapInteraction` + a second recorded state. Truth tables are
   a thin `TableMapInteraction(table)` (logic is math, so core is fine).
4. **Bilinear kinetic encounters on class grids** (`EncounterLaw`, `ProliferationLaw`, `DestructionLaw`):
   TumourGrowth, ImmuneCompetition. No brief interaction lowers to them. Proposed
   **`BilinearTransitionInteraction(candidate, field, rate[h,k], outcomes[h,k,i], mode='conserve'|'create'|'destroy',
   test=None)`** (it is `Reaction` vectorised over classes: `f_h + g_k -> f_i + g_k`). Could alternatively be a
   tensor form of `ReactionInteraction`.
5. **Agent <-> field exchange**: TumourCell (consume, secrete), Vessel (supply), Sprout (read gradient). The brief's
   `TransferInteraction` moves amounts between members, not between an agent and the field cell at its location
   (`model.field`, `model.deposit`). Either extend `Transfer` (see corrections) or add
   **`DepositInteraction(field_state, amount(agent, value_here))`** (signed deposit into the field at the agent cell).
6. **Biased (chemotactic) movement**: Sprout picks a neighbour cell by softmax of a field. Brief `Movement(step)` lists
   random walks and "searching the best cell", not logit choice over neighbour cells. Proposed: `Movement` takes a
   `utility(cell)` + `rule` from `Choice` (logit / best), so tactic walks are `Movement(utility=field, rule='logit')`.
7. **Core utils (not classes) missing**: `hill(x, K, n)` / `hill_repression` (5 private copies: `_hill`, `_saturation`
   x2, `_saturating` x2, HillRepression, Repression, Coagulation inline, Glucose inline), `logistic(x, center, width)`
   (`_switch`, `_step`, `_sigmoid`), and a finite `pulse_train(..., count=)` (`_schedule`, `_doses`, DrugToxicity's
   `pulse_train*(1-step)`, Adaptrode's `where(mod(t,..)<w, where(t<stop,..))`).
8. **Control inputs** (OptimalChemotherapy/Immunotherapy, `ParameterSpec(adaptive=True, bounds)`): no interaction in
   the brief; it is an engine/optimiser concern. Keep as spec, but note `Feedback` is the only brief class touching it.

## BRIEF CORRECTIONS

1. **`Rate(state, rate)` must bind several roles** and write a state of one role while reading others: Activation,
   Synthesis, HillRepression, Repression, VascularSupport (rate of `vasculature.K` reads `tumour.n`), DelayedSynapse.
   Signature: `RateInteraction(target, rate, noise=None, **roles)` or `state='role.name'` paths; `rate(ns)` sees all roles.
2. **Growth forms with a state capacity**: VascularSupport, AngiogenicTumour, AngiogenesisImmunotherapy use Gompertz
   with `K` = a state of another element (Hahnfeldt). Core `GrowthInteraction` takes `K` as a parameter only. The
   growth subclasses of `Rate` need `K` bindable to an expression of another role.
3. **`Coupling` read state != write state**: DelayedSynapse reads `s`, writes `v`; Activation reads `up.<name>`, writes
   `down.x`; GlobalInhibition reads `x` (via S), writes `z` and vice versa. Needs `source_state`/`target_state` (current
   core `CouplingInteraction` has one `state`).
4. **`Coupling` between different groups/elements (bipartite, hub)**: GlobalInhibition couples a single hub element to a
   lattice group with `reduce=max` one way and broadcast the other. `over=` needs `(source_group, target_group)` and
   `reduce` must include `max`/`min`, not only sum/mean.
5. **`Coupling` over a list of separate elements and `include_self`**: MutualInhibition sums over j != i across a list
   role (`units=['a','b']`); core 'mean-field' includes self. Add `include_self=False` and accept a list role.
6. **`Coupling` per-edge weights and delays on directed graphs**: delayed LIF network builds N^2 `DelayedSynapse`
   objects; with a directed `NetworkSpace` + per-edge `J`, `D` it is one `Coupling(delay=edge attr, weight=edge attr)`.
7. **`Coupling` lowering on fields must use the `laplacian` operator** (with lattice spacing and boundary), not per-edge
   laws: AlievPanfilov, FHN, Tissue, MinSkeleton (5 states, two diffusivities), TermanWang (`laplacian` of a FUNCTION of
   the state), ExchangingCell (network Laplacian plus `degree*x`, needs `degree` exposed in `ns`).
8. **Update-form `Coupling` cannot be summed with another update**: ContourCell's next value is a convex combination
   `(1-c) f(x_i) + c g(x_{i+1})`; two interactions each producing an `UpdateLaw` for `x` conflict. Either the update
   form takes the local map (`Coupling(form='update', local=f, weight=c)`) or discrete-time interactions lower to
   additive increments `x_{t+1} = x_t + sum_k delta_k` that the engine accumulates.
9. **`Transition` needs a discrete-time (per-step probability) lowering and independent-trials combination**:
   Kinouchi-Copelli excites with `1 - exp(-r) prod_j (1-p)^{[x_j = E]}` per step and advances refractory phases
   deterministically. Rate-based `Transition` (sum of rates) is only the continuous-time limit. Add
   `combine='independent'` (probability `1 - prod(1 - p_j)`) and deterministic transitions (`probability=1`). Lowers to
   `LocalRuleLaw` on lattices, not `JumpLaw`.
10. **`Reaction` needs real stoichiometry and modifiers**: GMA (`stoichiometry` real matrix, kinetic orders independent of
    stoichiometry), PerClock/Phosphorylation (MM propensity reading a catalyst not consumed), CellCycle (`A3 -> 2 A1`),
    DormantCellCycle (fractional daughters `b0`, `b1`), circulation (signed flow `drop/R` when no valve). Signature:
    `Reaction(changes={state: real}, propensity(ns))`, `mode='rate'` must allow negative propensity; `mode='jump'` must
    reject it. Reactants list is optional when `propensity` is given.
11. **`Transfer` needs a symbolic (continuous) form, not only agent RuleLaws**: circulation, two-compartment PK, PK chains,
    delayed migration (Metastasis, BoneMarrow: loss now, arrival delayed and attenuated). Symbolic lowering:
    `RateLaw(source, -q)`, `RateLaw(target, +yield * delayed(q, tau))`. Agent form must accept a field cell at the
    agent location as source/target (TumourCell, Vessel). Add `delay=` and `yield=` (fraction arriving).
12. **`LifeCycle(birth, death)` needs offspring group and placement**: Sprout creates a VESSEL agent (other group) at its
    old cell; TumourCell places the daughter in a FREE neighbour cell; both with a probability. Add
    `offspring_group=None`, `placement='here'|'free_neighbor'|callable`, `probability=`.
13. **Elements**: `CompartmentElement` needs a `state=` name (TumourElement `n`, VasculatureElement `K`, ChamberElement `V`,
    MetaboliteElement `x>0`, BooleanNode `x in {0,1}`) and a state domain argument. `DensityElement` is duplicated by
    `CellsElement` verbatim.
14. **Name clash**: physiology `TransferInteraction` (pressure-driven flow) has the same name as the brief's general
    `TransferInteraction`; rename to a thin subclass (e.g. `ResistiveFlowInteraction`) of the general one.

## FIELD-SHARED (`biology/common/<family>`)

- `biology/common/populations`: growth/decay realizations duplicated between oncology `_growth` and ecology
  `_populations` (Exponential, Logistic/GeneralizedLogistic, Gompertz, PowerLaw, Loss/Mortality, delayed logistic).
  All are thin subclasses of `Rate` growth forms; keep only ExponentialLinear and Hahnfeldt (VascularSupport) in
  oncology. Also `RateUnitElement` is logistic.
- `biology/common/regulation` (gene/biochemical regulation): HillRepression (systems_biology/_networks) ==
  Repression (systems_biology/_genes, with jump mode), Synthesis == Activation (linear activation), decaying stage
  elements (CascadeStage == GoodwinStage, Inhibitor), GeneElement. Thin subclasses of `Rate`/`Reaction`.
- `biology/common/pharmacokinetics`: first-order clearance + infusion/dose schedule + compartment exchange appear in
  physiology (TwoCompartment, DrugToxicity) and in ~20 oncology elements (drug `U' = u - k U`, Tamoxifen X-chain,
  AngiogenesisImmunotherapy 2-compartment `c<->q`). Realizations: `ClearanceInteraction` (decay), `DosingInteraction`
  (Rate with finite pulse train), `CompartmentExchangeInteraction` (Reaction pair), `EmaxEffect` helper (Hill).
- `biology/common/excitable`: FitzHughNagumoNeuronElement (neuroscience) and FitzHughNagumoElement (physiology) are the
  same kinetics (one adds diffusion, which should be a `Coupling`); AlievPanfilov, TermanWang, HH gates share the
  excitable fast/slow shape. Also spike-and-reset neurons (LIF, Izhikevich, SpikingNeuron) share a `Crossing*` reset.
- `biology/common/kinetics`: Michaelis-Menten / Hill-propensity reactions used by systems_biology (PerClock,
  Phosphorylation), physiology (Coagulation, Glucose, ExchangingCell), oncology (MM kill/recruitment). The functions
  themselves (hill, MM, logistic switch) belong in core utils (pure math); the reaction realizations
  (`MichaelisMentenReaction`, `HillSynthesis`) here.
- `biology/common/cell_quota`: Droop quota uptake shared by oncology CellQuotaProstate and ecology
  QuotaPhytoplanktonElement (check ecology form before merging).

## Field-flavoured core names these models rely on

- `EncounterLaw`, `ProliferationLaw`, `DestructionLaw`: docstrings speak of "active particles", "candidate", "field",
  "test" particles (Bellomo KTAP vocabulary). Operation names: bilinear conservative / creative / destructive
  transition laws. Used by oncology `_immune_competition`.
- `DensityElement`: "density of a population at one activity class" (KTAP); generic meaning is "member of a
  distribution over a class grid". Duplicated by oncology `CellsElement`.
- `CompartmentElement`: docstring examples `'S'`, `'E'` (epidemiology); arg `n` hard-wired.
- `GrowthInteraction`: argument `h` "harvest" (ecology), state default `n` "population"; neither oncology nor ecology
  actually uses it (both re-implement), which itself is a sign the core version is not reusable as is.
- `LocalRuleLaw` is fine; `CrossingLaw` doc example uses height/velocity only as illustration (fine).
- `ParticleElement`, `ConstantForceInteraction`, `DampingInteraction`, `PotentialInteraction(position, velocity, mass)`:
  physics-named, not used here.

## Other defects noticed

- `ExcitableElementElement` (neuroscience/_excitable): doubled suffix, should be `ExcitableElement`.
- `FitzHughNagumoElement` / `MackeyGlassElement` docstrings are placeholders ("Fitz hugh nagumo element.").
- `ChemovirotherapyTumourElement._get_laws` validates `infusion` and raises `ParameterError` inside `_get_laws`
  (should be `_check`); BooleanFunction/ProbabilisticFunction/GMA/S-system also validate shapes inside `_get_laws`.
- `OptimalChemotherapyElement._get_laws` only calls `super()` (redundant override).
