# Economics audit: elements and interactions vs the brief's general core

Scope: every element/interaction class in `src/skcomplex/models/economics/**` (98 classes; `common/` holds only
`learning/_learning.py` functions, `common/institutions` and `common/markets` are empty dirs). Paths below are relative
to `models/economics/`. "Built from" names brief classes (proposed) or existing core classes; `R`=RuleLaw,
`Rate`=RateLaw, `Noise`=NoiseLaw, `Upd`=UpdateLaw, `Tr`=TransferLaw, `Enc`=EncounterLaw, `Match`=MatchLaw.

## 1. Table, grouped by mechanism

### M1 Continuous rate on a scalar / group state (ODE, SDE)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| AssetElement | finance/_assets | none (state `price`) | state holder | `Element` w/ `price: StateSpec` | state declaration only |
| GeometricBrownianMotionInteraction | finance/_assets | Rate(mu x), Noise(sigma x) | GBM | `GeometricBrownianMotionInteraction(RateInteraction)` in core: state='price', rate=mu*x, noise=sigma*x | nothing - move to core (math name) |
| CorrelatedGeometricBrownianMotionInteraction | finance/_correlated_assets | Rate, Noise(process=market.dW[i]) | GBM driven by a component of a correlated Wiener process | same core GBM + `process=` (see Correction C1) | nothing - merge into core GBM |
| CapitalStockElement | macroeconomics/_capital | Rate(K, i a K - K/L) | linear growth minus decay | `GrowthInteraction(form='exponential', r=i*a, h=1/L)` on state K | parameter naming |
| EconomyElement | macroeconomics/_growth | none (state k) | state holder | `Element` | state declaration |
| SolowGrowthInteraction | macroeconomics/_growth | Rate(k, s k^a - d k) | power growth with proportional removal | `GrowthInteraction(form='power', r=s, p=alpha, h=delta, state='k')` | nothing - thin rename |
| RamseyEconomyElement | macroeconomics/_optimal_growth | Rate(k, A k^a - d k - c), c adaptive control | power growth - proportional - constant offtake | `GrowthInteraction(power, r=A, p=alpha, h=delta)` + `RateInteraction(state='k', rate=-c)` | objective/terminal value (solver), control bounds |
| QuadraticSupplyMarketElement | markets/_prices | Rate(P, k(beta - lam P - (delta + a P + g P^2))) | excess-demand adjustment (Riccati) | `RateInteraction(state='P', rate=k*(D(P)-S(P)))` | demand and supply polynomials |
| StockElement | resources/_harvesting | Rate(n, r n (1-n/K)) | logistic | `GrowthInteraction(form='logistic')` (element law today) | nothing |
| HarvestInteraction | resources/_harvesting | Rate(n, -e count(harvesters) n) | proportional removal whose rate reads another subsystem's group size | `GrowthInteraction(h=...)` or `RateInteraction(rate=-e*count(harvesters)*n)`; needs h as expression (C2) | nothing |
| HarvestedResourceElement | resources/_optimal_harvesting | Rate(x, logistic - q E x), E adaptive | logistic with controlled proportional removal | `GrowthInteraction(logistic, h=q*effort)` (h expression, C2) | discounted profit objective |
| ArbitrageDensityElement | finance/_arbitrage | Rate(rho, D lap - advection + c(t) rho) | drift-diffusion-reaction PDE on a FieldElement | `RateInteraction(state='rho', rate=...)` with `laplacian`/`advection` utils (or FluxLaw) | gauge-field coefficients A_x, E, A_t |
| SaverElement | finance/_wealth | Rate(W, J/<k> lap W), Noise(sigma W, stratonovich) | network diffusion + multiplicative noise (Bouchaud-Mezard) | `CouplingInteraction(state='W', K=J/<k>, mode='edges')` + `RateInteraction(noise=sigma*W, convention='stratonovich')` (C1) | nothing |
| MeanFieldSaverElement | finance/_wealth | Rate(W, J N/(N-1)(mean W - W)), Noise strat. | mean-field coupling (self excluded) + multiplicative noise | `CouplingInteraction(mode='mean-field', K=J*N/(N-1))` + GBM noise strat. | nothing |

### M2 Stocks and flows (system dynamics: TransferLaw with Source/Sink, delays, lookups)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| BassMarketElement | innovation/_bass | Tr(P->A, (p + q A/(P+A)) P) | spontaneous + mass-action conversion (mean-field SI) | `ReactionInteraction(reactants=P, products=A, propensity=...)` or `TransferInteraction(P, A, amount)` | p, q naming |
| CommodityMarketElement | markets/_commodities | 4 Tr (Source/Sink), lookup, distributed_delay, step | stock-flow with Erlang delays | `TransferInteraction` (symbolic, Source/Sink) + delay/lookup utils | price/investment lookup tables, demand step |
| CarDealershipElement | markets/_inventory | 2 Tr, distributed_delay, step | stock management with perception/delivery delays | `TransferInteraction` | ordering heuristic |
| EchelonElement | markets/_supply_chains | Rate(backlog), Rate(expected) | stock holder + adaptive expectation (first-order smoothing) | `RateInteraction`; expected = `OrnsteinUhlenbeckInteraction(sigma=0)` form (C3) | shipment rule |
| SupplyLinkInteraction | markets/_supply_chains | 3 Rate, 2 Tr, distributed_delay | flow between two elements' stocks with delays | `TransferInteraction(source=upstream.inventory, target=downstream.in_transit)` + RateInteraction | Sterman ordering (`_orders`) |
| RetailDemandInteraction | markets/_supply_chains | 2 Rate, Tr->Sink, step | exogenous demand input | `RateInteraction` + `TransferInteraction(..., Sink)` | demand step |
| BreweryInteraction | markets/_supply_chains | Rate, Tr(Source), Tr, delay | production pipeline | `TransferInteraction` x2 + Rate | nothing |
| OilEconomyElement | resources/_extraction | 3 Tr (Source/Sink), lookup | depletion + capital investment/depreciation | `TransferInteraction` x3 (depreciation = decay) | yield/price tables, capped investment |
| FisheryElement | resources/_extraction | 4 Tr, lookup | renewable stock + capital | `TransferInteraction` x4 | regeneration/yield/price tables |
| HilsaElement | resources/_fisheries | ~17 Tr in a 7-stage chain, Source/Sink | linear compartment chain with residence times and fractional losses | `TransferInteraction` per leg - repeated chain shape is gap G6 | stage parameters, nests cap |
| RiceSectorElement | resources/_food | 3 Tr, 6 Rate `(target - x)/tau`, lookup | stock-flow + first-order goal seeking | `TransferInteraction`, `OrnsteinUhlenbeckInteraction(theta=1/tau, mean=target, sigma=0)` x5 (C3) | lookup tables, productivity |
| PlantationElement | resources/_plantations | 4 Tr, 2 Rate (goal seek, logistic with control) | land-use stocks + pest logistic | `TransferInteraction`, `GrowthInteraction(logistic, r=expr, h=expr)`, OU(sigma=0) | profitability lookups |
| HouseholdsElement / FirmsElement / MonetaryBankElement | macroeconomics/_monetary | none (deposit/loan/sales states) | state holders | `Element` (or `CompartmentElement`) | declarations |
| CircuitInteraction | macroeconomics/_monetary | 7 Tr (discrete), 2 Upd | stock-flow-consistent circuit, discrete map | `TransferInteraction` x7 (update form) + `MapInteraction` x2 (G1) | behavioural coefficients |

### M3 Discrete-time map of scalar states

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| AdjacentPossibleElement | innovation/_diversity | Upd x2 | 2-d map | `MapInteraction` (G1) | `_new_entities` |
| IncomeElement | macroeconomics/_growth | Upd with `lag(Y,1)` | 2nd-order linear recurrence | `MapInteraction(state='Y', lags=1)` (G1) | Samuelson coefficients |
| TradingEconomyElement | macroeconomics/_trade | Upd x4 | static elasticity map | `MapInteraction` x4 (G1) | elasticity formulas |
| MarketElement | markets/_cobweb | none (price) | state holder | `Element` | declaration |
| CobwebInteraction | markets/_cobweb | Upd(p, (a-c-d p)/b) | excess-demand map (cobweb) | `MapInteraction(state='price')` (G1) | linear demand/supply |

### M4 Kinetic encounters on density classes

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| WealthClassElement | econophysics/_welfare | none (state f) | density of one class | `DensityElement` (thin rename) | nothing |
| WealthExchangeInteraction | econophysics/_welfare | Enc(f, f, eta, table) | binary-encounter transition kernel (KTAP) | no brief class (G2); today core `EncounterLaw` directly | `exchange_table` (coop/competition kernel) |
| WelfarePolicyInteraction | econophysics/_welfare | Enc(f, policy.f, tax*pattern, table) | encounter with an external field population | G2 | `policy_table` |

### M5 Agent random pairwise exchange (conserved transfers)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| TraderElement | econophysics/_exchange | R(give) | each member gives 1 unit to a random member | `TransferInteraction(source=traders, target=random member, amount=1, if w>0)` (C4) | nothing |
| SaverElement | econophysics/_kinetic | none | state holder | `Element` | declaration |
| SavingExchangeInteraction | econophysics/_kinetic | R(trade) | N random pairs/step, pool (1-lambda) and split by uniform eps | `TransferInteraction(pairing=random pairs, amount=f(i,j,rng), sweeps=N)` (C4) | saving split rule |
| DebtExchangeInteraction | econophysics/_kinetic | R(trade) | random payer->payee fixed amount with floor | `TransferInteraction(amount=a, feasible=w-a>=-debt)` (C4) | floor rule |

### M6 Agent discrete choice / learning from model history

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| PatronElement | econophysics/_el_farol | R(decide) reads history, storage | best predictor (argmax of discounted errors) -> action | `ChoiceInteraction` agent form, rule='best' (C5) + `ReinforcementInteraction` scores (G3) | Arthur's predictor pool, capacity threshold |
| BarInteraction | econophysics/_el_farol | Feedback: report attendance | aggregate count | an `ObservableSpec(sum attends)` readable by rules (C6) | nothing |
| DinerElement | econophysics/_kolkata | R(choose uniform) | uniform random choice of option | `ChoiceInteraction(utility=0, intensity=0)` | nothing |
| CrowdAvoidingDinerElement | econophysics/_kolkata | R(choose) | stay w.p. 1/crowd else random other | `ChoiceInteraction` agent form w/ stay prob (C5) | crowd-avoid rule |
| RestaurantsInteraction | econophysics/_kolkata | R(serve) | occupancy count per option + random rationing 1 per option | gap G4 (`OccupancyInteraction`) | n_restaurants |
| SpeculatorElement | econophysics/_minority | none | strategy codes + virtual scores | `Element` | declaration |
| MinorityGameInteraction | econophysics/_minority | R(play) | best-of-2 strategy table lookup on history bits, minority payoff, score update, periodic replacement of worst | `ChoiceInteraction` agent 'best' (C5) + `ReinforcementInteraction` (G3) + `EvolutionInteraction(selection='truncation')` (C7) | strategy tables keyed by history, minority payoff |
| SpeculatorElement | markets/_brock_hommes | R(choose: logit_choice over fitness history) | logit among predictors | `ChoiceInteraction` agent form, rule='logit' (C5) | fitness read from `pricing.fitness_*` |
| HeuristicPricingInteraction | markets/_brock_hommes | Feedback act | price = sum shares*forecasts/R + noise; fitness of predictors | `FeedbackInteraction` | Brock-Hommes pricing + fitness (uses common heuristics) |
| ConsumerElement | markets/_fair_food | R(shop) logit with neighbour share | binary logit with social term, budget gate | `ChoiceInteraction(options=(0,1), utility=..+w*mean(neighbors.fair), rule='logit')` needs neighbour reduction (C5) | utility weights, budget gate (utility=-inf) |
| CustomerElement / ProductElement / ShopElement | markets/_perishables | none | state holders | `Element` | declarations |
| PerishableTradeInteraction | markets/_perishables | R(trade) | N random-sequential visits/step: Fermi choice of shop, random product, EMA satisfaction, restock | `ChoiceInteraction` agent 'fermi' (C5) + `ReinforcementInteraction` (G3) + sub-step sequential activation (C8) | freshness exp(-age/tau) |
| HouseholdElement | innovation/_adoption | R(adopt) | income x(1+g); adopt if any neighbour adopted and income>=threshold | `MapInteraction(income)` + `TransitionInteraction(phone 0->1, prob=where(income>=thr, max(neighbors.phone),0))` discrete form (C9) | nothing beyond thresholds |

### M7 Matching markets (MatchingInteraction / MatchLaw + custom pairing)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| StockTraderElement | finance/_order_book | R(quote), R(account), R(imitate) | forecast quote (fundamental/trend+noise), mark-to-market, copy better neighbour | quote: common `heuristics` (G-F1); account: `MapInteraction`; imitate: `ImitationInteraction(rule='best', payoff='gain', state='strategy')` discrete agent form (C9) | quote mapping forecast->bid/ask |
| BookInteraction | finance/_order_book | Matching(PriceTimePriority) + own report | continuous double auction, last-trade price carried forward | `MatchingInteraction(protocol=PriceTimePriorityProtocol)` + report option (C10) | opening level |
| ZeroIntelligenceBuyerElement / SellerElement | markets/_double_auction | R(quote uniform) | redraw limit each round | `MapInteraction` with a uniform draw (G1/C11) | budget-constrained bounds |
| AuctionLedgerInteraction | markets/_double_auction | Feedback act | surplus/efficiency report + reset quantities | Observable (C6) + per-round reset `MapInteraction` | `maximum_surplus` |
| (model) MatchingInteraction auction | markets/_double_auction | Matching + on_match `_book_profit` | DA / call market | core `MatchingInteraction` + Continuous/BatchClearing protocol | `_book_profit` (shared, F2) |
| FirmElement | macroeconomics/_credit | R(plan), R(invest) | demand = leverage*worth; random return, interest, default -> reentry | `MapInteraction` + `LifeCycleInteraction` replacement (C12) + `TransferInteraction` to partner bank (reference state, G5) | Delli Gatti return/default |
| BankElement | macroeconomics/_credit | R(quote) | ask = a e^-a, supply = e/ratio, reset | `MapInteraction` | pricing formula |
| CreditMarketInteraction | macroeconomics/_credit | Matching(PartnerSwitch) + spread report | partner-switch credit matching | `MatchingInteraction` + extra report | spread report |
| EmployerElement | markets/_labour | R(post) | Bernoulli vacancy + uniform bid | `TransitionInteraction` (discrete prob) + draw `MapInteraction` | nothing |
| JobseekerElement | markets/_labour | R(search) | separation w.p.; reservation decays geometrically | `TransitionInteraction(employer -> -1, prob)` + `MapInteraction(ask)` | nothing |
| SearchCountInteraction | markets/_labour | Feedback report | count searching | Observable (C6) | nothing |
| (model) MatchingInteraction labour | markets/_labour | Matching(StableMatching) + `hire` | Gale-Shapley | core | `hire` on_match |
| ResidentElement | markets/_shops | R(plan) | demand reset, Bernoulli online -> travel cost | `MapInteraction` with Bernoulli draw | nothing |
| StoreElement | markets/_shops | R(open), R(settle) | reset supply; profit; exit if money<0; reverse price direction when profit falls | `MapInteraction` + `LifeCycleInteraction(death)` + `SignAdjustmentInteraction` (G7) | cost accounting |
| ShoppingInteraction | markets/_shops | Match(delivered_price_choice) + R(buy) | random-order requesters take the cheapest feasible provider by `ask + travel*distance` | `MatchingInteraction(protocol=GreedyChoiceProtocol(score))` (G8) | delivered-cost score |
| LabourMarketInteraction | macroeconomics/_metaphoria | Match(_offers) + R(hire) | random-order seekers take highest-wage firm within sight, capacity=vacancies | G8 with score=wage, candidates within radius, capacity state | hire effect |
| FoodMarketInteraction | macroeconomics/_metaphoria | Match(_shops) + R(sell) | cheapest firm within sight; quantity min(want, stock, money/price) | `MatchingInteraction(G8)` with quantity as expression (C13), transfers money(-1), food(+1) | gdp report |
| GrazingInteraction | resources/_commons | Match(richest_pasture) + R(graze) | each cow takes richest free neighbouring cell, eats, metabolism, death, owner income | `MovementInteraction(best cell)` + `ForagingInteraction(uptake=appetite)` + `TransferInteraction(owner income)` (G5 owner ref) | milk income |

### M8 Bilateral barter (pairing + mutually improving trades)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| BartererElement | markets/_barter | none | holdings + Cobb-Douglas weights | `Element` | declaration |
| ExchangeInteraction | markets/_goods_traders | Match(pairing) + R(trade) | pairs trade units at geometric-mean MRS while both utilities rise | `MatchingInteraction`/MatchLaw + `SwapInteraction` (G9) | Cobb-Douglas, MRS price |
| GoodsTraderElement | markets/_goods_traders | R(wander) | random neighbour step w.p. mobility | `RandomWalkInteraction(step='neighbor', probability=mobility)` | nothing |
| BarterTraderElement | markets/_barter_networks | R(appraise) | mrs = good2/good1 | `MapInteraction` (or observable) | nothing |
| BilateralBarterInteraction | markets/_barter_networks | Match(_best_neighbors) + R(trade) | pair with neighbour of largest log-MRS gap, unit trades while both gain | G8 (score=MRS gap, candidates=neighbors) + G9 | Wilhite trade rule |

### M9 Lattice resources, foraging, life cycle (agents)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| SugarPatchElement | markets/_sugarscape | R(grow) min(K, r+g) | linear regrowth | `RegrowthInteraction(form='linear', rate=growback)` | nothing - delete |
| ForagerElement | markets/_sugarscape | R(forage) | move to best visible unoccupied cell (axes, vision), eat all, metabolism, die | `MovementInteraction(best cell, vision, axes, unoccupied)` (C14) + `ForagingInteraction(uptake=inf, gain=1)` | nothing |
| CensusInteraction | markets/_sugarscape | Feedback report | population, gini | Observables (C6) + `DistributionAnalysis('gini')` | nothing |
| PlotElement | macroeconomics/_metaphoria | R(grow) | linear regrowth rate*K | `RegrowthInteraction(linear, rate=regrowth*capacity)` | nothing |
| SettlerElement | macroeconomics/_metaphoria | R(live) | age, child fed by parent, need, death (starve/old), birth w/ mutated traits, quit/found firm, wander | `LifeCycleInteraction` (birth w/ inheritance+mutation, death by condition, C12), `TransferInteraction(parent->child)` (G5), `ForagingInteraction` metabolism, `RandomWalkInteraction` (mixture, C14) | enterprise/founding, monetary switch |
| BusinessElement | macroeconomics/_metaphoria | R(produce) | labour harvest of patch, wages, borrowing (money+loan legs), debt service, price/wage adjustment, dividends | `TransferInteraction` multi-leg (C4), `SignAdjustmentInteraction` (G7), Foraging-like harvest | vacancy target, production |
| CentralBankElement | macroeconomics/_metaphoria | R(distribute) | pay income to a random holder | `TransferInteraction(target=random member)` | nothing |
| CommonsInteraction | macroeconomics/_metaphoria | R(harvest) | pooled harvest of a cell split equally among co-located members | `ForagingInteraction(share='equal')` (C15) | labour - resistance |
| CensusInteraction | macroeconomics/_metaphoria | R(count) | reports + estate transfer to heir + removal | Observables + `LifeCycleInteraction(death, estate=transfer)` (C12) | heir rule |
| PastureElement | resources/_commons | R(grow) | logistic regrowth with reseeding floor | `RegrowthInteraction(logistic)` + floor (C16) | nothing |
| CowElement | resources/_commons | none | owner ref + energy | `Element` | declaration |
| HerderElement | resources/_commons | R(invest) | add/remove own cows by income/quota | `LifeCycleInteraction` acting on another group via owner ref (C12, G5) | invest rule |
| RegulatorInteraction | resources/_commons | Feedback act | quota halve if share<low, +1 if >high (AIMD band) | `FeedbackInteraction` or `SignAdjustmentInteraction` (G7) | band thresholds |

### M10 Multiplicative / random maps on agents

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| GibratFirmElement | finance/_growth_processes | R(grow) | x <- x N(mu, sigma), keep previous | `MapInteraction` with random draw (G1, C11) | nothing |
| KestenFirmElement | finance/_growth_processes | R(grow) | x <- lambda x + b, uniform | `MapInteraction` with draws | nothing |
| ReflectingFloorInteraction | finance/_growth_processes | R(reflect) | x <- max(x, c mean(previous)) | `BoundaryInteraction`/clip with group-mean bound (G10) | nothing |
| InvestingFirmElement | macroeconomics/_firms | R(trade) | deterministic map + random price + bankruptcy reset | `MapInteraction` + `LifeCycleInteraction` reset (C12) | toy-firm equations |
| DuopolistElement | markets/_duopoly | none | state holder | `Element` | declaration |
| CournotAdjustmentInteraction | markets/_duopoly | R(respond) | simultaneous best response to other's output with belief noise | `CouplingInteraction(form='update', over=others, function=response)` (C17) | Cournot response |

### M11 Network cascade, banking, queues, innovation (agents)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| BankElement | finance/_contagion | none | h, status, value | `Element` | declaration |
| DebtRankCascadeInteraction | finance/_contagion | R(spread) | h <- min(1, h + W^T(h*1[D])); U->D if h>0, D->I | `CouplingInteraction(form='update', weights=impact, reduce=sum)` (C17) + clip + `TransitionInteraction` discrete (C9) | nothing |
| DepositorElement | macroeconomics/_banking | R(pay), R(repay) | payment to random member with mirrored bank legs; repayment | `TransferInteraction` multi-leg via `bank` ref (C4, G5) | spending/repay fractions |
| ReserveBankElement | macroeconomics/_banking | R(lend) | lend excess reserves (credit creation, 6 legs) | multi-leg `TransferInteraction` | reserve rule |
| RegulatorInteraction | macroeconomics/_banking | Feedback act | parameter switches at a time | parameter schedule `step(t, change_at, ...)` (C18) | nothing - delete |
| CarElement / TillElement | markets/_queues | none | queue item / accounts | `Element` | declarations |
| WasherElement | markets/_queues | R(work) | serve front of queue per stage, crew cap 2 | gap G11 (`QueueInteraction`) or `MatchingInteraction(QueueProtocol(order='sequential'))` | crew size |
| CarWashOperationInteraction | markets/_queues | R(operate) | Bernoulli arrivals, balking, stage transition, departure + revenue | G11: LifeCycle birth/death + TransitionInteraction(stage) | prices |
| StallElement | markets/_produce | none | bitmask stock | `Element` | bitmask encoding |
| ShopperElement | markets/_produce | R(shop) | visit random unvisited provider, buy set intersection | `MatchingInteraction(RandomSearchProtocol)` if set-valued states existed (G12) | shopping-list semantics |
| PartElement | innovation/_technology | none | cost | `Element` | declaration |
| RedesignInteraction | innovation/_technology | R(redesign) | propose new costs on dependents, accept if total decreases | `SampleLaw`-backed `LocalSearchInteraction(rule='greedy')` (G13) | dependency matrix proposal |
| TechnologyElement | innovation/_technology | none | arity/table/cost | `Element` | declaration |
| InventionInteraction | innovation/_technology | R(invent) | random circuit from pool; create member if goal met | `LifeCycleInteraction(birth by construction)` | circuit logic, GOALS |

## 2. GAPS (no brief class covers)

- **G1 `MapInteraction(state, map, lags=0)`** - discrete-time analogue of RateInteraction: `x_{t+1} = map(ns)`, lowering
  to `UpdateLaw` (symbolic) or a per-member `RuleLaw` (agent). Used by AdjacentPossibleElement, IncomeElement (lag),
  TradingEconomyElement, CobwebInteraction, CircuitInteraction (loans, sales), and every agent rule that is a pure map
  (Gibrat, Kesten, InvestingFirm, ZI quotes, Resident plan, Bank quote, BarterTrader mrs). Could also be
  `RateInteraction(form='update')`, mirroring CouplingInteraction's `form`. Needs random-draw primitives (C11).
- **G2 `EncounterInteraction(population, field, rate, kernel)`** - quadratic master equation on class densities
  `df_i/dt = sum_hk eta B^i_hk f_h g_k - f_i sum_k eta g_k`; WealthExchange and WelfarePolicy use `EncounterLaw`
  directly. Mathematically a binary-collision kernel; name `BinaryKernelInteraction` to avoid "encounter of particles".
- **G3 `ReinforcementInteraction(scores, payoff, discount)`** - per-member vector of option scores updated by the
  realized payoff of every option, `s_k <- d s_k + (1-d) payoff_k` (or `+=`). El Farol (discounted predictor errors),
  Minority game (virtual scores), Perishables (satisfaction EMA), Brock-Hommes fitness. Pairs with ChoiceInteraction.
- **G4 `OccupancyInteraction(members, state, capacity, rationing='random')`** - count members per value of a discrete
  state (crowd), serve at most `capacity` per value at random, write count and served flag. Kolkata; also attendance
  of El Farol/Minority is the same group-by count. Needs a core "count by value" reduction.
- **G5 Reference states** - members holding the id of a member of another group (bank, partner, employer, owner,
  parent, heir). Every model scans `next(b for b in banks if b.index == agent.bank)`. Proposed core primitive: a
  `StateSpec(reference='group')` resolved by `agent.ref(name)`; TransferInteraction/LifeCycle then take
  `target='bank'` meaning "the referenced member". Used by banking, credit, labour, metaphoria, commons.
- **G6 `ChainInteraction(stages, times, losses)`** - linear compartment chain: outflow `x_k/tau_k` split into
  survival to `x_{k+1}` and named sinks. HilsaElement repeats the shape 6 times; same as Erlang delays (supply chains,
  commodity) and SEIR-like chains in other fields.
- **G7 `SignAdjustmentInteraction(state, signal, step, form='multiplicative'|'additive')`** - `x <- x(1 + step
  sign(signal))`, with optional "reverse direction when objective falls" (hill climbing). StoreElement price,
  BusinessElement price/wage, commons quota band (AIMD), Brock-Hommes/PriceAdjustmentProtocol iterations.
- **G8 `GreedyChoiceProtocol(score, candidates='all'|'neighbors'|radius, capacity)`** - requesters in random order take
  the feasible provider with best `score(requester, provider)`, capacity from a state. Six hand-written `Pairing`s
  have exactly this shape: `delivered_price_choice` (shops), `_shops`, `_offers` (metaphoria), `richest_pasture`
  (commons), `_best_neighbors` (barter networks), `neighbor_pairing` (score=random). Lets MatchLaw-based interactions
  become `MatchingInteraction`.
- **G9 `SwapInteraction(pairing, move, accept)`** - repeated pairwise moves between paired members while an acceptance
  (both objectives increase) holds; ExchangeInteraction (`trade_goods`), BilateralBarterInteraction (`_barter`).
- **G10 Agent bound / clip** - `BoundaryLaw` is symbolic only; ReflectingFloor needs `x <- max(x, c mean(prev))` per
  member with a group-reduction bound. Also elementwise `minimum/maximum/positive` are written as `where(...)` in 6
  modules (plantations `_positive`, fisheries `_smaller`, extraction `_investment`, supply chains, inventory, diversity):
  core `maximum`/`minimum` are group reductions, not elementwise.
- **G11 `QueueInteraction(arrivals, servers, service, balk)`** - FIFO service of stages by a pool of servers with
  per-item crew cap, arrivals at a probability, balking at queue length. Car wash. `QueueProtocol(order='sequential')`
  covers the allocation only.
- **G12 Set-valued states** - produce market encodes sets as bitmask floats (`wanted`, `visited`, `stock`). No core
  support; stays own unless a set/array state appears.
- **G13 `LocalSearchInteraction(proposal, objective, rule='greedy'|'metropolis', temperature)`** - wraps core
  `SampleLaw` for agent groups with a global objective; RedesignInteraction (NK technology). No brief class wraps
  SampleLaw.

## 3. BRIEF CORRECTIONS

- **C1 RateInteraction noise** - needs `process=` (component of a `WienerProcessSpec`, CorrelatedGBM) and
  `convention='ito'|'stratonovich'` (Bouchaud-Mezard). `noise=None` alone is insufficient.
- **C2 Growth/removal coefficients as expressions** - HarvestInteraction's removal reads `count(harvesters)` of another
  subsystem; optimal harvesting uses `q*effort` (adaptive control). `GrowthInteraction.h` and `r` must accept
  callables of the namespace, and GrowthInteraction should allow a constant offtake (Ramsey `-c`).
- **C3 OU/relaxation target as expression, sigma=0** - six goal-seeking laws `(target - x)/tau` (food, plantations,
  supply chains `expected`). Covered by OU only if `mean` takes an expression and `sigma=0` is allowed; otherwise add a
  `RelaxationInteraction` - but that name clashes with core `RelaxationLaw` (sandpile), see field names below.
- **C4 TransferInteraction(source, target, amount)** is too narrow for agents:
  needs `pairing` (random member, random pairs with N sweeps, neighbours, matches, referenced member G5),
  `amount(source_agent, target_agent, rng)` possibly signed (saving split), `feasible` guard (debt floor, w>0),
  multiple legs `[(state, sign, holder)]` like MatchingInteraction `transfers` (bank deposits mirror customer deposits;
  money+loan legs of credit creation), and `Source`/`Sink` targets in agent form (central bank money creation).
- **C5 ChoiceInteraction agent form** - the brief's symbolic signature `utility(namespace, option)` cannot express any
  economics choice: utilities read `model.history(...)` (Brock-Hommes fitness, El Farol attendance), `model.storage`
  (predictor pool, strategy tables), per-agent tables, or neighbour reductions (fair food). Add
  `utility(agent, model, option)` lowering to `RuleLaw` (as MatchingInteraction does), keep the symbolic form lowering to
  `UpdateLaw`; add rule `'best'` (argmax with random tie break = logit at infinite intensity) and a "stay with
  probability" option (crowd avoiding). Then `common/learning.logit_choice/logit_probabilities` are redundant
  (ChoiceInteraction already has `_draw_softmax`). Symbolic form must allow neighbour reductions, i.e. accept a
  `layer`/neighbour binding like CouplingInteraction.
- **C6 Reporter-only feedbacks** - BarInteraction, SearchCountInteraction, sugarscape CensusInteraction and part of
  AuctionLedger/metaphoria Census only report group reductions so that rules can read them with `model.history`. If
  rules could read `ObservableSpec` history, these classes vanish. Either make observables readable by rules or give
  FeedbackInteraction a `reports={name: reduction}` argument.
- **C7 EvolutionInteraction** - needs `selection='truncation'` (replace worst fraction) and "re-draw at random" as
  mutation, plus an `every=period` cadence (Minority game).
- **C8 Activation** - Perishables does N random-sequential sub-visits per step at fractional times; SavingExchange does
  N random pairs sequentially. A phase activation `'random-sequential', sweeps=N` would let generic interactions do it.
- **C9 Discrete-time probability form of Transition/Imitation/Spreading** - these lower to `JumpLaw` (rates). Agent
  models need per-step probabilities in DiscreteTime: separation (labour), vacancy (employer), adoption with
  `max(neighbors.phone)` gate (phone adoption), DebtRank status, order-book imitation (`rule='best'`, payoff state
  `gain`, probability). Add `form='rate'|'probability'` and allow agent groups (not only network nodes).
- **C10 MatchingInteraction reports** - BookInteraction and CreditMarketInteraction subclass only to change reports:
  add `price='mean'|'last'`, carry the last value when nothing matched (`opening`), and an `extra reports` hook; then
  BookInteraction is deleted. Also the field-flavoured names (`price`, `budget`, "payment", "delivery") - see below.
- **C11 Random draws in maps** - `MapInteraction` (or UpdateLaw) needs draw primitives (`uniform(a,b)`, `normal`,
  `bernoulli(p)`) for ZI quotes, Gibrat/Kesten, vacancy posting, online shoppers; today all are agent RuleLaws.
- **C12 LifeCycleInteraction(birth, death)** - also needs: birth with inheritance (`traits` copied with mutation),
  endowment transferred from parent; death with estate transfer to an heir and "remove only when settled"
  (metaphoria); replacement in place (credit firm reentry, InvestingFirm reset); a member creating/removing members of
  ANOTHER group (herder->cows, settler->business, invention->technologies, car wash arrivals).
- **C13 MatchingInteraction quantity/limit as expressions** - FoodMarket quantity is `reserve*need - food` (derived),
  limit of buyers is +inf; brief names states only. Accept callables `(agent) -> float`.
- **C14 MovementInteraction(step)** - sugarscape "best" step must scan along lattice axes up to `vision`, skip occupied
  cells, tie-break by distance; commons "best" step chooses among Moore cells by a resource with a capacity; metaphoria
  wander is a mixture (jump within `sight` w.p. exploration, else neighbour). Brief mentions "searching the best cell"
  without these arguments.
- **C15 ForagingInteraction share** - metaphoria Commons harvests a cell once (labour - resistance) and splits it
  equally among co-located members; foraging today is per-forager uptake. Add `share='each'|'equal'` and an uptake
  expression of the group at the cell.
- **C16 RegrowthInteraction logistic floor** - Pasture regrows from `max(r, seed*K)` (reseeding floor). Add `floor`.
- **C17 CouplingInteraction(form='update')** - needs edge weights from a matrix (DebtRank impact on a directed weighted
  network), `over='others'` (Cournot best response to the rival, mean-field excluding self; MeanFieldSaver uses the
  N/(N-1) trick), and noise on the read value (Cournot belief error).
- **C18 Parameter schedules** - banking RegulatorInteraction only switches a parameter at a time; a piecewise-in-time
  parameter (core `step`) removes the feedback class.
- **Agent rules reading model history (general fit)** - every agent form of the brief's interactions should take
  callables of `(agent, model)` with `model.history/observable/storage/rng` available (the MatchingInteraction/
  FeedbackInteraction pattern), lowering to `RuleLaw(phase=...)`; symbolic forms take `namespace` callables. Several
  brief signatures (Choice, Transition, Imitation, Transfer, Map) are symbolic-only today.

### Protocol renames (field-neutral, by operation)

| current | proposed | why |
|---|---|---|
| PriceAdjustmentProtocol | `IterativeClearingProtocol` | iterates a clearing level until requested ~ provided, then rations at it |
| PriceTimePriorityProtocol | `ContinuousClearingProtocol` (alt `LimitTimePriorityProtocol`) | each arriving offer matches resting offers by limit then arrival |
| UniformPriceProtocol | `BatchClearingProtocol` (alt `UniformLimitProtocol`) | collects all offers, one clearing level |

Also rename `ClearingSpec` -> `MatchingRoundSpec` (or keep with the new "clearing" vocabulary consistently), and
`MatchSpec.price` / report `price` / `budget` -> `level`/`limit`-family names (`value`, `budget` -> `capacity`?).

### Field-flavoured core names/meanings (flag)

- `MatchingInteraction`, `MatchSpec`, `_Offer`, protocols: `price`, `budget`, "payment of the requester", "delivery to
  the requester", "spend"; `PRICINGS`/`pricing` option names.
- `ClearingSpec` ("clearing" is market jargon); `PartnerSwitchProtocol` docs cite Caiani (credit), fine but "cheaper".
- `GrowthInteraction.h` "proportional harvest" (economics/ecology): call it removal.
- `ForagingInteraction` (forager, metabolism, uptake), `RegrowthInteraction` (resource), `RandomWalkInteraction`
  `cost`/`budget` - ecology/economics flavoured; mathematically uptake-with-constant-drain-and-death.
- `DensityElement` "activity class" and `EncounterLaw`/`DestructionLaw`/`ProliferationLaw` "active/candidate/test
  particle" (KTAP jargon) - WealthClassElement shows the use is non-physical.
- `TransferLaw` `Source` docs "a bank that creates money"; `StateSpec(quantity='money'|'credit')` is user binding, fine.
- `RelaxationLaw` is sandpile-specific ("relaxation"), clashing with goal-seeking relaxation `(target-x)/tau`.
- `DriveLaw` ("dropping a grain"), `FluxLaw` ("physical flux"), `ParticleElement`, `ConstantForceInteraction`,
  `DampingInteraction`, `PotentialInteraction`/`PairPotential`/`PairForce` (`position`, `velocity`, `mass` args) -
  physics names as already noted in the brief.

## 4. FIELD-SHARED (`economics/common/<family>`, thin subclasses)

- **F1 `common/learning`** (exists): forecasting `heuristics` are used by econophysics/_el_farol and
  markets/_brock_hommes, while finance/_order_book `StockTraderElement.quote` re-implements `fundamental` and `trend`
  inline - switch it to `heuristics`. `logit_choice`/`logit_probabilities` should be replaced by core ChoiceInteraction
  (C5). A thin `PredictorChoiceInteraction(ChoiceInteraction)` (utilities = predictor fitness from history) would serve
  El Farol, Minority game, Brock-Hommes (econophysics + markets).
- **F2 `common/markets`** (empty dir): (a) `on_match` surplus accounting (`_book_profit` in double_auction; credit
  `_lend`, labour `hire` are field-specific effects) as a shared `SurplusMatchingInteraction(MatchingInteraction)`;
  (b) report-extended matchings (finance BookInteraction last price, macro CreditMarketInteraction spread) become one
  `MatchingInteraction` with options (C10); (c) concentration measures `_largest_share` (duplicated in markets/_shops and
  macroeconomics/_credit) and `_herfindahl`; (d) greedy pairings (G8) used by markets/_shops, markets/_barter_networks,
  macroeconomics/_metaphoria, resources/_commons if G8 is not taken into core; (e) `cobb_douglas`/`marginal_rate`
  (markets only today - stays in markets).
- **F3 `common/institutions`** (empty dir): banks with balance sheets and credit creation: macroeconomics/_banking
  (ReserveBank, Depositor), macroeconomics/_credit (BankElement), macroeconomics/_metaphoria (CentralBank, `_borrow`,
  `_service`), finance/_contagion (BankElement). Shared `LendingInteraction(TransferInteraction)` with legs money/loan
  (C4) and a `BankElement` state holder (deposits, reserves, loans, equity). Regulators: `RegulatorInteraction` exists in
  both resources/_commons (quota band) and macroeconomics/_banking (time switch) - different mechanisms, same name;
  banking's becomes a schedule (C18), commons' a `SignAdjustmentInteraction` (G7).
- **F4 `common/wealth`**: holders/exchanges of wealth in econophysics (_exchange, _kinetic, _welfare) and finance
  (_wealth): `SaverElement` is defined twice with different states (econophysics `wealth`, finance `W`); one shared
  wealth holder plus the exchange realizations as `TransferInteraction` subclasses (C4).
- **F5 `common/firms`** (multiplicative random growth with entry/exit): finance Gibrat/Kesten, macroeconomics
  InvestingFirm and credit `FirmElement.invest` (random return, default -> reentry) as `MapInteraction` +
  `LifeCycleInteraction` thin subclasses.
- **F6 `common/resources`** (renewables on lattices): markets/_sugarscape SugarPatch, macroeconomics/_metaphoria Plot
  (both linear regrowth = core `RegrowthInteraction`, delete both), resources/_commons Pasture (logistic+floor); labour
  harvest `min(food, labour - resistance)` duplicated in metaphoria Commons and Business.
- **F7 Stock-flow helpers** (system dynamics): `demand + step(t, step_time, demand_step)` in markets/_commodities,
  _inventory, _supply_chains; elementwise clip helpers in resources/_plantations, _fisheries, _extraction (G10).
  Markets/resources share them -> `common/dynamics` or core utils.
- **Name clashes across packages to resolve**: `SaverElement` (econophysics, finance), `SpeculatorElement`
  (econophysics/_minority, markets/_brock_hommes), `BankElement` (finance, macroeconomics), `CensusInteraction`
  (markets, macroeconomics), `RegulatorInteraction` (resources, macroeconomics). Three gini implementations (markets
  `gini`, econophysics symbolic `gini_coefficient`, metaphoria via `DistributionAnalysis`) -> use the analysis.
