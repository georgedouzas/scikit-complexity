# Design

## Context

The inventory records 537 entries, each tagged after `redesign-system-model` with its time kind, space kind, component
kinds, interaction bindings, and law classes, and with its priority. Citation counts per source show which books
were mined: Bertin is cited 109 times, while eight books are cited four times or fewer.

## Goals / Non-Goals

**Goals:**
- Make the coverage of every source explicit, so a thin citation count means a thin source and not a skipped one.
- Find primitives the core lacks before more models of covered kinds are added.

**Non-Goals:**
- Implementing the surveyed models; their entries go pending or blocked on the changes that implement them.

## Decisions

### Survey record

The design records, per source, each chapter or section read, the models found, and the tags of each in the system
vocabulary: time kind, space kind, component kinds, interaction bindings, and law classes. A section with no model is recorded as such, with a reason such as "conceptual" or
"covered by an existing entry".

### Priorities

New entries take the inventory's rule: `core` when they exercise a category or a combination of categories that no
implemented model exercises, or do not map to the vocabulary, and `example` when they repeat a covered pattern. An entry
that does not map is blocked on the change that adds the general primitive it needs; if no change adds it, the survey
proposes one. A field-specific need, such as a named potential, a pairing algorithm, or a learning heuristic, is not a
missing primitive: the entry maps to the general primitive, and the realization is part of implementing the model in
`skcomplex.models.<field>`.

## Risks / Trade-offs

- [Conceptual books yield few models] → The survey records the sections it read, so the low count is explained.

## Survey of the sources

Every source was read section by section from text extracted with `pdftotext`. The records below list each chapter or
section read, the models found, and the reason when none was. New entries are in the inventory; existing entries that
a source also describes gained the source and section.

| Source group | Chapters or papers read | New entries | Existing entries given the source |
|---|---|---|---|
| Barsky 2020, Mihailović et al. 2023, Starkovich | 23 chapters | 4 | 12 |
| Mobus 2022, Satz 2022, Konieczny et al. 2023 | 37 chapters | 5 | 17 |
| Gooding 2019, Schwichtenberg | 21 chapters and the appendices | 4 | 3 |
| Papers, first half | 10 papers | 16, one shared with Barsky | 24 |
| Papers, second half | 10 papers | 11 | 16 |

Of the 39 new entries, 24 map to the vocabulary and wait on the newly proposed `add-surveyed-models`, and 15 are
blocked on the pending change that adds the general primitive they need: `add-higher-order-interactions` (6),
`add-link-fields` (2), `add-static-models` (2), `add-multiscale-networks` (2), `add-data-driven-dynamics` (2), and
`add-nonlocal-operators` (1).

### Needs outside the vocabulary

- Spatially embedded network generators, the Waxman model and random geometric graphs, which the spatial variant of
  `triadic_percolation` needs (2311.14877). Routed to `add-adaptive-networks`, which adds the other generators. No new
  entry is blocked on it.
- Noise that conserves a field, the divergence of a random flux, for `active_model_b` and noisy Cahn-Hilliard models.
  `add-surveyed-models` checks whether `Noise` with a `Flux` law covers it before implementing them.
- Analysis measures without models of their own, recorded for later changes: detrended fluctuation analysis and
  skewness as early-warning indicators, a replica-comparison ergodicity coefficient and damage spreading
  (2604.15669), Kolmogorov, Lempel-Ziv, and change complexity (Mihailović), Simonian structural complexity (Mobus),
  persistent homology of activity patterns (2311.14877), and Koopman isochrons and transfer operators (2102.12086).

### Barsky 2020 (Bs): Entropy of Complex Processes and Systems

| Chapter | Sections read | Models or reason none |
|---|---|---|
| 1 Brief history: entropy | 1.1–1.9 | 1.1–1.3, 1.5–1.9: conceptual (Clausius and Boltzmann entropy, ensembles, Gibbs paradox, gas separation, second law and biology). 1.4: **Ehrenfest urn model** of balls drawn at random between two boxes → new `ehrenfest_urn`. |
| 2 Statistical entropy component | 2.1–2.14 | 2.1–2.7: probability background and measures (composition entropy of polyfractional mixtures). 2.8–2.11: separation efficiency as an entropy difference, the binary optimality condition, and a worked four-component example. These measures need a separation model to probe them, and the existing `cascade_classifier` covers it. 2.12: model of separation into n components, an optimal ordering of boundaries over a graph, run on a cascade apparatus; covered by `cascade_classifier` (add source). 2.13–2.14: entropy measures of construction completeness and of resource processing; static data evaluation, no model. |
| 3 Dynamic component of entropy | 3.1–3.15 | 3.1–3.2: conceptual (analogy in modelling). 3.3–3.6: statistical model of a critical two-phase flow, with N particles each oriented up or down and a binomial distribution of the separation factor; covered by `two_phase_zone_exchange` (add source Bs 3.3). 3.7–3.13: maximum-entropy stationary states, phase-space volume, mobility, invariants, and canonical ensembles; analytical statistical mechanics of the same model, no new model. 3.14: already the source of `two_phase_zone_exchange`. 3.15: cellular (Fermi-like occupancy) treatment of mass transfer; an analytical derivation, no new model. |
| 4 Verification for two-phase flows | 4.1–4.10 | 4.1: redistribution of a mixture in a multistage cascade (balance equations over stages); covered by `cascade_classifier` (add source Bs 4.1). 4.2–4.8, 4.10: experimental checks, unified separation curves, distribution coefficients, and generalizing invariants; data fitting. 4.9: correlation of the structural and cellular models (similarity criteria for flow regimes); no new model. |
| 5 Entropy in technical sciences | 5.1–5.7 | Conceptual (complexity growth, Darwinism, world view). |

### Mihailović et al. 2023 (Mi): Physics of Complex Systems

| Chapter | Sections read | Models or reason none |
|---|---|---|
| 1 Prolegomenon | 1.1–1.4 | Conceptual. |
| 2 Gödel's incompleteness | 2.1–2.4 | Conceptual and mathematical logic; Gödel's universe only mentioned. |
| 3 Time in physics | 3.1–3.4 | 3.1–3.2: conceptual. 3.3: environmental-interface energy balance map `X_{n+1} = A X_n − B X_n²`, reduced to the logistic map; covered by `logistic_map` (add source). 3.4: time-scale separation (Michaelis-Menten, a figure of competition-limited diffusion quoted from another paper, not described); functional-time barcode of synchronization in cells on a ring, which is the model of 10.2 → `closed_contour_cell_exchange`. |
| 4 Model and theory | 4.1–4.5 | 4.1, 4.3: conceptual (discrete versus continuous, the computational uncertainty principle). 4.2: **Couzin et al. (2005) model** of point particles with repulsion, alignment and attraction plus noise, for fish shoals → new `couzin_leadership`. 4.4: Lyapunov time and Kolmogorov time on the logistic map with sinusoidally modulated r; covered by `logistic_map` (add source). Lorenz's 12-equation baroclinic model is only mentioned historically. 4.5: coupled soil and deep temperature maps, already the source of `soil_temperature`. |
| 5 Inaccessible information | 5.1–5.5 | 5.1–5.3, 5.5: conceptual (information, chance, Shannon). 5.4: rare events and a superposed Poisson signal count for flerovium data; statistical data analysis, no model. |
| 6 Kolmogorov and change complexity | 6.1–6.4 | Measures (Lempel-Ziv Kolmogorov complexity, Aksentijevic-Gibson change complexity, the KC spectrum) applied to observed data (ENSO, LIGO, Bell experiments, river flows); no model. |
| 7 Separation of scales | 7.1–7.4 | 7.1: conceptual (scaling, universality). 7.2: renormalization group; Feigenbaum universality of the logistic map, so add source to `logistic_map`; 1D Ising decimation is only mentioned. 7.3: Ising and Heisenberg models; Heisenberg is already `quantum_heisenberg` (Mi 7.3), so add Mi 7.3 to `ising`. 7.4: "breaking" point from running Kolmogorov complexity on the logistic map (add source to `logistic_map`), refers back to `soil_temperature`; neutron and gamma detector data. |
| 8 Turbulent flows | 8.1–8.4 | Kolmogorov 1941 theory as concept; KC and randomness measures of measured mixing-layer, canopy and river data; no model. |
| 9 Complex systems and art | 9.1–9.4 | 9.1: overview of brain models naming Hodgkin-Huxley, Wilson-Cowan and large-scale brain networks; covered by `hodgkin_huxley` and `wilson_cowan` (add source). 9.2–9.4: conceptual, and complexity measures of paintings and perception. |
| 10 Biophysical systems | 10.1–10.4 | 10.1: **intercellular exchange through gap junctions and tunnelling nanotubes** on 100-cell Newman-Gastner and modular Erdős-Rényi networks: `dx_i/dt = −α_i x_i/(β_i + x_i) + Σ_j [g_ij (x_j − x_i) + ξ δ_ij x_j]` → new `nanotube_gap_junction_exchange`. 10.2–10.3: **closed-contour cell substance exchange**, a ring of logistic cells each coupled to the next by a power-map uptake `c_i x_{i+1}^{p_i}`; stability regions → new `closed_contour_cell_exchange`. 10.4: change complexity of EEG in schizophrenia; data analysis. |

### Starkovich 2021 (Sm): The Structures of Mathematical Physics

| Chapter | Sections read | Models or reason none |
|---|---|---|
| 1 Sets, relations and maps | 1.1–1.5 | Mathematical background. |
| 2 Groups | 2.1–2.8 | Mathematical background (cyclic, symmetric, dihedral and continuous groups). Group-defined spaces are the pending `group-spaces` primitive, but the book names no model. |
| 3 Rings and fields | 3.1–3.4 | Mathematical background (complex numbers, quaternions). |
| 4 Vector and tensor spaces | 4.1–4.6 | Mathematical background. 4.4.2 Example 4.11: standing wave with fixed ends; covered by `standing_waves` (add Sm 4.4). Spherical harmonics, Hermite functions and the quantum oscillator appear only as orthonormal bases, not as dynamics. |
| 5 Algebras and operators | 5.1–5.8 | Mathematical background (Lie and Poisson algebras, Hermitian and unitary operators). Problem 5.17: two masses and three springs, normal modes; covered by `two_mass_spring_damper` (add source). |
| 6 General topology | 6.1–6.8 | Mathematical background. |
| 7 Differentiable manifolds | 7.1–7.7 | Mathematical background (forms, Jacobians). 7.6: Hamilton's equations, phase portraits of the free particle, falling particle and pendulum (Fig. 7.12), and Liouville's theorem; covered by `pendulum` and `projectile` (add source). Problems 7.13–7.14: already `particle_in_a_well` (add Sm 7.14). |
| 8 Integration and Lie groups | 8.1–8.6 | Mathematical background (Stokes, homotopy, Lie groups, integral transforms). |

### Mobus 2022 (`Mo`)

A conceptual systems-science book. Its only formal content is the system definition (Ch 4), a handful of equations,
and stock-and-flow sketches. It has 16 chapters.

| Chapter | Sections read | Models or reason none |
|---|---|---|
| 1 Introduction | 1.1–1.6 | Conceptual (framework, deep systems analysis outline). |
| 2 Principles of Systems Science | 2.1–2.7 (12 principles, SD, what goes wrong) | Conceptual. |
| 3 System Ontology | 3.1–3.7, incl. 3.3.1 ontogenic cycle, auto-organization | Conceptual. Auto-organization of particles with affordances is described qualitatively only, with no rules. |
| 4 A Model of System and a Language | 4.1–4.6, incl. 4.3.3 formal definition (Eqs 4.1–4.10) | Mathematical background (the system tuple ⟨C,N,G,B,T,H,Δt⟩ matches the core's system model). Simonian complexity (Eq 4.10) is a structural measure, not a model. |
| 5 Process of Understanding Systems | 5.1–5.4 | Conceptual (process and knowledgebase pipeline). |
| 6 Deep Systems Analysis | 6.1–6.10, incl. 6.5.3 transformation function (Eq 6.1), 6.6.2 sources and sinks (Eqs 6.2–6.3), 6.9.1 fuzzy membership (Eqs 6.4–6.5), 6.9.2 homeostasis, autopoiesis, adaptive response, 6.9.4 Adaptrode (Eq 6.6) | Adaptrode is covered by existing `adaptrode`. Homeostasis/thermostat (6.9.2.1): add source to `thermostat`. Adaptive response (6.9.2.3): part of the new `adaptive_response_costs`. System identification and fuzzy membership are analysis or conceptual. |
| 7 Analyzing Example Systems | 7.1–7.7 (computer, neuron, brain, organization, HSS) | Conceptual decompositions. The neuron and synapse decomposition (7.3) has no dynamics beyond the Adaptrode. |
| 8 Capturing Knowledge | 8.1–8.5 | Conceptual (database schema). |
| 9 HSS Economy | 9.1–9.8, incl. 9.6.1.1 finite-reserve extraction dynamics, 9.6.2.1 self-replicating PV exercise | Covered by `energy_return` (Mo 9.6). The PV exercise is an accounting framework (A+B+C+D ≪ T), covered by the same entry. |
| 10 Model Archetypes for CAS, CAES | 10.1–10.5, incl. 10.3.2.1 adaptivity (Eq 10.1 cost objective, stimulus-response, adaptive response, anticipation), 10.4 VSM, Living Systems | **New `adaptive_response_costs`** (Mo 10.3.2.1, 6.9.2). The rest is conceptual. |
| 11 The Agent Model | 11.1–11.6 (agent types, decision models, learning) | Conceptual; no decision rules are specified. |
| 12 The Governance Model | 12.1–12.6, incl. 12.3.2 feedback and requisite variety, 12.3.3 cooperation between near neighbours with buffers and feedforward, 12.3.3.4 PID, 12.3.4 coordination, 12.4 biological examples | PID is covered by `pid_loop` (Mo 12.3). Homeostat (12.3.2.1): add source to `thermostat`. Process chains with buffer stocks and feedforward messages (12.3.3): add source to `beer_game`. Hierarchical governance is conceptual. |
| 13 The Economy Model | 13.1–13.5, incl. 13.2.2 supply chain, 13.2.3 market dynamics and buffering, 13.4.2 demand-driven versus speculative production | Supply chain with buffering is covered by `beer_game` (add Mo 13.2.3). The rest is conceptual. |
| 14 Complex Artifactual Systems | 14.1–14.4 | Conceptual (design cycle). |
| 15 Designing CAES | 15.1–15.4, incl. 15.2.1.5 simulation, 15.3 agri-village | Conceptual; the agri-village is a design narrative without quantities. |
| 16 Societal Systems | 16.1–16.4 | Conceptual (normative HSS design). |

### Satz 2022 (`Sa`)

A popular-science book on complexity in physics. It names many classic models; most are already in the inventory. It
has 16 chapters.

| Chapter | Sections read | Models or reason none |
|---|---|---|
| 1 Introduction | all | Conceptual overview of the book. |
| 2 The Flow of Time | Time invariance, Joule's experiment, entropy, Layzer | Joule free expansion with balls in boxes: add source to `ideal_gas`. Layzer's scenario is conceptual. |
| 3 Global Connections | coins on a table, forest fire on a site lattice, bond percolation | `continuum_percolation` already cites Sa 3. Burning a dilute forest at p_c ≈ 0.59: add source to `site_percolation`. |
| 4 The Nature of Forces | Newton and Coulomb, galaxy rotation and dark matter, entropic force, black-hole entropy, Verlinde | Conceptual or mathematical background. |
| 5 The Formation of Structure | gravitational clustering, CMB, star formation | Conceptual (qualitative gravitational clustering; `n_body` covers the mechanics). |
| 6 The Energy of Space | Friedmann–Lemaître, cosmological constant, Hubble law, energy balance | Covered by `friedmann_expansion` (Sa 6). |
| 7 Critical Behavior | critical exponents, Ising, Kadanoff blocking, renormalization, lattice QCD | Add Sa 7 to `ising` and `lattice_gauge_fields`. Block-spin coarse-graining is an analysis of Ising configurations. |
| 8 Self-Organized Criticality | correlation power laws, Gutenberg–Richter, slip-stick block, sandpile | Add Sa 8 to `burridge_knopoff` (slip-stick) and `sandpile`. |
| 9 Fractal Dimensions | box dimension, Koch curve, abelian sandpile from a central source | Add Sa 9 to `koch_curve` and `sandpile`. |
| 10 Bifurcation and Chaos | Lorenz, logistic map and Feigenbaum, Fibonacci rabbits, bees, tiling | `logistic_map` already cites Sa 10. Add Sa 10 to `lorenz` and `fibonacci_rabbits`. The tiling is not a model. |
| 11 Brownian Motion | coin sequences, 1D random walk, diffusion, Einstein–Smoluchowski | Add Sa 11 to `random_walk` and `brownian_motion`. |
| 12 Turbulence and Convection | Rayleigh–Bénard rolls, Bénard–Marangoni hexagons, Libchaber–Maurer period doubling, Fourier analysis | Add Sa 12 to `rayleigh_benard` (rolls, period doubling). **New `benard_marangoni`**. Fourier analysis is a measure. |
| 13 Intermittency | balls-in-cells moments, Ising clusters, products of random variables | Add Sa 13 to `multiplicative_process`. Normalized moments are a measure. |
| 14 Words and Numbers | Zipf for words, prime factors, cities, Gutenberg–Richter | Empirical laws and a number-theoretic count, with no generative model (Simon's model is not described). |
| 15 Quantum Complexity | bosons and fermions, Bose–Einstein condensate, superfluidity, superconductivity, Cooper pairs | Conceptual (qualitative, no model specified). |
| 16 Conclusion | all | Summary. |

### Konieczny et al. 2023 (`Ko`)

A descriptive systems-biology text. Its quantitative content is enzyme kinetics, the fuzzy oil drop folding model,
and information measures. Its regulation chapter names concrete mechanisms that are standard ODE motifs. It has 5
chapters.

| Chapter | Sections read | Models or reason none |
|---|---|---|
| 1 Structure and Function | 1.1.1–1.1.5 (incl. 1.1.4 dissociation and Michaelis–Menten constants, inhibition), 1.2 self-organization, 1.3.1 late-stage folding hypothesis (Eqs 1.1–1.9) | Covered by `michaelis_menten` (Ko 1.1.4) and `fuzzy_oil_drop` (Ko 1.3.1). The rest is descriptive biochemistry. |
| 2 Energy in Biology | 2.1–2.8 (thermodynamics, water synthesis, ATP, photosynthesis, efficiency, entropy) | Mathematical background (ΔG relations); no dynamics. |
| 3 Information | 3.1 Shannon information, 3.2 replication, expression, epigenetics, development, evolution genetics, 3.3–3.6, 3.5.1.2 immune repertoire P = 1−(1−p)^k, 3.7.1 early-stage folding (Eq 3.1) | Covered by `early_stage_folding_intermediate` (Ko 3.7.1). Information and repertoire probabilities are measures or static formulas. The rest is descriptive. |
| 4 Regulation | 4.1, 4.2 receptors, effectors, allosteric negative feedback, 4.3, 4.4.1–4.4.10 (encoding, amplification, cascade amplifier, positive feedback, attenuation, inactivation, discrimination, coordination, extracellular control, cell population), 4.5–4.7, 4.8.1 proteome hypothesis | **New `hormone_cascade_amplifier`** (4.4.2, 4.4.3, 4.4.6), **`coagulation_positive_feedback`** (4.4.4, 4.4.7), and **`glucose_dual_hormone_control`** (4.4.5). Allosteric end-product inhibition (4.2): add source to `goodwin_oscillator`. Cell population control (4.4.10): add source to `stem_cell_differentiation`. Discrimination by receptor clustering and the proteome hypothesis are conceptual. |
| 5 Interrelationship | 5.1–5.8 (cooperation, enzyme activation and inhibition, gene-expression coupling, common metabolites, safety valves, microbiome, bioengineering), 5.9.1 carcinogenesis, 5.9.2 criteria of life | The insulin/glucagon coordination of metabolism (5.4) supports `glucose_dual_hormone_control`. Carcinogenesis by mutation accumulation is qualitative. The rest is conceptual. |

### Gooding 2019

*Economics for a Fairer Society: Going Back to Basics using Agent-Based Models* (source key `Go`). A NetLogo book:
almost all models are variants of the Toy Trader and of Metaphoria, and the inventory already holds the Toy Trader
family.

| Chapter | Sections read | Models or reason none |
| --- | --- | --- |
| 1 Introduction | all | Conceptual. |
| 2 First Nation Transition | all | Conceptual (field narrative). |
| 3 Complexity, Emergence, and Evolution | 3.1, 3.2, 3.3 (3.3.1-3.3.3), 3.4 | Conceptual: definitions of emergence, fitness test, evolutionary solutions. |
| 4 Agent-Based Model History and Development | 4.1-4.4 | Conceptual. Boids, FERMIAC, Sugarscape, and the minority game are only mentioned; Sugarscape and the minority game are existing entries. |
| 5 Netlogo | 5.1-5.4 | Software background (patches, toroidal topology, Mersenne Twister, plots). |
| 6 Evolution in Netlogo | 6.1 Genetic algorithms, 6.2 Evolution by circumstance, 6.3 Evolutionary strength | RobbyGA (6.1) and its variant with lifetimes, starvation culling of the worst fraction, random breeding, and the culling switched on and off (6.2-6.3): covered by existing entry `robby_ga` (already cites `Go 6`). |
| 7 Verifying Model Results | 7.1, 7.2 (7.2.1-7.2.2), 7.3 | Methodology: conservation checks and correlation of Gini series across runs. Covered by existing entry `goods_traders`. |
| 8 Money Distribution | 8.1-8.5 | Simple Economy (8.1): existing `boltzmann_wealth` (cites `Go 8.1`). Toy Trader (8.2-8.3): existing `goods_traders` (cites `Go 8–12`). Gini coefficient (8.4): analysis measure, probed by these models. |
| 9 System Efficiency | 9.1-9.5 | El Farol (9.2): add source `Go 9.2` to existing `el_farol`. Rand/Stonedahl evolutionary El Farol (9.3): a variant of `el_farol`, same source. Axelrod tournaments (9.3): add source `Go 9.3` to existing `prisoners_dilemma_tournament`. Wilhite (2001) bilateral trade on four network topologies (9.3): **new entry `wilhite_trade_networks`**. Santa Fe double-auction tournament and DeDeo-Miller prisoner's dilemma (9.3): only mentioned, no specification. Toy Trader efficiency with market-sight and density (9.4): existing `goods_traders`. |
| 10 Price Basics | 10.1, 10.2 (10.2.1-10.2.2), 10.3, 10.4 | Toy Trader with mutation and imitation of the top 5% under money or asset fitness tests: existing `goods_traders`. 10.3 is empirical data. |
| 11 Evolutionary Price Robustness | 11.1-11.4 | Pricing variants and the Three-Goods Trader (11.1), supply and demand shocks (11.2): existing `goods_traders`. Double auction, short- and long-run (11.3): existing `double_auction` (cites `Go 8–12`). |
| 12 Factors Impacting Money Distribution | 12.1-12.5 | Toy Trader with number of trades, market-sight, and movement switched off: existing `goods_traders`. |
| 13 Introducing Life | 13.1-13.6 | Metaphoria: mortal agents on patches of food and harvesting resistance that pool labour, evolved propensities (fertility, exploration, job quitting) inherited with mutation, birth, ageing, starvation; a hunter/gatherer mode with harvest sharing and a monetary mode with businesses, wages, shop prices, a central bank and interest-bearing loans, dividends, and stock-flow consistent money. **New entry `metaphoria`**. |
| 14 Long-Term Economic Outcome | 14.1, 14.2, 14.3 (14.3.1-14.3.2), 14.4, 14.5 | Metaphoria with eternal versus mortal populations, interest switched off, eternal populations with mutation, food inheritance and children, and with a money fitness test. Configurations of `metaphoria` (source `Go 14` added to it). |
| 15 Summary and Final Comments | all | Conceptual. |
| Appendix: Model Descriptions | Toy Trader, Three-Goods Trader, Double Auction, Metaphoria 2019 and its two modifications, RobbyGA with starvation (ODD) | Specifications of the models above; nothing new. |

### Schwichtenberg 2020

*Physics from Finance* (source key `Sc`). The only runnable content is the financial toy model of chapters 1-2 (exchange
rates as a connection on a lattice, arbitrage as curvature, traders as covariant diffusion) and its translations into
Maxwell and Schrödinger equations in chapter 4. Its exchange-rate network is a lattice of countries, not a general
currency network, and the models need link variables (`add-link-fields`) once the exchange rates are dynamical.

| Chapter | Sections read | Models or reason none |
| --- | --- | --- |
| Preface | all | Conceptual. |
| 1 Finance Intuitively | 1.1-1.7 | Conceptual: gauge symmetry of currency rescaling, money space, exchange rates as connection, arbitrage gain around a currency loop as curvature, banks making the curvature dynamical. The four-currency loop (1.5) is a holonomy computation, an analysis measure of `add-link-fields`, not a model. The dynamics are stated in chapter 2. |
| 2 Finance Mathematically | intro, 2.1 Arbitrage dynamics, 2.2 Trader dynamics, 2.2.1 Trader dynamics and arbitrage | Intro: exchange rates `A_i` on lattice links, gauge transformations, plaquette gain `F_ij`, copper current `J_mu = q(phi(n+e_mu) - phi(n) - A_mu(n))`, continuum limit. 2.1: copper continuity and the discrete inhomogeneous Maxwell equation `sum_nu F_mu_nu(n) - F_mu_nu(n-e_nu) = mu0 J_mu(n)` for the exchange rates: existing `lattice_gauge_fields` (cites `Sc 2.1`). 2.2: diffusion from a random walk, and 2.2.1: covariant diffusion `(d_t - A_t) rho = D (d_x - A_x)^2 rho` with a prescribed connection: existing `arbitrage_traders` (cites `Sc 2.2`). The coupled system of 1.7, 2.1, and 2.2.1, banks adjusting exchange rates to the trader current while traders follow the arbitrage: **new entry `gauge_arbitrage_market`** (blocked `add-link-fields`). |
| 3 Elementary Particles | all | Conceptual (particle table and charges). |
| 4 Quantum Mechanics | intro, 4.1, 4.2, 4.3, 4.4 (4.4.1-4.4.3) | Intro: Schrödinger equation as imaginary diffusion, probability current: existing `schrodinger` (cites `Sc 4`). 4.1 Charge space: conceptual (U(1) phase). 4.2 Electrodynamics, Maxwell's equations with a current: add source `Sc 4.2` to existing `lattice_gauge_fields`. 4.3 Schrödinger equation with minimal coupling `(d_0 - i q A_0) psi = i D (d_i - i q A_i)^2 psi`: **new entry `charged_schrodinger`** (prescribed potential, as `arbitrage_traders`). 4.4 Isospin and color, SU(2) and SU(3) connections and non-abelian field strength: conceptual here; SU(2) link fields covered by existing `lattice_gauge_fields` (cites `Sc 4.4`). |
| 5 Special Relativity and General Relativity | 5.1, 5.2, 5.2.1, 5.3 | Conceptual: Minkowski metric, Lorentz group, Christoffel connection, Riemann tensor, Einstein equation stated without a solvable setup, Noether charges, torsion. No simulable model is specified. |
| 6 Closing Words | 6.1 Reading recommendations, 6.2 Outlook | Conceptual. |
| Appendices A-E | A Statistics, B Metrics, C Group theory, D Fiber bundles (D.1-D.5), E Taylor expansion | Mathematical background (coin-flip expectation, sphere metric, Lie groups and algebras, parallel transport on a sphere). |

### Notes on Gooding and Schwichtenberg

- `wilhite_trade_networks` is described in Gooding only as a summary of Wilhite (2001): 500 agents with random
  two-good bundles trading bilaterally to raise utility, on a complete network, ten isolated groups of 50, groups
  chained by shared end agents, and the chain with a few random long links. The trade rule comes from Wilhite 2001,
  which is cited as a second source.
- `metaphoria` combines a resource lattice with mortal, reproducing agents and a monetary economy of firms, a labour
  market, shops, and banks. Every need is either a core primitive the library has (variable groups, lattice agent
  spaces, phases, match and transfer laws, invariants for stock-flow consistency) or a field realization in
  `skcomplex.models.economics.realizations` (wage and price adjustment, loans), so it is `pending SURVEY`. Its
  primitive set is not a subset of any implemented entry's, so it is `core`.
- `gauge_arbitrage_market` couples a site group (log price or density of the good) to an edge group (log exchange
  rates) whose rate law is the discrete inhomogeneous Maxwell equation with the current on links, and checks Gauss's
  law as an invariant. It needs the cell complex, edge and hyperedge groups, and the exterior calculus of
  `add-link-fields`; the change already lists real-time lattice Maxwell evolution, so no new primitive is needed.
- `add-link-fields` lists U(1), Z_N, and SU(2) compact groups. Schwichtenberg 4.4 names SU(3), but only conceptually,
  so no entry needs it.

Each paper was extracted with `pdftotext` (layout and plain) and read section by section. "Existing" means an entry of
`docs/overview/inventory.toml`; "new" means an entry in `entries-papers-a.toml`.

### 2102.12086 — Modern Koopman Theory for Dynamical Systems (Brunton, Budišić, Kaiser, Kutz)

- 1.1, 1.3–1.5 Overview, data, objectives, organisation: review of methods, no model.
- 1.2 Illustrative example: unforced Duffing `ẍ = x − x³`. Covered by existing entry `duffing` (add source).
- 2.1–2.3 Koopman definitions, eigenfunctions, mode decomposition: review of methods. Notes: Koopman eigenfunctions,
  Koopman mode decomposition (analysis, `add-data-driven-dynamics`).
- 2.4 Simple Koopman embedding: `ẋ1 = μx1, ẋ2 = λ(x2 − x1²)` with a slow manifold and a three-dimensional
  Koopman-invariant subspace. New entry `koopman_slow_manifold` (example).
- 2.5 Analytic series expansions of eigenfunctions (linear, quadratic, polynomial systems): methods, no model.
- 3.1–3.2 DMD algorithm and extensions (DMDc, randomized, multiresolution, streaming, tensor): methods. Notes: all are
  fits covered by `add-data-driven-dynamics`.
- 3.3 Domain applications (fluids: cylinder wake, cavity, jets; epidemiology; neuroscience; video): applications to
  data, no concretely defined model.
- 3.4 Connections (system identification, ARIMA, LIM, PCR, resolvent analysis): methods. Note: ARIMA/LIM are linear
  stochastic surrogates; no entry.
- 4.1 Eigenfunctions as coordinate changes (Hartman–Grobman, isostables and isochrons of FitzHugh–Nagumo): analysis on
  existing `fitzhugh_nagumo_ode`; no new model. Note: isochrons/isostables are an analysis measure.
- 4.2 Phase portrait and ergodic partition (Chirikov standard map figure): existing `standard_map`; analysis only.
- 4.3 Perron–Frobenius operator; 4.4 continuous spectrum: methods (transfer-operator analysis).
- 4.5 Nonautonomous and stochastic Koopman operators: methods.
- 4.6 PDEs: viscous Burgers with Cole–Hopf linearisation, covered by existing `burgers` (add source); KdV and nonlinear
  Schrödinger via inverse scattering are only named, not defined; Kuramoto–Sivashinsky only cited (existing
  `kuramoto_sivashinsky`).
- 5.1 EDMD, 5.2 time-delay coordinates (HAVOK on Lorenz; covered by existing `lorenz`, add source), 5.3 diffusion maps,
  5.4 neural-network embeddings, 5.5 convergence: methods of `add-data-driven-dynamics`.
- 6.1–6.2 Model-based and data-driven control (MPC, DMDc, EDMDc): methods. 6.3 Koopman-based control strategies with
  MPC of a bilinear DC motor, the forced Duffing, and Van der Pol (Tab. 6.1): new entry
  `koopman_model_predictive_control` (blocked `add-data-driven-dynamics`, also needs `add-optimal-control`); add source
  to existing `van_der_pol`.
- 6.4–6.6 Stability, observability and controllability, sensor placement: methods.
- 7 Discussion: no model.

### 2104.06634 — Irreversibility and biased ensembles in active matter (Fodor, Jack, Cates)

- 1 Introduction: no model.
- 2.1 Modelling active matter: Langevin particles with AOUP (Eq. 2) and ABP (Eq. 3) self-propulsion, covered by existing
  `active_ornstein_uhlenbeck_particles` and `active_brownian_particles` (add sources). Scalar conserved field with
  non-integrable active currents ∇(∇φ)² and (∇φ)∇²φ (Eq. 4, Active Model B/B+): new entry `active_model_b`. Polar field
  coupled to density (Eq. 5): covered by existing `toner_tu` (add source).
- 2.2 Path probabilities and informatic entropy production (IEPR): analysis (`entropy_production`,
  `langevin_entropy_production` exist).
- 2.3 Work and heat at particle and hydrodynamic level; the linear-irreversible-thermodynamics embedding of Active Model
  B with a chemical flux ṅ (Eq. 17): a variant of `active_model_b`, recorded in its entry's scope, no separate entry.
- 2.4 Spatial and spectral decompositions of IEPR, Harada–Sasa, scaling of IEPR: analysis only. Note: local IEPR
  density of a field and the Harada–Sasa spectral decomposition are analysis measures.
- 3.1 Biased ensembles and optimal control (auxiliary/driven process): analysis (`tilted_generating_function`; cloning
  in `extend-dynamics-analysis`).
- 3.2 Phase transitions in biased ensembles of ABPs biased by active work: phase-separated arrest and collective motion;
  controlled ABPs with mean-field alignment torques. Covered by existing `active_brownian_particles`,
  `motility_induced_phase_separation`, and `aligning_active_brownian_particles` as configurations plus the cloning
  analysis of `extend-dynamics-analysis`; add source to `motility_induced_phase_separation`. Coarse-grained polar
  theory (Eq. 28): covered by `toner_tu`.
- 4 Conclusion: no model.

### 2203.06601 — Dynamics on higher-order networks: A review (Majhi, Perc, Ghosh)

- I Introduction: review; growing simplicial complexes (network geometry with flavour), simplicial activity-driven
  model, simplicial configuration model are cited. The triangle-attachment growth is defined concretely in IV.C, so it
  is recorded once as new entry `growing_simplicial_complex`.
- II Basic concepts (hyperlinks, simplices, simplicial complexes): definitions, no model.
- III Synchronization: Kuramoto with three-body coupling (Skardal–Arenas), covered by existing `hypergraph_kuramoto`
  (add source); explosive topological Kuramoto of node and link phases through boundary operators, with the NL and NLT
  couplings (Eqs. 1–9): new entry `topological_kuramoto` (blocked `add-link-fields`). Coupled hypergraph maps, MSF on
  hypergraphs, Hindmarsh–Rose with three-body coupling, Hebbian triadic coupling: cited only.
- IV.A Contagion: simplicial contagion (Iacopini et al.), covered by existing `simplicial_contagion` (add source); SIS on
  hypergraphs with heterogeneous exposure and nonlinear infection kernel (St-Onge et al.), defined concretely: new entry
  `hypergraph_exposure_sis`. Competing SIS on simplices, critical-mass hypergraph contagion: cited only.
- IV.B Consensus: nonlinear three-body consensus on temporal 3-uniform hypergraphs (Eq. 10): new entry
  `hypergraph_consensus`. Hypergraph bounded confidence, simplicial adaptive voter: cited only.
- IV.C Evolutionary games: games on 1- and 2-simplices with Fermi updates (Eq. 11): new entry
  `simplicial_social_dilemmas`; public goods game on hypergraphs: new entry `hypergraph_public_goods`. Adaptive
  simplicial snowdrift, signalling games: cited only.
- V Random walks and diffusion: size-biased random walk on hypergraphs (Eq. 12): new entry `hypergraph_random_walk`.
  Markov stability community detection: analysis note. Random walks on simplicial complexes (Hodge Laplacian): cited.
- VI Summary: no model.

### 2207.03581 — Gradients of O-information (Scagliarini et al.)

- Main text (single section): O-information and its first- and second-order gradients (analysis, pending
  `add-information-decomposition`). Example 1: frustrated 7-spin Ising on a hexagon with a central spin, Boltzmann
  distribution by exact enumeration: covered by existing `graph_ising` (add source); exact enumeration is a static
  computation (note). Example 2: US macroeconomic data: data analysis, no model.

### 2304.12482 — Information Theory for Complex Systems Scientists (Varley)

- I What are complex systems; II entropy, relative entropy, mutual information, total and dual total correlation,
  co-information; III information dynamics (entropy rate, storage, transfer, modification, excess entropy); IV partial
  information decomposition and extensions (PED, GID, ΦID); V network inference (functional and effective
  connectivity, null models, higher-order frameworks); VI integration, segregation, complexity (TSE, O-information,
  Φ); VII estimators: review of methods throughout. Toy systems (logic gates, elementary cellular automata, flocking)
  are only cited as applications. No new entry and no source added. Notes: local information measures, PID, PED, ΦID,
  TSE complexity, O-/S-information, and estimators belong to `add-information-decomposition`.

### 2306.05635 — Theoretical foundations of studying criticality in the brain (Tian et al.)

- Introduction, overview, key concepts (fine tuning vs self-organisation, standard vs non-standard, classification):
  review, no model.
- Neural avalanches and their phases (absorbing/active, synchronous/asynchronous, quasi-critical region): concepts.
- Instances of OC, qC, SOC, SOqC: conceptual; cortical branching model (Williams-García, Fosque), Levina
  integrate-and-fire with depressing synapses, Girardi-Schappo SOqC network: cited only, not defined. Kinouchi–Copelli
  is only cited (existing `kinouchi_copelli` already cites this paper).
- Mean-field and stochastic field theories: Reggeon-type Langevin field for the active density with multiplicative
  √ρ noise coupled to an energy field with drive and bulk dissipation (Eqs. 3–4), its mean-field, SOC, and SOqC limits
  (Eqs. 5–22): new entry `directed_percolation_langevin` (core: noise with boundaries on a field).
- Neural avalanche exponents: critical branching process (Eqs. 23–34, Box 1), covered by existing `galton_watson` (add
  source).
- Scaling relation, universal collapse (ABBM model only cited), slow decay of autocorrelation, estimation of exponents
  and collapse: analysis (`avalanches`, `fit_avalanches`, `crackling_noise`, `scaling_collapse`, `autocorrelation`).
- Discussion and future directions: no model.

### 2306.08868 — Nonreciprocal pattern formation of conserved fields (Brauns, Marchetti)

- I Introduction, I.A NRCH equations: covered by existing `nonreciprocal_cahn_hilliard` (add source). I.B summary.
- II FitzHugh–Nagumo with nonreciprocal coupling: covered by existing `fitzhugh_nagumo_ode` (add source).
- III Prototypical model; Table I classes Allen–Cahn (Model A), Cahn–Hilliard (Model B), FHN reaction–diffusion
  (existing `fitzhugh_nagumo`, add source), NRCH: new entries `cahn_hilliard` and `allen_cahn`. III.A–C linear stability,
  generalized Maxwell construction, interface mode: analysis (dispersion relation in `extend-dynamics-analysis`).
- IV Interrupted coarsening and V 2D undulating fronts and chaos: behaviour of NRCH, no new model.
- VI Concrete physical systems: active/passive particle mixtures (mapping only, cited), Min-protein mass-conserving
  reaction–diffusion (new entry `min_protein_skeleton`, App. F), 1D active viscoelastic gel with ECM (Eqs. 13–15) and
  active poroelastic models (App. G): new entry `active_viscoelastic_gel` (blocked `add-nonlocal-operators`), chemotactic
  bacteria and catalytic droplets (cited only).
- VII Conclusion: no model. App. A two-compartment reduction: an ODE covered by `fitzhugh_nagumo_ode`. App. B–E:
  analysis and numerics.

### 2311.14877 — Triadic percolation induces dynamical topological patterns (Millán, Sun, Torres, Bianconi)

- 1 Introduction: no model.
- 2 Triadic percolation on spatial networks: Waxman structural network, distance-dependent signed regulatory
  interactions, two-step percolation-regulation update. Covered by existing `triadic_percolation` (add source); the
  spatial (Waxman) generator is a missing general primitive, see needs file.
- 3 Emergence of patterns (TDA classification, permutation entropy and LMC complexity of 2D patterns), 4 dynamics
  (overlaps, blinking), 5 phase diagram (entropy rate, return probability): analysis only. Notes: persistent homology
  with Wasserstein distances, 2D permutation entropy and statistical complexity, Lempel–Ziv entropy rate.
- 6 Conclusions; Methods (network construction, TDA, entropy, temporal analyses); Supplementary (surface-tension
  mechanism, rectangular torus): no further model.

### 2312.16815 — Emergence and Causality in Complex Systems: a survey on causal emergence (Yuan et al.)

- 1 Introduction, 2 Background (causality, emergence): review.
- 3.1–3.2 Early works and computational mechanics (ε-machines): methods.
- 3.3 Seth's G-emergence tested on Boids: covered by existing `boids` (add source).
- 3.4 Other quantitative theories: methods.
- 3.5 Hoel's causal emergence: Markov chain TPM examples and a stochastic 4-node Boolean network (Fig. 7–8): covered by
  existing `probabilistic_boolean_network` (add source); continuous-system causal emergence: methods; 3.5.6 random walks
  on networks (Klein–Hoel): covered by existing `network_random_walk` (add source).
- 3.6 Rosas ΦID emergence (Ψ, Δ, Γ) on Game of Life and flocking: covered by existing `game_of_life` and `boids` (add
  source to `game_of_life`).
- 4 Causal emergence and machine learning (NIS, NIS+): methods; test systems SIR with noise (existing `sir`, add source)
  and Boids. Note: NIS+ is a learned coarse-graining, close to `add-data-driven-dynamics` and `add-emergence-analysis`.
- 5 Discussion and applications: no model.

### 2402.09090 — Software in the natural world: hierarchical emergence (Rosas et al.)

- I Introduction; II computational mechanics (causal states, ε- and υ-machines); III theory (computational,
  informational, causal closure, lumpability, hierarchies): methods. Notes: closure tests and ε-machines are in
  `add-information-decomposition`.
- IV.A Invariant quantities in elementary cellular automata (rules 60 and 150): existing `elementary_automaton` (add
  source).
- IV.B Ehrenfest two-chamber diffusion model: new entry `ehrenfest_urn` (example).
- IV.C Ising under Glauber dynamics, energy causally closed: existing `ising` (add source).
- IV.D Random walk over a modular weighted network: existing `network_random_walk` (add source).
- IV.E Agent mobility on a graph (refugee movement) with population lumpability: existing `network_random_walk` (add
  source).
- IV.F Stochastic Hopfield network with pattern overlaps as closed macro-variables: existing `hopfield` (add source).
- V Discussion: no model. Appendices (proofs, counterexamples on small Markov chains): existing `markov_source` (add
  source).

### 2404.02642 — Urban Scaling Laws (Ribeiro, Netto 2024)

- §1 Introduction: Y = Y0 N^β, Kleiber's law. This is an analysis (allometric fit), not a model.
- §2 History: West-Brown-Enquist fractal distribution network, Zipf's law, agglomeration. These are named only, without a definition. Zipf
  city sizes are covered by the existing entries `simon_yule` and `gibrat_process`.
- §3 Empirics: three classes of urban variables. This section is empirical, with no model.
- §4 Possible explanations: the Bettencourt model (Y ∼ N^(2−β_infra)) is covered by the existing entry `urban_scaling_interactions`, which gains section 4.
  The gravity interaction model of Ribeiro et al. (p(r) ∼ r^−γ on a population of fractal dimension D_P, β = 2 − γ/D_P) gives the new entry
  `gravity_urban_scaling`. Louf-Barthelemy congestion and Molinero-Thurner geometry are named only.
- §5 Criticism, §6 Open issues, §7 Summary: conceptual.

### 2405.15751 — A review of active matter reviews (te Vrugt, Wittkowski 2024)

A metareview that cites reviews. It defines no model by equations, so only canonical models named with a mechanism are recorded.
- §I–VI (introduction, books, general reviews, colloids, special particles, materials): review of literature, with no model.
- §VII Methods: active Brownian particles are covered by the existing entry `active_brownian_particles`. Field theories and DDFT are named as method families, not
  models.
- §VIII Collective dynamics: the existing entries `motility_induced_phase_separation`, `vicsek`, and `toner_tu` cover MIPS, Vicsek, and Toner-Tu (`nonreciprocal_vicsek`
  already cites the paper). Keller-Segel chemotactic collapse gives the new entry `keller_segel`, and swarmalators give the new entry `swarmalators`.
  Hydrodynamic synchronization of cilia is named only.
- §IX Active nematics: covered by the existing entry `dry_active_nematics`.
- §X–XV (microswimmers, environments, biological, molecular, artificial, medical): review of literature. Jeffery orbits and squirmers are named only.
- §XVI Conclusions.

### 2407.02181 — A mathematical definition of complex adaptive system as interaction space (Giordano 2024)

- §1–5 (CAS after Zipf, global states, generalized evolution principle, entropy, formal definition): conceptual and
  mathematical, with no runnable model.
- §6 Power law after Mandelbrot: minimise cost over diversification entropy on the probability simplex, which yields
  Zipf-Mandelbrot. This gives the new entry `mandelbrot_least_effort`, and §6.1 lists its cost functions.
- §7 Von Thünen model: impedance zones (§7.1–7.2) and adapted economy (§7.3) are covered by the existing entry `von_thunen`, which already cites §7.
- §8 Conclusions.

### 2409.10990 — Higher-order interactions in random Lotka-Volterra communities (Sidhom, Galla 2024)

- §I Introduction: review.
- §II Model definitions: generalized LV with quenched Gaussian couplings of order p (mean μ_p, variance σ_p², symmetry γ_p).
  The order p ≥ 3 is covered by the existing entry `higher_order_random_lotka_volterra`. The pairwise random ensemble (p = 2) is covered by the existing entry `generalized_lotka_volterra`, and both
  gain sources.
- §III Mathematical analysis: generating functional, dynamic mean-field effective process, fixed points, and the stability threshold z1. This is an
  analysis; see the needs file.
- §IV Results for p = 2, 3, 4 and mixed orders: these are variants of the same model, covered by the two entries above.
- §V Conclusions. The supplement (S1–S6) is a derivation, with no new model.

### 2412.12988 — Network Renormalization (Gabrielli, Garlaschelli, Patil, Serrano 2024)

- Introduction: RG background.
- §2 Coarse-graining approaches: box covering, spectral, topological, symmetry-based, ML, and information-theoretic methods. These are
  analysis methods. Watts-Strogatz is covered by the existing entry `random_graph_ensembles`.
- §3 Geometric renormalization: the S1/H2 model is covered by the existing entry `popularity_similarity_network`. The geometric branching growth (GBG) fine-graining model gives the new
  entry `geometric_branching_growth`. The WSD weighted variant is a variant of S1. Bond percolation on replicas is an analysis.
- §4 Laplacian renormalization: heat diffusion on a network is covered by the existing entry `network_diffusion`. The LRG and the entropic susceptibility are an analysis.
- §5 Multiscale renormalization: the MultiScale Model of aggregation-invariant fitness graphs, quenched and annealed (Lévy fitness), gives the new
  entry `multiscale_fitness_network`.
- §6 Discussion: conceptual, covering the Fisher-information and sloppy-model view.

### 2509.01581 — A gauge theory of complex adaptive systems (Mihaylov, Cacciatori 2025)

- §1 Introduction (frustration, main idea, motivation): conceptual. The spin glasses it mentions are covered by existing entries.
- §2–8 (discrete k-paths, integral structures, principal, associated bundles, connections, local expressions): mathematical construction,
  with no runnable model.
- §9 Complexity ("exasperation", topological complexity): conceptual.
- §10 A gauge theory: a static toy action (Mahalanobis holonomy functional) and an iterative dynamic version. These are covered by the existing entry
  `gauge_network_agents`. The Z2-gauge reading of the Ising model is illustrative only and is covered by the existing entry `ising`.
- §11 A network model: the gauge-fitness network with SO(n) transport of Gaussian internal states is covered by `gauge_network_agents`.
- §12 Conclusions and appendices A–E: mathematical.

### 2509.11338 — Next-Generation Reservoir Computing for Dynamical Inference (Cestnik, Martens 2025)

- §I Introduction.
- §II.1 Traditional RC: covered by the existing entry `echo_state_network`. §II.2 NGRC gives the new entry `next_generation_reservoir`.
- §III Methods (random projection, ridge regression, error tracking, validation): part of `next_generation_reservoir`.
- §IV.A Attractor reconstruction: covered by the existing entries `lorenz` and `rossler`.
- §IV.B Bifurcation diagram: Rössler with an Ornstein-Uhlenbeck-driven parameter (Eq. 13) gives the new entry `ou_driven_rossler`.
- §IV.C Asymptotic phases: the same noisy Rössler in a limit-cycle regime, covered by `ou_driven_rossler`.
- §V Discussion.

### 2511.01168 — Tipping points and cascading transitions (Fang, Wang, Kurths, Fan 2025)

- Introduction, Tipping point and tipping element: conceptual.
- Bifurcation and tipping types: the Sutera-Fraedrich stochastic energy balance under B-, N-, and R-tipping (Eq. 3, Table 1) gives the new entry
  `sutera_fraedrich_tipping`. It differs from the existing deterministic `energy_balance`.
- Early warning signals: the stochastic cubic fold with a ramped control parameter (Eq. 4) gives the new entry `ramped_stochastic_fold`. AC1, variance,
  DFA, and the Table 2 indicators are analysis measures, recorded only as notes.
- Cascading transitions: the leader-follower pair (Eq. 10) and the Wunderling tipping network (Eq. 11) are covered by the existing entry `tipping_cascade_network`.
- Challenges, Opportunities: conceptual. Climate networks and interdependent networks are named only.

### 2604.15669 — Self-organization to the edge of ergodicity breaking (Lesmana, Feng, Chen, Lai 2026)

- §I Introduction.
- §II.A SK game: Q-learning softmax agents on quenched SK couplings give the new entry `sk_game` (the non-evolutionary baseline).
- §II.B EvoSK: covered by the existing entry `evolutionary_sk_game`.
- §III Optimality, ergodicity edge, avalanches: analysis of EvoSK (ergodicity coefficient, avalanche distribution, branching
  ratio as a Jacobian spectral radius), recorded as notes.
- §IV Conclusions. Appendix A covers methods. Appendix B covers inheritance variants V1–V3 of EvoSK, a variant of the existing entry. Appendix C covers theory.

### 2604.19827 — More Is Different: emergence in AI-native software ecosystems (Russo 2026)

- §1–3 (introduction, deterministic assumption, AI ecosystems as CAS after Holland): conceptual.
- §4 Measuring emergence: effective information and PID over repository micro- and macro-states. This is an analysis protocol on
  empirical data, with no model. It is covered by the pending `add-emergence-analysis` and `add-information-decomposition`.
- §5 Propositions P1–P7: empirical hypotheses. The NK model and Perrow's tight coupling are named only.
- §6–8 (ecosystem science, implications, conclusion): conceptual. The paper has no runnable model.
