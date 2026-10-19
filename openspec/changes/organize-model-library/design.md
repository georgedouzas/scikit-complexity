# Design

## Context

`skcomplex.models` grew one change at a time. Each change added a package named after its technique, such as
`lattices`, `fields`, `fluids`, and `nonlinear`, beside the earlier science packages such as `ecology` and `physics`.

The nine domain-agnostic packages sit beside `models` at the top of `skcomplex`, and the model library imports several
of them. The import check forbids private imports only.

Five survey passes covered the 28 books in `references/` and 15 arXiv reviews:
- 0710.3256, social dynamics;
- 1408.2701, network epidemics;
- 1010.5017, collective motion;
- 1207.2929, soft active matter;
- 1602.00081, active particles;
- 0805.2976, network synchronization;
- 1902.05307, oscillator mean fields;
- 2112.12147, brain dynamics;
- cond-mat/0607344, evolutionary games;
- cond-mat/0012229, traffic;
- 0905.1518, money and wealth;
- 0909.1974, econophysics;
- 1608.06582, stochastic biochemistry;
- 2111.04508, biological circuits;
- 1910.00583, climate variability.

The passes found about 390 model entries, or about 230 distinct models. The inventory below records them.

## Goals / Non-Goals

**Goals:**

- The package shows the split between the general framework, `skcomplex.core`, and the domain models.
- A reader finds a model by its science, and every model reads as an example of the general interface.
- The library covers every surveyed model the interface supports.
- Every surveyed model the interface cannot express is visible as blocked, with the change that would unblock it.
- Missing features are built on mature libraries.

**Non-Goals:**

- Implementing the missing features in this change. Each becomes its own change, created when the first blocked model
  is reached during implementation.
- Models without dynamics, such as static equilibria, qualitative systems diagrams, and analysis algorithms like
  centrality. The inventory leaves them out.
- Compatibility aliases for the old package paths, of the framework or the models. The library is pre-1.0, so the move
  is a clean break.

## Decisions

### Core and models

Every domain-agnostic package moves under `skcomplex.core`:

| Old path | New path |
|---|---|
| `skcomplex.core` | `skcomplex.core.base` |
| `skcomplex.model` | `skcomplex.core.model` |
| `skcomplex.compile` | `skcomplex.core.compile` |
| `skcomplex.solvers` | `skcomplex.core.solvers` |
| `skcomplex.runtime` | `skcomplex.core.runtime` |
| `skcomplex.viz` | `skcomplex.core.viz` |
| `skcomplex.experiments` | `skcomplex.core.experiments` |
| `skcomplex.analysis` | `skcomplex.core.analysis` |
| `skcomplex.backends` | `skcomplex.core.backends` |

The layering inside the core stays as it is, running downward from `base` through `model`, `compile`, `solvers`,
`runtime`, `viz`, `experiments`, `analysis`, and `backends`. `skcomplex.core.__init__` carries only its docstring, like
the top-level package. Users import from the package that owns a name, for example
`from skcomplex.core.model import Element`.

`base` keeps today's content of `core`: symbols, domains, variables, visual specs, registries, and errors. It is the
general foundation of the core, and the name says so without repeating `core`.

The catalog in `core.runtime` keeps its registry. Model packages register their factories with `register_model` when
`skcomplex.models` is imported. The core never imports `skcomplex.models`, so `catalog` lists the library models only
once they are imported, as today. A layout test enforces the boundaries with an AST scan:
- core modules never import `skcomplex.models`;
- model modules import only public names of the core;
- no model package imports another.

Model packages are orthogonal and self-contained. Today several share components across packages:
- `Particle`, `Gravity`, and `ConstantForce` are used by physics models in different subfields;
- `Population` is used by ecology factories and by the agent-based socio-ecological model.

Each such component is either domain-agnostic, and moves into the core, or belongs to one package, and the other
packages define their own. The migration decides each case:
- the generic point particle with position, velocity, and mass, together with its common forces, is domain-agnostic
  mechanics of the core and moves to `skcomplex.core.model`;
- science-specific components, such as epidemiological compartments or ecological populations, stay in their
  subfield, and other subfields declare their own.

### Taxonomy

Packages follow fields, then subfields:

| Field | Subfields |
|---|---|
| `physics` | `mechanics`, `celestial_mechanics`, `nonlinear_oscillators`, `synchronization`, `statistical_mechanics`, `criticality`, `soft_matter`, `active_matter`, `fluids`, `waves`, `pattern_formation`, `heat`, `cosmology` |
| `chemistry` | `kinetics`, `pattern_formation` |
| `biology` | `ecology`, `epidemiology`, `evolution`, `neuroscience`, `systems_biology`, `physiology`, `collective_behaviour`, `oncology` |
| `earth` | `climate`, `oceanography`, `seismology` |
| `economics` | `macroeconomics`, `markets`, `finance`, `econophysics`, `resources`, `innovation` |
| `social` | `opinion_dynamics`, `segregation`, `cooperation`, `spreading`, `crowds`, `organizations` |
| `engineering` | `traffic`, `circuits`, `control`, `power_systems`, `robotics`, `processes` |
| `mathematics` | `dynamical_systems`, `stochastic_processes`, `networks`, `fractals` |
| `computer_science` | `automata`, `evolutionary_computation`, `neural_networks` |

Each subfield is a self-contained package whose private modules hold its component types and factories, and whose
`__init__` re-exports them. A field package re-exports its subfield packages, and nothing else. A model lives where the science that poses it
lives. Tags carry the cross-field reach, such as `chaos` on the Lorenz model in `mathematics/dynamical_systems` and
`atmosphere` on the same entry.

### Migration of the existing models

| Old location | New location |
|---|---|
| `ecology` | `biology.ecology` |
| `epidemiology` | `biology.epidemiology` |
| `neuroscience` | `biology.neuroscience` |
| `economics.solow`, `economics.multiplier_accelerator` | `economics.macroeconomics` |
| `economics.cobweb` | `economics.markets` |
| `finance` | `economics.finance` |
| `social.schelling` | `social.segregation` |
| `social.voter_model` | `social.opinion_dynamics` |
| `social.boltzmann_wealth` | `economics.econophysics` |
| `social.harvesting` | `economics.resources` |
| `fields.heat_diffusion`, `fields.heat_l_shape` | `physics.heat` |
| `fields.fisher_kpp` | `biology.ecology` |
| `fields.gray_scott`, `fields.brusselator`, `fields.schnakenberg` | `chemistry.pattern_formation` |
| `fields.fitzhugh_nagumo`, `nonlinear.mackey_glass` | `biology.physiology` |
| `fields.kuramoto_sivashinsky` | `physics.fluids` |
| `fields.complex_ginzburg_landau`, `fields.swift_hohenberg` | `physics.pattern_formation` |
| `fields.pheromone_foraging` | `biology.collective_behaviour` |
| `fluids` | `physics.fluids` |
| `lattices.ising`, `lattices.potts`, `lattices.xy` | `physics.statistical_mechanics` |
| `lattices.sandpile`, `lattices.forest_fire` | `physics.criticality` |
| `lattices.earthquakes` | `earth.seismology` |
| `lattices.elementary_automaton`, `lattices.game_of_life`, `lattices.langtons_ant` | `computer_science.automata` |
| `lattices.majority_rule` | `social.opinion_dynamics` |
| `lattices.hodgepodge` | `chemistry.kinetics` |
| `lattices.nagel_schreckenberg` | `engineering.traffic` |
| `nonlinear.lorenz`, `rossler`, `sprott_jerk`, `logistic_map`, `henon_map` | `mathematics.dynamical_systems` |
| `nonlinear.duffing`, `van_der_pol`, `henon_heiles` | `physics.nonlinear_oscillators` |
| `nonlinear.delayed_kuramoto` | `physics.synchronization` |
| `nonlinear.stochastic_resonance`, `physics.brownian_near_wall` | `physics.soft_matter` |
| `physics.ideal_gas`, `physics.lennard_jones_fluid` and their components | `physics.statistical_mechanics` |
| `physics.rlc_circuit` | `engineering.circuits` |
| the remaining `physics` models | `physics.mechanics` and `physics.celestial_mechanics` |

### Documentation as patterns

Each field gets a documentation page generated from the catalog. For every model the page shows:
- the factory's docstring;
- the component types and their laws as written, the source of their `laws`;
- `equations_of` for the forms that have equations.

The gallery keeps short runnable scripts per field, so a user sees the model, its equations, and a run side by side.

### Inventory format and checks

The inventory is a data file in the documentation sources, one table per field path. Each row holds:
- the name, the sources, and the status: `implemented`, `pending <change>`, or `blocked <change>`;
- the missing feature, for blocked models.

A test checks the inventory against the catalog and against the OpenSpec changes, active or archived. The documentation
build renders the inventory as the coverage page, linking implemented models to their field pages.

### Missing features become changes built on mature libraries

When implementation reaches a blocked model, the contributor:
1. creates the change that specifies the missing feature, unless it already exists;
2. surveys the libraries that implement the feature in that change's design;
3. records the model as blocked on that change.

The expected changes and their candidate libraries:

| Change | Missing feature | Candidate libraries |
|---|---|---|
| `add-stochastic-maps` | Random draws from the simulation's stream inside update laws, and discrete-time Markov chains | NumPy `Generator` |
| `add-anomalous-processes` | Renewal processes with power-law waiting times, and fractional Gaussian noise | `stochastic`, `fbm`, SciPy `levy_stable` |
| `add-complex-states` | Complex-valued states and fields | NumPy complex dtypes, Numba complex support |
| `add-nonlocal-operators` | Convolution kernels, reads at shifted positions, and modified-Helmholtz inversion on fields | SciPy `signal.fftconvolve` and `ndimage` |
| `add-lattice-geometries` | Hexagonal and triangular lattices, local rules and Monte Carlo on arbitrary graphs, and extremal updates | networkx lattice generators |
| `add-phase-space-fields` | Fields over position and velocity or orientation, for kinetic equations | `py-pde`, `pylbm` |
| `add-tensor-fields` | Tensor-valued fields coupled to flow, such as the active nematic Q-tensor | `py-pde`, Dedalus |
| `extend-agent-spaces` | Three-dimensional continuous agent spaces and geographic spaces | Mesa continuous spaces, `mesa-geo` |

Some models stay blocked with no change planned, because they fall outside a dynamical-systems framework. The
inventory names a library a future change could wrap:
- quantum spin operators, QuTiP;
- lattice gauge link fields;
- gradient training of recurrent networks, PyTorch or JAX;
- static Nash equilibria, `nashpy`;
- geometric substitution fractals;
- cellular automata on groups that are not finitely presented;
- continuum percolation clusters, SciPy `spatial`.

### Audit of the framework's own numerics

The constitution gains a rule: a feature that a popular, mature, or performant library implements MUST be built on
that library, unless the design states why none fits.

This change audits the framework's existing numerics against established libraries and records each replacement worth
making as a follow-up change:
- finite-difference and spectral fields against `py-pde` and Dedalus;
- finite volumes against PyClaw;
- stiff and exponential steppers against SciPy, `diffrax`, and Dedalus' IMEX schemes;
- Gillespie against `gillespy2`.

The audit compares accuracy, speed, and fit with the symbolic laws, and keeps the in-house code only where it wins on
the comparison or no library fits the interface.

## Risks / Trade-offs

- [Breaking every import path, framework and models] → The library is pre-1.0. The changelog and user guide list the
  moves, the catalog finds every model by name whatever its path, and the move happens in one change before the
  pending changes are applied, so they are written against the new paths.
- [The library grows to about 230 models] → Each model is a short factory on shared components, the conformance suite
  stays generic, and slow scenarios run in the benchmark suite.
- [Inventory drift] → The inventory check fails when the catalog and the inventory disagree.
- [Fields with few models] → The taxonomy is the place where future models land. A subfield with one model is fine.

## Inventory

Status codes:
- `✓`: implemented today;
- `+`: supported by the interface, to implement in this change;
- `P:<change>`: pending a planned change;
- `B:<change>`: blocked until the named change exists;
- `B:out`: blocked with no change planned.

Source keys:
- Ar Argun et al. 2021; Ba Bala et al. 2017; Bs Barsky 2020; Be Bellomo et al. 2017; Ca Caiani et al. 2016;
- Cm Carmichael et al. 2019; CC Ceccherini-Silberstein and Coornaert 2023; Ch Charbonneau 2017; Cy Chesney 2021;
- DP Datseris and Parlitz 2022; Di Dilão 2023; Fo Forrest 2018; Go Gooding 2019; Gr Gros 2015;
- Ha Hamill and Gilbert 2016; Je Jensen 2023; Li Lindgren 2024; Mi Mihailović et al. 2023; Mo Mobus 2022;
- Ry Ryzhov et al. 2021; Sa Satz 2022; Sc Schwichtenberg; Sm Starkovich; TC Downey, Think Complexity; TS Meadows,
  Thinking in Systems; Th Thurner et al. 2018; Tr Trzaska 2023;
- arXiv identifiers as above.

### physics

- **mechanics:**
  - `✓`: pendulum (Ar 1, Ry 1.1.2, Tr 2.2.2), harmonic and driven damped oscillators (Ar 1, Ry 1.1.1),
    projectile (Ry 1.2.1), Newton's cradle (Ry 1.2.4), bouncing ball;
  - `+`: double pendulum (Ar 1.7, Ry 1.2.8), coupled pendulums (Ry 1.2.7), pendulum on a sliding cart (Ry 1.2.6),
    rotating swing (Ry 1.2.9), rotating balls on rods (Ry 1.2.5), ballistic pendulum (Ry 1.2.3), gyroscope in a
    gimbal (Ry 1.2.10), magnetic pendulum (DP 5.3.3), parametric pendulum (Di 6.1), Huygens clocks and pendulums on
    coupled supports (Di 7.1–7.2), two-mass spring-damper between walls (Ry 1.2.2), particle in the well `k x^2 e^-x`
    (Sm 7.13).
- **celestial_mechanics:**
  - `✓`: two-body orbit;
  - `+`: N-body and solar system (Ar 1.1, Di 8), restricted three-body and Sitnikov problems (Di 8), Störmer charge
    in a dipole field (Di 6.4).
- **nonlinear_oscillators:**
  - `✓`: Duffing (DP 9.1, Tr 3.9), Van der Pol (DP 9.2, Tr 3.6), Hénon-Heiles (DP 8.2);
  - `+`: Stuart-Landau (Di 4.14, Gr 2.2), forced Van der Pol (Tr 3.6), Adler equation (DP 9.2.2), dynamical billiards
    of Sinai and mushroom shapes (DP 8.1).
- **synchronization:**
  - `✓`: delayed Kuramoto;
  - `+`: Kuramoto (DP 9.2.3, Gr 9.2, Je 3.3), Kuramoto-Sakaguchi chimera (1902.05307 §4.1, Je 7.2), nonlocal ring
    chimera (DP 10.2.2), Ott-Antonsen reduction (1902.05307 §3.1), Winfree model (1902.05307 §2), Mirollo-Strogatz
    pulse-coupled oscillators (0805.2976 §3.2), coupled Rössler and the master stability function (DP 9.3, 10.2.1,
    0805.2976 §4.1), aggregate-averaging synchronization (Gr 9.4);
  - `P:add-adaptive-networks`: Kuramoto on networks (0805.2976 §3.1, Je 8.5.3), coupled map lattices on networks
    (0805.2976 §3.3).
- **statistical_mechanics:**
  - `✓`: Ising (Ar 2, Je 3.1, Th 3.3.1), Potts, XY (Je 6.7), ideal gas, Lennard-Jones fluid (Ar 1.3);
  - `+`: Q2R reversible Ising automaton (Li 5.6), site percolation (Ch 4, TC 7, Th 3.3.1), diffusion-limited
    aggregation (Ch 3), Landau mean-field relaxation (Gr 6.1), contact process (1408.2701 §II.3), TASEP
    (cond-mat/0012229 §I.E.1), two-phase zone exchange (Bs 3.14), discrete master equation (Je 10.7);
  - `B:add-lattice-geometries`: Ising on a graph (Ar 2.1);
  - `B:out`: quantum Heisenberg model (Mi 7.3), continuum percolation (Sa 3).
- **criticality:**
  - `✓`: Bak-Tang-Wiesenfeld sandpile (Ch 5, Gr 6.4, TC 8, Je 12.1), Drossel-Schwabl forest fire (Ar 3, Ch 6, Je
    12.1.4);
  - `+`: Manna sandpile (Je 12.5);
  - `B:add-lattice-geometries`: forest fire on hexagonal lattices and graphs (Ar 3).
- **soft_matter:**
  - `✓`: Brownian particle near a wall (Ar 7.6), stochastic resonance (Ar 5, Gr 3.3);
  - `+`: inertial and overdamped Brownian motion (Ar 5.3), optical trap including oscillating and blinking traps
    (Ar 5.4), Kramers escape (Ar 2.1, Gr 3.3), multiplicative noise conventions (Ar 7.1–7.5), scaled Brownian motion
    (Ar 6.3), diffusion in a speckle field (Ar 6.6);
  - `B:add-anomalous-processes`: continuous-time random walk, Lévy walk, annealed transient time motion (Ar 6.3),
    fractional Brownian motion (Ar 6.1).
- **active_matter:**
  - `+`: active Brownian particles (Ar 9.1, 1602.00081 §II.1), chiral active particles (1602.00081 §II.2.1),
    run-and-tumble (1602.00081 §II.2.2), active Ornstein-Uhlenbeck particles (1602.00081 §II.2.2), motility-induced
    phase separation (1602.00081 §IV), living crystals (Ar 9.3), Cucker-Smale flocking (1010.5017 §4.4.1, Be 6.7.2),
    depot-pumped Brownian particles (cond-mat/0012229 §I.F.1), Szabó cell model (1010.5017 §5.1.2), Toner-Tu
    hydrodynamics (1207.2929 §II.1), flocking with active and passive agents (Ch 10);
  - `P:add-adaptive-networks`: Vicsek model (Ar 8, 1010.5017 §4.1, Je 11.1), Grégoire-Chaté cohesive flocks,
    Ginelli nematic particles (1010.5017 §4.2);
  - `B:add-tensor-fields`: dry active nematics, active nematic and polar gels (1207.2929 §II.2, §III);
  - `B:add-phase-space-fields`: Bertin-Droz-Grégoire kinetic flocking (1207.2929 §V.2.3).
- **fluids:**
  - `✓`: Burgers, shallow-water dam break, decaying turbulence, Rayleigh-Bénard, Kuramoto-Sivashinsky (DP 11.3);
  - `+`: nonlinear advection and wave breaking (Fo 14.3.3), discrete-velocity Boltzmann (Be 2.3.2);
  - `B:add-lattice-geometries`: FHP lattice gas (Di 5.16);
  - `B:add-phase-space-fields`: Boltzmann, BGK, Vlasov, and Enskog equations (Be 2.2–2.3).
- **waves:**
  - `+`: spring chains and guitar string (Ar 1.5–1.6), standing waves on a string (Sm 4.3), Schrödinger equation as
    real and imaginary fields (Sc 4);
  - `B:out`: lattice Maxwell and gauge link fields (Sc 2.1, 4.4).
- **pattern_formation:**
  - `✓`: complex Ginzburg-Landau, Swift-Hohenberg.
- **heat:**
  - `✓`: heat diffusion, heat on an L-shaped domain;
  - `+`: cooling coffee (TS 1).
- **cosmology:**
  - `+`: Friedmann expansion (Sa 6).

### chemistry

- **kinetics:**
  - `✓`: hodgepodge machine (Ch 11);
  - `+`: Brusselator ODE (DP 2.2, Tr 4.2), Oregonator and the FKN mechanism of the BZ reaction (Tr 4.5–4.6), cubic
    autocatalysis (Tr 4.3), Michaelis-Menten as ODEs and as a jump process (Tr 4.4, 1608.06582 §5.1), chemical
    Langevin equation (1608.06582 §4.1), bistability on a catalyst pellet (Tr 4.7), pulsed electrochemical reactor
    (Tr 4.8), cyclic cellular automaton (Di 5.17).
- **pattern_formation:**
  - `✓`: Gray-Scott (Gr 4.2, TC 7, Li 7), Brusselator fields (DP 11.1.3), Schnakenberg.

### biology

- **ecology:**
  - `✓`: exponential and logistic growth, Hutchinson (Tr 3.2), Lotka-Volterra (Ar 14.1, Gr 3.1.4, Th 5.4.2), Fibonacci
    rabbits, wolf and sheep, Fisher-KPP (Gr 4.1);
  - `+`: competition, mutualism, and food webs (Ar 14.2–14.4), Rosenzweig-MacArthur and rate tipping (DP 12),
    Nicholson-Bailey (Ar 14.7), Lotka-Volterra with harvesting (Tr 3.3), logistic growth with a varying capacity
    (Tr 3.2), mangrove succession (Ba 2.5.2), agent-based Lotka-Volterra (Ar 14.1), stigmergic predator-prey automaton
    (Cm 8), carrying capacity of a meadow (Ha 10.2), neutral macroecology (Gr 8.6), intrinsic-instability populations
    (Je 13.2), generalized stochastic predator-prey (Cm 1);
  - `P:add-adaptive-networks`: sequentially linear ecosystem (Th 5.5.2).
- **epidemiology:**
  - `✓`: SIR;
  - `+`:
    - SIS and SIRS, and agent-based SIR with mortality, waning immunity, incubation, and lockdown (Ar 11);
    - random-walker epidemics on a grid and on a small world (Ch 9);
    - SIR on a network as an agent model (DP 10.3, Th 4.8.6);
    - heterogeneous mean-field SIS (1408.2701 §V.1);
    - metapopulation reaction-diffusion (1408.2701 §IX);
    - discrete-time SIRS with a recovery delay (Gr 9.6);
  - `P:add-adaptive-networks`: quenched mean-field SIS (1408.2701 §IV.1), adaptive SIS with rewiring (1408.2701
    §VII.2.7, Th 4.8.7), activity-driven temporal networks (1408.2701 §VIII).
- **evolution:**
  - `+`: replicator dynamics and hawks and doves (Gr 8.7, Th 5.3.1, cond-mat/0607344 §3.3), Moran process
    (cond-mat/0607344 §4.3, Je 10.6), Eigen quasispecies (Gr 8.3), hypercycles (Gr 8.5.2), NK fitness landscapes
    (Th 5.4.3, TC 11), Tangled Nature (Je 11.3), adaptive climbing in finite populations (Gr 8.4), rock-paper-scissors
    (cond-mat/0607344 §7);
  - `B:add-lattice-geometries`: Bak-Sneppen, which needs the extremal update (Gr 6.6, Th 5.4.4, Je 4.2);
  - `P:add-adaptive-networks`: Jain-Krishna autocatalytic networks (Th 5.5.1.4), Solé-Manrubia extinctions
    (Th 5.5.3.1).
- **neuroscience:**
  - `✓`: integrate-and-fire, Izhikevich;
  - `+`: Hodgkin-Huxley (Tr 5.3, 2112.12147 §1.2), FitzHugh-Nagumo ODE (Tr 5.4, DP 2.2.4), Wilson-Cowan
    (2112.12147 §1.3.2), theta neurons and QIF networks (1902.05307 §2.3), networks of leaky integrate-and-fire neurons
    with delays (2112.12147 §1.3.4), polyhomeostatic neuron (Gr 10.3.1), winners-take-all network (Gr 10.4.2), LEGION
    oscillators (Gr 9.5), Adaptrode synapse (Mo 6.9);
  - `B:add-nonlocal-operators`: Amari neural field (2112.12147 §1.3.1).
- **systems_biology:**
  - `+`: genetic toggle switch and repressilator (2111.04508), negative autoregulation and bursty gene expression
    (1608.06582 §3), Drosophila circadian clock (Tr 5.2), Kauffman random Boolean networks and the yeast cell cycle
    network (Gr 7).
- **physiology:**
  - `✓`: Mackey-Glass (Di 6.3), FitzHugh-Nagumo spirals (DP 11.2.1);
  - `+`: Aliev-Panfilov cardiac tissue (DP 11.3.3), lumped cardiovascular model (Tr 5.5.2).
- **collective_behaviour:**
  - `✓`: pheromone foraging;
  - `+`: boids in two dimensions (TC 10), bird swarm with pair potentials (Gr 4.3.3), binary ant decision
    (Gr 4.3.2), active walkers and trail formation (cond-mat/0012229 §I.F), calm and nervous rats (Gr 4.3.1);
  - `B:extend-agent-spaces`: boids in three dimensions (TC 10).
- **oncology:**
  - `+`: multiscale tumour angiogenesis (Cm 8);
  - `P:add-kinetic-models`: immune competition (Be 5.5).

### earth

- **climate:**
  - `+`: zero-dimensional energy balance with ice albedo (DP 2.1, 1910.00583 §III.1), Budyko-Sellers latitude model
    (1910.00583 §III.1), Daisyworld (Je 3), Hasselmann stochastic climate (1910.00583 §III.7.3), ENSO delay oscillator
    (1910.00583 §V.5), Saltzman glaciation and Van der Pol ice ages (DP 12.3.3), Rothman excitable carbon cycle
    (DP 12.3.1), Lorenz-84 and Lorenz-96 (DP 4, 12), Gissinger geodynamo (DP 4), coupled soil and deep temperature
    maps (Mi 4.5);
  - `P:add-stocks-and-flows`: declining energy return on investment (Mo 9.6).
- **oceanography:**
  - `+`: Stommel thermohaline boxes (DP 12, 1910.00583 §III.3.1);
  - `B:add-nonlocal-operators`: quasi-geostrophic double gyre (1910.00583 §III.6.1).
- **seismology:**
  - `✓`: Olami-Feder-Christensen (Ch 8);
  - `+`: Burridge-Knopoff spring-block fault (Ch 8.1).

### economics

- **macroeconomics:**
  - `✓`: Solow, multiplier-accelerator;
  - `+`: toy firm investment (Ca 2.1), exchange rates in international trade (Ha 8), capital stock and bank account
    (TS 2);
  - `P:add-optimal-control`: Pontryagin capital and consumption (Di 9.1);
  - `P:add-agent-economies`: firm and bank credit networks (Ca 2.2), fractional-reserve banking (Ha 9).
- **markets:**
  - `✓`: cobweb;
  - `+`: Cournot duopoly with beliefs (Ha 6.2), Riccati price adjustment (Fo 16.1), produce market and household demand
    (Ha 2–3), car-dealer inventory with delays (TS 2), car wash queue (Cy 6.4);
  - `P:add-stocks-and-flows`: commodity cycle with capacity delay (Ba 4.11.4);
  - `P:add-agent-economies`: double auction, toy and three-goods traders (Go 8–12), shops and digital markets
    (Ha 6.3–6.4), Edgeworth and Red Cross barter (Ha 5), labour market (Ha 7), Ajani's choice and fair food (Cy 3, 7),
    Sugarscape (Ar 16.2, TC 9);
  - `B:out`: static oligopoly equilibria (Fo 16, 20, 21).
- **finance:**
  - `✓`: geometric Brownian motion;
  - `+`: multiplicative, Gibrat, and Kesten processes (Ca 4.3.5, Th 3.3.3), Bouchaud-Mézard wealth (0905.1518 §III.2),
    DebtRank cascades (Th 4.10.1), random-walk traders with arbitrage (Sc 2.2);
  - `P:add-agent-economies`: order book with imitation (Ca 3).
- **econophysics:**
  - `✓`: Boltzmann wealth exchange (Go 8.1);
  - `P:add-agent-economies`: exchange with debt limit and with saving propensity (0905.1518 §II, 0909.1974 §VIII),
    minority game and its evolutionary version, El Farol, and Kolkata Paise Restaurant (0909.1974 §IX, Cm 9).
- **resources:**
  - `✓`: harvesting farmers;
  - `+`: oil economy and renewable fishery with capital (TS 2);
  - `P:add-agent-economies`: tragedy of the meadow commons (Ha 10.2);
  - `P:add-stocks-and-flows`: cocoa, hilsa fishery, and food security models (Ba 8–10).
- **innovation:**
  - `+`: NK technology model (Th 5.4.3.5), Arthur-Polak technology evolution (Th 5.6.3), combinatorial creative phase
    transition (Th 5.6.4), fixed-line phone adoption (Ha 4.5).

### social

- **opinion_dynamics:**
  - `✓`: voter model, majority rule automaton;
  - `+`: Sznajd, Galam majority, social impact theory (0710.3256 §III), Deffuant bounded confidence (0710.3256 §III.6,
    Gr 4.3.4), naming game (0710.3256 §V), belief diffusion with influentials (Cy 4), cognitive consistency agents
    (Cm 2), threshold adoption (Ha 4.3), voter variants (Th 4.8.8.1);
  - `P:add-adaptive-networks`: Hegselmann-Krause (0710.3256 §III.6), coevolving voter (0710.3256 §X, Th 4.8.8.2),
    join-a-club spins (Th 6.4.3.2);
  - `P:add-agent-economies`: Axelrod culture dissemination (0710.3256 §IV).
- **segregation:**
  - `✓`: Schelling (Ar 16.1, TC 9, Je 11.2);
  - `+`: stochastic Schelling (Je 11.2).
- **cooperation:**
  - `+`: spatial prisoner's dilemma with best-neighbour and Fermi imitation (Ar 13, Gr 8.7, cond-mat/0607344 §6),
    public goods game (cond-mat/0607344 App. A.8), iterated prisoner's dilemma tournament (TC 12).
- **spreading:**
  - `+`: Maki-Thompson rumours and Watts threshold cascades (1408.2701 §X), Simon-Yule growth (Th 3.3.4).
- **crowds:**
  - `+`: social force pedestrians (cond-mat/0012229 §V.B);
  - `P:add-kinetic-models`: crowd dynamics and evacuation (Be 6.3–6.5), criminality and security (Be 5.4).
- **organizations:**
  - `+`: Bonabeau hierarchies and Barabási priority queues (0710.3256 §VII–VIII), employment takeover (Cy 1.4),
    trafficking and brothel movement models (Cy 5), insider sabotage and obesity models (Cm 4, 6);
  - `B:extend-agent-spaces`: the Kauai population model on GIS data (Cm 3).

### engineering

- **traffic:**
  - `✓`: Nagel-Schreckenberg (cond-mat/0012229 §III.B.1);
  - `+`: Charbonneau traffic (Ch 7), optimal velocity (cond-mat/0012229 §III.A.2), intelligent driver model
    (§III.A.3), Lighthill-Whitham-Richards (§III.D.1), Payne and Kerner-Konhäuser models (§III.D), car following with
    delay (Gr 4.4), ring-road driver agents (TC 10);
  - `B:add-nonlocal-operators`: GKT non-local traffic (§III.E.4);
  - `P:add-agent-economies`: ridesharing (Cm 7).
- **circuits:**
  - `✓`: RLC circuit;
  - `+`: Chua double and multiple scrolls (Tr 3.4), modified Chua with mixed-mode oscillations (Tr 2.4), memristor
    oscillator (Tr 2.2.2), tunnel-diode oscillator (Tr 2.2.2), circuits under periodic impulses (Tr 2.3), vibration
    energy harvester (Tr 3.8).
- **control:**
  - `+`: thermostat (TS 2), PID governance loop (Mo 12.3), pendulum swing-up and satellite transfer (Di 9.2–9.3).
- **power_systems:**
  - `+`: swing-equation power grid with Braess paradox (DP 10.2.3), regenerative braking (Tr 6.4), Faraday-disk charger
    (Tr 6.5), wheel and rail contact (Tr 6.6).
- **robotics:**
  - `+`: light-sensitive robots with sensory delay (Ar 10), swarmtaxis (Cm 10).
- **processes:**
  - `+`: bathtub (TS 1), cascade classifier (Bs 4);
  - `P:add-stocks-and-flows`: rice milling and solid waste systems (Ba 11–12), food relief and irrigation systems
    (Ba 4.11).

### mathematics

- **dynamical_systems:**
  - `✓`: Lorenz (Gr 3.1.2, Tr 2.6), Rössler (DP 3, Tr 3.7), Sprott jerk, logistic map (Ch 1.3, DP 1.1, Sa 10), Hénon
    map;
  - `+`: standard, circle, tent, baker, and towel maps (DP 8–9, Di 2.2, Li 8), intermittency maps (Di 1.19, Je 12.3),
    Newton-Raphson and cat maps (Di 3.16, 5.7), saddle-node, Hopf, Takens-Bogdanov, and cusp normal forms (Gr 2.2–2.3,
    Di 4.14), finite-time blow-up (Fo 14.2).
- **stochastic_processes:**
  - `+`:
    - random walks (Ar 5.1, Ch C, Je 10.1);
    - Ornstein-Uhlenbeck with weakening restoring force (Je 10.5);
    - Pólya urn (Th 2.5.3);
    - Galton-Watson branching (Gr 6.5, Je 5);
    - sample-space-reducing processes and fragmentation (Th 2.5.4, 3.3.5);
    - ageing random walk (Th 6.4.3.1);
    - Bernoulli and Poisson processes (Th 2.5.1);
  - `B:add-stochastic-maps`: persistent random walk (Je 10.4), Markov and hidden Markov sources (Li 3.2).
- **networks:**
  - `+`: random walks, diffusion, and PageRank iterations on networks (Th 4.8, Je 8.5.2);
  - `P:add-adaptive-networks`: Barabási-Albert and Holme-Kim growth (Ar 12.5, Gr 1.5, TC 4, Th 4.5.4).
- **fractals:**
  - `B:add-stochastic-maps`: iterated function systems by the chaos game (Di 5.4);
  - `B:add-complex-states`: Mandelbrot and Julia sets (Di 5.5);
  - `B:out`: Koch, Cantor, and Sierpiński substitutions (Th 3.5.2, Tr 3.5).

### computer_science

- **automata:**
  - `✓`: elementary automata (Ar 4.1, Ch 2.1, TC 5, Li 4), Game of Life (Ar 4.2, TC 6), Langton's ant (Ch 2.4);
  - `+`: Highlife and the planar automaton zoo (Ch 2.2, TC 6), noisy automata (Li 4.3.3), diffusion automaton (TC 7),
    unbounded-alphabet automaton (CC 3.3.3);
  - `P:add-group-cellular-automata`: automata on the free group, lamplighter, Heisenberg, and Baumslag-Solitar groups,
    linear automata over rings, and subshifts (CC 4–8);
  - `B:out`: automata on the Grigorchuk group (CC 6.9).
- **evolutionary_computation:**
  - `+`: RobbyGA can collector (Go 6), bit-string evolution on a fitness landscape (TC 11).
- **neural_networks:**
  - `+`: Hopfield network (Tr 7.3, 2112.12147 §3.1.2);
  - `B:out`: Elman recurrent network trained by backpropagation (Gr 10.5.1).

These models are not in the inventory, because they are formalisms or already planned rather than distinct models:
- the system-dynamics demonstrations of Bala chapter 4 and Meadows, beyond those listed above;
- the kinetic theory of active particles in general (Be 3), which `add-kinetic-models` covers.
