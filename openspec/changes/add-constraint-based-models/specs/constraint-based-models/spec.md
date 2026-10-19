# Spec Delta

## Purpose

Lets a user declare constraint-based models, which are linear constraint systems over flux variables with bounds and
objectives, solve them at a point in time, couple the solve to dynamics, regulation, lattices, and communities, analyse
them with established methods, and enumerate the elementary modes and extreme pathways of their flux cone.

## ADDED Requirements

### Requirement: Constraint networks are declared from symbols
The modelling layer SHALL declare a constraint network from pools, processes with coefficients per pool and bounds,
extra linear constraints, switches, Boolean rules over switches that gate processes, and objectives. Each process SHALL
expose a flux symbol that laws and observables read. Invalid declarations SHALL raise a named error that names the
offending part.

#### Scenario: Stoichiometric matrix
- **WHEN** a user declares a network of 8 pools and 10 processes
- **THEN** its stoichiometric matrix is 8 by 10 with the declared coefficients, in the declared order

#### Scenario: Inverted bounds
- **WHEN** a process is declared with a lower bound of 5 and an upper bound of 1
- **THEN** construction raises a named validation error that names the process and both bounds

#### Scenario: Rule over an undeclared switch
- **WHEN** a rule names a switch that the network does not declare
- **THEN** construction raises a named validation error that names the process and the switch

### Requirement: Networks solve in the default install
A network SHALL be solved as a linear program in the default install, without the optional extra, and SHALL return the
fluxes, the objective values, the status, the pool duals, and the reduced costs. Several objectives SHALL be optimised
in order, each keeping the earlier optima.

#### Scenario: Toy network optimum
- **WHEN** Raman's toy network of Table 8.2 is solved for its growth objective
- **THEN** the optimum equals the value of Table 8.2 within 1e-9, and every pool balance is zero within 1e-9

#### Scenario: Domain-agnostic flow network
- **WHEN** a transport network with capacities on its routes is declared as a network that maximises the flow from a
  source pool to a sink pool
- **THEN** the optimum equals the maximum flow of the same graph computed by NetworkX

#### Scenario: Lexicographic objectives
- **WHEN** a network maximises growth and then minimises the total absolute flux
- **THEN** the growth equals the single-objective optimum within 1e-9, and the total flux is no larger than that of
  the single-objective solution

### Requirement: Switches and rules gate processes
A process whose rule is false for the current switch values SHALL have both bounds set to zero. Switches SHALL change
as parameters at construction, as live parameter changes, and through switch laws.

#### Scenario: Isozymes
- **WHEN** a process has the rule `either(g1, g2)` and only `g1` is switched off
- **THEN** the process keeps its bounds, and switching off `g2` as well sets them to zero

#### Scenario: Live knock-out
- **WHEN** a running static network has the switch of an essential process set to 0 between frames
- **THEN** the next frame's objective is zero within 1e-9

### Requirement: Interactions bound fluxes from other elements
Interactions SHALL return bound laws whose expressions read states, parameters, and inputs. The bounds in effect at a
solve SHALL be the intersection of the declared bounds and every bound law on that flux.

#### Scenario: Michaelis-Menten uptake
- **WHEN** a medium's glucose concentration is 0.015 with a half-saturation constant of 0.015 and a maximal uptake of
  10
- **THEN** the glucose uptake flux is bounded by 5 at that solve

#### Scenario: Two bound laws
- **WHEN** two interactions give one flux the upper bounds 8 and 6
- **THEN** the solve uses the upper bound 6

### Requirement: Infeasible solves follow a declared policy
A network SHALL declare whether an infeasible solve raises a named error, sets fluxes and objectives to zero, or keeps
the last fluxes. The error SHALL name the network, the time, the site if any, and the processes and pools of an
irreducible infeasible subset.

#### Scenario: Raised infeasibility
- **WHEN** a network with the policy `raise` requires a maintenance flux of 8 while its substrate uptake is bounded by 0
- **THEN** solving raises a named error that names the network, the time, the maintenance process, and the uptake
  process

#### Scenario: Zero policy
- **WHEN** the same network has the policy `zero`
- **THEN** the solve returns zero fluxes and a zero objective, and the run continues

### Requirement: Solves run inside ODE right-hand sides
A network in the same subsystem as rate laws SHALL be solved at every evaluation of the drift, once per evaluation,
before the rates that read its fluxes. This is the direct approach of dynamic FBA.

#### Scenario: One solve per evaluation
- **WHEN** three rate laws read fluxes of one network and the drift is evaluated once
- **THEN** the network is solved exactly once

### Requirement: Separate subsystems hold fluxes between synchronisation points
A network in a sibling subsystem of rate laws SHALL form its own partition, solved at every synchronisation point. Its
fluxes SHALL be held as inputs until the next one, which is the static optimisation approach of dynamic FBA.

#### Scenario: Static optimisation step
- **WHEN** biomass and substrate are co-simulated with held fluxes over one synchronisation interval
- **THEN** they match the closed-form update of Raman's Eqs. 9.8–9.9 within 1e-8 relative error

#### Scenario: Agreement with the direct approach
- **WHEN** the E. coli core batch culture runs with the static optimisation approach and intervals 0.1, 0.05, and
  0.025 h
- **THEN** the biomass at 6 h approaches the direct approach's biomass, with an error that falls at least linearly with
  the interval

### Requirement: Networks solve per lattice site
A network on a lattice SHALL be solved at every site where its activity condition holds, with bound laws read per site
from fields. Inactive sites SHALL have zero fluxes and SHALL NOT be solved.

#### Scenario: Only occupied sites are solved
- **WHEN** a lattice of 25 by 25 sites has biomass above the threshold at 9 sites
- **THEN** one synchronisation step solves the network 9 times, and every other site's fluxes are zero

#### Scenario: Single site reproduces the well-mixed model
- **WHEN** a lattice of one site runs the batch culture without diffusion
- **THEN** its biomass matches the well-mixed dynamic FBA within 1e-6 relative error

### Requirement: Boolean regulation alternates with solves
Switch laws SHALL set switches from expressions such as Boolean node states, and a map partition of regulatory nodes
SHALL alternate with the network's solve at the regulatory time step.

#### Scenario: Regulation gates a process
- **WHEN** a regulatory node that represses a process is on
- **THEN** the next solve has that process's flux at zero, and after the node turns off the process may carry flux

### Requirement: Static analyses come with an optional extra
With the extra `metabolism` installed, the library SHALL analyse a network with:
- flux variability;
- parsimonious fluxes;
- single and double deletions of processes or switches;
- MoMA and ROOM in their linear and quadratic or mixed-integer forms;
- synthetic lethal pairs.

Results SHALL be indexed by process or switch name.

#### Scenario: Flux variability at the optimum
- **WHEN** flux variability runs on the E. coli core model at 100% of the optimum
- **THEN** the range of the biomass flux is 0.8739 within 1e-4 at both ends

#### Scenario: Parity with the core solve
- **WHEN** the E. coli core model is solved by the core and by the extra's analysis path
- **THEN** the growth agrees within 1e-7, and the parsimonious total flux agrees with the core's lexicographic total
  flux within 1e-6

#### Scenario: Adjustment after a knock-out
- **WHEN** the `pgi` gene (b4025) is knocked out against the wild-type solution
- **THEN** FBA gives growth 0.8632 within 1e-3, and MoMA and ROOM give positive growth below it

### Requirement: Strain designs are computed with the extra
With the extra installed, the library SHALL compute OptKnock, RobustKnock, OptCouple, and minimal cut set designs with
a maximum number of interventions. It SHALL return each design as sets of process or switch names.

#### Scenario: Growth-coupled succinate
- **WHEN** OptKnock designs succinate production in the E. coli core model with at most 3 knock-outs and growth of at
  least 0.1
- **THEN** a returned design, at maximal growth, secretes at least 9 mmol/gDW/h of succinate at its minimum, where the
  wild type secretes none

### Requirement: Flux scanning runs in the default install
The library SHALL scan fluxes with enforced objective flux (FSEOF). It SHALL enforce a target flux at increasing
levels, and report the processes whose absolute flux rises monotonically with it, without the optional extra.

#### Scenario: Scan toward a product
- **WHEN** FSEOF scans the succinate exchange of the E. coli core model in 10 steps
- **THEN** the reported processes carry a flux that rises with every step, and the growth falls from its optimum

### Requirement: Communities are modelled jointly and with trade-offs
The library SHALL:
- join member networks into one network with a shared compartment and an abundance-weighted growth objective;
- compute SteadyCom with one common growth rate, without the optional extra;
- compute MICOM's cooperative trade-off with the extra.

#### Scenario: Identical members
- **WHEN** two copies of the E. coli core model share one medium whose uptake allows each of them a glucose uptake of 10
- **THEN** joint FBA gives each member growth 0.8739 within 1e-4, and SteadyCom gives the common rate 0.8739 within
  1e-4 with abundances that sum to 1

#### Scenario: Trade-off at full community growth
- **WHEN** the cooperative trade-off runs at fraction 1
- **THEN** the community growth equals the joint FBA optimum within 1e-5

### Requirement: Models import and export standard formats
With the extra installed, the library SHALL read SBML and BiGG JSON files and BiGG identifiers into networks, keeping
the identifiers of reactions, metabolites, and genes, the bounds, the objective, and the gene rules. It SHALL write
networks back to both formats.

#### Scenario: Bundled E. coli core
- **WHEN** a user reads the bundled E. coli core model and solves it in the core
- **THEN** aerobic growth is 0.8739 within 1e-4, and with oxygen uptake bounded at 0 it is 0.2117 within 1e-4

#### Scenario: Round trip
- **WHEN** a network is written to SBML and to JSON and read back
- **THEN** both copies have the same matrix, bounds, rules, and optimum as the original

### Requirement: Elementary modes of a network are enumerated
With the extra `polyhedra` installed, the library SHALL enumerate the elementary modes of a network: the steady-state
flux vectors with minimal support in which every irreversible process runs in its declared direction. It SHALL take the
directions from the bounds and switches, compute exactly, and return a table indexed by mode and process, normalised to
a chosen process.

#### Scenario: Toy network modes
- **WHEN** the elementary modes of Raman's toy network of Table 8.2 are enumerated and normalised to the uptake `b1`
- **THEN** there are exactly 2 modes, one through `v1` and one through `v2` and `v3`, both with `v4 = 11/12`,
  `v5 = 1/12`, `v6 = 5/3`, `v7 = 1/6`, `b2 = 5/3`, and `vbio = 1/12`, and each satisfies every pool balance exactly

#### Scenario: Optimum as a combination of modes
- **WHEN** the FBA optimum of the toy network, growth 50, is decomposed over its 2 modes by non-negative least squares
- **THEN** the residual is below 1e-9, and the growth equals 1000 divided by the ratio `v6 / vbio = 20` of both modes,
  since the bound on `v6` binds

#### Scenario: Domain-agnostic circulation
- **WHEN** a network declares one pool per node and one irreversible process per arc of the complete directed graph on 4
  nodes
- **THEN** its 20 elementary modes are the 20 simple cycles that NetworkX finds in that graph

#### Scenario: Knock-out removes modes
- **WHEN** the switch that gates `v1` is off
- **THEN** only the mode through `v2` and `v3` remains

#### Scenario: Non-zero balance
- **WHEN** a network whose pool `A` has a balance of 5 is enumerated
- **THEN** a named error names the network and the pool, and says that modes are defined for steady state only

### Requirement: Extreme pathways of a network are enumerated
With the extra `polyhedra` installed, the library SHALL compute the extreme pathways of a network: the minimal set of
generators of its flux cone after every reversible internal process is split into two irreversible ones, with each
exchange process kept as one column in its declared direction. A process SHALL count as an exchange when it touches one
pool, unless the network names its exchanges. Generators of a cone that is not pointed SHALL be returned as lines,
apart from the pathways.

#### Scenario: Pathways and modes
- **WHEN** the extreme pathways and elementary modes of the toy network are computed
- **THEN** every extreme pathway is an elementary mode, every elementary mode is a non-negative combination of extreme
  pathways, and for this network both sets hold the same 2 vectors

### Requirement: Enumeration is bounded in time
An enumeration SHALL stop after a time limit, 60 s by default, and SHALL then raise a named error that names the
network, its number of processes, and the limit, and suggests closing exchanges or enumerating a subnetwork.

#### Scenario: Genome-scale network
- **WHEN** the elementary modes of the full E. coli core model are enumerated with a limit of 5 s
- **THEN** the enumeration raises the named error within 6 s

### Requirement: Enumeration runs as a simulation
A network declared for enumeration SHALL compile to an enumeration problem whose run records one frame per mode or
pathway, ordered by a declared ratio such as the growth yield on a substrate, with the mode's fluxes and the ratio.

#### Scenario: Modes as frames
- **WHEN** the toy network's modes run through `Simulation`
- **THEN** the trajectory's axis is `mode`, it has 2 frames, and the fluxes of frame `k` equal row `k` of the table the
  analysis function returns

### Requirement: The extras are optional and named
The default install SHALL NOT install or import any library that the extras provide. Every capability that needs an
extra SHALL raise a named error when it is missing. The error SHALL name the capability and its extra, `metabolism` or
`polyhedra`, give the install command, and list the capabilities that run without it. Labelling simulation and flux
estimation SHALL need neither extra.

#### Scenario: Missing extra
- **WHEN** a user without the extra calls flux variability
- **THEN** a named error says that flux variability needs `pip install 'scikit-complexity[metabolism]'`, and that
  network solves, FSEOF, SteadyCom, labelling simulation, and flux estimation run without it

#### Scenario: Missing polyhedra extra
- **WHEN** a user without the extra `polyhedra` enumerates elementary modes
- **THEN** a named error says that elementary modes need `pip install 'scikit-complexity[polyhedra]'`

#### Scenario: Default install stays importable
- **WHEN** the package and its model catalog are imported without the extras
- **THEN** no module of the extras is imported, and every model that needs an extra is listed in the catalog

### Requirement: Fluxes are recorded in trajectories
A run SHALL record each network's fluxes, objective values, and switch values per frame under dotted names, with one
value per site for lattice networks. It SHALL record the pool duals only on request.

#### Scenario: Flux series
- **WHEN** a user asks a dynamic FBA trajectory for `cell.flux.EX_glc__D_e`
- **THEN** it returns the glucose exchange flux at every frame, and it is zero after glucose depletion
