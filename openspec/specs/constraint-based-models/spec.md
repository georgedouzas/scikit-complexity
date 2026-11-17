# constraint-based-systems Specification

## Purpose

Lets a user set algebraic states to optimal solutions of linear programs, couple the solve to dynamics, regulation,
lattices, and communities, and enumerate the elementary modes and extreme pathways of their steady-state cone.

## Requirements

### Requirement: Linear program laws set algebraic states
`LinearProgramLaw` SHALL set its variables, algebraic states, to an optimal solution of the linear program whose
constraints are its Sage relations and whose objectives are its `LinearObjectiveSpec`s, solved in order with each optimum kept
within a relative 1e-9. Relations SHALL be linear in the variables with constant coefficients, and their other terms MAY
read states, parameters, and the time.

#### Scenario: Toy network optimum
- **WHEN** Raman's toy network of Table 8.2 is solved for its growth objective with every bound at 1000
- **THEN** the growth is 50 within 1e-9, and every balance is zero within 1e-9

#### Scenario: Domain-agnostic flow network
- **WHEN** a transport network with capacities on its routes is written as a program that maximises the flow from a
  source node to a sink node
- **THEN** the optimum equals the maximum flow of the same graph computed by NetworkX

#### Scenario: Lexicographic objectives
- **WHEN** a program maximises growth and then minimises the sum of the absolute values of its variables
- **THEN** the growth equals the single-objective optimum within 1e-9, and the sum is no larger than that of the
  single-objective solution

#### Scenario: Variable that is not algebraic
- **WHEN** a linear program law names a state that is not declared algebraic as a variable
- **THEN** a named error names the state and says to declare it with `StateSpec(algebraic=True)`

### Requirement: Programs that share variables merge
The linear program laws whose variables overlap, from any element or interaction, SHALL form one program whose
constraints add, and at most one of them SHALL hold objectives.

#### Scenario: Michaelis-Menten uptake
- **WHEN** an interaction bounds the glucose uptake of a cell by `10 S / (0.015 + S)` and the medium holds `S = 0.015`
- **THEN** the uptake is bounded by 5 at that solve

#### Scenario: Two bounds
- **WHEN** two interactions give one variable the upper bounds 8 and 6
- **THEN** the solve uses the upper bound 6

### Requirement: Switches gate variables through adaptive parameters
A bound SHALL be able to read adaptive parameters with the domain `{0, 1}` through `either` and `both`, so that a
knock-out sets a parameter to 0 at construction, between runs, or by an `UpdateLaw` of another component.

#### Scenario: Isozymes
- **WHEN** a reaction is bounded by `1000 * either(g1, g2)` and only `g1` is switched off
- **THEN** the reaction keeps its bound, and switching off `g2` as well sets it to zero

#### Scenario: Live knock-out
- **WHEN** the switch of an essential reaction is set to 0 between two runs of a program
- **THEN** the objective of the second run is zero within 1e-9

#### Scenario: Regulation gates a reaction
- **WHEN** an `UpdateLaw` sets the switch of a reaction from a Boolean node that represses it, and the node is on
- **THEN** the next solve has that reaction at zero, and after the node turns off the reaction may carry flux

### Requirement: Infeasible solves follow a declared policy
A linear program law SHALL declare whether an infeasible solve raises `InfeasibleError`, sets the variables and
objectives to zero, or keeps the last values. The error SHALL name the path, the time, the member if any, and the
constraints and variables of an irreducible infeasible subset.

#### Scenario: Raised infeasibility
- **WHEN** a program with the policy `'raise'` requires a maintenance flux of 8 while its substrate uptake is bounded by
  0
- **THEN** solving raises `InfeasibleError` that names the program, the time, the maintenance variable, and the uptake
  variable

#### Scenario: Zero policy
- **WHEN** the same program has the policy `'zero'`
- **THEN** the solve returns zero variables and a zero objective, and the run continues

### Requirement: Programs solve where their variables are read
A program beside rate laws under continuous time SHALL solve once per evaluation of the drift, before the rates that
read it. A program alone or beside update laws under discrete time SHALL solve once per step, before the updates. A
program alone under static time SHALL solve once per frame in the static form of `add-static-models`.

#### Scenario: One solve per evaluation
- **WHEN** three rate laws read variables of one program and the drift is evaluated once
- **THEN** the program is solved exactly once

#### Scenario: Static program
- **WHEN** a system under static time holds only a program
- **THEN** each frame solves it once from the current parameters

### Requirement: Co-simulated programs hold their values between synchronisation points
A program in a discrete-time nested system beside a sibling with rate laws SHALL solve at every synchronisation point,
and the sibling SHALL read its variables as inputs held until the next one, which is the static optimisation approach of
dynamic FBA.

#### Scenario: Static optimisation step
- **WHEN** biomass and substrate are co-simulated with held fluxes over one synchronisation interval
- **THEN** they match the closed-form update of Raman's Eqs. 9.8 and 9.9 within 1e-8 relative error

#### Scenario: Agreement with the direct approach
- **WHEN** the E. coli core batch culture runs with the static optimisation approach and intervals 0.1, 0.05, and
  0.025 h
- **THEN** the biomass at 6 h approaches that of the direct approach, with an error that falls at least linearly with
  the interval

### Requirement: Programs solve per member
A program of a group on the sites of a lattice SHALL solve per member where its activity holds, with bounds read per
member, and inactive members SHALL have zero variables and SHALL NOT be solved.

#### Scenario: Only active sites are solved
- **WHEN** a lattice of 25 by 25 sites has biomass above the threshold at 9 sites
- **THEN** one evaluation solves the program 9 times, and every other site's variables are zero

#### Scenario: Single site reproduces the well-mixed system
- **WHEN** a lattice of one site runs the batch culture without diffusion
- **THEN** its biomass matches the well-mixed dynamic FBA within 1e-6 relative error

### Requirement: Program values are recorded
A run SHALL record the variables of every program, its objective values under `<path>.objective_<k>`, and, when the law
asks for them, the duals of its constraints under `<path>.dual_<k>`, with one value per member for groups.

#### Scenario: Recorded values
- **WHEN** a user asks a dynamic FBA trajectory for the glucose exchange variable of the cell
- **THEN** it returns the exchange at every frame, and it is zero after glucose depletion

### Requirement: Misused programs are reported before compilation
`PassCompiler` SHALL report a relation that is not linear in the variables or whose coefficient reads a state
(`linear-program-form`) and two merged laws that both hold objectives (`linear-program-objectives`), each with its path
and fix.

#### Scenario: Nonlinear relation
- **WHEN** a constraint holds the product of two variables
- **THEN** the report holds an issue with the code `linear-program-form` that names the relation and both variables

#### Scenario: Two objectives
- **WHEN** two laws over shared variables both hold objectives
- **THEN** the report holds an issue with the code `linear-program-objectives` that names both paths

### Requirement: Elementary modes and extreme pathways are enumerated
`elementary_modes` SHALL enumerate exactly, for the kind `'modes'`, the steady-state vectors of a program with
minimal support in which every irreversible variable runs in its declared direction, and, for the kind `'pathways'`,
the generators of the cone with every reversible internal variable split and each exchange kept as one variable, of
either sign when it is reversible, none of which is a non-negative combination of the others (Raman 2021, Section 9.8).
It SHALL take the directions from the bounds at the initial values, and SHALL stop after a time limit with
`EnumerationLimitError`.

#### Scenario: Toy network modes
- **WHEN** the elementary modes of Raman's toy network are enumerated and normalised to the uptake `b1`
- **THEN** there are exactly 2 modes, one through `v1` and one through `v2` and `v3`, both with `vbio = 1/12` and
  `v6 = 5/3`, and each satisfies every balance exactly

#### Scenario: Optimum as a combination of modes
- **WHEN** the optimum of the toy network, growth 50, is decomposed over its 2 modes by non-negative least squares
- **THEN** the residual is below 1e-9, and the growth equals 1000 divided by the ratio `v6 / vbio = 20` of both modes

#### Scenario: Domain-agnostic circulation
- **WHEN** a program has one balance per node and one irreversible variable per arc of the complete directed graph on 4
  nodes
- **THEN** its 20 elementary modes are the 20 simple cycles that NetworkX finds in that graph

#### Scenario: Knock-out removes modes
- **WHEN** the switch that gates `v1` is off
- **THEN** only the mode through `v2` and `v3` remains

#### Scenario: Non-zero balance
- **WHEN** a program whose balance of `A` has the right side 5 is enumerated
- **THEN** a named error names the balance and says that modes are defined at steady state

#### Scenario: Pathways and modes
- **WHEN** the extreme pathways and elementary modes of the toy network are computed
- **THEN** every pathway is a mode, every mode is a non-negative combination of pathways, and both sets hold the same 2
  vectors

#### Scenario: Figure 9.2 of Raman
- **WHEN** the elementary modes and extreme pathways of the network of Figure 9.2 of Raman (2021) are enumerated
- **THEN** there are 7 modes and 4 pathways, every pathway is a mode, and the pathways from `C` to `G` yield 1 and 2

#### Scenario: Genome-scale network
- **WHEN** the elementary modes of the E. coli core system are enumerated with a limit of 5 s
- **THEN** the analysis raises `EnumerationLimitError` within 6 s

### Requirement: The extra is optional and named
The default install SHALL NOT install or import COBRApy, StrainDesign, or MICOM. Every class that needs them SHALL raise
`MissingExtraError` naming the class, the extra `metabolism`, and the install command.

#### Scenario: Missing extra
- **WHEN** a user without the extra calls `cobra_measures(system, kind='variability')`
- **THEN** `MissingExtraError` says that it needs `pip install 'scikit-complexity[metabolism]'`

#### Scenario: Default install stays importable
- **WHEN** the package and its system catalog are imported without the extra
- **THEN** no module of the extra is imported, and every system that needs it is listed in the catalog
