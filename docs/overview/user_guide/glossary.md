# Glossary

This page defines the mathematical and library terms that the user guide uses, in alphabetical order. Each entry ends
with a link to the page or section where the term is at work.

## Agent

An agent is an element whose laws are rules that read and change its own states, its neighbours, or the system, one
member at a time. Agents move, choose, trade, and die, which equations of averages cannot express. The tutorial
[Agent-based systems](build_a_custom_system/agent_based_systems.md) builds them.

## Algebraic state

An algebraic state has no derivative of its own, because the equations of its system fix its value at every time. It
takes an equation or a linear program, never a rate. See [EquationLaw](core_components/laws.md#equationlaw).

## Approximate Bayesian computation

Approximate Bayesian computation estimates a posterior without a likelihood. It draws parameters from the prior, runs
the system, and keeps the draws whose summaries lie close to the data. See
[Calibrate](calibrate.md#approximate-bayesian-computation).

## Attractor

An attractor is the set of states that the runs of a dissipative system settle on, such as a fixed point, a cycle, or
the strange set of a chaotic flow. Its dimension and its Lyapunov exponents describe it. See
[Analyse](analyse.md#invariants-of-dynamics).

## Backend

A backend turns the laws of a compiled problem into numeric functions. NumPy is the default, Numba compiles large
systems, and JAX writes functions that can be differentiated. See [Simulate](simulate.md#backends).

## Basin stability

Basin stability is the share of random initial states whose runs reach a given behaviour. It measures how stable the
behaviour is against large disturbances. See [Run experiments](run_experiments.md#basin-stability).

## Calibration

Calibration estimates the parameters of a system from data. A point fit returns one value per parameter, and a
posterior method returns samples of the posterior. See [Calibrate](calibrate.md).

## Chaos

A chaotic system is deterministic, yet two runs that start close apart separate exponentially fast, so its long-term
path cannot be predicted. A positive Lyapunov exponent is its signature. See
[Analyse](analyse.md#chaos-and-long-term-behaviour).

## Co-simulation

Co-simulation runs the partitions of one system, each with its own solver, and exchanges their values at
synchronisation points. It joins parts that no single solver runs, such as agents beside a differential equation. See
[Simulate](simulate.md#co-simulation).

## Compiler

The compiler reads the laws of a prepared system and builds one problem of a mathematical form, such as a system of
ordinary differential equations. A solver then advances that problem. See
[How the parts fit](core_components/index.md#a-run).

## Credible interval

A credible interval of level 90% holds the parameter with probability 0.9 under the posterior. A calibrated method
covers the true value at about that rate over many studies. See [Evaluate](evaluate.md#posterior-coverage).

## Deterministic

A deterministic system draws no random numbers, so one initial state gives one run. Its tag `randomness` is
`deterministic`, the opposite of `stochastic`. See [Run experiments](run_experiments.md#ensembles).

## Domain

The domain of a state is the set of values it may take, such as the real numbers, the integers, or the numbers from
zero up. A role names the domains it accepts, and a state of another domain does not fit it. See
[Check a system](check.md#tags).

## Dotted name

A dotted name joins the label of a part and the name of one of its values, such as `prey.x` or `predation.a`. A
trajectory, a schedule, and an experiment name states and parameters this way. See
[Simulate](simulate.md#the-trajectory).

## Element

An element holds states, the numbers that change as a system runs, such as the size of a population or the balance
of an account. It may also hold parameters and laws of its own. See [Elements](core_components/elements.md).

## Ensemble

An ensemble repeats the runs of a stochastic system with independent random streams. The spread of its outcomes shows
what chance alone can do. See [Run experiments](run_experiments.md#ensembles).

## Field realization

A field realization is a subclass of a core part that names it in the words of one field and fixes its arguments, such
as a compartment of an epidemic written as a `DeclaredElement`. The catalog writes its systems this way. See
[System classes](system_classes.md#field-realizations).

## Fixed point

A fixed point is a state where every rate is zero, so a run that starts there stays there. Its stability says whether
nearby states move toward it, away from it, or around it. See [Analyse](analyse.md#fixed-points-and-stability).

## Frame

A frame is one recorded instant of a run, with the time, the states, and the parameters. The time between frames is
`dt`, while the solver may take smaller steps inside it. See [Simulate](simulate.md#the-trajectory).

## Gillespie algorithm

The Gillespie algorithm runs random events exactly in continuous time. It draws the time of the next event from the
total rate and picks the event in proportion to its rate. See [Simulate](simulate.md#random-runs).

## Group

A group holds many members of one element class, one per site of a lattice, node of a network, or point in space. A
variable group lets members appear and disappear. See [GroupElement](core_components/elements.md#groupelement).

## Identifiability

A parameter is identifiable when the data fix its value. Practical identifiability asks this of the data at hand, by
refitting noisy copies or by profiling the likelihood. See [Evaluate](evaluate.md#practical-identifiability).

## Interaction

An interaction reads states through its roles and returns laws that change them, such as hunting that lowers the prey
and feeds the predators. See [Interactions](core_components/interactions.md).

## Invariant

An invariant is a quantity that must stay constant along a run, such as the energy of a frictionless oscillator. A run
given an invariant checks it at every frame and raises an error when it drifts beyond its tolerance. See
[Simulate](simulate.md#observables-and-invariants).

## Jacobian

The Jacobian of a system is the matrix of the derivatives of its rates with respect to its states. Its eigenvalues at
a fixed point decide the stability of that point. See [Analyse](analyse.md#fixed-points-and-stability).

## Jump

A jump is a random event that changes states by a fixed amount at a random time, such as one birth or one reaction.
Jump laws run under the Gillespie algorithm. See [Simulate](simulate.md#random-runs).

## Kind of data

A kind of data is what an analysis consumes, such as a series, a jump path, a field, agent snapshots, a network, or an
ensemble of runs. An analysis reads its kind from a trajectory or from measured arrays alike. See
[Analyse](analyse.md#kinds-of-data).

## Kind of element

The kind of an element says how many members it holds and where they sit, such as `element`, `group`, or
`site-group`. A role accepts some kinds and rejects the others. See
[Check a system](check.md#what-an-element-carries).

## Label

A label is the name of a part, unique among the parts of its system. Roles bind elements by label, and dotted names
start with it. See [Check a system](check.md#two-parts-with-one-label).

## Law

A law is one equation or rule for one state, such as a rate, the next value at a step, or a random jump. Elements and
interactions return laws, and the compiler joins them into a problem. See [Laws](core_components/laws.md).

## Law kind

The law kind of a part is the kind of law it returns, such as `rate`, `update`, or `jump`. The law kinds of a system
decide its mathematical form and its default solver. See [Check a system](check.md#tags).

## Likelihood

The likelihood of parameters is the probability of the observed data under them. Calibration maximises it or samples
the posterior it implies. See [Calibrate](calibrate.md#the-losses).

## Loss

A loss scores how far a candidate set of parameters lies from the data, such as the sum of squared residuals. A
calibration method minimises it or reads it as a likelihood. See [Calibrate](calibrate.md#the-losses).

## Lyapunov exponent

A Lyapunov exponent is the average rate at which two nearby paths separate along one direction. A positive exponent
means chaos, a negative one contraction, and zero a direction along which paths neither separate nor approach. See
[Analyse](analyse.md#chaos-and-long-term-behaviour).

## Main public methods

The main public methods of a part are `prepare`, which checks it against what it works on, and `run`, which sets what
it contributes. Both return the part itself. See [How the parts fit](core_components/index.md#the-main-public-methods).

## Markov chain Monte Carlo

Markov chain Monte Carlo draws samples of a posterior with a random walk whose long-run distribution is the posterior.
The No-U-Turn Sampler follows the gradient of the likelihood to move faster. See
[Calibrate](calibrate.md#markov-chain-monte-carlo).

## Mass action

The law of mass action makes the rate of a reaction proportional to the product of the amounts of its reactants.
Epidemiology calls the same assumption homogeneous mixing. See
[ReactionNetworkSystem](core_components/general.md#reactionnetworksystem).

## Member

A member is one instance of the element class of a group, such as one cell of a lattice or one agent of a flock. Each
member holds its own copy of the states of the class. See [GroupElement](core_components/elements.md#groupelement).

## Model parameter

A model parameter is an argument of a system class, such as the growth rate `r`, set with `set_params`. Changing it
builds new parts, while a dotted name changes a value of the built parts. See
[Change it](change_it.md#model-parameters).

## Neighbourhood

A neighbourhood says which members of a space count as neighbours of each member, such as the four nearest sites of a
lattice or every point within a radius. Interactions over neighbours read it. See
[Spaces, neighbourhoods, and times](core_components/spaces.md).

## Nested system

A nested system is a system held inside another as one of its parts. Its elements are reached by paths such as
`village/farmers`, and it becomes a partition when its laws fit one form. See [Simulate](simulate.md#co-simulation).

## Observable

An observable is a named value computed from the states at every frame, such as the total population. The trajectory
records it beside the states. See [Simulate](simulate.md#observables-and-invariants).

## Observer

An observer receives the frames of a run as they come. The recorder keeps them as a trajectory, and other observers
check invariants or stream the run to a viewer. See [Observe](observe.md).

## Ordinary differential equation

An ordinary differential equation gives the rate of change of each state as a function of the states and time. A
system of rate laws in continuous time compiles to one. See [Simulate](simulate.md#solver-and-backend).

## Orbit diagram

An orbit diagram shows the values a state visits after a transient, for every value of a parameter. One value means
a steady state, a few a periodic cycle, and many a cycle that does not repeat. See
[Run experiments](run_experiments.md#orbit-diagrams).

## Partition

A partition is a part of a system that runs with its own solver during co-simulation. Each nested system whose laws
fit one form becomes one. See [Simulate](simulate.md#co-simulation).

## Phase space

A phase space pairs every position of a lattice with a value on one or more class axes, such as a velocity. A density
on it describes a population, such as the molecules of a gas, by how much of it sits in each cell. See
[PhaseSpace](core_components/spaces.md#phasespace).

## Posterior

The posterior is the distribution of the parameters given the data and the prior. A posterior method of calibration
returns samples of it. See [Calibrate](calibrate.md#approximate-bayesian-computation).

## Posterior predictive check

A posterior predictive check simulates data from posterior draws and asks whether they look like the observed data.
The share of observed points inside the simulated band measures it. See
[Evaluate](evaluate.md#posterior-predictive-checks).

## Prior

The prior is the distribution of the parameters before the data are seen. A bound `(low, high)` reads as a uniform
prior. See [Calibrate](calibrate.md#approximate-bayesian-computation).

## Propensity

The propensity of a random event is its rate, the expected number of firings per unit of time. Under mass action it is
the rate constant times the product of the reactant amounts. See [Simulate](simulate.md#random-runs).

## Recipe

A recipe is a general system of `skcomplex.systems.general`, such as a reaction network or a stock and flow model,
that a field system subclasses or fills with data. See
[Build a custom system](build_a_custom_system/index.md#as-a-subclass-of-a-shared-recipe).

## Recurrence

A recurrence is a pair of times at which the states of a run lie close to each other. The share of recurrences that
form diagonal lines, the determinism, is close to 1 for a periodic series. See
[Analyse](analyse.md#series-of-a-run-and-of-the-records).

## Residual

A residual is the difference between a simulated value and an observed one at the same time. Losses and measures of
fit are built from residuals. See [Evaluate](evaluate.md#a-run-against-the-records).

## Role

A role is a named slot of an interaction, such as `predator` or `prey`, bound to an element by its label. It names the
kinds of element and the states it reads. See [Check a system](check.md#what-an-interaction-needs).

## Run settings

The run settings are the arguments of `simulate`, such as `until`, `dt`, and `random_state`. A system class tunes
defaults for those a call leaves out. See [Simulate](simulate.md#run-settings).

## Scenario

A scenario is a named variant of a system, such as an intervention, run under the same random draws as the baseline.
See [Run experiments](run_experiments.md#scenarios).

## Schedule

A schedule sets parameters from functions of time at every frame of a run, such as a slow ramp of a birth rate. See
[Simulate](simulate.md#changing-a-run).

## Section

A section reduces a flow to a map by keeping its states at chosen moments. A Poincaré section keeps the moments when a
state crosses a value, and a stroboscopic section samples once per period of a periodic force. See
[Analyse](analyse.md#sections).

## Sensitivity analysis

Sensitivity analysis ranks the inputs of a model by their effect on an output, one at a time or by splitting the
variance of the output. See [Run experiments](run_experiments.md#sensitivity).

## Solver

A solver advances a compiled problem in time, such as the LSODA method for ordinary differential equations or the
Gillespie algorithm for jumps. The tags of a system choose the default solver. See
[Simulate](simulate.md#solver-and-backend).

## Space

A space places the members of a group and says who neighbours whom, such as a lattice, a network, or continuous space.
See [Spaces, neighbourhoods, and times](core_components/spaces.md).

## Spec

A spec is a small frozen record that describes a value, such as a `StateSpec` that gives the domain and the order of a
state. Parts read specs for their declarations and settings. See
[How the parts fit](core_components/index.md#the-elements).

## State

A state is a number that changes as a system runs, such as the size of a population. Its initial value is the
default of its argument. See [How the parts fit](core_components/index.md#the-parts).

## Steady state

A steady state is a state that a run reaches and keeps, so its rates are zero there. It is a fixed point that
attracts nearby runs. See [Run experiments](run_experiments.md#orbit-diagrams).

## Stiff

A stiff system mixes fast and slow time scales, so an explicit solver needs tiny steps to stay stable. Implicit methods
such as BDF take large steps instead. See [Simulate](simulate.md#solver-and-backend).

## Stochastic

A stochastic system draws random numbers, so one initial state gives many possible runs. A `random_state` makes one of
them reproducible. See [Simulate](simulate.md#random-runs).

## Stoichiometry

The stoichiometry of a reaction is the number of units of each species it consumes and produces. Its stoichiometric
vector is the change of every amount when the reaction fires once. See
[ReactionInteraction](core_components/general.md#reactioninteraction).

## Surrogate

A surrogate is a system learned from data that runs, forecasts, and is analysed like a system written by hand. See
[Analyse](analyse.md#equations-learned-from-data).

## Sweep

A sweep runs a system once per row of a design of parameter values and collects the outputs in one table. See
[Run experiments](run_experiments.md#sweeps).

## System

A system composes elements, interactions, a space, and a time into one model that `simulate` runs. A system class
builds its parts from model parameters. See [Systems](core_components/systems.md).

## System class

A system class is a subclass of `BaseSystem` that builds its parts from model parameters and tunes its run settings.
The catalog holds ready-made system classes. See [System classes](system_classes.md).

## Tag

A tag records a capability of a part, such as its time, its law kinds, the domains of its states, or its randomness.
The tags of a system join those of its parts and choose its solver. See [Check a system](check.md#tags).

## Time

The time of a system says whether it flows continuously, steps in discrete increments, or is solved once without time.
See [ContinuousTime](core_components/spaces.md#continuoustime).

## Trajectory

A trajectory holds the frames of a run, with the times, the states, and the parameters of every frame. `simulate`
returns it. See [Simulate](simulate.md#the-trajectory).

## Variational inference

Variational inference fits a simple distribution, such as a multivariate normal, to the posterior instead of sampling
it. It is fast but can miss a skewed posterior. See [Calibrate](calibrate.md#variational-inference).

## Virtual population

A virtual population is a set of parameter rows whose runs look like real observations, drawn from priors and kept
when their outputs lie in plausible ranges. See [Calibrate](calibrate.md#virtual-populations).
