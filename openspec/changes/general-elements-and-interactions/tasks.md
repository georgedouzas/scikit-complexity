# Tasks

## 1. Core mechanisms

- [x] 1.1 Work on `development`, cutting one worktree branch per parallel agent from it, and verify `git status` is clean before the first merge.
- [ ] 1.2 Add `CompartmentElement` and `DensityElement` to `skcomplex.core.elements`, one module each, with docstrings and doctests; verify tests in `tests/core/elements/` for declared compartments and a density used by an `EncounterLaw` pass.
- [ ] 1.3 Add `ReactionInteraction` with the modes `'rate'`, `'jump'`, and `'langevin'`; verify a test that runs `S + I -> 2 I` and `I -> R` in rate and jump mode with 100,000 units, the jump means within 2 percent of the rate solution, and `S + I + R` constant.
- [ ] 1.4 Add `GrowthInteraction` with the forms `'exponential'`, `'logistic'`, `'gompertz'`, `'power'`, and `'bertalanffy'`, and the capacity, exponent, harvest, and delay; verify tests of each form against its closed solution and of the logistic inflection at half the capacity.
- [ ] 1.5 Add `RandomWalkInteraction` with the steps `'neighbor'`, `'gaussian'`, and `'levy'`, a step probability, a rate, and a cost; verify the mean squared displacement `4 D t` within 5 percent for 10,000 Gaussian walkers, and neighbour steps on a lattice and a network.
- [ ] 1.6 Add `SpreadingInteraction` over neighbours, with recovery and loss of immunity; verify the contact process dies out below and survives above its critical rate on a square lattice.
- [ ] 1.7 Add `CouplingInteraction` with a coupling function, lag, delay, and mean field; verify Kuramoto order above 0.8 for 500 oscillators above the critical strength on a complete graph, and the delayed and lagged variants against their reference tests.
- [ ] 1.8 Add `ThresholdInteraction` with reset and redistribution, recording cascades as events; verify sandpile avalanche sizes follow a power law over two decades on a 50 by 50 lattice.
- [ ] 1.9 Add `ChoiceInteraction` with the rules `'logit'`, `'heat_bath'`, `'metropolis'`, and `'fermi'`; verify heat-bath fully connected spins approach `m = tanh(2 m)`, and logit probabilities against `exp(beta u) / sum`.
- [ ] 1.10 Add `ImitationInteraction` with the rules `'random'`, `'majority'`, and `'best'`; verify voter consensus on a ring of 50, with the winning probability equal to the initial fraction over 200 runs.
- [ ] 1.11 Add `ForagingInteraction` and `RegrowthInteraction`; verify energy gain, metabolism, splitting reproduction, death at zero energy, and regrowth by delay and towards a capacity in unit tests.
- [ ] 1.12 Add `OrnsteinUhlenbeckInteraction`, additive and multiplicative; verify the stationary variance `sigma**2 / (2 theta)` within 5 percent, and the log-normal mean of the geometric case.
- [ ] 1.13 Add `AlignmentInteraction`, polar and nematic, with noise and repulsion and attraction zones; verify Vicsek polar order above 0.9 at low noise.
- [ ] 1.14 Add `EvolutionInteraction` with tournament and roulette selection, crossover, and mutation over an alphabet or Gaussian, as private methods that replace `evolve`; verify the mean fitness rises on the count of ones over 50 generations and two runs with one seed agree.
- [ ] 1.15 Add `StuartLandauElement`, `VanDerPolElement`, and `LogisticMapElement`; verify each against its reference trajectory (limit-cycle radius, relaxation period, and the logistic map's period doubling at `r = 3`).
- [ ] 1.16 Export every new class from its package, and verify the field-term test and `tests/test_types.py` pass with every new name.

## 2. Forces and potentials

- [ ] 2.1 Give `PotentialInteraction` and `PairPotentialInteraction` the `position`, `velocity`, and `mass` state names, defaulting to `x`, `v`, and `m`; verify a pair potential on states `r` and `u` and the existing potential tests.
- [ ] 2.2 Add `PairForceInteraction`, which takes an energy function of the distance and a cutoff, `DampingInteraction`, and `ConstantForceInteraction` moved from physics; verify forces against the gradient of the energy and damping against exponential decay.
- [ ] 2.3 Rebuild `physics/common/particles` on the core: Lennard-Jones, soft spheres, springs, gravity, and the harmonic potential on `PairForceInteraction` or `PotentialInteraction`, the thermostat on `OrnsteinUhlenbeckInteraction`, with their LAMMPS translations; verify the Lennard-Jones minimum at `2 ** (1 / 6)`, the thermostat's equilibrium temperature within 5 percent, and the LAMMPS agreement tests.

## 3. Matching, feedback, and learning in the core

- [ ] 3.1 Move the market protocols to the core as `BaseMatchingProtocol` and the seven protocols named in the proposal, `Order` to the private `_Offer`, and `Trade` to `MatchSpec`; verify the moved protocol tests, including price-time priority and stable matching, under the new names.
- [ ] 3.2 Add `MatchingInteraction` with the `requesters` and `providers` roles, the `limit`, `quantity`, and `budget` state names, and `transfers`; port the double auction, labour, order book, and credit models and the ride-sharing dispatch to it; verify their tests pass, and the ride-sharing test with `StableMatchingProtocol`.
- [ ] 3.3 Move `InstitutionInteraction` to the core as `FeedbackInteraction` and port its seven users; verify their tests and a feedback test that changes a parameter only at rounds where an observable exceeds a target.
- [ ] 3.4 Rename `economics/common/learning` to `economics/common/forecasting`, keep only the forecasting heuristics, and port the users of `logit_choice` and `evolve` to `ChoiceInteraction` and `EvolutionInteraction`; verify their tests, and that `markets` and `institutions` no longer exist in `economics/common`.

## 4. Model rewrites

Each task rewrites the classes of one cluster in `inventory.md` to the core class: pure classes bind it and keep only what is their own, composite classes bind it for their part, and state holders become the core element. Each verifies the tests and doctests of every touched model package, with stochastic models whose tolerance fails compared over 20 seeds against their previous trajectories.

- [ ] 4.1 Rewrite cluster 1, random walks, including the four copied `_wander` helpers; verify the touched packages' tests, and that `grep -r "def _wander" src/skcomplex/models` finds nothing.
- [ ] 4.2 Rewrite cluster 2, reactions and compartment flows, and its state holders; verify the touched packages' tests.
- [ ] 4.3 Rewrite cluster 3, spreading; verify the touched packages' tests.
- [ ] 4.4 Rewrite cluster 4, coupling; verify the touched packages' tests.
- [ ] 4.5 Rewrite cluster 5, thresholds and cascades; verify the touched packages' tests.
- [ ] 4.6 Rewrite cluster 6, choice by a rule; verify the touched packages' tests.
- [ ] 4.7 Rewrite cluster 7, foraging and regrowth, including the meadow and commons duplicates; verify the touched packages' tests, and that neither `_meadow.py` nor `_commons.py` defines a grazing or pasture class.
- [ ] 4.8 Rewrite cluster 8, evolution; verify the touched packages' tests.
- [ ] 4.9 Rewrite cluster 9, Ornstein-Uhlenbeck and geometric noise; verify the touched packages' tests.
- [ ] 4.10 Rewrite cluster 10, imitation; verify the touched packages' tests.
- [ ] 4.11 Rewrite cluster 11, pair forces and external potentials, in biology, social, and physics; verify the touched packages' tests.
- [ ] 4.12 Rewrite cluster 12, class densities, to `DensityElement`; verify the touched packages' tests.
- [ ] 4.13 Rewrite cluster 13, growth and decay, and its state holders, replacing both `ExponentialGrowthInteraction` classes; verify the touched packages' tests.
- [ ] 4.14 Rewrite cluster 14, alignment; verify the touched packages' tests.
- [ ] 4.15 Replace the duplicated equations of section D with the core `StuartLandauElement`, `VanDerPolElement`, and `LogisticMapElement`, and move FitzHugh-Nagumo to `biology/common/excitable`; verify the touched packages' tests, and that no model module defines another class of these equations.

## 5. Group mechanisms on hyperedges

- [ ] 5.1 Check whether a run can change a network's hyperedges through the model view, as reshuffled groups need, and record the answer in design.md; verify with a test that redraws the hyperedges of a small network between rounds, or report the gap.
- [ ] 5.2 Port the Galam discussions, public-goods groups, Kolkata restaurants, minority game, and El Farol bar of section H to `HyperedgeInteraction` subclasses; verify their tests, including the Galam majority rule, and keep any model that needs reshuffling on `RuleLaw` only if 5.1 reported the gap.

## 6. Specs, docs, and integration

- [ ] 6.1 Update the user guide's sections on elements, interactions, matching, institutions, learning, and particles to the core classes, with one example per mechanism; verify `pytest docs/overview` passes.
- [ ] 6.2 Update the model inventory and the examples that import moved names; verify `grep` finds no import of `MarketInteraction`, `InstitutionInteraction`, `logit_choice`, `evolve`, or `HopfElement` in `src`, `tests`, or `docs`.
- [ ] 6.3 Run the checks session (ruff, black, lint-imports, mypy, bandit, interrogate, pydoclint, deptry) and the full suite with doctests, and verify only the known timing test fails.
- [ ] 6.4 Merge every agent branch into `development` as it reports and delete its worktree, and verify `git worktree list` and `git branch` list none of them.

## Workflow follow-up

- Archive the change with `openspec archive general-elements-and-interactions -y` once merged.
