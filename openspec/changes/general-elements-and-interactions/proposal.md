# Proposal

## Why

The core gives models containers (`Element`, `GroupElement`, `FieldElement`, `LatticeElement`, `ParticleElement`) and
binding patterns (`Interaction`, `PairInteraction`, `NeighborInteraction`, `HyperedgeInteraction`), but no mechanism a
model can use as it is. So the models define about 500 element classes and 154 interaction classes, and the same
mechanisms come back across fields:
- random walks of agents in 27 classes across 5 fields, with one private `_wander` helper copied four times;
- mass-action reactions and compartment flows in 18 classes across 6 fields;
- coupling of states (Kuramoto, consensus, alignment, power lines) in 18 classes across 5 fields.

Some are copied outright: `CowElement`, `PastureElement`, and `GrazingInteraction` exist in both biology and
economics, and `StuartLandauElement` twice in physics. Two specs cause much of this:
- "Specific realizations live in the model library" makes a field that needs another field's shared code define its
  own copy.
- "The core is application agnostic" allows a core primitive only one neutral default.

So the logit choice of economics, the potentials of physics, and the market matching protocols are rewritten
wherever another field needs them.

## What Changes

- **General mechanisms in the core.** The core gains concrete, configurable elements and interactions, named by their
  mathematics, never by a field:

  | Mechanism | Core class |
  | --- | --- |
  | random walk on cells, nodes, or continuous space, with neighbour, Gaussian, or Lévy steps | `RandomWalkInteraction` |
  | mass-action reaction or compartment flow, as rates, jumps, or Langevin noise | `ReactionInteraction` |
  | spreading of a state to neighbours, with recovery and loss of immunity | `SpreadingInteraction` |
  | coupling of a state to others through a function, with lag, delay, or mean field | `CouplingInteraction` |
  | threshold crossing with reset or redistribution | `ThresholdInteraction` |
  | choice among options by the logit (heat-bath), Metropolis, or Fermi rule | `ChoiceInteraction` |
  | imitation of a neighbour's discrete state, at random, by majority, or of the best | `ImitationInteraction` |
  | foraging of a resource with an energy budget, and regrowth of the resource | `ForagingInteraction`, `RegrowthInteraction` |
  | Ornstein-Uhlenbeck and geometric Brownian noise | `OrnsteinUhlenbeckInteraction` |
  | growth and decay, exponential, logistic, Gompertz, power, or von Bertalanffy, with harvest and delay | `GrowthInteraction` |
  | alignment of headings or velocities within a radius | `AlignmentInteraction` |
  | evolution of strategies by selection, crossover, and mutation | `EvolutionInteraction` |
  | pair forces from a user-given potential of the distance | `PairForceInteraction` |
  | linear damping and a constant force | `DampingInteraction`, `ConstantForceInteraction` |
  | densities over activity classes, for encounter laws | `DensityElement` |
  | compartments whose states are counts or fractions | `CompartmentElement` |

- **BREAKING: matching moves to the core with neutral roles.** `MarketInteraction` becomes `MatchingInteraction`, whose
  roles are `requesters` and `providers` and whose states are `limit`, `quantity`, and `budget`, and the protocols
  are named for their mechanism:
  - `RandomSearchProtocol`
  - `PartnerSwitchProtocol`
  - `QueueProtocol`
  - `PriceAdjustmentProtocol`
  - `PriceTimePriorityProtocol`
  - `UniformPriceProtocol`
  - `StableMatchingProtocol`

  The economics models bind them to bids, asks, money, and goods.
- **BREAKING: feedback and learning move to the core.** `InstitutionInteraction` becomes `FeedbackInteraction`, an
  actor that reads observables and sets parameters once per round. `logit_choice` and `evolve` become
  `ChoiceInteraction` and `EvolutionInteraction`. The forecasting heuristics of economics stay in economics.
- **Potentials take state names.** `PotentialInteraction` and `PairPotentialInteraction` read the position, velocity,
  and mass from states the user names, defaulting to `x`, `v`, and `m`.
- **Physics realizations are built on the core.** Lennard-Jones, soft spheres, springs, gravity, the harmonic
  potential, and the Langevin thermostat stay in `physics/common/particles` as realizations built on
  `PairForceInteraction`, `PotentialInteraction`, and `OrnsteinUhlenbeckInteraction`, with their LAMMPS translations.
- **Group mechanisms run over hyperedges.** Galam discussion groups, public-goods groups, the Kolkata restaurants, and
  the minority game become `HyperedgeInteraction` subclasses over networks with hyperedges, so the primitive has
  model users.
- **Every implementing model is rewritten.** Every model class whose mechanism the core now provides uses the core
  class and keeps only what is its own, so the models become examples of the core. That is about 200 classes,
  listed in `inventory.md`.
- **Duplicates are removed.**
  - One `StuartLandauElement` in `physics/common` replaces the two physics copies. Named equations stay realizations
    of their fields, so mathematics and earth keep their own Hopf, Van der Pol, and logistic-map classes.
  - One FitzHugh-Nagumo element in a new `biology/common/excitable` family replaces the physiology and neuroscience
    copies.
  - The meadow and commons models share the core foraging and regrowth.
  - `GrowthInteraction` replaces both `ExponentialGrowthInteraction` classes.

## Capabilities

### New Capabilities

- `general-mechanisms`: the field-neutral elements and interactions the core provides for mechanisms that models
  across fields share, each configurable by states, parameters, functions, and named mathematical variants.

### Modified Capabilities

- `system-modeling`:
  - "The core is application agnostic" now allows a core mechanism standard variants named by their mathematics.
  - Matching protocols and choice rules count as general mechanisms, not as specific realizations.
- `model-library`:
  - "Specific realizations live in the model library": a mechanism that models of more than one field need lives in
    the core instead of being copied. A field's `common` package holds only realizations named by the field.
  - "Pair potentials for particle groups" and "Langevin thermostat" are realizations built on the core interactions.
  - "Economic and social agent models" use the core's matching, feedback, choice, and evolution.

## Impact

- **Code:**
  - `skcomplex.core.elements` and `skcomplex.core.interactions` gain about 20 classes.
  - `skcomplex.models.economics.common` loses `markets` and `institutions`, and keeps only the forecasting heuristics
    in `learning`.
  - `skcomplex.models.physics.common.particles` is rebuilt on the core.
  - About 200 model classes across all nine fields are rewritten.
- **API:** breaking for imports of `MarketInteraction`, the market protocols, `InstitutionInteraction`,
  `logit_choice`, `evolve`, the two physics `StuartLandauElement`, `ExponentialGrowthInteraction`, and the
  duplicated meadow/commons classes. Model factories keep their names and parameters, and their trajectories stay the
  same within the tolerance of their tests.
- **Specs:** the field-term test of the core keeps guarding every new name.
- **Overlap with pending changes:**
  - `add-higher-order-interactions` extends laws over hyperedges, and this change adds the first model users of
    `HyperedgeInteraction`. Whichever lands second rebases on the other.
  - `add-learning-dynamics` adds gradient learning, which is distinct from the selection-based
    `EvolutionInteraction`.
