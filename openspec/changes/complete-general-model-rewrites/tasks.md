# Tasks

## 1. Core capabilities

- [ ] 1.1 Couple a set of separate elements and a hub element to a group by any reduction in `CouplingInteraction`, and combine a map with an update-form coupling; rebuild mutual inhibition, global inhibition, the transmission lines, and the contour cell; verify each against its previous trajectory.
- [x] 1.2 Accept parametric stoichiometric coefficients and per-member reactions in the agent form in `ReactionInteraction`; rebuild the Oregonator and split the rats' nervousness into two reactions; verify the limit cycle and the rats' tests.
- [ ] 1.3 Add transfers along lattice and network edges in both directions at parametric rates with one draw per edge, and composite transfers that hold parameters; rebuild zero-range, exclusion, reservoir, and parallel-update TASEP processes, the beer game, and the stock-flow consistent circuit; verify the currents and the reporters.
- [ ] 1.4 Add phases of N visits drawn with replacement to the agent forms, with reinforcement of the chosen option only, and births whose probability reads the composition; rebuild the perishables market and tangled nature; verify each over 20 seeds.
- [ ] 1.5 Add the agent form of `CouplingInteraction` over several radii with priorities and clamps; rebuild boids and the Couzin model; verify each over 20 seeds.
- [ ] 1.6 Let agent rules read observable history; remove the reporter-only feedback interactions of El Farol and the other models that only report; verify their tests.
- [x] 1.7 Let experiments replicate systems that hold interactions; rebuild the named maps and flows of the orbit diagrams on `MapInteraction` and `RateInteraction`; verify the orbit-diagram tests and docs.

## 2. Rebuilds the core already allows

- [ ] 2.1 Rebuild the meadow, ants, trails, angiogenesis, agent predation, and stigmergic plants on the member schedule, cell transfers, site deposits, and life cycles; verify each over 20 seeds.
- [x] 2.2 Rebuild the degree-class mean fields on callable coupling weights; verify against their previous trajectories.
- [x] 2.3 Find why the commons herders differ after the life-cycle rebuild, and rebuild them or record why they stay; verify over 20 seeds.
- [ ] 2.4 Decompose the large single-element ODE models of oncology, neuroscience, physiology, and systems biology into rate and reaction interactions, with `biology/common/pharmacokinetics` and `biology/common/kinetics` where several packages share them; verify each against its previous trajectory.
- [ ] 2.5 Rebuild the remaining social classes (voter diffusion, coevolution, standardization, crowding, walls, recruitment, the spatial games, the organizations package, and segregation blocks) on the general classes; verify their tests.
- [x] 2.6 Rebuild the transitions of the epidemiology `PersonElement` on `TransitionInteraction` with the incubation timing of the general class, and state the convention in its docstring; verify its tests and the match with its compartment model.
- [ ] 2.7 Make `plot_bouchaud_mezard_wealth`, `plot_credit_networks`, and `plot_multiplicative_process` run within the gallery budget of 60 s, and restore the uniform band of `plot_unbounded_alphabet_automaton`; verify the docs build.

## 3. Conventions and integration

- [ ] 3.1 Split every model module that holds several classes into one class per private module, re-exported unchanged; verify the catalog test and every model test.
- [x] 3.2 Update `uv` to 0.13.4 or later in `pdm.lock` for PYSEC-2026-4321; verify pip-audit reports nothing.
- [ ] 3.3 Run the checks session and the full suite with doctests; verify only the known timing tests fail.
