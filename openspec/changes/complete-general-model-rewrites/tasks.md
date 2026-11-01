# Tasks

## 1. Core capabilities

- [x] 1.1 Couple a set of separate elements and a hub element to a group by any reduction in `CouplingInteraction`, and combine a map with an update-form coupling; rebuild mutual inhibition, global inhibition, the transmission lines, and the contour cell; verify each against its previous trajectory.
- [x] 1.2 Accept parametric stoichiometric coefficients and per-member reactions in the agent form in `ReactionInteraction`; rebuild the Oregonator and split the rats' nervousness into two reactions; verify the limit cycle and the rats' tests.
- [x] 1.3 Add transfers along lattice and network edges in both directions at parametric rates with one draw per edge, and composite transfers that hold parameters; rebuild zero-range, exclusion, reservoir, and parallel-update TASEP processes, the beer game, and the stock-flow consistent circuit; verify the currents and the reporters.
- [x] 1.4 Add phases of N visits drawn with replacement to the agent forms, with reinforcement of the chosen option only, and births whose probability reads the composition; rebuild the perishables market and tangled nature; verify each over 20 seeds.
- [x] 1.5 Add the agent form of `CouplingInteraction` over several radii with priorities and clamps; rebuild boids and the Couzin model; verify each over 20 seeds.
- [x] 1.6 Let agent rules read observable history; remove the reporter-only feedback interactions of El Farol and the other models that only report; verify their tests.
- [x] 1.7 Let experiments replicate systems that hold interactions; rebuild the named maps and flows of the orbit diagrams on `MapInteraction` and `RateInteraction`; verify the orbit-diagram tests and docs.

## 2. Rebuilds the core already allows

- [x] 2.1 Rebuild the meadow, ants, trails, angiogenesis, agent predation, and stigmergic plants on the member schedule, cell transfers, site deposits, and life cycles; verify each over 20 seeds.
- [x] 2.2 Rebuild the degree-class mean fields on callable coupling weights; verify against their previous trajectories.
- [x] 2.3 Find why the commons herders differ after the life-cycle rebuild, and rebuild them or record why they stay; verify over 20 seeds.
- [x] 2.4 Decompose the large single-element ODE models of oncology, neuroscience, physiology, and systems biology into rate and reaction interactions, with `biology/common/pharmacokinetics` and `biology/common/kinetics` where several packages share them; verify each against its previous trajectory.
- [x] 2.5 Rebuild the remaining social classes (voter diffusion, coevolution, standardization, crowding, walls, recruitment, the spatial games, the organizations package, and segregation blocks) on the general classes; verify their tests.
- [x] 2.6 Rebuild the transitions of the epidemiology `PersonElement` on `TransitionInteraction` with the incubation timing of the general class, and state the convention in its docstring; verify its tests and the match with its compartment model.
- [x] 2.7 Make `plot_bouchaud_mezard_wealth`, `plot_credit_networks`, and `plot_multiplicative_process` run within the gallery budget of 60 s, and restore the uniform band of `plot_unbounded_alphabet_automaton`; verify the docs build.
- [x] 2.8 Move the parameters of the economics transfer subclasses (circuit, supply link, retail demand, brewery) to their elements and build each remaining mechanism as its own `RateInteraction` or `MapInteraction` in the factory; verify the beer game and the stock-flow consistent economy are identical.
- [x] 2.9 Add the choice variant whose proposals include the current option, the agent form of `MapInteraction` on vector states, periodic schedules, and counts grouped by cell; rebuild standardization, segregation blocks, beliefs, the employee, obesity, and trafficking on them and on the visits activation; verify their tests, over 20 seeds where draws change.
- [x] 2.10 Let the agent forms of the general interactions resolve their roles through the subclass, so a thin subclass names its own roles in both forms; rebuild the meadow with grazing as a transfer subclass with roles `pasture` and `cows`; verify its unchanged tests and 20 seeds against the current model.

## 3. Conventions and integration

- [ ] 3.1 Split every model module that holds several classes into one class per private module, re-exported unchanged; verify the catalog test and every model test.
- [x] 3.2 Update `uv` to 0.13.4 or later in `pdm.lock` for PYSEC-2026-4321; verify pip-audit reports nothing.
- [ ] 3.3 Derive the random stream of `Simulation` from `SeedSequence(random_state).spawn(...)`, so dynamics never reuse the numbers a user or factory drew from `default_rng(random_state)` for initial states; update the tests whose expected values depend on the old stream; verify the Fermi regression test passes with one shared seed. In the same pass, replace the hand-built generators with library calls: `xgi.random_hypergraph` in `hypergraph_kuramoto`, `xgi.random_simplicial_complex` in `simplicial_contagion`, `reservoirpy.mat_gen.normal` for the echo-state weights, `scipy.stats.cauchy.ppf` for the three Lorentzian quantiles, and `networkx.gnp_random_graph` for the neuroscience random connections; let `MesaSolver` read phases from nested systems as the symbolic path does.
- [x] 3.4 Fix the periodic merging of `ClusterAnalysis` with `scipy.sparse.csgraph.connected_components` over every wrapped neighbour offset, so diagonal and corner contacts across the boundary join; add the regression test.
- [x] 3.5 Count motifs of `MotifAnalysis` with `igraph` (`Graph.motifs_randesu`), mapping isoclasses to the current names and keeping the networkx null models; add `igraph` as a dependency in `pyproject.toml` and the Project Profile.
- [ ] 3.6 Run the checks session and the full suite with doctests; verify only the known timing tests fail.
