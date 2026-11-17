## 1. Dependency

- [ ] 1.1 Add `ribs>=0.12.1` to the runtime dependencies in `pyproject.toml` and the Project Profile, and verify
  `pdm install` resolves and `import ribs` works

## 2. Diversity search

- [ ] 2.1 Add `diversity_search` and `DiversitySearchResult` in `src/skcomplex/core/experiments/_diversity_search.py`
  on the runner of `sweep`, export them, and verify the unknown-method and same-seed scenarios in
  `tests/core/experiments/test_diversity_search.py`
- [ ] 2.2 Add `'map-elites'` and `'cma-me'` on pyribs grid archives, and verify the map-of-a-known-function scenario
- [ ] 2.3 Add `'novelty'` on the proximity archive, and verify the novelty-without-an-objective scenario
- [ ] 2.4 Add `'goal-exploration'` with nearest neighbours of scikit-learn, and verify the goals-reached scenario
- [ ] 2.5 Accept shaped parameters by their entries, and verify a search over a two by two matrix parameter
- [ ] 2.6 Once `add-learning-foundation` is applied, accept the weights of a learned function or a policy as a searched
  parameter, and verify the diverse-cart-pole-policies scenario, marked slow
- [ ] 2.7 Write the docstring with a runnable example and a section of `docs/overview/user_guide/run_experiments.md`,
  and verify both run

## 3. Open-endedness

- [ ] 3.1 Add `open_endedness` and `OpenEndednessResult` in `src/skcomplex/core/analysis/_open_endedness.py` with
  `'historical-novelty'`, and verify the fixed-point and orthogonal-frames scenarios in
  `tests/core/analysis/test_open_endedness.py`
- [ ] 3.2 Add `'observer'` with the loss matrix and the two tests, and verify the noise and drifting-stream scenarios
- [ ] 3.3 Export the function, declare its analysis tags, and document it in `docs/overview/user_guide/analyse.md`
  with a runnable example

## 4. Particle life

- [ ] 4.1 Add `ParticleLifeSystem` in `src/skcomplex/systems/computer_science/automata/_particle_life.py`, export it,
  and verify the no-interaction and chasing-species scenarios in `tests/systems/computer_science/automata/`
- [ ] 4.2 Add the inventory entry `particle_life` with the source 2412.17799 and the primitive `diversity-search` with
  `change = "add-diversity-search"`, and verify the inventory check passes
- [ ] 4.3 Add the gallery example `docs/examples/computer_science/automata/plot_particle_life.py`, which maps particle
  life by MAP-Elites over its matrix with two descriptors and scores its runs by historical novelty, and verify the
  strict gallery build runs it
- [ ] 4.4 Add the gallery example `docs/examples/computer_science/automata/plot_lenia_goal_exploration.py`, which
  explores the growth parameters of `LeniaSystem` by goal exploration over the mass and the speed of the pattern, and
  verify it runs

## 5. Integration

- [ ] 5.1 Commit the work as conventional `feat` commits and verify `nox -s changelog` lists them
- [ ] 5.2 Run `nox` formatting, checks, tests of the touched modules, and the strict docs build, and fix every finding
