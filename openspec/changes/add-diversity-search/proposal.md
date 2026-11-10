## Why

Artificial-life research has turned from running one system to searching the space of its parameters for behaviour
that is diverse, novel, or open-ended. Curiosity-driven goal exploration finds moving, self-maintaining creatures in
Lenia (2402.10236). Searches over Lenia, Boids, particle life, and Life-like automata, scored in the embedding of a
pretrained model, find target behaviours, open-ended runs, and atlases of diverse runs (2412.17799). Hughes et al.
define open-endedness as a stream of artefacts that is both novel and learnable to an observer (2406.04268). The library
has `sweep`, `ensemble`, and `calibrate`, which cover a grid, replicates, and a fit to data, but no search that fills a
space of behaviours, and no measure of novelty or learnability of a run.

## What Changes

- New experiment `diversity_search(system, parameters, *, descriptors, objective, method, ...)` in
  `skcomplex.core.experiments`. It searches the bounds of the parameters for runs whose descriptors, functions of a
  trajectory, cover a space of behaviours. The methods are `'map-elites'`, `'cma-me'`, and `'novelty'` on pyribs, and
  `'goal-exploration'`, the intrinsically motivated goal exploration of 2402.10236. It returns a frozen
  `DiversitySearchResult` with the archive of parameters, descriptors, and objectives, the coverage, the quality-diversity
  score, the nearest-neighbour diversity of ASAL, and the history over iterations.
- New analysis `open_endedness(source, *, kind, embedding, observer, ...)` in `skcomplex.core.analysis`. The kind
  `'historical-novelty'` gives the novelty of every frame against its closest earlier frame in an embedding, and the
  score of 2412.17799. The kind `'observer'` fits an observer to the artefacts up to every time and gives the loss
  matrix `l(t, T)`, the novelty and learnability tests of 2406.04268, and the verdict.
- `embedding` takes any function from a frame to a vector, so a pretrained vision model can be plugged in by the user.
- New system `ParticleLifeSystem` in `computer_science/automata`, particles of several species that attract or repel
  one another through an asymmetric matrix, one of the substrates of 2412.17799.
- New runtime dependency `ribs` (pyribs).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `experiments`: a diversity search over the parameters of a system.
- `complexity-measures`: the open-endedness of a run by historical novelty and by an observer.
- `model-library`: the particle life system.

## Impact

- Core: a new module `core/experiments/_diversity_search.py` and a new module `core/analysis/_open_endedness.py`, with
  their results, exported and tagged. No change to the engine.
- Systems: a new module in `skcomplex.systems.computer_science.automata`.
- Dependencies: `ribs` 0.12.1, released 2026-11-07. scikit-learn, already a runtime dependency, gives the observers and
  the nearest neighbours.
- Docs: the experiments and analysis guides, a gallery example of particle life searched by MAP-Elites, a gallery
  example of goal exploration on Lenia, and the inventory with the primitive `diversity-search`.
- Independent of the other changes. `add-torch-surrogates` makes torch a runtime dependency, after which a later change
  can add a CLIP embedding beside the user function.
