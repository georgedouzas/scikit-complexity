# Tasks

## 1. Cell complexes

- [ ] 1.1 Add `CellComplexSpace` to `skcomplex.core.spaces` for lattices with every boundary and for networks with
  given, hyperedge, minimum-cycle-basis, and clique faces, with `incidences_` and `cofaces_`, and verify the periodic
  square lattice, fixed edges, faces of a network, clique complex, and invalid face scenarios in
  `tests/core/spaces/test_cell_complex.py`
- [ ] 1.2 Verify the potential and field strength scenario for groups on the edges and hyperedges of a cell complex

## 2. Exterior calculus and coface reductions

- [ ] 2.1 Add `exterior_derivative`, `codifferential`, and `hodge_star` in a module `_cells` of
  `skcomplex.core.laws.utils`, with their lowering to incidence products on NumPy and gather loops on Numba, and verify
  the curl of a gradient and divergence of a curl scenarios on both backends
- [ ] 2.2 Add `CofaceNeighborhood` to `skcomplex.core.spaces` and its lowering in coface reductions, and verify the
  abelian plaquette scenario
- [ ] 2.3 Add the issue `law-degree` to `PassCompiler` and the `law-space` issue of a coface reduction outside a cell
  complex, and verify the several link issues at once scenario
- [ ] 2.4 Verify the Gauss law and lattice dispersion scenarios with rate laws on edge groups

## 3. Sampling edge groups

- [ ] 3.1 Let `SamplingInteraction` target edge groups of a cell complex, with the colored schedule over the graph of
  links that share a face, and verify the two-dimensional U(1), reproducible sampling, and colored schedule on an odd
  lattice scenarios

## 4. Analysis and visualization

- [ ] 4.1 Add `CellComplexAnalysis` with the kinds `'hodge'`, `'betti'`, and `'curvature'` to `skcomplex.analysis`, and
  verify the consistent exchange rates, flow around a torus, and flat and curved connections scenarios
- [ ] 4.2 Add `CellVisual` to `skcomplex.engine.viz`, and verify the plaquette image scenario with a recorded Rerun
  stream

## 5. Library

- [ ] 5.1 Add `WilsonSamplingInteraction`, `GaugeAnalysis`, and `lattice_gauge_fields` to
  `skcomplex.models.physics.waves`, and verify the plaquette against the exact result, gauge-invariant run, Maxwell
  dynamics, invariance of loops, and integer charge scenarios and the conformance suite
- [ ] 5.2 Add `topological_kuramoto` to `skcomplex.models.physics.synchronization`, and verify the link synchronization
  scenario and the conformance suite
- [ ] 5.3 Add `gauge_arbitrage_market` to `skcomplex.models.economics.finance`, and verify the conservation scenario and
  the conformance suite
- [ ] 5.4 Add `TransportedCouplingInteraction` and `gauge_network_agents` to
  `skcomplex.models.social.opinion_dynamics`, and verify the frustration scenario and the conformance suite
- [ ] 5.5 Mark the four entries implemented in `docs/overview/inventory.toml`, add their gallery examples, and verify
  the inventory check and the gallery coverage check pass

## 6. Documentation and gate

- [ ] 6.1 Document cell complexes, the exterior calculus, coface reductions, sampling of links, and
  `CellComplexAnalysis` in the user guide with runnable examples, and verify the documentation test passes
- [ ] 6.2 Add a changelog entry for the new public names, and verify the changelog session passes
- [ ] 6.3 Run `nox -s formatting`, `nox -s checks`, `nox -s docs`, and `nox -s tests` in that order, and verify all of
  them pass
