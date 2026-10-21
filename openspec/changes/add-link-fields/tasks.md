# Tasks

## 1. Cell complexes and cell states

- [ ] 1.1 Add `CellComplex` for lattices with every boundary and for networks with given or minimum-cycle-basis faces,
  with SciPy coboundary matrices and NetworkX incidence, and verify the periodic square lattice, faces of a network,
  and invalid face scenarios
- [ ] 1.2 Add the `degree` keyword of `state` and `LinkField` with block storage and trajectory reshaping, and verify the
  potential and field strength and the values of the wrong shape scenarios
- [ ] 1.3 Add the named errors of complexes, degrees, and schedules to the errors module of `skcomplex.core.base`, and
  verify each error message names the value to fix

## 2. Exterior calculus

- [ ] 2.1 Add `exterior_derivative`, `codifferential`, `hodge_star`, and `curl`, and the dispatch of `gradient`,
  `divergence`, and `laplacian` on cell states, and verify the degree mismatch scenario and that every site-based
  model's tests still pass
- [ ] 2.2 Lower the operators to sparse matrices on NumPy and gather loops on Numba, and verify the curl of a gradient
  and the divergence of a curl scenarios on both backends

## 3. Gauge groups and holonomies

- [ ] 3.1 Add `GaugeGroup` with U(1), Z_N, and SU(2) products, inverses, traces, and projections, and verify the angles
  wrap scenario
- [ ] 3.2 Add `holonomy` and `staples` with their index tables on both backends, and verify the abelian plaquette and
  non-abelian order scenarios

## 4. Link Monte Carlo

- [ ] 4.1 Add link energies and `WilsonAction`, the `group` and `overrelax` moves, and `CompiledLinkMonteCarlo` with
  Metropolis acceptance, and verify the reproducible sampling scenario
- [ ] 4.2 Add the exact heat-bath draws, von Mises for U(1) through NumPy's `Generator`, categorical for Z_N, and
  Kennedy-Pendleton with the Creutz fallback for SU(2), and verify the two-dimensional U(1) and SU(2) scenarios and the
  SU(2) stays unitary scenario
- [ ] 4.3 Add the colored schedule for lattice links and for network links through NetworkX greedy coloring, and verify
  the schedule on an odd lattice scenario and that no two links of one class share a plaquette

## 5. Analysis, real-time fields, and visualization

- [ ] 5.1 Add `gauge_transform`, `wilson_loop`, `polyakov_loop`, `topological_charge`, `gauss_residual`, and the mean
  plaquette observable, and verify the invariance of loops and integer charge scenarios
- [ ] 5.2 Run rate laws on cell states with the existing ODE steppers, and verify the Gauss law and lattice dispersion
  scenarios
- [ ] 5.3 Add `hodge_decomposition` over SciPy's `lsqr`, and verify the consistent exchange rates and flow around a
  torus scenarios
- [ ] 5.4 Add `LinksVisual` under the kind `'links'` with its default blueprint, and verify the plaquette image scenario
- [ ] 5.5 Add `CellComplex.clique_complex` over NetworkX clique enumeration, and verify the complete-graph scenario
- [ ] 5.6 Add `frustration`, `betti_numbers` over sparse boundary ranks, and the topological charge over closed surfaces of
  a complex, and verify the flat and frustrated connections and torus scenarios

## 6. Library

- [ ] 6.1 Add `lattice_gauge_fields` to `skcomplex.models.physics.waves` as a registered factory with Monte Carlo and
  Hamiltonian dynamics, and verify the plaquette against the exact result, Maxwell dynamics conserve the Gauss law, and
  gauge-invariant run scenarios and the conformance suite
- [ ] 6.2 Move the inventory entry of `lattice_gauge_fields` from out of scope to pending this change, then to
  implemented, and verify the inventory check passes
- [ ] 6.3 Add `docs/examples/physics/waves/plot_lattice_gauge_fields.py` with its Rerun blueprint, caption, and the
  mean plaquette against `I_1(β) / I_0(β)` figure, and verify the gallery page scenario and the recording budget
- [ ] 6.4 Add `gauge_network_agents` to `skcomplex.models.social.opinion_dynamics`, verify the frustration scenario and
  the conformance suite, and add its curated gallery example with a review row

## 7. Documentation and gate

- [ ] 7.1 Document cell complexes, cell states, exterior calculus, gauge groups, link Monte Carlo, and the Hodge
  decomposition in the user guide, with the library survey's outcome, and verify the documentation test passes
- [ ] 7.2 Add a changelog entry for the new public names, and verify the changelog session passes
- [ ] 7.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
