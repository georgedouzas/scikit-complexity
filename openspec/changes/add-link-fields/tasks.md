# Tasks

## 1. Cell complexes and edge groups

- [ ] 1.1 Add `CellComplex` to `skcomplex.core.space` as a network space for lattices with every boundary and for
  networks with given, hyperedge, or minimum-cycle-basis faces, with SciPy coboundary matrices and NetworkX incidence,
  and verify the periodic square lattice, faces of a network, and invalid face scenarios
- [ ] 1.2 Support values and trajectory reshaping of groups on the edges and hyperedges of a cell complex, and verify
  the potential and field strength and the values of the wrong shape scenarios
- [ ] 1.3 Add the named errors of complexes to the errors module of `skcomplex.core.base`, and verify each error message
  names the value to fix

## 2. Exterior calculus

- [ ] 2.1 Add `exterior_derivative`, `codifferential`, `hodge_star`, and `curl` to `skcomplex.core.laws`, and the
  dispatch of `gradient`, `divergence`, and `laplacian` on edge and hyperedge groups, and verify that every site-based
  model's tests still pass
- [ ] 2.2 Lower the operators to sparse matrices on NumPy and gather loops on Numba in `skcomplex.core.backends`, and
  verify the curl of a gradient and the divergence of a curl scenarios on both backends

## 3. Compact groups and holonomies

- [ ] 3.1 Add `DomainKind.COMPACT_GROUP` and the domain `CompactGroup` with U(1), Z_N, and SU(2) products, inverses,
  traces, projections, and heat-bath draws, and verify the angles wrap scenario
- [ ] 3.2 Add `holonomy` and `staples` with their index tables on both backends, and verify the abelian plaquette and
  non-abelian order scenarios

## 4. Link Monte Carlo

- [ ] 4.1 Extend `Sample` to edge groups with link energies, the `group` and `overrelax` moves, and
  `CompiledLinkMonteCarlo` with Metropolis acceptance, and verify the reproducible sampling scenario
- [ ] 4.2 Add the exact heat-bath draws, von Mises for U(1) through NumPy's `Generator`, categorical for Z_N, and
  Kennedy-Pendleton with the Creutz fallback for SU(2), and verify the two-dimensional U(1) and SU(2) scenarios and the
  SU(2) stays unitary scenario
- [ ] 4.3 Add the colored schedule for lattice links and for network links through NetworkX greedy coloring, and verify
  that no two links of one class share a plaquette

## 5. Declarations and validation

- [ ] 5.1 Declare the requirements of `CellComplex`, `CompactGroup`, the exterior calculus, and holonomies, and the
  support of the Monte Carlo solver and the NumPy and Numba backends for compact group domains and edge groups, and
  verify the declaration test passes
- [ ] 5.2 Add the validation issues `law-degree` and `sample-schedule`, and verify the degree mismatch, rate on a
  compact group, schedule on an odd lattice, and several link issues at once scenarios

## 6. Analysis, real-time fields, and visualization

- [ ] 6.1 Run rate laws on edge groups under continuous time with the existing ODE steppers, and verify the Gauss law
  and lattice dispersion scenarios
- [ ] 6.2 Add `hodge_decomposition` over SciPy's `lsqr` to `skcomplex.core.analysis`, and verify the consistent exchange
  rates and flow around a torus scenarios
- [ ] 6.3 Add `LinksVisual` under the kind `'links'`, and verify the plaquette image scenario
- [ ] 6.4 Add `CellComplex.clique_complex` over NetworkX clique enumeration, and verify the complete-graph scenario
- [ ] 6.5 Add `frustration` and `betti_numbers` over sparse boundary ranks to `skcomplex.core.analysis`, and verify the
  flat and frustrated connections and torus scenarios

## 7. Library

- [ ] 7.1 Add the Wilson action, `gauge_transform`, `wilson_loop`, `polyakov_loop`, `topological_charge`, and
  `gauss_residual` to `skcomplex.models.physics.waves`, and verify the invariance of loops and integer charge scenarios
- [ ] 7.2 Add `lattice_gauge_fields` to `skcomplex.models.physics.waves` as a registered factory with Monte Carlo and
  Hamiltonian dynamics and the standard observables of its catalog entry, and verify the plaquette against the exact
  result, Maxwell dynamics conserve the Gauss law, and gauge-invariant run scenarios and the conformance suite
- [ ] 7.3 Move the inventory entry of `lattice_gauge_fields` from out of scope to pending this change, then to
  implemented with its system tags, and verify the inventory check passes
- [ ] 7.4 Add `docs/examples/physics/waves/plot_lattice_gauge_fields.py` with its Rerun blueprint, caption, and the
  mean plaquette against `I_1(β) / I_0(β)` figure, and verify the gallery page scenario and the recording budget
- [ ] 7.5 Add `gauge_network_agents` to `skcomplex.models.social.opinion_dynamics` with its own alignment energy,
  verify the frustration scenario and the conformance suite, and add its curated gallery example with a review row

## 8. Documentation and gate

- [ ] 8.1 Document cell complexes, edge and hyperedge groups, exterior calculus, compact groups, link Monte Carlo, and
  the Hodge decomposition in the user guide, with the library survey's outcome, and verify the documentation test passes
- [ ] 8.2 Verify that the field-term test and the import-graph test of the core still pass with the new names
- [ ] 8.3 Add a changelog entry for the new public names, and verify the changelog session passes
- [ ] 8.4 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
