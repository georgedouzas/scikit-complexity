## 1. Neighbour hyperedge reduction

- [x] 1.1 Add `over` to `hyperedges` and the field `over` to `HyperedgeReductionSpec`, with a `ValidationError` for a missing size, and verify the doctest and the missing-size scenario with tests in `tests/core/components/laws/`
- [x] 1.2 Add the set rows to `NeighborhoodKernel` on NumPy and dispatch hyperedge reductions with `over` to it in `_reduction_lowering.py`, and verify the three-body loop, four-body count, and too-few-neighbours scenarios with tests on NumPy
- [x] 1.3 Add the Numba loop over neighbour sets in `_numba_neighbours.py` and its printing in `_numba_printer.py`, and verify the three-body loop scenario on Numba within 1e-12
- [x] 1.4 Report the `law-space` issue for a neighbourhood outside a continuous space and an issue on the JAX backend, and verify both with tests
- [x] 1.5 Add a test that a member whose neighbour leaves the radius has a zero 3-body term from the next step

## 2. Coupling over neighbour hyperedges

- [x] 2.1 Pass `neighborhood` of `CouplingInteraction` to `hyperedges` in the mode `'hyperedges'` and update its docstring, and verify the flanking-alignment scenario with a test
- [x] 2.2 Document neighbour hyperedges beside the hyperedge section of `docs/overview/user_guide/build_a_custom_system/networks.md`, with a runnable example and the cost, and verify the page builds

## 3. Many-body alignment swarm

- [x] 3.1 Add `ManyBodyAlignmentSwarmSystem` in a new module of `src/skcomplex/systems/biology/collective_behaviour/` with `size`, `dimension`, `alpha`, `beta`, the Morse constants, `gamma`, `l_v`, `l_3`, `kappa_v`, `kappa_4`, `R`, and `random_state`, exported from the subfield, and verify its doctest
- [x] 3.2 Add tests in `tests/systems/biology/collective_behaviour/` for the pairwise collective, the body-order terms against a direct evaluation of Eq. 2, and the bounded motion with defaults
- [x] 3.3 Add the primitive `neighbour-hyperedges` and the entry `many_body_alignment_swarm` under `biology.collective_behaviour` with source `2611.08314`, and verify the inventory check passes
- [x] 3.4 Add the gallery example `docs/examples/biology/collective_behaviour/plot_many_body_alignment_swarm.py` that compares the pairwise, 3-body, and 2+3+4-body collectives, and verify the gallery check passes
- [x] 3.5 Commit the work as conventional `feat` commits, and verify `nox -s changelog` lists them

## 4. Verification

- [x] 4.1 Run `nox -s checks`, the tests of the touched modules, and the strict docs build, and fix every finding
