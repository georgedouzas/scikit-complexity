## 1. Graph rewriting

- [x] 1.1 Delete `HostGraphElement`, `GraphProductionLaw`, the graph kernel, the graph structure spec, and the graph drawing, with their tests and the user guide section.
- [x] 1.2 Rename the model elements that reuse core names: `HabitatPatchElement`, `SugarPatchElement`, `GrowingWordElement`.

## 2. The compiler setting

- [x] 2.1 Remove `claims_`, `_claims`, and the claim loop. Add the `compiler` setting to `compile_system` and `Simulation`. Add `_get_solver(problem)` and `solver_` to `BaseCompiler`, and implement them in `PassCompiler` and `QuantumCompiler`. Replace the form list of `Simulation._check_dt` by a `BaseSolver._is_stepwise()` hook. Verify with `pytest tests/engine`.
- [x] 2.2 Make quantum systems name their compiler: `PassCompiler` refuses them with an error that names `compiler=QuantumCompiler()`. Add a `compiler` to the model catalog entries, and update the quantum models, analyses, tests, gallery, and user guide. Verify with the quantum tests and the user guide.

## 3. Rewriting in the models library

- [ ] 3.1 Move word and patch rewriting into `skcomplex.models.mathematics.fractals`:
  - the elements, laws, and word utilities;
  - the module, prototile, affine, and geometry specs, and the structures;
  - the kernels, compiler, solver, problem spec, and visual;
  - `RewritingError`.

  Remove `FormKind.REWRITING` and the rewriting laws from the backends and the co-simulation solver. Add the test that the core, engine, specs, and types name none of it. Verify with lint-imports, the fractal tests, and the full test session.
- [ ] 3.2 Update the gallery examples and the user guide's rewriting section, and run the curated gallery.
