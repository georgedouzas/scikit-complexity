# Proposal

## Why

Field laws read their own site and, through `laplacian`, `gradient`, `divergence`, `advection`, `partial`, and
`upwind`, their nearest neighbours. The one global operator, `inverse_laplacian`, solves a periodic Poisson equation on
spectral fields only. Models need more than that:
- the Amari neural field integrates the firing rate over the whole domain against a lateral-inhibition kernel
  (2112.12147 §1.3.1);
- the quasi-geostrophic double gyre inverts potential vorticity into a streamfunction through a modified Helmholtz
  equation, `(laplacian - 1/Rd²) psi = q - beta y`, in a closed basin with walls (1910.00583 §III.6.1);
- the active viscoelastic gel solves its force balance by inverting `gamma - eta d²/dx²` at every step
  (2306.08868 VI);
- the gas-kinetic-based traffic model reads density and velocity at an advanced position, a look-ahead distance that
  depends on the local velocity (cond-mat/0012229 §III.E.4);
- the Terman-Wang oscillators of `legion` and the exchanging cells of `nanotube_gap_junction_exchange` sum a function of
  both ends over the neighbours of a field, which `CouplingInteraction` cannot lower on a `FieldElement`, so both
  elements keep their own laws and rewrite the sum by hand with `laplacian` and a degree term.

These are general field operators, convolution, shifted reads, and screened Poisson inversion, so they belong in the
core, built on SciPy, which is already a runtime dependency. The current core already gives `FieldElement` with its
discretizations and boundaries, `FieldLowering` with its neighbour tables and the global operators of
`FieldOperatorKernel`, and the `law-space` issue of the compiler, so this change builds on them and adds no element,
law kind, or dependency.

## What Changes

- Add `convolve(expression, kernel, radius=None)` and the displacement symbols `displacement(axis)` to
  `skcomplex.core.laws.utils`. The kernel is a symbolic function of the displacement and of parameters, sampled on the
  lattice and applied by FFT, circular on periodic lattices and truncated at fixed or zero-flux edges.
- Add `shifted(expression, displacement)`, which reads a field expression at a physical displacement per axis. A
  displacement of whole sites reads the neighbour tables, and any other displacement, including one that depends on the
  states, interpolates and follows the field's boundary conditions.
- Add `inverse_helmholtz(expression, screening)`, the solution of `(laplacian - screening²) psi = expression`, on
  spectral fields, finite-difference lattices with periodic, zero-flux, or fixed edges, networks, and finite-element
  meshes. `inverse_laplacian` becomes its case with zero screening, so it also works outside spectral fields.
- Let the stencil operators read the value of a global operator at neighbouring sites, so `gradient` and `advection`
  apply to an inverted streamfunction on a finite-difference field.
- Lower `CouplingInteraction` on a `FieldElement`: its sum over neighbours of a function of both ends becomes a sum of
  whole-site shifted reads on a lattice, a gather over the neighbour table on a network, and `laplacian` when the
  function is a constant times the difference of the ends.
- Report misuse as issues of the compiler's report, with the law's path and a fix.
- Add the Amari neural field, the quasi-geostrophic double gyre, and the gas-kinetic-based traffic model, implement the
  blocked active viscoelastic gel, and rebuild the couplings of `legion` and `nanotube_gap_junction_exchange` on
  `CouplingInteraction`.

## Capabilities

### New Capabilities

- `nonlocal-operators`: convolution kernels, shifted and look-ahead reads, modified-Helmholtz inversion, their
  composition with the local operators, and couplings lowered on fields.

### Modified Capabilities

- `model-library`: the nonlocal field models and the field couplings of the Terman-Wang and exchanging-cell models.

## Impact

- `skcomplex.core.laws.utils` gains `convolve`, `displacement`, `shifted`, and `inverse_helmholtz`, and keeps
  `inverse_laplacian` as the zero-screening case.
- `skcomplex.engine.compile` extends `FieldLowering` with the new operators, the neighbour reads of global operators,
  and the lowering of couplings on fields, and the pass compiler gains the issues of their misuse.
  `skcomplex.engine.problems.FieldOperatorKernel` gains the displacement operand. The numerics come from
  `scipy.fft`, `scipy.signal`, `scipy.ndimage`, `scipy.sparse.linalg`, and the scikit-fem matrices on meshes.
- `skcomplex.models` gains `amari_neural_field` in `biology/neuroscience`, `double_gyre` in `earth/oceanography`,
  `gkt_traffic` in `engineering/traffic`, and `active_viscoelastic_gel` in `physics/active_matter`, and rebuilds
  `TermanWangElement` and `ExchangingCellElement` without their coupling terms.
