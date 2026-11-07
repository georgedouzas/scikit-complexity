# Design

## Context

Laws of fields write their nonlocal operators as Sage functions of `skcomplex.core.components.laws.utils`, such as
`convolve`, `inverse_helmholtz`, and `reduce_axes`, and `FieldLowering` replaces every call by a derived value, a
`FieldOperatorKernel` whose `apply` maps the values of the operand to the values of the result. A phase space is a
lattice whose class axes have cell coordinates, which laws read through `coordinate(name)`, and finite-volume fields on
it run through `FiniteVolumeKernel`, which walls the ends of interval class axes. See proposal.md for the motivation.

## Goals / Non-Goals

**Goals:**

- One two-point integral operator over class axes, one cumulative integral along a lattice axis, and convolutions
  along the position axes of a phase space, all as derived values like the existing nonlocal operators.
- One simplex domain whose partial cells the finite-volume fluxes and the reductions respect.

**Non-Goals:**

- Two-point kernels over position axes. Position kernels of the displacement stay with `convolve`.
- Local stencils, spectral operators, and finite elements on a simplex. Only finite-volume fluxes, reductions, and
  integrals run there.
- Low-rank or fast summation of two-point kernels. The kernel is a dense matrix of the cells of the integrated axes.

## Decisions

### Operators are law utilities, not new classes

The existing nonlocal operators are Sage functions that the compiler lowers, named after the operation, `convolve`
and `inverse_helmholtz`. The new ones follow them: `integral_transform`, `cumulative_integral`, and the symbol
`integration_variable`. A new interaction class would duplicate `RateInteraction`, which already takes any rate.

### The kernel reads the coordinates of both points

The kernel of `integral_transform` is a Sage expression of `coordinate(name)`, the point a law updates, and
`integration_variable(name)`, the point integrated over. The lowering samples it on the outer grid of the cell centres
with the parameter values of the compilation, exactly as `convolve` samples its kernel of `displacement(axis)`, so the
two share one helper, `_sampled_kernel`. The coordinates of the kernel are excluded from the parameter substitution,
since the layout declares them as hidden parameters. The alternative, a Python callable of arrays, would bypass the
symbolic layer and the `kernel-state` check.

### Convolutions on a phase space act along positions

A convolution on a phase space today convolves along every axis, which no law uses, since class axes are integrated by
`reduce_axes` or `moment`. The kernel is now sampled on the position lattice and given unit length along the class
axes, so every class slice is convolved along the positions with the edges of the position lattice. Periodic
positions use the FFT along the position axes alone. This matches `inverse_helmholtz`, which already acts along the
position axes of a phase space.

### The simplex is a phase space with cut cells

`SimplexSpace` derives from `PhaseSpace` with the same arguments, so it is a lattice of the box `[0, 1]^d` of the free
frequencies, and adds two derived attributes. `volumes_` holds the fraction of every cell inside the simplex, and
`apertures_` the fraction of every face inside it, both from the cumulative distribution of the Irwin-Hall
distribution of `scipy.stats`, the volume of a unit cube cut by a hyperplane. In two dimensions this is the scheme of
Cooney 2024, appendix E.2: whole squares, half triangles along the slanted face, and closed faces beyond it.

`FiniteVolumeKernel` takes the volumes and apertures, multiplies every interface flux by its aperture, and divides
every cell's flux difference by its volume, with no rate in empty cells. Its `max_speed` divides by the smallest
positive volume, so `SSPRKSolver` steps stably. The derived values of `reduce_axes`, `integral_transform`, and
`cumulative_integral` multiply the values by the volumes. Other spaces give the volume one.

The alternatives were a new space of the simplex cells alone, which every reshape to the lattice shape would break, and
a mask without partial cells, which leaves a staircase that loses the total along the slanted face.

### Libraries

No library holds two-point integral operators of symbolic kernels on the fields of this core. SciPy provides the
Irwin-Hall distribution, the FFTs, and the windowed convolutions, and NumPy the cumulative sums. `scipy.stats.irwinhall`
appeared in SciPy 1.15, so the lower bound of `scipy` rises from 1.13 to 1.15, whose release of January 2025 is within
the age limit.

### Systems

The two systems of Cooney 2024 are factories in a new module of `social/cooperation`, on a one-position phase space with
one class axis `z` and on a one-position simplex space with the axes `x` and `y`. The forest of Grilletta et al. 2026
is a factory in a new module of `biology/ecology`, on a phase space of a position lattice with fixed zero edges and a
height axis. Its flux is the conservative `I g f`, so the shading term of Eq. 10 comes from the product rule, and
recruitment is a source in the lowest height cell. The kernel along the strip is the one-dimensional exponential,
because the printers have no modified Bessel function for the kernel of Eq. 36.

## Risks / Trade-offs

- [Dense kernel matrices grow with the square of the cells of the integrated axes] → The trimorphic system defaults to
  30 cells per axis, a matrix of 900 by 900, and the docstrings say the integral is a dense product.
- [Laws with source terms fill the empty cells of a simplex] → The guide states that densities start at zero there, and
  the systems are proportional to the density.
- [Partial cells use the cell-centre coordinate rather than the centroid] → The error is of the order of the cell
  size, as in the reference scheme, which averages over the cell.
