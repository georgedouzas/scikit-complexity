# Design

## Context

`add-correlated-noise` lets noise laws draw named, shared, and correlated Wiener processes, lowered to columns of a
noise matrix and drawn by the SDE steppers. Fields are site groups on a lattice whose laws read stencils of their
neighbours.

## Goals / Non-Goals

**Goals:**
- Conserved noise on finite-difference fields on lattices, exact to rounding in the total of the field.

**Non-Goals:**
- Conserved noise on meshes and networks, and multiplicative conserved noise whose amplitude depends on the field at
  the link; a later change can add them on the same link processes.

## Decisions

### Link processes

On a lattice of spacing `h` in `d` dimensions, the link between site `i` and `i + e_k` carries one Wiener process
`W_{i,k}`. A conserved noise of amplitude `a` adds `(a / h) (dW_{i - e_k, k} - dW_{i, k})` to site `i`, summed over
`k`, the finite-difference divergence of the random flux. Every link adds and subtracts the same increment, so the
sum over a periodic lattice is unchanged. Fixed and zero-flux edges carry no process on their outer links, so the total
is conserved there too. The amplitude is a number or an expression of parameters, the same at every link.

The lowering adds the link processes as columns of the noise matrix of `add-correlated-noise`, two nonzero entries per
column, so every stepper that draws independent processes draws them.

### Implementation choices

- A conserved noise law lowers, for every dimension `k`, to two noise outputs per target component: `-a / h_k` on the
  link to the next neighbour and `a / h_k` on the link from the previous one. The links of one target are numbered
  `k n + i` within one block of `d n` processes, so the columns are those of `add-correlated-noise` with a table of
  links per site.
- A site on an edge that is not periodic draws its missing link from one process of zero variance, a block of
  `add-correlated-noise` with the factor `0`, so the steppers and backends need no change.
- `cahn_hilliard` takes `temperature`, 0 by default. A positive temperature switches its field from the spectral to
  the finite-difference discretization, since the dealiasing of spectral rates would leave the highest modes of the
  link noise undamped, and the run then needs an SDE solver.
- `active_model_b` is Active Model B+, with `lam = 1`, `zeta = 4`, and mean 0.4 by default, where the bubbles of the
  dilute phase stop coarsening, and `zeta = 0` gives Active Model B.
- Conserved noise holds on finite-difference fields of square lattices. A conserved law that names a process is
  rejected when it is built, and one whose amplitude reads a state fails to compile, as the non-goals state.

### Declarations and validation

`NoiseKind` gains `CONSERVED`. Euler-Maruyama, Heun, and the NumPy and Numba backends support it. A conserved noise
on a state that is not a field on a lattice, or under a solver or backend without the kind, is a validation issue
with the code `noise-unsupported`, naming the solvers that support it.

## Risks / Trade-offs

- [The number of processes grows with `d` per site] → The columns are sparse, two entries each, so drawing them costs
  one normal number per link.
