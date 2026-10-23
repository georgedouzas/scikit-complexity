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

### Declarations and validation

`NoiseKind` gains `CONSERVED`. Euler-Maruyama, Heun, and the NumPy and Numba backends support it. A conserved noise
on a state that is not a field on a lattice, or under a solver or backend without the kind, is a validation issue
with the code `noise-unsupported`, naming the solvers that support it.

## Risks / Trade-offs

- [The number of processes grows with `d` per site] → The columns are sparse, two entries each, so drawing them costs
  one normal number per link.
