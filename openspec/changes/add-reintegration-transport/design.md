## Context

`convolve`, `shifted`, `gradient`, `advection`, and `upwind` in `core/components/laws/utils/_operators.py` give the
nonlocal and differential operators of fields on lattices. `convolve` takes a kernel of `displacement(axis)` and a
radius, and the backends lower it as a sum over a window of offsets with the boundary of the lattice. `shifted` reads a
field at a fractional displacement by interpolation, which is semi-Lagrangian but not conservative. `LeniaSystem` in
`computer_science/automata/_lenia.py` writes Lenia as an `UpdateLaw` with `convolve`, `where`, and `clip`.

## Goals / Non-Goals

**Goals:**
- One operator that moves a field along a flow with its total conserved to round-off, on every backend, and with
  gradients on JAX.
- Flow-Lenia with localised parameters as the probe.

**Non-Goals:**
- The softmax sampling of parameters (equation 10 of the paper). It is a random choice per cell and adds a draw that
  reads the incoming shares of every neighbour. The mass-weighted mean covers the multi-species runs of the paper.
- Unstructured meshes. Reintegration is defined on lattices, where cells are squares.
- The evolutionary optimisation of Flow-Lenia creatures. `calibrate` and the diversity search of
  `add-diversity-search` cover it.

## Decisions

### Placement

| Component | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `reintegrate` | 2, an operator the backends lower | `skcomplex.core.components.laws.utils` | artificial life, conservative tracer advection in oceanography and climate, densities of cells and morphogens in biology, crowd densities in social systems, sediment and granular transport in physics |
| `FlowLeniaSystem` | 3, one system | `computer_science/automata/_flow_lenia.py` | computer science |

### The operator

- `reintegrate(expression, flow, *, dt, spread=0.5, reach=None)` returns a call of a new symbolic operator, like
  `convolve`. `flow` is a vector expression with one component per axis, in sites per unit time. `reach` defaults to
  `ceil(spread) + 1` sites.
- The lowering is a gather. For every offset `k` in the window `[-w, w]^d` with `w = ceil(reach + spread)`, the source
  site `p - k` has the target centre `c = p - k + clip(dt F(p - k))`, and the share it gives to site `p` is
  `prod_a max(0, min(p_a + 1/2, c_a + s) - max(p_a - 1/2, c_a - s)) / (2 s)^d`. The new value is the sum over offsets
  of the source value times the share. The window is static, so JAX traces it as a fixed number of rolls, Numba as a
  loop, and NumPy as rolled arrays.
- The clip of the displacement to `reach - spread` per axis keeps every square inside the window, which is what
  makes the shares of each source sum to one and the total conserved.
- At fixed edges the target centre is clipped to `[spread - 1/2, n_a - 1/2 - spread]` per axis, so no share falls
  outside the lattice. Zero-flux edges behave the same. Periodic lattices wrap the rolls.
- The operator is an update, not a rate. A law that uses it is an `UpdateLaw` under discrete time. Using it inside a
  rate law raises `ValidationError` naming the law, since it already contains the step `dt`.
- Its derivative with respect to its arguments is left to JAX tracing. The symbolic derivative is not defined, as for
  `convolve`.

### Carried fields

The mass-weighted mean of equation 9 equals `reintegrate(A * P, F) / reintegrate(A, F)`, since both sums share the
shares `I(p', p)`. A system writes it with `where(reintegrate(A, F) > eps, ratio, P)`, which the guide shows. No second
operator is added.

### Flow-Lenia

- `FlowLeniaSystem(n_channels=1, kernels=None, theta_A=1.0, n=2, dt=0.2, spread=0.65, size=128, localized=False,
  mutation_rate=0.0, random_state=None)`. A `kernels` of `None` draws the parameters of every kernel from the ranges
  of Table 1 of the paper: `R` in `[2, 25]`, `r` in `[0.2, 1]`, `a`, `b`, and `w` of three bumps, `h` in `[0, 1]`, `mu`
  in `[0.05, 0.5]`, and `sigma` in `[0.001, 0.2]`, with an adjacency matrix of channels.
- A site element holds the mass of every channel. One update interaction computes `U_j = sum_i h_i G_i(K_i * A_c_i)`
  over the kernels that target `j`, the Sobel gradients of `U_j` and of `A_sum` by `convolve` with the Sobel kernel,
  the flow, and the new mass by `reintegrate`.
- With `localized=True` the weights `h` become states of the sites, the affinity reads them per site (equation 8), and
  they move as carried fields. Mutation adds Gaussian noise of a given scale to the weights in a random square zone,
  drawn per step with probability `mutation_rate`.
- Primitives: `fields`, `nonlocal-operators`, `lattices`, `updates`, `draws`, and the new `mass-transport`.

### Libraries surveyed

The reference code of Flow-Lenia is a JAX script, not a package. No maintained Python library offers reintegration
tracking or conservative semi-Lagrangian transport on lattices. `pyclaw` solves conservation laws by finite volumes
with time-step limits that Flow-Lenia does not respect. So the operator is written in the core.

## Risks / Trade-offs

- [The window grows with the reach] → The cost is `(2w + 1)^d` per site. The paper's flows stay within a few sites, and
  the default reach covers them. The docstring states the cost.
- [Clipping changes the flow of fast regions] → The clip is the price of exact conservation, and the guide shows how to
  raise `reach` when it bites.
- [Float rounding over long runs] → Conservation holds per step to round-off, and the mass test runs 500 steps.

## Migration Plan

None.

## Open Questions

- Should a stochastic carried field, the softmax sampling of the paper, follow once a model needs it?
