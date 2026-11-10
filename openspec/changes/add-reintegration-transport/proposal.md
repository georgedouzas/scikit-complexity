## Why

Flow-Lenia makes Lenia conserve mass: the growth of Lenia becomes an affinity whose gradient moves matter, and the
matter moves by reintegration tracking, a semi-Lagrangian scheme that sends the mass of every cell to a square around
its target and sums what lands in each cell (2212.07906). Because mass is conserved, almost every pattern stays
localised, and because the parameters of the rule can travel with the matter, creatures with different rules meet and
compete in one world, a step towards open-ended evolution. The core convolves fields, reads them at shifted positions,
and has upwind and advection terms of rate laws, but no update that moves a field along a flow while it conserves its
total exactly. Conservative transport of this kind is used beyond artificial life, in the advection of tracers in
oceans and atmospheres and in the transport of densities of cells and crowds.

## What Changes

- New law function `reintegrate(expression, flow, *, dt, spread, reach)` in `skcomplex.core.components.laws.utils`.
  It returns the field after its mass in every cell moves to a square of side `2 spread` centred at the cell plus
  `dt` times the flow, with the share of each target cell given by the overlap of the square with the cell. The total
  of the field is conserved on periodic lattices and at fixed edges.
- The mass-weighted mean of a field carried with the matter is the ratio of two reintegrations, which the guide shows,
  so parameters travel with the mass with no second operator.
- The JAX, NumPy, and Numba backends lower the operator as a gather over a fixed window of offsets, so a Flow-Lenia run
  is differentiable on JAX.
- New system `FlowLeniaSystem` in `computer_science/automata`, Flow-Lenia with several channels, bump kernels from
  any channel to any channel, the flow of the paper, and an option to carry the kernel weights with the matter and
  mutate them in random zones.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `nonlocal-operators`: fields move along a flow by mass-conserving reintegration.
- `model-library`: the Flow-Lenia system.

## Impact

- Core: a new operator in `core/components/laws/utils/_operators.py` and its lowering in the JAX, NumPy, and Numba
  printers. No new law kind. Systems that do not use it are not affected.
- Systems: a new module in `skcomplex.systems.computer_science.automata`, which reuses nothing of `_lenia.py` except
  what both may import from the core.
- No new dependency.
- Docs: the section on nonlocal operators in the user guide, a gallery example, and the inventory with the primitive
  `mass-transport`.
- The neural cellular automaton of 2509.11131 is added to `add-hybrid-neural-laws`, since it needs neural functions in
  laws and not transport.
