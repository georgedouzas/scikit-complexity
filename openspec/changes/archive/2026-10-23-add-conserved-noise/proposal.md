# Proposal

## Why

Stochastic field theories of conserved quantities, such as Model B, active Model B, and noisy Cahn-Hilliard
coarsening, add thermal noise that conserves the field: the divergence of a random flux, `∂ρ/∂t = ... + ∇·(√(2D) η)`,
with `η` a vector white noise. The core gives every site of a field its own Wiener process, and a noise law cannot
contain the noise itself, so the divergence of a random flux cannot be written. The survey of the references
(`survey-thin-references`) found `active_model_b` (arXiv 2104.06634) blocked on it.

## What Changes

- Let a noise law on a state of a field on a lattice conserve the field: `Noise(target, amplitude, conserved=True)`
  adds `∇·(amplitude η)` with one Wiener process per link of the lattice, so every increment a site gains another
  site loses, and the total of the field changes only by its rate laws.
- Declare the noise kind `conserved` in `NoiseKind`, supported by the SDE solvers and backends that draw independent
  processes, and validated like the other noise kinds.
- Add `active_model_b` to the library, and noise to `cahn_hilliard`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `spatial-fields`: noise that conserves a field.

## Impact

- `skcomplex.core.laws.Noise` gains `conserved`; the compiler lowers it to link processes on lattices; `NoiseKind`
  gains `conserved`.
