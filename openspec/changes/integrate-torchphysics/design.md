## Context

See proposal.md for the motivation and the reason the change is postponed. The relevant state:

- `redesign-core-api` gives `System`, `prepare`, `simulate`, and `ObjectiveLaw`. An `ObjectiveLaw` outside a pair
  interaction is an objective to minimise, so a field whose laws are one `ObjectiveLaw` already states its energy.
- `add-torch-surrogates` adds torch with CPU wheels in CI, `PinnSolver` on deepxde for single physics-informed solves,
  and `neural_operator` on neuraloperator for maps learned from runs.
- Surrogate results follow `add-learning-foundation`: `system`, `training_error`, `validation_error`.
- TorchPhysics 1.1.2 (Apache-2.0, released 2026-06-12, active repository at Qewton-Labs/torchphysics) describes a
  problem by spaces, domains, samplers, and conditions, and trains with PyTorch Lightning. It provides the Deep Ritz
  method and parameters as input variables of a network, beside PINNs and DeepONet.

## Goals / Non-Goals

**Goals:**

- Lower a prepared field system to a TorchPhysics problem: its space to a domain, its laws to conditions, its fixed
  boundary values to boundary conditions, and its named parameters to parameter variables.
- One adapter module per method, holding only that lowering and the reading of the trained network back to values.

**Non-Goals:**

- Single physics-informed solves of rate laws, which stay with `PinnSolver` on deepxde.
- DeepONet and Fourier operators learned from runs, which stay with `neural_operator` on neuraloperator.
- Inverse problems with TorchPhysics. Calibration of fields keeps `calibrate(method='pinn')` of
  `add-torch-surrogates`.
- Meshes. Both methods sample the domain of a regular space. Unstructured meshes keep finite elements.

## Decisions

### TorchPhysics for Deep Ritz and parametric training

Libraries surveyed for the two methods:

- deepxde: single solves are its strength and it already serves `PinnSolver`. It has no first-class energy
  minimisation, and parameters as inputs would be written by hand on top of it, which this codebase would then own.
- NVIDIA PhysicsNeMo: covers parametric training, but is a large framework aimed at CUDA clusters with its own
  configuration system. Too heavy as a core dependency for two methods.
- neuraloperator: learns from runs, not from equations, so it cannot train without solves.
- TorchPhysics: both methods are first-class, and its problem description by domains and conditions maps one to one
  onto a prepared field system. Chosen.

### The method follows from the laws and the solver

`DeepRitzSolver` is a solver like any other, chosen with `solver=` on the system. It accepts a system whose only laws
are one `ObjectiveLaw` over a field and fixed boundary values, and the census of `prepare` checks this, so a wrong
system fails at `prepare` with the names of the laws it cannot read. `parametric_surrogate` is an analysis function,
not a solver, because its product is a new system over a range of parameters, the same shape as the other surrogate
functions.

### Lowering through the prepared problem

Both adapters read the prepared problem of the system, its symbolic laws, space, and boundary values, never the
classes the user wrote. The energy or the rate laws are turned into torch functions from their symbolic form, the
same path `PinnSolver` uses for deepxde, so the conversion from symbolic laws to torch exists once and both adapters
share it.

### The surrogate system

The `system` of a `parametric_surrogate` result is a `System` whose single element holds the trained network and the
parameter ranges. Simulating it at a parameter value evaluates the network on the points of the space at the frames
asked for. Its parameters keep the names of the original system, so `set_params(k=1.3)` works on both.

### Seeds and devices

Training takes `random_state` and seeds torch from it, so the scenarios run deterministically on CPU. The device
follows the device choice of `add-device-execution` when that change has landed, and defaults to CPU otherwise.

## Risks / Trade-offs

- [The upstream pin may never be lifted] → The change stays postponed and costs nothing. `dependency-freshness`
  reviews the library at each release. If TorchPhysics has had no release for 24 months by then, the change is
  dropped and recorded as such.
- [Hard `jupyter` requirement] → Ask upstream to move it to an extra in the same pull request as the numpy pin. If it
  stays, record the extra install size in the dependency review before adding the library.
- [PyTorch Lightning brings its own logging and checkpoint files] → Run its trainer with logging and checkpoints off,
  and in a temporary directory, so a run leaves nothing behind.
- [Network training is slow for tests] → Scenarios use small networks and few steps on CPU, and the long accuracy
  checks are marked slow.
- [Two physics-informed libraries] → Each owns distinct methods, written down in the Non-Goals, so no method has two
  implementations.

## Migration Plan

1. Open a pull request to Qewton-Labs/torchphysics that allows numpy 2 and moves `jupyter` to an extra, and wait for
   a release that contains it.
2. Add the dependency, then the shared symbolic to torch conversion if `add-torch-surrogates` has not landed it.
3. Add `DeepRitzSolver`, then `parametric_surrogate`, each with its tests, doctests, and gallery example.

Nothing existing changes, so rollback is removing the two modules and the dependency.
