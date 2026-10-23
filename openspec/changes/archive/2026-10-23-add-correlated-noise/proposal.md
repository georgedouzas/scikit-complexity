# Proposal

## Why

A noise law drives every state it targets with its own independent Wiener process. A reaction that changes several
species therefore cannot pass one random increment to all of them. The chemical Langevin equation (Gillespie 2000)
needs exactly that, so the library builds it only on the Schlögl reaction, where every channel changes one species.
Correlated asset returns and Langevin particles in a common bath need correlated Wiener processes, a diffusion
matrix, which the framework cannot state either.

**Rebased on `redesign-system-model`.** The redesign moves the noise convention from the system onto each noise law,
`Noise(convention=...)`, and checks solvers and backends against declared support before compilation. This change keeps
its scope: it adds named and correlated Wiener processes, declares which solvers and backends support them, and lets
validation report a solver or backend that does not, instead of raising inside a stepper.

## What Changes

- Noise laws name the Wiener process they use. Laws of one component that name the same process share its increment,
  and a law that names no process keeps today's independent process for every component and member.
- Add `Wiener` to `skcomplex.core.components`, an element that declares a set of Wiener processes with an optional
  constant covariance matrix. Laws of any component in the system name its processes through a role, so processes are shared and correlated across
  components and group members.
- The compiler lowers the noise of a system to a sparse diffusion matrix: one amplitude, state row, and process column
  per output, plus a factor of every covariance block. NumPy computes the factor, and the steppers draw standard normal
  increments from the run's stream and scale them by the factor.
- The Stratonovich and anti-Itô drift corrections, chosen per law by `Noise(convention=...)`, use the full diffusion
  matrix and the covariance. A state that several noise laws drive no longer raises the ambiguity error, because each
  law now has a known process. Validation reports laws that share a process but declare different conventions.
- The declaration vocabulary of `skcomplex.core.base` gains the noise structures `'diagonal'`, `'shared'`, and
  `'correlated'`, and `Supports` gains a `noise` field. `EulerMaruyama` and `Heun` declare all three on NumPy and Numba.
  `Milstein` declares only diagonal noise, since its derivative-free correction holds only for it, and the LAMMPS
  backend declares only diagonal noise. Validation reports a system whose noise structure the chosen solver or backend
  does not support, before compilation.
- Systems that name no process draw the same random numbers as before, so seeded runs reproduce their old trajectories.
- `Binding` and `Dissociation` gain a chemical Langevin mode in which one Wiener process per channel drives every
  species the channel changes. Add the library models `chemical_langevin_network` and
  `correlated_geometric_brownian_motion`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: noise laws name shared Wiener processes, and `Wiener` elements declare correlated processes.
- `hybrid-dynamics`: the per-law noise conventions correct the drift for shared and correlated processes.
- `system-validation`: solvers and backends declare the noise structures they support, and validation reports the
  misuse of named and correlated processes.
- `simulation`: SDE steppers draw shared and correlated increments from the run's random stream, compiled or not.
- `model-library`: chemical Langevin equations of reaction networks and correlated geometric Brownian motions.

## Impact

- `skcomplex.core.laws`: `Noise` gains a `process` field.
- `skcomplex.core.components`: `Wiener` is new, with its requirement declaration.
- `skcomplex.core.base`: the noise structure terms and the `noise` field of `Supports`.
- `skcomplex.core.system`: validation issues for unsupported noise structures, invalid covariances, misused process
  handles, and mixed conventions on one process.
- `skcomplex.core.compile`: the noise lowering, the problem's diffusion layout, and the convention corrections.
- `skcomplex.core.solvers`: `EulerMaruyama`, `Heun`, `Milstein`, and the hybrid SDE loop, with their noise support.
- `skcomplex.core.backends`: the Numba Euler-Maruyama loop, and the noise support of the NumPy, Numba, and LAMMPS
  backends.
- `skcomplex.models.chemistry.kinetics` and `skcomplex.models.economics.finance`: the Langevin mode and two models,
  each with a new entry in `docs/overview/inventory.toml`.
- No new dependency. NumPy's linear algebra factors the covariance, and NumPy's `Generator` supplies the increments.
