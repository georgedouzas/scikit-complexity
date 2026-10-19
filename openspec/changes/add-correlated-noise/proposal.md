# Proposal

## Why

A noise law drives every state it targets with its own independent Wiener process. A reaction that changes several
species therefore cannot pass one random increment to all of them. The chemical Langevin equation (Gillespie 2000)
needs exactly that, so the library builds it only on the Schlögl reaction, where every channel changes one species.
Correlated asset returns and Langevin particles in a common bath need correlated Wiener processes, a diffusion
matrix, which the framework cannot state either.

## What Changes

- Noise laws name the Wiener process they use. Laws of one component that name the same process share its increment,
  and a law that names no process keeps today's independent process for every component and member.
- Add `Wiener`, an element that declares a set of Wiener processes with an optional constant covariance matrix. Laws
  of any component in the system name its processes through a role, so processes are shared and correlated across
  components and group members.
- The compiler lowers the noise of a system to a sparse diffusion matrix: one amplitude, state row, and process column
  per output, plus a factor of every covariance block. NumPy computes the factor, and the steppers draw standard normal
  increments from the run's stream and scale them by the factor.
- The Stratonovich and anti-Itô drift corrections use the full diffusion matrix and the covariance. A state that
  several noise laws drive no longer raises the ambiguity error, because each law now has a known process.
- `EulerMaruyama` and `Heun` handle shared and correlated processes on NumPy and Numba. `Milstein` raises a named error
  for noise that is not diagonal, since its derivative-free correction holds only for diagonal noise.
- Systems that name no process draw the same random numbers as before, so seeded runs reproduce their old trajectories.
- `Binding` and `Dissociation` gain a chemical Langevin mode in which one Wiener process per channel drives every
  species the channel changes. Add the library models `chemical_langevin_network` and
  `correlated_geometric_brownian_motion`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: noise laws name shared Wiener processes, and `Wiener` elements declare correlated processes.
- `hybrid-dynamics`: the noise conventions correct the drift for shared and correlated processes.
- `simulation`: SDE steppers draw shared and correlated increments from the run's random stream, compiled or not.
- `model-library`: chemical Langevin equations of reaction networks and correlated geometric Brownian motions.

## Impact

- `skcomplex.core.model`: `Noise` gains a `process` field, and `Wiener` is new.
- `skcomplex.core.compile`: the noise lowering, the problem's diffusion layout, the convention corrections, and the
  Numba Euler-Maruyama loop.
- `skcomplex.core.solvers`: `EulerMaruyama`, `Heun`, `Milstein`, and the hybrid SDE loop. A backend that cannot honour
  shared processes, such as LAMMPS, rejects them with a named error.
- `skcomplex.models.chemistry.kinetics` and `skcomplex.models.economics.finance`: the Langevin mode and two models,
  each with a new entry in `docs/overview/inventory.toml`.
- No new dependency. NumPy's linear algebra factors the covariance, and NumPy's `Generator` supplies the increments.
