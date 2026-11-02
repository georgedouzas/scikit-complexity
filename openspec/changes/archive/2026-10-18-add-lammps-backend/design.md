# Design

## Context

Probes of the LAMMPS 2025.7.22 wheel on macOS arm64 established the following.

- The wheel links `libmpi.12.dylib` but does not bundle it, so `import lammps` fails until the `mpich` wheel is
  installed, which places the library where LAMMPS looks for it.
- `pair_style lepton` accepts expression strings such as `4.0*eps*((sig/r)^12-(sig/r)^6);eps=1.0;sig=1.0`, so a
  symbolic pair energy can run in LAMMPS without a hand-written style. The `PYTHON` package is absent, so Python pair
  styles are not available.
- `fix nve`, `fix langevin`, `fix wall/reflect`, `fix enforce2d`, `fix addforce`, and `fix external` are available.
- `lmp.numpy.extract_atom('x')` returns a NumPy view of positions, so reading state each frame copies nothing.
- 9612 Lennard-Jones atoms in two dimensions ran 1000 steps in 1.36 seconds, and 100 runs of 10 steps took 1.45
  seconds, so chunked runs per frame cost little.

## Goals / Non-Goals

**Goals:**

- A backend contract general enough for LAMMPS, Mesa, and later libraries such as SUNDIALS.
- Particle groups with pair potentials, walls, thermostats, and external forces in LAMMPS, matching native runs.

**Non-Goals:**

- Bonds, molecules, many-body potentials, and long-range electrostatics. The translation table can grow to them.
- Running LAMMPS across MPI ranks. The backend runs one process.
- Writing LAMMPS input scripts for users.

## Decisions

### Backend contract

`ExternalBackend` extends `Stepper` with `check(problem)`, which returns the unsupported items, `initialize`, which
builds the library model, and `advance`, which runs the library to `until` and returns the state vector. Because it is a
`Stepper`, `Simulation` and the runtime need no change except running `check` before the first frame.

- External backends live in a new `skcomplex.backends` subpackage, one module per library, and register in the solver
  registry under their name, for example `solver='lammps'`.

### Translation table

| Framework | LAMMPS |
| --- | --- |
| particle group of type `k` | atom type `k`, `create_atoms` from positions, `velocity set` per atom |
| `Particle.m` | `mass` per type, or per atom with `atom_style` that holds masses |
| two-dimensional space | `dimension 2` and `fix enforce2d` |
| `ReflectingBox`, `PeriodicBox` | `boundary f`, `fix wall/reflect`, `boundary p` |
| pair interaction with a symbolic energy | `pair_style lepton` with the energy printed in Lepton syntax and the cutoff |
| several pair interactions | `pair_style hybrid/overlay` |
| `ConstantForce` | `fix addforce` |
| `LinearDrag`, `LangevinThermostat` | `fix viscous`, `fix langevin` |
| other single-particle symbolic forces | `fix external` with a callback that evaluates the compiled kernel |

- Units are `lj`, so framework values pass through unchanged.
- The Lepton printer is a SymPy printer that maps `**` to `^` and functions to Lepton names, and raises the backend's
  unsupported error for anything else.

### Implementation notes

- `backends` imports `models.physics` to recognize the interactions it translates, so it is the top layer, above
  `models`. The runtime does not import it, and importing `skcomplex.backends` registers the solver name `'lammps'`
  for the ODE and SDE forms. A problem now keeps its system and its interactions with their role bindings, which the
  translation reads.
- LAMMPS assigns atoms to a `group ... type` group when the command runs, so groups are defined after the atoms are
  created.
- Open dimensions use the `m` boundary, which shrink-wraps without going below the initial box, and the initial box is
  padded by the largest cutoff, so small systems keep a box larger than the neighbour cutoff.
- Periodic runs keep ghost atoms after the local atoms, so every per-atom read is cut to `nlocal`.
- Interactions without a native fix run through `fix external`: the backend compiles a copy of the system that keeps
  only those interactions, and the callback evaluates its drift and returns mass times acceleration.
- The thermostat's temperature parameter is named `temperature`, in energy units.

### Time stepping

`advance(t, x, p, until)` runs `run N pre no post no` with `N = round((until - t) / dt)` and a backend `dt` parameter,
then reads positions and velocities through NumPy views into the state vector. A parameter change marks the commands
that depend on it, and they are reissued before the next run.

### Dependencies

`lammps` and `mpich; sys_platform != 'win32'` become runtime dependencies, and the `lammps` extra is removed. The
constitution's Project Profile records both and the reason for `mpich`.

## Risks / Trade-offs

- [The Windows wheel's MPI linkage is not verified] → A CI job on Windows checks that LAMMPS loads. If it does not, the
  dependency gains a platform marker and the Project Profile documents the limitation.
- [Lepton expressions are slower than native pair styles] → Known potentials such as Lennard-Jones translate to their
  native `lj/cut` style, and Lepton is the general fallback.
- [`fix external` calls Python every step] → It only handles forces that no native fix covers, and the compiled kernels
  keep the callback cheap.
- [Chaotic dynamics diverge between solvers] → Agreement is specified over a short horizon only.
