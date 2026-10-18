# external-backends Specification

## Purpose

Lets a model described once in the framework run inside a third-party solver library, such as LAMMPS for particle
systems, while frames, trajectories, live parameters, and Rerun visualization keep working.

## Requirements

### Requirement: External backends act as solvers
A simulation SHALL accept an external backend as its solver, and the run SHALL produce frames, trajectories, and Rerun
logs with the same names as a native solver.

#### Scenario: Trajectory from LAMMPS
- **WHEN** a Lennard-Jones gas is simulated with the LAMMPS backend
- **THEN** its trajectory returns `gas.x` and `gas.v` with the same shapes as a native run

### Requirement: Unsupported laws are rejected before running
An external backend SHALL check every element, interaction, and law of a problem before the first frame and SHALL
raise a named error that lists each unsupported one.

#### Scenario: Population model on LAMMPS
- **WHEN** a Lotka-Volterra system is given to the LAMMPS backend
- **THEN** a named error is raised that names the backend and the unsupported elements and interactions

### Requirement: LAMMPS is available after installation
Installing the package SHALL install a LAMMPS build that loads on macOS and Linux without further steps.

#### Scenario: LAMMPS loads
- **WHEN** the package is installed with pip in a new virtual environment on macOS or Linux
- **THEN** the LAMMPS backend starts a LAMMPS instance without a missing library error

### Requirement: Symbolic pair energies run in LAMMPS
The LAMMPS backend SHALL translate the symbolic energy of a pair interaction into a LAMMPS expression potential with
the interaction's cutoff.

#### Scenario: Custom pair potential
- **WHEN** a user-defined pair interaction with energy `eps * exp(-r / sigma)` runs with the LAMMPS backend
- **THEN** the forces match the native backend within 1e-8 relative error on the same state

### Requirement: LAMMPS agrees with the native solver
The LAMMPS backend and the native solver SHALL produce the same trajectory for the same system over a short horizon.

#### Scenario: Short Lennard-Jones run
- **WHEN** a 100-atom Lennard-Jones gas runs for 0.1 time units with the same time step on both solvers
- **THEN** the positions agree within 1e-6

### Requirement: Parameters change live in LAMMPS
A running LAMMPS simulation SHALL apply parameter changes before the next frame by reissuing the affected LAMMPS
commands.

#### Scenario: Thermostat temperature change
- **WHEN** a user raises the Langevin thermostat temperature during a LAMMPS run
- **THEN** the kinetic temperature of the gas rises towards the new value in the following frames

### Requirement: LAMMPS runs large systems in real time
The LAMMPS backend SHALL advance a Lennard-Jones gas of 10000 atoms in two dimensions fast enough to keep a real-time
clock of speed 1 with frames every 0.05 time units.

#### Scenario: Real-time Lennard-Jones gas
- **WHEN** a gas of 10000 Lennard-Jones atoms runs for 5 seconds with a real-time clock and the LAMMPS backend
- **THEN** the run falls behind by no more than 0.2 seconds
