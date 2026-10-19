# Spec Delta

## ADDED Requirements

### Requirement: Visuals declare their views
Every visual SHALL declare the Rerun views that show its entities: scalars in time series views, points, agents,
lattices, and fields in 2D or 3D spatial views by their dimension, and networks in graph views. A visual registered by
a user SHALL be able to declare its views the same way.

#### Scenario: Views of the built-in visuals
- **WHEN** the views of the scalars, points in three dimensions, image, and agents on a network visuals are listed
- **THEN** they are a time series view, a 3D spatial view, a 2D spatial view, and a graph view, each with the visual's
  entity path as origin

### Requirement: Systems get a default blueprint
`default_blueprint(system)` SHALL return a Rerun blueprint built from the views of the system's visuals. Its scalar
views SHALL be grouped by element, its parameters left out unless asked for, its side panels collapsed, and its time
panel following the latest frame, so the views show the end of the run.

#### Scenario: Blueprint of an ODE model
- **WHEN** the default blueprint of `lotka_volterra` is built
- **THEN** it holds one time series view with the prey and predator populations, no parameter view, collapsed side
  panels, and a time panel that follows the latest frame

#### Scenario: Blueprint of a field model
- **WHEN** the default blueprint of `gray_scott` is built
- **THEN** it holds one 2D spatial view per field state, `u` and `v`, side by side

#### Scenario: Parameters on request
- **WHEN** the default blueprint is built with `parameters=True`
- **THEN** it also holds a time series view of the parameters

### Requirement: The observer sends and saves blueprints
`RerunObserver` SHALL accept a `blueprint` argument: `None` for today's behaviour, `'default'` for the default
blueprint of the simulated system, or a Rerun blueprint. It SHALL send the blueprint with the recording and save it
next to a saved recording. `save_blueprint(blueprint, path)` SHALL write a `.rbl` file.

#### Scenario: Saved recording with its blueprint
- **WHEN** a simulation runs with `RerunObserver(save_path='run.rrd', blueprint='default')`
- **THEN** `run.rrd` and `run.rbl` exist, and opening both in the viewer shows the default layout

#### Scenario: No blueprint
- **WHEN** a simulation runs with `RerunObserver()` and no blueprint
- **THEN** the recording is the same as before this change
