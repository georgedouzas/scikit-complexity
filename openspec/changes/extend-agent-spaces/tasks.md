# Tasks

## 1. Three-dimensional continuous spaces

- [ ] 1.1 Add the `depth` argument and the `n_dim` property to `ContinuousSpace`, and widen `SpaceError` to locations
  that do not fit their space, and verify the three-dimensional box and wrong number of coordinates scenarios
- [ ] 1.2 Pass three-dimensional bounds to Mesa's `ContinuousSpace` in the Mesa stepper, and place, move, and query
  agents over every coordinate, and verify the depth boundary and depth distance scenarios
- [ ] 1.3 Record the trajectories of an existing two-dimensional continuous model before the change and compare them
  after it, and verify the two-dimensional space unchanged scenario

## 2. Geographic spaces and the `geo` extra

- [ ] 2.1 Add the `geo` optional extra with `mesa-geo` to `pyproject.toml`, install it in the development and CI
  environments, and verify `pdm install`, `deptry`, and `pip-audit` pass on Linux and macOS for Python 3.12 and 3.13
- [ ] 2.2 Add `GeoSpace` to `skcomplex.core.model` and its adapter to `mesa_geo.GeoSpace` in the Mesa stepper, with the
  lazy import and layer reprojection, and verify the layer reprojected, unknown reference system, and extra missing
  scenarios
- [ ] 2.3 Add `AgentType.from_layer` and the placement of agents at random points inside features from the model's
  generator, and verify the agents per census region and reproducible placement scenarios
- [ ] 2.4 Add the distance neighbours, `features_here`, `move_to`, and `move_to_feature` behaviour of `AgentView` in
  geographic spaces, and verify the distance neighbours and move to a favourite place scenarios

## 3. Trajectories and visualization

- [ ] 3.1 Report three-dimensional and geographic locations in `AgentSnapshot`, and verify the three coordinates per
  agent scenario
- [ ] 3.2 Draw three-dimensional agents as `Points3D`, and geographic layers as static `LineStrips2D` with agents as
  `Points2D`, in `AgentsVisual`, and verify the boids in 3D and island map scenarios with a recording in memory

## 4. Models

- [ ] 4.1 Add three-dimensional boids to `biology/collective_behaviour` as a registered factory, and verify the boids
  align in three dimensions scenario
- [ ] 4.2 Create `social/organizations` with the Kauai sentiment model and the synthetic island GeoJSON layers, and
  verify the population follows the census layer and positive event raises sentiment scenarios
- [ ] 4.3 Mark both models as implemented in the inventory, and verify the inventory test against the catalog passes
- [ ] 4.4 Run the model conformance and layout tests, and verify both new models pass and import only public core names

## 5. Documentation and gate

- [ ] 5.1 Document three-dimensional and geographic spaces, the `geo` extra, and loading real census shapefiles in the
  user guide with runnable examples, and verify the documentation test passes
- [ ] 5.2 Add the `geo` extra and its macOS floor to the Project Profile in a constitution amendment, and the change to
  the changelog, and verify the changelog session builds
- [ ] 5.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
