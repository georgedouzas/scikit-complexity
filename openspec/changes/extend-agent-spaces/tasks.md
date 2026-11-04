# Tasks

## 1. Three-dimensional continuous spaces on `MesaSolver`

- [x] 1.1 Record the trajectories of an existing two-dimensional continuous agent model with random state 0, and verify
  they load in a test
- [x] 1.2 Pass one bound per side of a `ContinuousSpace` to Mesa in `MesaSolver._build_space`, and place, measure, and
  report agents over every side in `_place`, `ModelView.dimensions`, and `AgentView.displacement_to`, and verify the
  three-dimensional box, depth boundary, depth distance, Gaussian steps in depth, and two-dimensional space unchanged
  scenarios
- [x] 1.3 List continuous spaces of more than three sides or without an extent in `MesaSolver._unsupported`, add the
  `location-dimension` issue to `PassCompiler`, widen `SpaceError` to locations that do not fit their space, and
  verify the wrong number of coordinates and unsupported dimension scenarios
- [x] 1.4 Report three coordinates per agent in `AgentSnapshotSpec.locations`, and verify the three coordinates per
  agent scenario

## 2. Region layers

- [x] 2.1 Add `regions` to `ContinuousSpace` with the derived `regions_` index of Shapely `STRtree`s and the check of
  valid polygons, and verify the regions are indexed and invalid region scenarios
- [x] 2.2 Raise the Shapely requirement to `shapely>=2.2` in `pyproject.toml` and the Project Profile, let
  `GroupElement(location=...)`, `ModelView.create`, and `AgentView.move_to` take a polygon, placed at a uniform point
  inside it drawn from the model's generator over the area-weighted triangles of Shapely's
  `constrained_delaunay_triangles`, and verify the agents per census region, reproducible placement, and uniform
  placement in a concave polygon scenarios
- [x] 2.3 Add `AgentView.regions_here`, `ModelView.region`, and `ModelView.regions`, and verify the move to a favourite
  place scenario with a `MovementInteraction`

## 3. Visualization

- [x] 3.1 Draw three-dimensional agents as `Points3D`, and region layers as static `LineStrips2D` under the agents'
  `Points2D`, in `AgentsVisual`, and verify the boids in 3D and island map scenarios with a recording in memory

## 4. Models

- [ ] 4.1 Add three-dimensional boids to `biology/collective_behaviour` as a registered factory built on an agent-form
  `MapInteraction` and a `MovementInteraction` under the `'member'` schedule, and verify the boids align in three
  dimensions scenario
- [ ] 4.2 Create `social/organizations` with the Kauai sentiment model, built on `MovementInteraction` and agent-form
  interactions, and the synthetic island GeoJSON layers, and verify the population follows the census layer and
  positive event raises sentiment scenarios
- [ ] 4.3 Mark `boids_3d` and `kauai_population` as implemented in `docs/overview/inventory.toml`, rename the primitive
  `geographic-spaces` to `region-layers` with its description, retag both models, and verify the inventory check against the catalog passes
- [ ] 4.4 Run the model conformance and layout tests, and verify both new models pass and import only public names

## 5. Documentation and gate

- [ ] 5.1 Document three-dimensional spaces, region layers, and how to reproject census shapefiles before passing their
  polygons in the user guide with runnable examples, add the changelog entry, and verify the documentation build runs
  the examples
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
