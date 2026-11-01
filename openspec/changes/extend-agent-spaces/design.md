# Design

## Context

An agent system is an ordinary `System` under `DiscreteTime`: groups of elements whose laws or interactions return
agent `RuleLaw`s, run by `MesaSolver`, an `ExternalSolver` in `skcomplex.engine.solvers`. Agents act through the
general interactions:
- `MovementInteraction(target, step, probability, rate, condition, accept, exclusive, score, scale, exponent, cost,
  vision, schedule)` moves them by `'neighbor'`, `'gaussian'`, `'levy'`, or `'best'` steps or a user function
  `step(agent, model)` that returns the destination, under the `'sequential'`, `'random-sequential'`, or `'member'`
  schedule. Its Gaussian and Lévy steps already draw vectors of the length of the agent's location.
- The agent forms (`form='agent'`) of `MapInteraction`, `TransitionInteraction`, `TransferInteraction`, and the other
  agent interactions run per member under the `'member'` schedule, composed by the order of the system's interactions.
- Rules see agents through `AgentView` (`location`, `neighbors(radius)`, `distance_to`, `displacement_to`, `move_to`,
  `agents_here`) and the model through `ModelView` (`dimensions`, `torus`, `rng`, `create`, `field`, `parameter`).

`MesaSolver._build_space` lowers `LatticeSpace` to `OrthogonalMooreGrid` or `OrthogonalVonNeumannGrid`, `NetworkSpace`
to `Network`, and `ContinuousSpace` to `mesa.experimental.continuous_space.ContinuousSpace` with the bounds
`[[0, extent[0]], [0, extent[1]]]`. `_place` draws random positions in those two sides, and `ModelView.dimensions`
returns the width and the height. `ContinuousSpace(extent=(...), n_dim=None, boundary=...)` itself takes any number of
sides. `MesaSolver` declares its `SupportSpec` over space classes, which cannot state a number of dimensions.

Mesa 3.5, the pinned version, accepts any number of dimensions in its continuous space. Its bounds are an `(n, 2)`
array, and its distance and torus code is vectorised over all coordinates. The three-dimensional case is therefore a
change to the adapter, not a new engine.

## Goals / Non-Goals

**Goals:**

- One continuous space for agents in two and three dimensions, and regions of a map in the same space, with rules
  written the same way in every space.
- Agents move through `MovementInteraction` and act through agent-form interactions, with no agent class of their own.
- Every combination `MesaSolver` cannot run rejected before the first frame.

**Non-Goals:**

- Coordinate reference systems and reprojection. The caller passes polygons in the coordinates of the space, such as a
  projected metric system, and reprojects them with the tools of its choice.
- Raster layers, and agents that are raster cells.
- Routing along road networks. A road graph is already a `NetworkSpace`.
- Agents in continuous spaces with more than three dimensions. Rerun cannot draw them, and no model needs them.
- Shipping real Kauai census or LEHD data. The model takes region layers as arguments, and its package ships a small
  synthetic island so tests and examples run offline.

## Decisions

- **Regions are Shapely geometries.** Shapely is a runtime dependency, and neither mesa-geo nor pyproj is added; a
  user projects coordinates before building the space. Decided on 2026-11-01.

### Library survey

Three-dimensional continuous space:
- **Mesa `ContinuousSpace` 3.5, chosen.** Mesa is the reference Python agent-based framework, Apache-2.0, and already a
  dependency. Its bounds have any number of dimensions, and its torus correction and radius queries are vectorised.
- **SciPy `cKDTree`, rejected for now.** It answers radius queries faster for large flocks but is a query structure,
  not an agent space, so it would duplicate Mesa's agent bookkeeping. It stays the performance fallback inside the
  adapter.

Regions of a map:
- **Shapely 2, chosen.** It is the reference geometry library, BSD-3, already a runtime dependency, and gives
  polygons, containment, and the `STRtree` index the region queries need.
- **`mesa-geo` 0.9.3, rejected.** It needs `geopandas`, `pyproj`, `rasterio` with GDAL, `ipyleaflet`, and `folium`.
  `pyproj` 3.8 ships macOS wheels for macOS 14 on Apple silicon and macOS 15 on Intel only, above the macOS 13 floor of
  the Project Profile, and the profile makes every library `skcomplex.engine` uses a runtime dependency, so it would
  raise the floor for every user. Its CRS handling is the part this change leaves to the caller.
- **GeoPandas alone, rejected.** It reads shapefiles and reprojects, which the caller can do before passing polygons,
  and it carries the same `pyproj` floor.
- **GAMA, Repast HPC, and the NetLogo GIS extension, rejected.** They are Java or C++ platforms and cannot run inside
  `MesaSolver`.

### Three-dimensional continuous spaces on `MesaSolver`

`ContinuousSpace(extent=(10.0, 5.0, 2.0), boundary='periodic')` is a three-dimensional box today. `MesaSolver`
passes one bound `[0, side]` per side to Mesa, with `torus=True` for a periodic boundary. `_place` draws random
positions from all bounds, `ModelView.dimensions` returns every side, and `AgentView.displacement_to` takes the minimum
image on every coordinate. `MovementInteraction` needs no change, since its Gaussian and Lévy steps already draw
vectors of the location's length, and a user step returns a point of any length.

`MesaSolver._unsupported` lists a continuous space of more than three sides or without an extent, so `_prepare`
raises one `BackendError` before the first frame, as the external-backends contract requires. `PassCompiler` reports
an initial location whose length differs from the dimension of its space as the issue `location-dimension`, whose path
is the group and whose message names the dimension and the location's length. During a run, `move_to` with a location
of the wrong length raises `SpaceError`, whose docstring widens to cover a location that does not fit its space.

Rejected alternative: a `depth` argument or a three-dimensional space class. The space already takes any number of
sides.

### Regions in a continuous space

`ContinuousSpace` gains `regions: Mapping[str, Sequence[Polygon]] | None = None`, stored unmodified. `prepare`
derives `regions_`, one Shapely `STRtree` per layer, and checks that every region is a valid polygon in two
dimensions on a space of two sides, raising `SpaceError` that names the layer and the region otherwise. Region layers
need no new space kind, since they are a property of the plane the agents already live in, and the symbolic solvers
ignore them.

Rules reach the regions through the views of `MesaSolver`:
- `AgentView.regions_here(layer)` returns the indices of the regions of the layer that contain the agent, by the
  layer's `STRtree`,
- `ModelView.region(layer, index)` returns the polygon of a region, and `ModelView.regions(layer)` the number of
  regions of a layer,
- a location may be a polygon, in `GroupElement(location=...)`, `ModelView.create`, and `AgentView.move_to`, and the
  solver places the agent at a random point inside it.

Random points inside polygons use rejection sampling in the polygon's bounds with the model's NumPy generator, so runs
stay reproducible under the simulation's seed. After a fixed number of tries, sampling falls back to the polygon's
`representative_point`.

A group drawn from a region layer is then plain `GroupElement` arguments: the model package computes the size and the
values from the attributes of the regions, such as one member per resident of a census region, and passes the region
polygon of every member as its location.

### Moving and acting in regions

Movement between regions is a `MovementInteraction` with a user step, such as
`step=lambda agent, model: model.region('places', agent.favourite)` returning a polygon, or a point drawn in it, under
the `'member'` schedule, so it composes with the member's other agent interactions. What an agent does where it
stands is an agent-form interaction that reads `agent.regions_here(layer)` or `agent.neighbors(radius)`, such as an
agent-form `MapInteraction` that moves a sentiment towards the mean of the neighbours.

### Trajectories

`AgentSnapshotSpec.locations` stays an `(n, n_dim)` float array, with three columns in a three-dimensional space. The
regions are not states, so trajectories do not record them.

### Rerun

`AgentsVisual` logs `rr.Points3D` when the space has three sides. A space with regions logs every layer once as static
`rr.LineStrips2D` under `<path>/regions/<layer>`, and the agents as `rr.Points2D` over them.

### Models

Three-dimensional boids live next to the two-dimensional ones in `biology/collective_behaviour`, as a separate
factory over a three-sided periodic `ContinuousSpace`. The heading update is an agent-form `MapInteraction` of the
velocity reading the neighbours within a radius, and the flight is a `MovementInteraction` whose step adds the
velocity to the position, both under the `'member'` schedule.

The Kauai model creates `social/organizations`. It takes the boundary, census, places, and facility layers as region
polygons, and its package ships the synthetic island as small GeoJSON files in a metric coordinate system, read through
`importlib.resources` and `shapely.geometry.shape`. Residents move between home, work, and favourite places through a
`MovementInteraction`, and their sentiment, topic weights, and events follow Carmichael et al. (2019 ch. 3 §4) through
agent-form interactions. The model is registered under its field path, its catalog entry declares its observables,
and its package stays self-contained.

## Risks / Trade-offs

- [Mesa's radius query is linear per agent, so a step is quadratic in the number of boids] → It is vectorised and fast
  enough for the thousands of boids the examples use. A SciPy `cKDTree` index can replace it inside the adapter later
  without changing the spec.
- [Users with shapefiles in geographic coordinates] → The user guide shows how to reproject them before passing the
  polygons. Distances in an unprojected space are in degrees, which the guide states once.
- [Random points in thin polygons make rejection sampling slow] → Sampling falls back to a point on the polygon's
  `representative_point` after a fixed number of tries, and the fallback is recorded in the docstring.
- [Real census and commuting data cannot ship or be downloaded in tests] → The synthetic island has the same layer
  schema, and the user guide shows how to load public census shapefiles.

## Migration Plan

Existing two-dimensional models keep their signatures and results. No rollback steps beyond reverting the change are
needed.

## Open Questions

- None. Regions are an argument of `ContinuousSpace`, so no geographic space kind and no extra are added.
