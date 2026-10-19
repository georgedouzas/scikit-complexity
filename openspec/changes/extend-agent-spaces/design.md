# Design

## Context

The Mesa stepper in `skcomplex.core.solvers._agents` lowers the framework spaces to Mesa:
- `GridSpace` to `OrthogonalMooreGrid` or `OrthogonalVonNeumannGrid`;
- `NetworkSpace` to `Network`;
- `ContinuousSpace(width, height, torus=...)` to `mesa.experimental.continuous_space.ContinuousSpace` with bounds
  `[[0, width], [0, height]]`.

Rules see agents through `AgentView`, whose `neighbors(radius)`, `move_to`, and `location` dispatch on the space.
`AgentSnapshot.locations` holds one row per agent, and `AgentsVisual.log_agents` draws grid and continuous agents as
`Points2D` and network agents as `GraphNodes`. Agent rules are Python callables run by Mesa. They are not Sage laws,
so the symbolic compile pipeline only infers the agent form and collects the groups, parameters, and space.

Mesa 3.5.1, the pinned version, already accepts any number of dimensions in its continuous space. Its bounds are an
`(n, 2)` array, and its distance and torus code is vectorised over all coordinates. The three-dimensional case is
therefore a change to the adapter, not a new engine.

## Goals / Non-Goals

**Goals:**

- One adapter per space kind in the Mesa stepper, with rules written the same way in every space.
- Geographic data entering through GeoPandas, the format GIS users already have.

**Non-Goals:**

- Raster layers as agent spaces, and agents that are raster cells. `mesa-geo` supports them, and a later change can
  expose them when a model needs one.
- Routing along road networks. OSMnx and networkx shortest paths cover it, and no inventory model needs it.
- Continuous spaces with more than three dimensions. Rerun cannot draw them, and no model needs them.
- Shipping real Kauai census or LEHD data. The model takes layers as arguments, and its package ships a small
  synthetic island so tests and examples run offline.

## Decisions

### Library survey

Three-dimensional continuous space:
- **Mesa `ContinuousSpace` 3.5.1, chosen.** Mesa is the reference Python agent-based framework, with about 3k GitHub
  stars, the Apache-2.0 licence, and monthly releases. It is pure Python and already a dependency. Its bounds have any
  number of dimensions, and its torus correction and radius and k-nearest queries are vectorised with NumPy and SciPy
  `cdist`.
- **SciPy `cKDTree`, rejected for now.** It is mature, BSD-licensed, and already installed, and answers radius queries
  faster for large flocks. It is a query structure, not an agent space, so using it would duplicate Mesa's agent
  bookkeeping. It stays the performance fallback inside the adapter.
- **Mesa discrete grids with property layers, rejected.** They hold cells, not real positions.

Geographic space:
- **`mesa-geo` 0.9.3, released February 2026, chosen.** It is the Mesa organisation's GIS extension, with about 240
  stars, the Apache-2.0 licence, and releases in 2024, 2025, and 2026. It requires `mesa[rec]~=3.0`, which accepts
  Mesa 3.5. It is pure Python, and its dependencies GeoPandas, Shapely, pyproj, rasterio, and rtree ship binary
  wheels for Linux x86_64 and macOS. It provides `GeoSpace` with a CRS and reprojection, `GeoAgent` with a Shapely
  geometry, `AgentCreator` from GeoDataFrames and files, and R-tree backed `get_neighbors_within_distance`,
  `get_intersecting_agents`, and `get_relation`. It plugs into the Mesa model the stepper already builds.
- **GeoPandas and Shapely alone, rejected.** They are very mature and are the base `mesa-geo` builds on. Using them
  alone would mean writing the agent index, the CRS bookkeeping, and agent creation in house.
- **OSMnx, rejected for this change.** It builds street networks from OpenStreetMap, which needs the network, and a road
  graph can already be a `NetworkSpace`.
- **GAMA, Repast HPC, and the NetLogo GIS extension, rejected.** They are Java or C++ platforms and cannot run inside
  the Mesa stepper.

The resolver test with `uv pip compile --only-binary :all:` for `mesa==3.5.1` and `mesa-geo==0.9.3` resolves on Linux
x86_64 and on macOS with Python 3.12 and 3.13, given a macOS 14 deployment target on arm64.

### `ContinuousSpace` gains an optional depth

`ContinuousSpace(width, height, *, depth=None, torus=False)` stores `depth` unmodified. `n_dim` becomes a property
that returns 3 when `depth` is set and 2 otherwise. The stepper passes bounds `[[0, width], [0, height], [0, depth]]`
to Mesa, and random placement draws from all bounds. `AgentView.move_to` and `_place` validate the location's length
against `n_dim` and raise the existing `SpaceError` of `skcomplex.core.base`, whose docstring widens to cover a
location that does not fit its space.

Alternatives: a separate `ContinuousSpace3D` class duplicates every dispatch, and a `bounds` sequence changes the
signature of every existing model. An optional depth is backward compatible and keeps one class.

### `GeoSpace` wraps `mesa_geo.GeoSpace`

`GeoSpace(layers, *, crs='epsg:3857')` holds a mapping from layer name to a GeoDataFrame or a path, stored unmodified.
The stepper builds a `mesa_geo.GeoSpace(crs=crs)`, reads paths with GeoPandas, reprojects each layer with `to_crs`, and
adds it with `add_layer`. A layer without a CRS raises `SpaceError`. `n_dim` is 2.

Agents in a `GeoSpace` are Mesa agents of a framework-generated subclass of `mesa_geo.GeoAgent`, as continuous agents
are generated subclasses of `ContinuousSpaceAgent` today. Their geometry is a Shapely point. Rules see them through
the same `AgentView`:
- `location` returns the point's coordinates;
- `neighbors(radius)` calls `get_neighbors_within_distance`;
- `features_here(layer)` returns the indices of the layer's features that contain the point;
- `move_to(point)` replaces the geometry;
- `move_to_feature(layer, index)` places the agent at a random point inside the feature.

Random points inside polygons use rejection sampling in the polygon's bounds with the model's NumPy generator, so runs
stay reproducible under the simulation's seed.

### Agent groups from layers

`AgentType.group(label, size=..., space=GeoSpace(...))` keeps working with explicit locations. A new classmethod,
`AgentType.from_layer(label, layer, *, count=None, values=None)`, returns a `Group` whose `values` come from feature
attributes, whose size is the feature count or the sum of the `count` attribute, and whose initial locations are
`('feature', index)` markers. The stepper resolves those markers to random points inside the features at placement.
The group stays a plain `Group`, so compilation and trajectories need no new form.

### Compile pipeline and runtime

Agent rules are Python, so nothing is lowered to NumPy or Numba. The compiler already copies the agent space into the
`Problem`. It gains one check, that every agent group with initial locations matches the space dimension. The
`Simulation`, clocks, observers, and experiments run unchanged, because the Mesa stepper keeps its interface and
`AgentSnapshot.locations` stays an `(n, n_dim)` float array for both new spaces.

### Rerun

`AgentsVisual.log_agents` logs `rr.Points3D` when the space has three dimensions. For a `GeoSpace` it logs each
polygon or line layer once as static `rr.LineStrips2D` under `<path>/layers/<name>`, and the agents as `rr.Points2D`
in the space's projected coordinates. Projecting to a metric CRS such as EPSG:3857 keeps the drawn map undistorted
enough for a single island or city.

### Optional `geo` extra and lazy import

`mesa-geo` pulls in rasterio with GDAL, pyproj with PROJ, ipyleaflet, and folium, and pyproj's current wheels raise
the macOS floor above the macOS 13 the Project Profile promises. It therefore goes in a `geo` extra. The stepper
imports `mesa_geo` lazily inside the `GeoSpace` adapter, the one lazy import the constitution allows, with a
suppression that names the extra. Without it, building the space re-raises the `ImportError` with a message that names
`pip install scikit-complexity[geo]`, as `ExperimentResult.to_frame` does for pandas. The development and CI
environments install the extra, so the gate covers geographic spaces.
`GeoSpace` itself imports nothing from `mesa_geo`, so it can be declared without the extra.

### Models

Three-dimensional boids live next to the two-dimensional ones in `biology/collective_behaviour`, as a separate
factory over a `ContinuousSpace` with depth. The Kauai model creates `social/organizations`. It takes the boundary,
census, places, and facility layers as arguments, and its package ships the synthetic island as small GeoJSON files
read through `importlib.resources`. Its sentiment, topic weights, events, and favourite-place schedule follow
Carmichael et al. (2019 ch. 3 §4). The model is registered under its field path, and each model package stays
self-contained.

## Risks / Trade-offs

- [Mesa's radius query is linear per agent, so a step is quadratic in the number of boids] → It is vectorised and fast
  enough for the thousands of boids the examples use. A SciPy `cKDTree` index can replace it inside the adapter later
  without changing the spec.
- [`mesa-geo` is smaller than Mesa and pinned to `mesa~=3.0`] → It is maintained by the Mesa organisation, and the
  adapter uses only `GeoSpace`, `GeoAgent`, and its distance and intersection queries. A Mesa 4 release would need
  both pins moved together.
- [The `geo` extra needs macOS 14 on arm64 and macOS 15 on x86_64] → The default install keeps macOS 13, and the
  Project Profile states the higher floor for the extra.
- [Random points in thin polygons make rejection sampling slow] → Sampling falls back to a point on the polygon's
  `representative_point` after a fixed number of tries, and the fallback is recorded in the docstring.
- [Real census and commuting data cannot ship or be downloaded in tests] → The synthetic island has the same layer
  schema, and the user guide shows how to load the public US Census shapefiles.

## Migration Plan

Existing two-dimensional models keep their signatures and results. No rollback steps beyond reverting the change are
needed.
