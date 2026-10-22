# Design

## Context

After `redesign-system-model`, an agent system is an ordinary `System`: elements with `Rule` laws in variable groups,
under `DiscreteTime`, on a space of the same kinds every system uses. The Mesa backend in `skcomplex.core.backends`
lowers those spaces to Mesa:
- `LatticeSpace` with its neighbourhood and capacity to `OrthogonalMooreGrid` or `OrthogonalVonNeumannGrid`;
- `NetworkSpace` to `Network`;
- `ContinuousSpace(width, height, boundary=...)` to `mesa.experimental.continuous_space.ContinuousSpace` with bounds
  `[[0, width], [0, height]]`, a periodic boundary becoming Mesa's torus.

`ContinuousSpace(*sides, boundary=...)` already takes any number of sides, so a three-dimensional box exists in the
core; only the Mesa backend reads two of its sides. Rules see agents through the backend's agent view, whose
`neighbors(radius)`, `move_to`, and `location` dispatch on the space. `AgentSnapshot.locations` holds one row per
agent, and `AgentsVisual.log_agents` draws lattice and continuous agents as `Points2D` and network agents as
`GraphNodes`. Agent rules are Python callables run by Mesa. They are not Sage laws, so the symbolic compile pipeline
only collects the groups, parameters, and space, and validation checks them against the backend's declared support.

Mesa 3.5.1, the pinned version, already accepts any number of dimensions in its continuous space. Its bounds are an
`(n, 2)` array, and its distance and torus code is vectorised over all coordinates. The three-dimensional case is
therefore a change to the adapter, not a new engine.

## Goals / Non-Goals

**Goals:**

- One adapter per space kind in the Mesa backend, with rules written the same way in every space.
- Geographic data entering through GeoPandas, the format GIS users already have.
- Every combination the backends cannot run reported by validation before compilation.

**Non-Goals:**

- Raster layers as agent spaces, and agents that are raster cells. `mesa-geo` supports them, and a later change can
  expose them when a model needs one.
- Routing along road networks. OSMnx and networkx shortest paths cover it, and no inventory model needs it.
- Agents in continuous spaces with more than three dimensions. Rerun cannot draw them, and no model needs them.
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
  `get_intersecting_agents`, and `get_relation`. It plugs into the Mesa model the backend already builds.
- **GeoPandas and Shapely alone, rejected.** They are very mature and are the base `mesa-geo` builds on. Using them
  alone would mean writing the agent index, the CRS bookkeeping, and agent creation in house.
- **OSMnx, rejected for this change.** It builds street networks from OpenStreetMap, which needs the network, and a road
  graph can already be a `NetworkSpace`.
- **GAMA, Repast HPC, and the NetLogo GIS extension, rejected.** They are Java or C++ platforms and cannot run inside
  the Mesa backend.

The resolver test with `uv pip compile --only-binary :all:` for `mesa==3.5.1` and `mesa-geo==0.9.3` resolves on Linux
x86_64 and on macOS with Python 3.12 and 3.13, given a macOS 14 deployment target on arm64.

### The Mesa backend runs three-dimensional continuous spaces

The space itself needs no change: `ContinuousSpace(10.0, 5.0, 2.0, boundary='periodic')` is a three-dimensional box.
The Mesa backend passes one bound `[0, side]` per side to Mesa, with `torus=True` for a periodic boundary, and random
placement draws from all bounds. Its `Supports` declaration states continuous spaces of two and three dimensions. A
continuous space of more dimensions, or an unbounded one, on the Mesa backend is reported by validation with the issue
`space-unsupported`, whose fix names the dimensions and boundaries the backend runs.

Initial locations of the members of a group are checked by validation against the space's dimension, with the issue
`location-dimension`, whose path is the group and whose message names the space dimension and the location's length.
During a run, `move_to` with a location of the wrong length raises the existing `SpaceError` of `skcomplex.core.base`,
whose docstring widens to cover a location that does not fit its space.

Rejected alternative: an optional `depth` argument or a `ContinuousSpace3D` class. The redesigned space already takes
any number of sides, so either would duplicate it.

### `GeoSpace` is a space kind the Mesa backend wraps with `mesa_geo.GeoSpace`

`GeoSpace(layers, *, crs='epsg:3857')` lives in `skcomplex.core.space` and holds a mapping from layer name to a
GeoDataFrame or a path, stored unmodified. Its `space_kind` is a new `SpaceKind.GEOGRAPHIC`, and `n_dim` is 2. It is a
kind of its own rather than a `ContinuousSpace`, because no solver other than the Mesa backend can run layers and
reference systems, and a continuous kind would let validation accept it for the NumPy and Numba solvers.

Declarations:
- `GeoSpace` requires discrete time, and the groups on it hold members with point locations;
- the Mesa backend's `Supports` gains `SpaceKind.GEOGRAPHIC`;
- no other solver or backend supports it, so validation reports a geographic space with any other solver as
  `space-unsupported` and lists the Mesa backend in the fix;
- the availability check reports a geographic space without `mesa_geo` installed as `backend-unavailable`, whose fix
  names `pip install scikit-complexity[geo]`.

The backend builds a `mesa_geo.GeoSpace(crs=crs)`, reads paths with GeoPandas, reprojects each layer with `to_crs`, and
adds it with `add_layer`. A layer without a CRS raises `SpaceError` when the space is built.

Members of groups in a `GeoSpace` are Mesa agents of a backend-generated subclass of `mesa_geo.GeoAgent`, as continuous
agents are generated subclasses of `ContinuousSpaceAgent` today. Their geometry is a Shapely point. Rules see them
through the same agent view:
- `location` returns the point's coordinates;
- `neighbors(radius)` calls `get_neighbors_within_distance`;
- `features_here(layer)` returns the indices of the layer's features that contain the point;
- `move_to(point)` replaces the geometry;
- `move_to_feature(layer, index)` places the agent at a random point inside the feature.

Random points inside polygons use rejection sampling in the polygon's bounds with the model's NumPy generator, so runs
stay reproducible under the simulation's seed.

### Groups from layers

A variable group on a `GeoSpace` keeps working with explicit locations. A new classmethod of `Group` in
`skcomplex.core.components`, `Group.from_layer(label, element, layer, *, count=None, values=None)`, returns a
`Group(variable=True)` whose `values` come from feature attributes, whose size is the feature count or the sum of the
`count` attribute, and whose initial locations are `('feature', index)` markers. The backend resolves those markers to
random points inside the features at placement. The group stays a plain `Group`, so compilation and trajectories need
no new form. Validation reports a group created from a layer that the system's space does not hold, or in a system
whose space is not geographic, as `layer-missing`, with the group's path and the layers the space holds.

### Compile pipeline and runtime

Agent rules are Python, so nothing is lowered to NumPy or Numba. The compiler already copies the space into the
`Problem`. The checks on the space, the dimension of locations, and the backend's support are validation issues, so
`Simulation` reports them before the backend builds anything. The clocks, observers, and experiments run unchanged,
because the Mesa backend keeps its interface and `AgentSnapshot.locations` stays an `(n, n_dim)` float array for both
new spaces.

### Rerun

`AgentsVisual.log_agents` logs `rr.Points3D` when the space has three dimensions. For a `GeoSpace` it logs each
polygon or line layer once as static `rr.LineStrips2D` under `<path>/layers/<name>`, and the agents as `rr.Points2D`
in the space's projected coordinates. Projecting to a metric CRS such as EPSG:3857 keeps the drawn map undistorted
enough for a single island or city.

### Optional `geo` extra and lazy import

`mesa-geo` pulls in rasterio with GDAL, pyproj with PROJ, ipyleaflet, and folium, and pyproj's current wheels raise
the macOS floor above the macOS 13 the Project Profile promises. It therefore goes in a `geo` extra. The Mesa backend
imports `mesa_geo` lazily inside its `GeoSpace` adapter, the one lazy import the constitution allows, with a
suppression that names the extra. Without it, validation reports the backend unavailable for the geographic space,
and building the space directly re-raises the `ImportError` with a message that names
`pip install scikit-complexity[geo]`, as `ExperimentResult.to_frame` does for pandas. The development and CI
environments install the extra, so the gate covers geographic spaces. `GeoSpace` itself imports nothing from
`mesa_geo`, so it can be declared without the extra.

### Models

Three-dimensional boids live next to the two-dimensional ones in `biology/collective_behaviour`, as a separate
factory over a three-sided periodic `ContinuousSpace`. The Kauai model creates `social/organizations`. It takes the
boundary, census, places, and facility layers as arguments, and its package ships the synthetic island as small
GeoJSON files read through `importlib.resources`. Its sentiment, topic weights, events, and favourite-place schedule
follow Carmichael et al. (2019 ch. 3 §4). The model is registered under its field path, each catalog entry declares its
observables, and each model package stays self-contained.

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

## Open Questions

- Does a geographic space belong in the core? This design puts `GeoSpace` in `skcomplex.core.space` as its own space
  kind, since GIS layers and reference systems serve social, ecological, and epidemic models alike, and the field-term
  test of `redesign-system-model` lists no geographic term. The alternative is a library realization in
  `skcomplex.models.social.realizations` over `ContinuousSpace`, which would need the Mesa backend to read layers from
  a registry the realization fills, as the LAMMPS backend reads pair styles. The choice decides whether `SpaceKind`
  grows a geographic kind.
