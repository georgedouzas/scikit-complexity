# Proposal

## Why

Agents today run on the Mesa backend on a lattice, on a network, or in a two-dimensional continuous box. Two models in
the library inventory need more:
- Downey's boids fly in three dimensions (Think Complexity ch. 10).
- Zanbaka et al. place one agent per resident of Kauai from census shapefiles. Agents live and work at points on the
  map and move to favourite places drawn from another shapefile (Carmichael et al. 2019 ch. 3).

Both models are recorded as blocked on this change, and neither can be written in the model package without support
the core does not have. Mesa, already a dependency, has an N-dimensional continuous space. Its companion project
`mesa-geo` adds geographic spaces on top of GeoPandas and Shapely. The core only needs the adapters.

**Rebased on `redesign-system-model`.** The redesign made agent spaces the same space kinds as every other system, and
`ContinuousSpace(*sides, boundary=...)` already takes any number of dimensions with periodic, reflecting, or open
boundaries, so this change no longer adds a `depth` argument or a three-dimensional variant of the space. Agents are
elements with `Rule` laws in variable groups, and Mesa is a backend, so there is no `AgentType` to extend. This change
still adds: three-dimensional continuous spaces on the Mesa backend, a geographic space kind, groups created from
layer features, geographic queries for rules, their trajectories and drawing, the `geo` extra, and the two models.

## What Changes

- Let the Mesa backend run agent groups in a continuous space of three dimensions, so positions, radius neighbour
  queries, periodic wrapping, and random placement work on three coordinates, and declare that support. Two-dimensional
  spaces keep working unchanged.
- Add a geographic space, `GeoSpace`, in `core.space`, of a new space kind. It has a coordinate reference system, and
  vector layers loaded from GeoPandas data frames or files. Members of groups on it are located at geometries in that
  system. Only the Mesa backend declares support for it.
- Let groups be created from the features of a vector layer, one member per feature or a given number of members per
  feature. Feature attributes become initial state values, and the members are placed at points inside the feature.
- Let rules query a geographic agent's neighbours within a distance in the space's units, find the layer features
  that contain the agent, and move the agent to a point or to a feature.
- Report agent locations in three-dimensional and geographic spaces in trajectories, and draw them in Rerun as
  `Points3D` and as projected `Points2D` over the layer outlines.
- Report through validation a geographic space with a solver or backend that does not support it, a location whose
  length does not fit its space, and a missing `geo` extra.
- Add the `geo` optional extra, which installs `mesa-geo` and with it GeoPandas, Shapely, pyproj, and rasterio.
- Add the library models this unblocks: boids in three dimensions and a Kauai population sentiment model on
  geographic layers.

## Capabilities

### New Capabilities

- `agent-spaces`: three-dimensional continuous spaces and geographic spaces for agent groups, with placement,
  neighbour queries, movement, validation, trajectories, and visualization of agents in them.

### Modified Capabilities

- `model-library`: boids in three dimensions and the Kauai population model on geographic data.

## Impact

- `skcomplex.core.space` gains `GeoSpace`, and `SpaceKind` in `skcomplex.core.base` gains the geographic kind.
  `skcomplex.core.components.Group` gains `from_layer`.
- The Mesa backend in `skcomplex.core.backends` maps continuous spaces of three dimensions to Mesa's
  `ContinuousSpace` and `GeoSpace` to `mesa_geo.GeoSpace`, declares support for both, and its agent view gains
  `features_here` and `move_to_feature`.
- `skcomplex.core.viz.AgentsVisual` draws three-dimensional and geographic agents.
- New optional extra `geo` with `mesa-geo`. The Project Profile lists it, and states that the extra needs
  macOS 14 on Apple silicon and macOS 15 on Intel, the oldest systems with pyproj wheels for Python 3.13.
- `skcomplex.models.biology.collective_behaviour` gains three-dimensional boids, and
  `skcomplex.models.social.organizations` is created with the Kauai model and a small synthetic island dataset.
