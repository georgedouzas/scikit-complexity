# Proposal

## Why

Agents today live on a grid, on a network, or in a two-dimensional rectangle. Two models in the library inventory need
more:
- Downey's boids fly in three dimensions (Think Complexity ch. 10).
- Zanbaka et al. place one agent per resident of Kauai from census shapefiles. Agents live and work at points on the
  map and move to favourite places drawn from another shapefile (Carmichael et al. 2019 ch. 3).

Both models are recorded as blocked on this change, and neither can be written in the model package without a space
the core does not have. Mesa, already a dependency, has an N-dimensional continuous space. Its companion project
`mesa-geo` adds geographic spaces on top of GeoPandas and Shapely. The core only needs the adapters.

## What Changes

- Extend the continuous agent space to three dimensions with an optional depth, so positions, radius neighbour queries,
  toroidal wrapping, and random placement work on three coordinates. Two-dimensional spaces keep working unchanged.
- Add a geographic agent space. It has a coordinate reference system, and vector layers loaded from GeoPandas
  data frames or files. Agents are located at geometries in that system.
- Let agent groups be created from the features of a vector layer, one agent per feature or a given number of agents
  per feature. Feature attributes become initial state values, and the agents are placed at points inside the feature.
- Let rules query a geographic agent's neighbours within a distance in the space's units, find the layer features
  that contain the agent, and move the agent to a point or to a feature.
- Report agent locations in three-dimensional and geographic spaces in trajectories, and draw them in Rerun as
  `Points3D` and as projected `Points2D` over the layer outlines.
- Add the `geo` optional extra, which installs `mesa-geo` and with it GeoPandas, Shapely, pyproj, and rasterio.
- Add the library models this unblocks: boids in three dimensions and a Kauai population sentiment model on
  geographic layers.

## Capabilities

### New Capabilities

- `agent-spaces`: three-dimensional continuous agent spaces and geographic agent spaces, with placement, neighbour
  queries, movement, trajectories, and visualization of agents in them.

### Modified Capabilities

- `model-library`: boids in three dimensions and the Kauai population model on geographic data.

## Impact

- `skcomplex.core.model` gains the `depth` argument of `ContinuousSpace` and a new `GeoSpace`. The Mesa stepper in
  `skcomplex.core.solvers` maps them to Mesa's `ContinuousSpace` and to `mesa_geo.GeoSpace`. `AgentView` gains
  `features_here` and `move_to_feature`.
- `skcomplex.core.viz.AgentsVisual` draws three-dimensional and geographic agents.
- New optional extra `geo` with `mesa-geo`. The Project Profile lists it, and states that the extra needs
  macOS 14 on Apple silicon and macOS 15 on Intel, the oldest systems with pyproj wheels for Python 3.13.
- `skcomplex.models.biology.collective_behaviour` gains three-dimensional boids, and
  `skcomplex.models.social.organizations` is created with the Kauai model and a small synthetic island dataset.
