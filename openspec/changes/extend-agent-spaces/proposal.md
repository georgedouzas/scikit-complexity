# Proposal

## Why

Two models of the inventory are blocked on agent spaces:
- Downey's boids fly in three dimensions (Think Complexity ch. 10).
- Zanbaka et al. place one agent per resident of Kauai from census regions. Agents live and work at points on the map
  and move to favourite places drawn from another layer of regions (Carmichael et al. 2019 ch. 3).

The core already gives agents everything but the space, so this change builds on it and drops it from its scope:
`ContinuousSpace(extent, boundary=...)` takes any number of sides, `MovementInteraction` moves agents by Gaussian,
Lévy, best-cell, or user steps with a vision and the `'member'` schedule, and the agent forms of `MapInteraction`,
`TransitionInteraction`, `TransferInteraction`, and the other agent interactions act on the members of a group through
the `AgentView` and `ModelView` of `MesaSolver`. Two gaps remain. `MesaSolver` builds Mesa's continuous space from the
first two sides of the extent alone, and places, measures, and reports agents in two dimensions. And no space holds the
regions of a map, so an agent cannot be placed in a region, ask which regions contain it, or move to one.

## What Changes

- Let `MesaSolver` run agent groups in a `ContinuousSpace` of three sides: Mesa's bounds, random placement, periodic
  wrapping, radius queries, `ModelView.dimensions`, and the agent locations of a trajectory cover every side. Two
  dimensional spaces keep their results.
- Reject before the first frame a continuous space on `MesaSolver` with more than three sides or no extent, and report
  in the compile report an initial location whose length differs from the dimension of its space.
- Add `regions` to `ContinuousSpace`, a mapping from a layer name to a sequence of Shapely polygons in the coordinates
  of the space. `prepare` indexes them with Shapely's `STRtree`.
- Let a location of a member, in `GroupElement(location=...)`, in `ModelView.create`, and in `AgentView.move_to`, be a
  Shapely polygon, which places the agent at a point drawn uniformly inside it from the simulation's generator, over
  the area-weighted triangles of Shapely's `constrained_delaunay_triangles`.
- Add `AgentView.regions_here(layer)`, the indices of the regions of a layer that contain the agent, and
  `ModelView.region(layer, index)`, the polygon of a region, so a `MovementInteraction` step moves an agent to a region
  and an agent-form interaction reads where it stands.
- Draw agents in a three-dimensional space as `Points3D`, and the regions of a space once as static `LineStrips2D`
  under the agents' `Points2D`.
- Add the library models this unblocks, built on `MovementInteraction` and agent-form interactions: boids in three
  dimensions and a Kauai population sentiment model on region layers, with a small synthetic island in its package.

## Capabilities

### New Capabilities

- `agent-spaces`: three-dimensional continuous spaces and region layers for agent groups, with placement, neighbour
  and region queries, movement, validation, trajectories, and visualization of agents in them.

### Modified Capabilities

- `model-library`: boids in three dimensions and the Kauai population model on region layers.

## Impact

- `skcomplex.core.spaces.ContinuousSpace` gains `regions` and the derived `regions_` index. No space class and no
  dependency are added, since Shapely is already a runtime dependency; its minimum rises from 2.1 to 2.2 for
  `constrained_delaunay_triangles`, which places agents uniformly in a polygon.
- `MesaSolver` in `skcomplex.engine.solvers` builds Mesa's continuous space from every side, rejects unsupported
  continuous spaces in `_unsupported`, places agents in polygons, and its views gain `regions_here` and `region`.
- `PassCompiler` gains the issue `location-dimension`. `SpaceError` widens to a location that does not fit its space.
- `AgentsVisual` in `skcomplex.engine.viz` draws three-dimensional agents and region outlines.
- `skcomplex.models.biology.collective_behaviour` gains three-dimensional boids, and
  `skcomplex.models.social.organizations` is created with the Kauai model and a small synthetic island dataset.
