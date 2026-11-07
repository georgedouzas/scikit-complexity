## Why

The mathematics systems change left out several network models because topology requests cannot choose among
candidate edges, close triangles, contract edges, duplicate nodes, or act on exactly one random node per step, and
because triadic regulation reads one regulator per link (see `add-audited-mathematics-systems`, design, Left out).
These are general graph operations, so the requests gain them and the left-out models become probe systems.

## What Changes

- `'add-edge'` chooses among `candidates` edges the one of the smallest `score`, and without a target runs once per
  step between two nodes drawn uniformly, the Achlioptas process.
- New request kinds: `'close-triangle'` joins two neighbours of a node, `'contract-edge'` merges a node with a
  neighbour into one node that keeps the union of their links, `'duplicate-node'` adds a copy of a node that inherits
  each of its links with a probability, and `'replace-node'` removes a node and adds a new one in one request.
- `one=True` makes a request run for one member drawn uniformly among the members present at every step.
- `TopologyInteraction` takes the new actions and options. Its `'replace-node'` fires one request, so a replacement
  with a probability removes and adds together.
- Links regulated by several signed regulators: the hyperedge of a link lists its negative and then its positive
  regulators, whose numbers the link holds, read by the interaction of `spatial_triadic_percolation`.
- New systems in `mathematics/networks`: `explosive_percolation`, `davidsen_ebel_bornholdt`, `node_merging`,
  `duplication_deletion`, `spatial_triadic_percolation`, and `network_superposition`, each in a module of its own
  that defines its elements, interactions, and helpers, with tests against a known result, an inventory entry, and a
  gallery example.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `adaptive-networks`: topology requests choose among candidates, close triangles, contract edges, duplicate and
  replace nodes, and act on one drawn member per step.
- `model-library`: the new network systems and the regulation of links by several signed regulators.

## Impact

`skcomplex.core.specs.TopologyRequestSpec`, `skcomplex.core.components.interactions.TopologyInteraction`, the topology
kernel and the compiler passes of topology requests, `skcomplex.systems.mathematics.networks`, their tests, the
networks guide, `docs/overview/inventory.toml`, and the gallery of `mathematics/networks`. No new package and no new
dependency.
