# Design

## Context

Variable groups on the sites, edges, and hyperedges of a network run with capacity rows, and topology requests change
their rows between steps through `TopologyKernel`. A variable group on hyperedges takes its width from the initial
hyperedges, all of one size. Neighbourhood reductions, `neighbors` and `components`, run in `NeighborhoodKernel` on the
pairs of the present edges. Agents on a `NetworkSpace` sit on Mesa network cells and read `agents_here(group)` and
`neighbors()`.

## Goals / Non-Goals

**Goals:**
- Name the new parts after the mathematics: shortest paths, rank of a hypergraph, autocatalytic sets.
- Extend existing concepts (agents on nodes, topology requests, neighbourhood reductions) rather than add parallel ones.

**Non-Goals:**
- Compiled member sets in symbolic laws, laws per hyperedge on groups of several sizes, and a Numba betweenness.

## Decisions

- **Member sets are agents on nodes.** The audit proposed an owner index on one variable group. Agents of a variable
  group placed on network nodes already carry it as their location, and `create`, `remove`, and `agents_here` already
  act on it. The missing read is the collections of the neighbouring nodes, so `neighbors` takes `group`. A compiled
  owner index read through reductions would duplicate this for symbolic laws, which no probe needs.
- **Rank of a variable group on hyperedges.** `GroupElement(rank=k)` reserves `k` member columns, and a smaller
  hyperedge fills the rest with -1, which `_remove_node`, snapshots, and `AutocatalyticSetAnalysis` read. The gather
  tables of laws per hyperedge assume one size, so a hyperedge interaction on a ranked group raises `RoleError`.
  Alternative: one variable group per size, which would turn the hyperedges of `NetworkRowsSpec` and of snapshots into
  mappings and touch every reader.
- **Hyperedge requests.** `'add-hyperedge'` runs for a node row and `'remove-hyperedge'` for a hyperedge row, the rows
  `'hyperedge'` of `CompiledRequestSpec` that `TopologyInteraction` gets by binding the group on hyperedges as its target.
  A new node with a hyperedge keeps the edge-anchored attachment when it draws edges, and with `edges=0` or
  `independent` draws every member by its weights, since no edge then joins the first two members.
- **Independent and directed ends.** TAP assigns catalysis with probability `pc` to every ordered pair at the creation
  of the later good, a random number of edges both ways. `independent=True` joins every candidate with the weight as
  its probability, and `direction` chooses `'out'`, `'in'`, or `'both'` on directed layers.
- **Shortest paths wrap NetworkX.** NetworkX (3.6, released in 2025 and actively maintained, already a dependency)
  gives betweenness and closeness. `NeighborhoodKernel` keeps the last active links and their values in `paths_`, so a
  map and a condition that read the reduction in one step, or steps without a change, cost one computation. The Numba
  backend reports the reduction as unprinted, so laws that read it run in Python with a `BackendWarning`, the existing
  fallback.
- **TAP in continuous time.** Algorithm 1 draws Poisson numbers of goods per step. The system fires a jump per new good
  at the rate `alpha^i C(M, i)` from the current count, with Gillespie, which the docstring states.
- **RAF analysis in the core.** The maximal RAF is a fixed point on catalytic reaction networks, general to chemistry
  and economics, and no maintained Python library computes it, so `AutocatalyticSetAnalysis` implements the
  Hordijk-Steel reduction in a few lines.

## Risks / Trade-offs

- The Mesa urns are slow for long runs on large networks, so the gallery runs 150 steps on the karate club.
- Eq. (4) of Iacopini et al. is a mean-field estimate, about 15 percent above the simulated novelties on the karate
  club after 300 steps, so the tests compare the isolated urn exactly and only the ranking on a star.
