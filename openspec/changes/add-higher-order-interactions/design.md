# Design

## Context

Roles bind interactions to pairs of components, groups vectorise laws over members, and networks give neighbourhoods. The pending `add-link-fields` adds cell complexes with states on links and faces, and `add-adaptive-networks` adds network generators and edge states. None declares a law over a hyperedge.

## Goals / Non-Goals

**Goals:**
- Declare laws over hyperedges of any size, with the same symbolic and compiled path as pairwise laws.
- Let a node gate a link, so structure and dynamics regulate each other.

**Non-Goals:**
- Persistent homology of hypergraphs, which belongs to analysis libraries.

## Decisions

### Hypergraphs through XGI

XGI (CompleXGroupInteractions) is the maintained library for hypergraphs and simplicial complexes, with generators, statistics, and incidence matrices. The space wraps it, as `NetworkSpace` wraps NetworkX; HyperNetX is the fallback the survey compares.

### k-body laws lower to gathers

A hyperedge interaction compiles to an index table of members per hyperedge, so a k-body law is a gather and a scatter-add, the pattern pair interactions already use. Reductions over the hyperedges of a node use the incidence matrix.

### Triadic regulation

A regulation maps a node state to the activity of a link, applied between steps as an update of the link state, so percolation alternates with the regulation as in Millán et al.

## Risks / Trade-offs

- [Large hyperedges make gathers wide] → Hyperedges are grouped by size, one index table per size.
