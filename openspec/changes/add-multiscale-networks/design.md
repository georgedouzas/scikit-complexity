# Design

## Context

`TopologyInteraction` draws the new ends of a node it adds with probability proportional to `toward(namespace)`, a
function of the namespace of a candidate node, and gives the new node its `values`, each a value or a function of the
random number generator. The weight cannot read the new node, so a node cannot attach by its distance to the others,
which popularity-similarity growth needs. Analyses are classes of `skcomplex.analysis` with `check`, `prepare`, and
`run`, and a family that computes one kind of quantity is one class with a `kind` setting. `NetworkAnalysis` reads the
graph of every frame of a trajectory, or a graph, and `ScalingAnalysis` already reads a mapping from size to samples for
its `'variance-mean'` kind.

## Goals / Non-Goals

**Goals:**
- Grow networks whose nodes attach by hyperbolic distance with the general topology interaction.
- Measure networks across scales.
- Fit scaling laws across sizes from experiment sweeps.

**Non-Goals:**
- Embedding real networks in hyperbolic space by maximum likelihood, which dedicated tools such as Mercator do.
- A static hyperbolic generator. A static graph from any library enters through `NetworkSpace`.

## Decisions

- **Reference numbers are checked before they become thresholds.** The superlinear exponent of the urban scaling
  scenario is first reproduced from the cited source; when a run disagrees, the scenario uses the measured value with
  a tolerance from the spread over seeds, and design.md records the source, the measured value, and why they differ.
  Decided on 2026-11-01.

### Attachment weights that read the new node

In the action `'add-node'`, the namespace that `toward` receives holds the candidate node, as it does today, and the
values drawn for the node being added as `namespace.source`. The condition and the probability of the action read the
same namespace. Existing weights that read only the candidate keep their meaning.

### Popularity-similarity growth on the core

`popularity_similarity_network` follows Papadopoulos et al. (2012) as the review writes it. The nodes form a variable
group on the sites of the network with the states `angle`, drawn uniformly on the circle for every new node, and `age`,
which a `MapInteraction` in the increment mode raises by one every step. A `TopologyInteraction(action='add-node')`
without a target adds one node per step, which joins `m` existing nodes drawn in proportion to the Fermi-Dirac
probability of the hyperbolic distance `x = r_s + r_t + 2 ln(theta_st / 2)` between the candidate and the new node, at
temperature `T`, with the radial coordinates `r = 2 ln(t - age)` faded by the popularity exponent. The probability is
`1 - logistic(x, R, 2 T)` with the core utility `logistic`.

`urban_scaling_interactions` grows each city the same way and adds internal links between existing residents with
`TopologyInteraction(action='add-edge')` toward the same weight, as the extended model of Papadopoulos et al. (2015)
does, so the mean degree grows with the size. Its residents meet each neighbour with a probability per step through a
`MapInteraction` in the agent form that counts the meetings. The two packages belong to different fields, so each writes
the one-line distance of the published model.

### Renormalization

`RenormalizationAnalysis(kind='laplacian' | 'geometric', steps=3, time=None, block=2, angle='angle')` reads a graph or
the network of the last frame of a trajectory, with the node states as node attributes:
- `'laplacian'` follows Villegas et al. (2023): the diffusion operator `exp(-time L)` at the time where the specific
  heat peaks, or at `time`, merges every pair of nodes whose entry is at least the smaller of their diagonal entries.
- `'geometric'` follows García-Pérez et al. (2018): blocks of `block` consecutive nodes in the order of their angle
  merge, and two supernodes link when any of their nodes link.

Both merge with `networkx.quotient_graph` and set `graphs_`, the network at every scale, and `degrees_`, `exponents_`,
and `clustering_` per scale, with the exponents fitted as `NetworkAnalysis(kind='degrees')` fits them. The diffusion
uses `scipy.linalg.expm` below a few thousand nodes and `scipy.sparse.linalg.expm_multiply` above. No maintained library
on PyPI implements either renormalization, so the merge rules are written in house on NetworkX and SciPy.

### Scaling fits

`ScalingAnalysis(kind='allometric')` reads a mapping from size to samples of a quantity, such as the results of a
`GridDesign` sweep over size, or two columns. It regresses the logarithm of every sample on the logarithm of its size
with `scipy.stats.linregress`, sets `exponent_`, and sets `interval_` from `scipy.stats.bootstrap` over the pairs, with
the new settings `confidence` and `resamples` and the existing `random_state`.

## Risks / Trade-offs

- [Renormalization is costly on large networks] → The sparse exponential acts on vectors above a few thousand nodes.
- [The superlinear exponent depends on the internal links] → The scenario fixes the rate of internal links, and the user
  guide shows the exponent against it.
