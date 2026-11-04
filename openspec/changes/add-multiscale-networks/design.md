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

### Implementation notes and measured values

Recorded on 2026-11-04 when the change was applied.

- **Urban exponent.** Ribeiro and Netto (2024) state that socioeconomic exponents are distributed around 1.15. The
  model at `internal=0.002` over the sizes 100 to 800, with three city seeds and three experiment seeds, measures
  1.16, 1.11, and 1.13. The 95% interval contains 1.15 for two of the three and ends at 1.148 for the other. It
  measures from 0.98 to 1.03 without internal links, and 1.26 at `internal=0.005`. The scenario stays "above 1", which
  every run at `internal=0.002` meets, with intervals that start above 1.06.
- **Hyperbolic degrees.** The scenario asked for a fit within 0.2 of 2.5. With `edges=2` and `temperature=0.2` at 2000
  nodes, `NetworkAnalysis('degrees')` measures from 2.67 to 2.72 over seven seeds, mean 2.70, which straddles the
  limit of 2.7. Plain shifted preferential attachment, `Pi(k) ~ k + m (gamma - 2)` on the links received, which
  Papadopoulos et al. (2012) show has the same degree distribution, fits a mean of 2.60 with a spread from 2.39 to
  2.78 over eight seeds, so the fit at this size reads above the exponent of the law. The scenario uses the measured
  value, 2.70 within 0.06, three times the spread over seeds.
- **Exponents away from 2.5.** The gap to the reference grows as the exponent nears 2. At the exponent 2.2 the model
  fits 2.78 with a largest degree near 360, and the reference fits 2.19 with a largest degree from 318 to 556. At the
  exponent 3 the model fits 2.89 and the reference 2.68. An independent NumPy implementation of Papadopoulos et al. (2012), with two edges, temperature 0.2,
  and 2000 nodes, fits the same values through `NetworkAnalysis('degrees')` over three seeds: 2.78 to 2.79 at the exponent
  2.2, 2.71 to 2.73 at 2.5, and 2.88 to 2.96 at 3, with the same largest degrees. The model does not differ from the
  published rule. The fit at this size follows the exponent weakly, so the largest degrees are the better guide.
- **Self-similar network.** The exponent stays within 0.18 of the original at three scales for the three seeds tested.
  The fit at one scale varies by about 0.1 over seeds, so the limit of 0.2 holds for these seeds and is not a margin.
- **Peak of the specific heat.** The two blocks of the Laplacian scenario have two peaks, as the review states for
  block models. The taller peak, at the time 0.2, merges nothing, and the merge happens at the second, at the time 27.
  The kind takes the smallest peak at which at least one pair of nodes joins, and a network with no such peak stays as
  it is.
- **Supernode angle.** A geometric supernode takes the mean angle of its nodes. The review weighs it by the hidden
  degrees, which an analysis of a graph does not have. The mean keeps the order of the blocks, which the next scale
  reads.
- **Meetings.** A `MapInteraction` in the agent form cannot share a system with `TopologyInteraction`, since no form
  takes `RuleLaw` next to `UpdateLaw` (`laws-mixed`), and a partition cannot help because both target the residents.
  The meetings are a symbolic map with the draw `binom(degree, meeting)`, which has the same distribution as one
  meeting per neighbour with the given probability. The map replaces the count every step, so `total_meetings` counts
  the meetings of one step. A cumulative count would grow with the time a city took to grow.
- **Mesa.** `MesaSolver` does not run `UpdateLaw` or `TopologyRequestSpec`, so no Mesa path reads the new node. A test
  records the rejection.
- **Targeted additions.** Only `'add-node'` without a target reads the drawn values in its condition and probability.
  An addition with a target keeps reading the node it runs for as `namespace.source`, which its documented contract
  and `'replace-node'` rely on. Its weight reads the new node.
- **Rates and reductions.** A rate cannot read the new node, since no node exists before the jump, and a run raises an
  error. A reduction cannot read it, and compilation raises an error.
- **Weights of edges.** `'add-edge'` and `'rewire'` read the adding node and the kept end as `node.source`, because
  the internal links of `urban_scaling_interactions` need the distance from the adding node.
- **Stopping.** The growth and the internal links of `urban_scaling_interactions` stop at `population`, since a
  `GridDesign` sweep runs every size to one `until`, and a variable group raises an error beyond its capacity.
- **Zero weights.** The existing draw falls back to uniform weights where fewer candidates have a positive weight than
  the edges to draw. It stays, since existing models rely on it, so the attachment scenario holds where a candidate
  lies within the distance.

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
