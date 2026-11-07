# Design

## Context

A `TopologyRequestSpec` is the target of an update or jump law, and `TopologyKernel` applies the fired requests to the
capacity rows of the variable groups of a network between steps. A request runs for every node, edge, or hyperedge
row its law runs for, or once per step for `'add-node'` without a target. The weights `toward` compile into a kernel
over the node rows that reads the node a new edge starts from as `source`, set apart by `TowardsPass`.
`TriadicRegulationInteraction`, in `systems/mathematics/networks`, reads one regulator per link as the third member of
a hyperedge, and the sign of the regulator from the link, and it cannot read several regulators per link.

## Goals / Non-Goals

**Goals:**
- Name the new requests after the graph operation and keep every option general, so the probe systems only compose
  them.
- Keep the existing requests and their systems unchanged in behaviour, except the pairing of `'replace-node'`.

**Non-Goals:**
- Choice among candidates for requests other than `'add-edge'`, and scores that read reductions of the source node.
- Topological data analysis of the spatial patterns, which the paper does with giotto-tda.

## Decisions

- **Choice is an option of `'add-edge'`.** `candidates` and `score` choose the edge of the smallest score among drawn
  candidates, a maximum being the minimum of the negated score. The score compiles like `toward`, as a kernel over the
  node rows that reads the start of the edge as `source`, so `_WeightSpec` names which of the two it holds. A kind
  `'choose-edge'` would duplicate the drawing and the values of `'add-edge'`. Without a target `'add-edge'` runs once
  per step and draws every candidate as two distinct unjoined nodes, by rejection among uniform pairs, as Achlioptas
  et al. draw their edges.
- **Triadic closure as a kind.** `'close-triangle'` draws a pair of unjoined neighbours uniformly by rejection among
  the pairs of neighbours, after counting the joined pairs from the present edges. Common-neighbour counts in `toward`
  would need a reduction over pairs of neighbours, which the neighbourhood kernels do not compute, and the closing
  edge is requested by neither end. The fallback of Davidsen et al. for a node with fewer than two neighbours is the
  join of `'add-edge'`, also for a node whose neighbours are all joined, so a closure always adds one edge, as the
  balance `1 = gamma (k - 1)` of Thurner et al. assumes, and `edges=0` turns the fallback off.
- **Contraction runs for a node.** Kim et al. draw a node and one of its neighbours, so `'contract-edge'` runs for a
  node and draws the neighbour by `toward` among its neighbours. The edges of the neighbour move to the node on every
  layer, an edge that would repeat a link or close a loop is freed, and the neighbour then leaves like a removed node.
- **Duplication copies values and links.** `'duplicate-node'` writes the values of the node into the copy, copies each
  edge with its values with probability `inherit`, joins the node with probability `parent`, and joins further ends
  among the nodes joined to neither, as a new node does. The two probabilities are numbers of the request, the options
  of the duplication-divergence family.
- **One drawn member.** `one=True` draws one present row uniformly per step and fires the request there with the
  probability of the law. It applies to update laws, since a jump channel already belongs to one row, and the pass
  raises `ValidationError` for a jump. Flags drawn by maps lag one step and reductions over free rows break the choice.
- **`'replace-node'` is one request.** The interaction expanded a replacement into an addition and a removal decided
  apart, which fire together only with probability one. One kind removes and adds in one request. The condition keeps
  reading the replaced node and the weights the new one.
- **Self-contained system modules.** Every new system defines its elements, interactions, and helpers in its own
  module, so it reads as an example. `spatial_triadic_percolation` has its own regulation and percolation
  interactions rather than extending `TriadicRegulationInteraction`, which `triadic_percolation` keeps unchanged. The
  three systems of node turnover each define their own node element with its degree, since the `common` package of
  mathematics holds families of realizations and a new family would be a new package.
- **Several regulators in one hyperedge.** Laws per hyperedge write the link through the edge of the first two
  members, so several hyperedges per link would write one link several times. One hyperedge per link lists its
  negative and then its positive regulators, the link holds their numbers, and further members pad the hyperedges to
  one size and regulate nothing. One size keeps one kernel. Hyperedges of several sizes would write the link from one
  kernel per size, combined update laws that the Numba backend scatters wrongly through gather tables today.
- **Links without positive regulators.** They need only their negative regulators inactive, which keeps the one
  regulator model of Sun et al. and changes the spatial model only for links without positive regulators, about
  `exp(-2 pi rho c+ dr²)` of them, below 0.1 percent at the paper's parameters.
- **Spatial network with NumPy.** NetworkX's Waxman graph uses the distance only through the largest distance and
  iterates over the pairs in Python, so the system draws the links of Eq. 1 and the regulators of Eq. 2 with NumPy on
  the torus distances, a node regulating a link with probability `c+ e^{-d/dr}` or `c- e^{-d/dr}`, never both.
- **Superposition per subgraph.** The book's step draws a linking probability per attempted link, which gives an
  Erdős-Rényi graph of the mean probability. The user decided on one probability per subgraph, so the network is the
  union of Erdős-Rényi subgraphs of `n_nodes` nodes, each with a probability drawn from `Q(p) ~ p^-exponent`, built
  with NetworkX, whose degrees mix Poisson laws as Eq. 4.34 states.

## Risks / Trade-offs

- [The requests run in Python between steps] → the probe systems run a few thousand steps on networks of a few
  thousand nodes, and the gallery examples stay smaller.
- [Padding makes hyperedges as wide as the most regulated link] → the gathers grow with the largest number of
  regulators, about 35 at the paper's parameters, which costs less than one kernel per size.
- [Rejection sampling of pairs] → it stops after a bounded number of draws and then adds no edge, which happens only on
  networks close to complete.
