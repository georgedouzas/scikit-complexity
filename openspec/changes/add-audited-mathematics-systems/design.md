# Design

## Context

The audit judged each listed model supported by the current core. Maps iterate with `MapInteraction` or update laws,
flows with rate laws, the delay equation with `delayed`, random maps with `bernoulli` draws, cascades of balls with
`LifeCycleInteraction`, observed growth with a renewal `JumpLaw` beside a rate law, and networks with topology
requests, neighbourhood reductions, components over layers, and NetworkX constructions.

## Goals / Non-Goals

**Goals:**
- Follow the equations, parameters, and defaults of the cited sections, and test a known result of each.
- Reuse existing elements, helpers, and systems, and add an option where the source presents a variant.

**Non-Goals:**
- Core changes. Models that need one are left out and listed below.

## Decisions

- **Separate systems for named maps.** The Lozi map, the conservative Hénon map, and the hyperchaotic Rössler flow
  have their own equations and reference parameters, so they are systems of their own rather than options of
  `henon_map` and `rossler`, whose defaults would not fit them.
- **Fitness is an option of Barabási-Albert growth.** Bianconi-Barabási attachment changes only the weight of new
  ends to `eta k` and draws `eta` for every node, so `barabasi_albert(fitness=True)` uses `FitnessVertexElement`.
- **CDN map follows Fig. 3.17 of Dilão.** Exercises 3.4 and 3.5 write `exp(-alpha² x² / 2π)`, but the chaotic band
  of the bifurcation diagram at `alpha = 2.8` belongs to `exp(-alpha² x²) + beta`, which the system implements.
  The docstring states the rescaling between both forms.
- **The conservation variant of cascading SSR is `fragmentation`.** Splitting a ball into two whose states sum to its
  own is the existing fragmentation system, so the cascade keeps the multiplication rule only.
- **Graphlet arrival in one step.** The first new node attaches by degree and the second, in the same step, to it with
  probability `alpha` or to its contact otherwise, read from the neighbours of the node of age 0.
- **Static networks.** `kronecker_graph` and `configuration_model` build their graphs with NetworkX and NumPy, as
  `network_structure_analysis` does, and carry the degree and clustering of every node. The configuration model
  rewires self-loops and multilinks crosswise, after Thurner et al. section 4.5.1.1.
- **Interdependent networks share their nodes.** A node stands for a dependent pair, the first network is the graph
  of the space and the second a layer, and the cascade alternates the giant components of both by the parity of the
  step.

## Left out

- `explosive_percolation`: the Achlioptas product rule compares two candidate edges by the sizes of the components
  they join and adds one of them. Topology requests draw the ends of one edge by weights and cannot choose among
  candidate edges.
- Davidsen-Ebel-Bornholdt network: triadic closure links two neighbours of a drawn node, an edge that neither end
  requests, and weights of common neighbours of the requesting node are not readable in `toward`.
- Node-merging network of Kim et al.: merging two linked nodes into one that inherits their links needs an edge
  contraction request.
- Duplication-deletion network of Jensen: one node removed and one parent duplicated per step need a request that acts
  on exactly one uniformly drawn node. Flags drawn by maps lag one step and reductions over inactive rows break the
  choice.
- Network superposition model of Thurner et al. section 4.5.2: the stated algorithm, a linking probability drawn for
  every attempted link, gives an Erdős-Rényi graph of the mean probability, which contradicts Eq. 4.34. The model
  needs a decision on which reading to implement.
- Weighted geometric soft configuration model (WSD): its renormalization flow needs the geometric renormalization of
  networks that `add-multiscale-networks` blocks, and the source states no weight rule.
- Spatial triadic percolation: not attempted in this change. Its links carry several signed regulators, so the
  regulation counts the active positive and negative regulators of every link, which `TriadicRegulationInteraction`
  cannot do for more than one regulator. Counts incremented from a variable group on signed regulatory hyperedges may
  express it in the current core, a follow-up to try.
