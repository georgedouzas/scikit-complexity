# Design

## Context

Network analysis wraps NetworkX for centralities, communities, and motifs. Experiments sweep parameters and collect results, so a sweep over system size gives the data a scaling fit needs.

## Goals / Non-Goals

**Goals:**
- Measure networks across scales.
- Fit scaling laws across sizes from experiment sweeps.

**Non-Goals:**
- Embedding real networks in hyperbolic space by maximum likelihood, which dedicated tools such as Mercator do.

## Decisions

### Renormalization

Laplacian renormalization follows Villegas et al.: diffusion at time tau defines blocks of nodes that merge. Geometric renormalization merges consecutive nodes on the similarity circle of an embedded network. Both return the network at each scale with its degree distribution and clustering.

### Scaling fits

A scaling fit regresses the logarithm of a quantity on the logarithm of size, with bootstrap confidence intervals, reading the sweep through the experiment results.

## Risks / Trade-offs

- [Renormalization is costly on large networks] → Diffusion uses sparse matrix exponentials over a few eigenvectors.
