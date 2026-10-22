# Design

## Context

Entropies and transfer entropy go through `antropy` and the project's estimators, and `add-emergence-analysis` adds partition coarse-graining and effective information. The new measures build on both.

This design is rebased on `redesign-system-model`. The measures live in `skcomplex.core.analysis` and read trajectories
through the existing adapters, whose observables come from simulations and catalog entries after the redesign. They add
no primitive, solver, or backend, so they need no requirement or support declaration.

## Goals / Non-Goals

**Goals:**
- Measure redundancy and synergy among many variables of any model.
- Test whether a macro feature of a model is emergent by the criteria the literature uses.

**Non-Goals:**
- Integrated information Φ of IIT, which PyPhi computes and which needs exhaustive bipartitions.

## Decisions

### Libraries

`dit` is the maintained library of discrete information theory with partial information decomposition and several redundancy measures, and the toolkit wraps it for discrete data. O-information and its gradients on continuous data use Gaussian estimators, which are closed-form covariance determinants, and the k-nearest-neighbour estimators already in the toolkit.

### Emergence criteria

Ψ, Δ, and Γ follow Rosas et al. (2020) with the minimum mutual information redundancy, so they need only mutual informations, which keeps them cheap; the redundancy is a keyword.

### Closure and ε-machines

A coarse-graining is closed when the macro series predicts its own future as well as the micro series predicts it, compared by conditional entropies. ε-machines are reconstructed from discrete series by the causal-state splitting algorithm.

## Risks / Trade-offs

- [Estimators are biased for short series] → Every measure reports its estimator and accepts a surrogate test.
