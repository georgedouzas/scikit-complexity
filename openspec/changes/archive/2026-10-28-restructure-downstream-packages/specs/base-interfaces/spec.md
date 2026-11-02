## MODIFIED Requirements

### Requirement: Configurable bases share three verbs
Every base on `BaseEstimator` SHALL define `check()`, `prepare(context)`, and `run(input, final=False)` once, each calling
a private hook a child overrides, with trivial defaults, so every configurable object answers all three. This holds for
the bases of the engine, of analyses, of designs, and of experiments. No configurable class SHALL add a public verb
beyond its constructors and the extension hooks a user implements.

#### Scenario: One vocabulary
- **WHEN** a user calls `check()`, `prepare(...)`, and `run(...)` on a space, a solver, a visual, an observer, a
  design, a surrogate, an analysis, and an experiment
- **THEN** every call works, and none of `initialize`, `advance`, `lower`, `on_frame`, `compile`, `forecast`, `fit`, or
  `compute` exists

#### Scenario: A new child through hooks
- **WHEN** a user writes a space that implements `_prepare` setting `size_`, `n_dim_`, and `positions_`, and `_run`
  returning neighbours
- **THEN** the space is checked, prepared, and run through the base's verbs, and the runtime uses it
