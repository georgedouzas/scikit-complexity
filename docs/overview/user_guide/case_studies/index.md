# Case studies

The case studies take one system of the catalog through the whole workflow, from selecting and checking it to
simulating, analysing, evaluating or calibrating, experimenting, and drawing a figure. Each study comes from a different
field and tests the model against data or a known result, which the page cites. The data enter as short inline arrays,
so the pages need no download, and every page runs in under a minute with small budgets and fixed seeds.

- [An epidemic calibrated to case data](epidemic_calibration.md) fits the SIR model to the daily counts of an influenza
  outbreak in an English boarding school in 1978, reported in the British Medical Journal. It checks the fit and the
  coverage of the posterior, asks whether the counts determine the two rates, and compares scenarios of earlier
  isolation.
- [An opinion model swept and analysed](opinion_sweep.md) sweeps the bounded confidence model of Deffuant et al. over
  its confidence bound, with an ensemble of runs per value. It counts the opinion clusters against the known result
  `1 / (2 d)` and measures how the final consensus depends on the bound and the convergence rate.
- [A lattice phase transition](ising_transition.md) sweeps the two-dimensional Ising model over temperature on lattices
  of three sizes. It finds the peak of the susceptibility near the exact critical temperature of Onsager and estimates
  the critical exponents by finite-size scaling.

The workflow tutorials explain each step on its own, such as [Simulate](../simulate.md), [Calibrate](../calibrate.md),
and [Run experiments](../run_experiments.md).
