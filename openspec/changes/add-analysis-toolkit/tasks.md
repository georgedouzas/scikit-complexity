# Tasks

## 1. Library

- [x] 1.1 Add the `nonlinear` module with Lorenz, Rössler, Hénon, logistic map, Duffing, Van der Pol, and Hénon-Heiles
  factories, and verify the strange attractor scenario and the conformance suite

## 2. Dynamics

- [x] 2.1 Add `fixed_points`, `jacobian`, `stability`, `nullclines`, and `divergence`, and verify the Lotka-Volterra,
  logistic map, and volume preservation scenarios
- [x] 2.2 Add `lyapunov_spectrum` for flows and maps, and verify the Lorenz and logistic map scenarios
- [x] 2.3 Add `orbit_diagram` and `basin_stability` over experiments, and verify the period doubling scenario

## 3. Statistics, events, and series

- [x] 3.1 Add histograms, distributions, power-law fitting with comparisons, Hill estimates, and Gini, and verify the
  exponent recovery and Gini scenarios
- [x] 3.2 Add event detection with sizes, durations, and waiting times, and verify the avalanche scenario
- [x] 3.3 Add autocorrelation, spectra, mean squared displacement, DFA, and early-warning indicators, and verify the
  Brownian motion and white noise scenarios

## 4. Information

- [x] 4.1 Add symbolisation and the information measures, and verify the periodic sequence and information flow
  scenarios

## 5. Spatial and networks

- [x] 5.1 Add cluster labelling, correlation functions, and box counting, and verify the percolation and Sierpinski
  scenarios
- [x] 5.2 Add network measures on space snapshots, and verify the degree distribution of a Barabási-Albert graph has an
  exponent within 0.3 of 3

## 6. Documentation and gate

- [x] 6.1 Document the toolkit by submodule in the user guide with one example per family, and verify the documentation
  test passes
- [x] 6.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
