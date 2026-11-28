## 1. Systems (#33, #34, #35, #38, #44)

- [x] 1.1 Correct the exponents in the `SimonYuleSystem` docstring, with a test of the rank slope
- [x] 1.2 Document the Poisson draw times of `PolyaUrnSystem` and add `time='discrete'`, with tests
- [x] 1.3 Add the offspring laws and `individuals=False` to `GaltonWatsonSystem`, with tests, and say in its
      docstring which seed drives its runs
- [x] 1.4 Profile `WestBrownEnquistSystem` over depths and fix the step that grows faster than the vessels, with a
      test kept out of the quick tests if it is slow
- [x] 1.5 Document how the threshold of `ContactProcessSystem` depends on `tau`

## 2. Analysis (#36, #37, #39, #40, #41, #43)

- [x] 2.1 Report the S0 location and a skewness of 0 at index 2 in `fit_law(law='stable')`, tested in
      `tests/core/analysis/test_fit.py`
- [x] 2.2 Silence mrestimator in `events(kind='branching')` and pass its messages on as warnings, tested in
      `tests/core/analysis/test_events.py`
- [x] 2.3 Raise for thin grids and count points in box counting, tested in `tests/core/analysis/test_scaling.py`
- [x] 2.4 Make the default fluctuation windows unique: they already are with the pinned nolds 0.6.3, whose
      logarithmic windows drop duplicates, so no change was needed
- [x] 2.5 State the denominator of the percolation strength in the `clusters` docstring

## 3. Trajectories (#42)

- [x] 3.1 Return one array per frame from `get_values` for per-agent states of groups whose size changes, tested in
      `tests/core/engine/runtime/test_trajectories.py`

## 4. Close

- [x] 4.1 Run the fast checks and the tests of the changed modules
- [x] 4.2 Close issues #33 to #44 with a comment naming the commit
