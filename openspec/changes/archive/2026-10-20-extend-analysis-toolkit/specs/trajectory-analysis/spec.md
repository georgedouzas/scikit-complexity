# Spec Delta

## MODIFIED Requirements

### Requirement: Heavy tails are fitted and tested
The analysis layer SHALL compute log-binned histograms, complementary cumulative distributions, maximum-likelihood
power-law fits of continuous or discrete samples with their lower cutoff, an optional upper cutoff, and bootstrap
goodness of fit, comparisons with exponential, log-normal, stretched exponential, and truncated power-law tails, Hill
estimates, and Gini coefficients.

#### Scenario: Recover an exponent
- **WHEN** a user fits 10,000 samples drawn from a power law with exponent 2.5 above 1
- **THEN** the fitted exponent lies within 0.1 of 2.5 and the goodness-of-fit p-value exceeds 0.1

#### Scenario: Discrete samples
- **WHEN** a user fits 10,000 integer samples drawn from a discrete power law with exponent 2.5 above 1
- **THEN** the fitted exponent lies within 0.1 of 2.5 with the exact discrete likelihood

#### Scenario: Truncated tail
- **WHEN** a user compares the fits of 10,000 samples of a power law with exponent 1.5 and an exponential cutoff at 100
- **THEN** the truncated power law is preferred over the pure power law with a p-value below 0.05

#### Scenario: Gini of equal wealth
- **WHEN** a user computes the Gini coefficient of equal values
- **THEN** the result is 0

### Requirement: Time-series measures characterise memory and diffusion
The analysis layer SHALL compute autocorrelation, power spectra, mean squared displacement over all lags with its
exponent over a chosen range, detrended fluctuation of any order, Hurst exponents by detrended fluctuation or rescaled
range, and rolling variance, autocorrelation, skewness, and spectral reddening as early-warning indicators.

#### Scenario: Brownian motion
- **WHEN** a user computes the mean squared displacement of 1000 simulated Brownian particles
- **THEN** the scaling exponent lies within 0.05 of 1

#### Scenario: White noise memory
- **WHEN** a user computes the Hurst exponent of white noise
- **THEN** the result lies within 0.05 of 0.5

#### Scenario: Persistent and integrated noise
- **WHEN** a user computes the detrended fluctuation exponent of fractional Gaussian noise with Hurst exponent 0.7 and
  of a random walk
- **THEN** the first lies within 0.05 of 0.7 and the second within 0.05 of 1.5

#### Scenario: Same early warning on two tipping points
- **WHEN** a user computes early-warning indicators of a noisy harvested population approaching its fold and of a noisy
  lake phosphorus model approaching its fold
- **THEN** in both the Kendall trends of variance and lag-1 autocorrelation exceed 0.5

### Requirement: Information measures quantify structure
The analysis layer SHALL compute Shannon entropy, Kullback-Leibler divergence, block entropies, entropy rate, excess
entropy, mutual information, transfer entropy, permutation entropy with an embedding delay, and Lempel-Ziv complexity,
from symbols or from series symbolised by bins, quantiles, or ordinal patterns of any length.

#### Scenario: Periodic sequence
- **WHEN** a user computes the entropy rate and excess entropy of the repeated sequence `0011`
- **THEN** the entropy rate is 0 and the excess entropy is 2 bits

#### Scenario: Direction of information flow
- **WHEN** a user computes transfer entropy between a random binary series `x` and `y` that copies `x` with a lag of one
- **THEN** the transfer entropy from `x` to `y` is close to 1 bit and from `y` to `x` close to 0

#### Scenario: Long ordinal patterns
- **WHEN** a user symbolises white noise with ordinal patterns of length 12
- **THEN** two distinct rank patterns never share a symbol

#### Scenario: Lempel-Ziv of a long sequence
- **WHEN** a user computes the normalized Lempel-Ziv complexity of 10^5 fair coin flips
- **THEN** the result lies within 0.1 of 1 and the call returns within one second
