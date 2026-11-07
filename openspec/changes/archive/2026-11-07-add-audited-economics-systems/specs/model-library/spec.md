# Spec Delta

## ADDED Requirements

### Requirement: Fishery with measurement errors
The library SHALL provide `fishery_measurement_errors` in `economics/resources`, after Argun et al. (2021, Problem
14.8): independent logistic fish stocks with a noisy growth rate, harvested by the quota `x~ - K / 2` of a measurement
with a Gaussian error, floored at zero unless restocking is allowed, as an Euler-Maruyama map.

#### Scenario: Deterministic quota
- **WHEN** the stocks run without noise
- **THEN** they settle at `K / sqrt(2)` with the quota `K / sqrt(2) - K / 2`

#### Scenario: Extinction
- **WHEN** the growth is noisy and the measurement error has a standard deviation of 0.3
- **THEN** more than 30 percent of the stocks go extinct, and none with restocking

### Requirement: Rice food balance
The library SHALL provide `rice_food_balance` in `economics/resources`, the food security model of Bala et al. (2017,
Section 4.11.3), with its equations and defaults.

#### Scenario: Food security peaks
- **WHEN** the model runs for 100 years
- **THEN** the area approaches `L / (1 + T f)`, and the food security peaks between 10 and 35 years and ends below its
  start

### Requirement: Crop livestock household
The library SHALL provide `crop_livestock_household` in `economics/resources`, the crop livestock model of Bala et al.
(2017, Section 4.11.6), as a yearly map with truncated purchases and sales and a pipeline delay of three years.

#### Scenario: Maturation delay
- **WHEN** the household runs for 12 years
- **THEN** the animal bought in year 1 joins the herd in year 5, and the livestock income starts after five years

### Requirement: Brock-Mirman growth
The library SHALL provide `brock_mirman_growth` in `economics/macroeconomics`, after Caiani et al. (2016, Section
1.1.4), the capital map `K' = alpha beta z K^alpha` with consumption `(1 - alpha beta) Y` and Markov productivity.

#### Scenario: Steady state and stationary mean
- **WHEN** the productivity is constant, or it switches between 0.9 and 1.1 for 4000 periods
- **THEN** the capital reaches `(alpha beta)^(1 / (1 - alpha))`, or the mean of `ln K` is
  `(ln(alpha beta) + E ln z) / (1 - alpha)` within 0.02

### Requirement: Network herding
The library SHALL provide `network_herding` in `economics/macroeconomics`, the choices `c_t = Theta c_t-1 + gamma x_t`
of Caiani et al. (2016, Section 1.1.6) on a weighted network.

#### Scenario: Aggregation through the out-degrees
- **WHEN** the choices run without shocks
- **THEN** the mean choice follows Eq. 1.12 at every step and the choices reach `(I - Theta)^-1 gamma x`

### Requirement: Categorized purchasing power
The library SHALL provide `categorized_purchasing_power` in `economics/macroeconomics`, the three-sector system
`z' = A z + B x`, `y = C z + D x` of Forrest (2018, Sections 11.4 and 22), with an option for the feedback component
that decouples the sectors.

#### Scenario: Decoupling
- **WHEN** a step policy hits the luxury sector
- **THEN** with feedback only the luxury index moves, to the steady state of its decoupled system, and without feedback
  the other indices move too

### Requirement: Mark-0 economy
The library SHALL provide `mark0` in `economics/macroeconomics`, the Mark-0 economy of Sharma et al. (2020, Section 2)
with a shock to consumption and productivity, naive and adaptive credit policies, and helicopter money.

#### Scenario: Recession and policy
- **WHEN** the consumption propensity falls by half for nine months
- **THEN** without policy the unemployment stays above 50 percent, and with adaptive credit it returns below 10 percent

### Requirement: Kirman exchange rate
The library SHALL provide `kirman_exchange_rate` in `economics/finance`, Eq. 1.36 of Caiani et al. (2016, Section
1.3.2) with the share of fundamentalists of Kirman's recruitment chain.

#### Scenario: Regimes
- **WHEN** the market runs for 40000 steps
- **THEN** the share of fundamentalists is mostly near 0 or 1, and the rate changes more when chartists dominate

### Requirement: Default cascades
The library SHALL provide `default_cascade` in `economics/finance`, the threshold cascade of Thurner et al. (2018,
Section 5.5.3.2) with a recovery rate, and the random network of Gai and Kapadia reviewed by Bardoscia et al. (2021,
Section 3.1).

#### Scenario: Contagion window
- **WHEN** one bank of networks of mean degree 0.5, 4, and 12 is shocked
- **THEN** cascades are large only at the intermediate degree

### Requirement: Eisenberg-Noe clearing
The library SHALL provide `eisenberg_noe_clearing` in `economics/finance`, the iteration of proportional payments to
the greatest clearing vector.

#### Scenario: Linear programme
- **WHEN** the clearing runs to convergence
- **THEN** the payments solve the linear programme of the clearing vector

### Requirement: Fire sales of overlapping portfolios
The library SHALL provide `overlapping_portfolio_fire_sales` in `economics/finance`, banks with common assets that
sell under a threshold or leverage targeting, with market impact, after Bardoscia et al. (2021, Section 3.2).

#### Scenario: Indirect contagion
- **WHEN** one asset loses 60 percent of its price
- **THEN** only its holders default without market impact, and more than twice as many with it

### Requirement: Guru order book
The library SHALL provide `guru_order_book` in `economics/finance`, the order-driven market of noise traders who follow
gurus on a network rewired towards wealth, of Caiani et al. (2016, Chapter 3).

#### Scenario: Coordination and fat tails
- **WHEN** the market runs for 1000 days
- **THEN** cash and stocks are conserved, a guru gathers more than 40 followers, and the returns have an excess kurtosis
  above 3

### Requirement: Household demand
The library SHALL provide `household_demand` in `economics/markets`, the Cobb-Douglas and the elasticity demand of
Hamill and Gilbert (2016, Sections 3.2 and 3.3) for households with unequal budgets.

#### Scenario: Price rise
- **WHEN** the price of food rises by 10 percent
- **THEN** Cobb-Douglas households keep their expenditure, and households with elasticities lower their demand by 3.5
  to 5 percent and raise their expenditure by 4.5 to 6 percent

### Requirement: Fair Food Program
The library SHALL provide `fair_food_program` in `economics/markets`, the four rules of Chesney (2021, Chapter 7).

#### Scenario: Tipping point
- **WHEN** the programme runs for 5000 periods
- **THEN** farms join only after most buyers commit, and then all farms and buyers join

### Requirement: Gooding double auction
The library SHALL provide `gooding_double_auction` in `economics/markets`, buyers and sellers who adapt their quotes,
after Gooding (2019, Section 11.3).

#### Scenario: Competitive price
- **WHEN** 100 buyers and 100 sellers of five units trade
- **THEN** the 500 units trade at a mean price within 5 of 70

### Requirement: Toy Trader
The library SHALL provide `toy_trader` in `economics/markets`, Gooding's Toy Trader with market sight, mutation, and
imitation under a fitness test, and the Three-Goods Trader as `n_goods=3`.

#### Scenario: Market sight
- **WHEN** traders see one unit or the whole torus for 300 iterations
- **THEN** the price falls below 9 or rises above 11, and money and toys are conserved

### Requirement: El Farol coalitions
The library SHALL provide `el_farol_coalitions` in `economics/econophysics`, Collins' strategic group formation in the
El Farol bar of Rand and Wilensky.

#### Scenario: Coalitions lower returns
- **WHEN** patrons run alone and in coalitions for 500 nights
- **THEN** lone patrons attend near capacity and coalitions lower the returns by more than 0.1

### Requirement: CCC model
The library SHALL provide `ccc_model` in `economics/innovation`, the co-evolving combinatorial critical model of
Thurner et al. (2018, Section 5.6.4).

#### Scenario: Regimes
- **WHEN** the model runs with intermediate, mostly productive, and mostly destructive tensors
- **THEN** the first is punctuated, the second fills, and the third stays nearly empty
