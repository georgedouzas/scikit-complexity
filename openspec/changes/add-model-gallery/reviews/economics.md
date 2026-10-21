# Economics review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| arbitrage_traders | space-time ridge of the good bending into a parabola | centre D E t^2, variance 1/2 + 2 D t plus the upwind diffusion dx D E t^2, mass exp(D E^2 t^3 / 3), all exact | 0.4 s, 176 frames, 0.36 MB |
| boltzmann_wealth | Gini coefficient jumps from 0 and settles near 0.62 | pooled wealth falls exponentially, exp(-w / 1.34); Lorenz curve, final Gini 0.62 | 5.8 s, 201 frames, 0.10 MB |
| bouchaud_mezard_wealth | Gini coefficient rises and settles near 0.5 | relative wealth follows the inverse gamma of mu = 1 + 2 J / sigma^2 = 2 in density and tail (model docstring says 1 + J / sigma^2, which is wrong for its noise) | 6.2 s, 151 frames, 0.74 MB |
| car_wash | cars on the forecourt over an eight-hour day | till grows at the expected 41.6 an hour; daily revenue saturates at the demand from three workers on, while the queue keeps shrinking | 22.7 s, 289 frames, 0.38 MB |
| cobweb | price oscillating in alternate directions and damping onto 2.5 | cobweb diagram spirals in; deviation scales exactly as (d / b)^t for d / b = 0.5, 1.0, 1.1 | 0.5 s, 31 frames, 0.03 MB |
| cournot_duopoly | total output overshooting, settling on the Nash total 95, jittering | trajectory zigzags to the crossing of the reaction curves; distance to Nash halves every period, 2^-t | 0.4 s, 61 frames, 0.04 MB |
| debtrank | value at risk climbing round by round to a plateau | DebtRank of all 50 banks equals a direct Battiston iteration on the impact matrix; DebtRank uncorrelated with bank size | 0.4 s, 11 frames, 0.02 MB |
| geometric_brownian_motion | one price path wandering from 100 | variance of log returns grows as sigma^2 tau over 200 years; one-step log returns normal with mean (mu - sigma^2 / 2) dt | 5.6 s, 201 frames, 0.04 MB |
| gibrat_process | largest firm pulling away from the mean | final log sizes normal with the Gauss-Hermite mean and variance; variance of log size grows as v t | 1.6 s, 151 frames, 0.09 MB |
| kesten_process | largest firm leaping in bursts above a flat mean | tail of P(X > x) parallels x^-1.47 from <lambda^alpha> = 1; mean relaxes to <b> / (1 - <lambda>) = 25 | 14.5 s, 251 frames, 0.70 MB |
| multiplicative_process | largest firm leaping as the mean drifts | relative sizes have a CCDF tail w^-1.43 = w^-1 / (1 - c); spread of log w stationary | 5.9 s, 251 frames, 0.28 MB |
| produce_market | items left falling and money spent saturating | items left match the exact hypergeometric expectation for the realised stocks (incl. the floor of unsold kinds); geometric law as reference | 0.8 s, 10 frames, 0.03 MB |
| riccati_prices | price lingering near 25 then falling onto 3.38 | phase line with stable and unstable equilibria; simulated prices match the closed-form Riccati solution from five starts | 0.3 s, 201 frames, 0.07 MB |
| capital_stock | capital stock growing exponentially from 100 | stocks for f = 0.1 to 0.3 follow exp((f o - 1 / L) t); measured net rates on the line, zero at f = 0.2 | 0.2 s, 51 frames, 0.03 MB |
| fishery_capital | fleet rising and fish falling into a damped equilibrium | stock-fleet plane spirals to a fixed point; a yield table of better technology overshoots and both collapse | 0.6 s, 151 frames, 0.05 MB |
| multiplier_accelerator | income swinging around 100 in damped cycles | one run in each Samuelson region behaves as predicted; periods 2 pi / arg(lambda) of 13.6 and 14.9 | 0.3 s, 41 frames, 0.03 MB |
| oil_economy | capital peaking and declining as the resource is drawn down | doubling the oil delays the extraction peak by about 16 years (38, 54, 70), the capital peak less (65, 72, 80) | 0.2 s, 101 frames, 0.04 MB |
| solow | capital per worker rising from 1 onto k* = 7.25 | paths from three starts match the closed form via z = k^(1 - alpha); steady-state consumption peaks at s = alpha | 0.2 s, 151 frames, 0.04 MB |
