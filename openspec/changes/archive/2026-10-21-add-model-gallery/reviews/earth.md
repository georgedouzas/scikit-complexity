# Earth review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| budyko_sellers | warm equator to cold pole gradient (history view shows no visible transient) | ice cap climate from 15 °C, snowball below -10 °C from -40 °C | 5 s script, 65 frames, 0.09 MB |
| burridge_knopoff | speckles of slipping blocks in the speed history | blocks trail the plate in jumps; 75 of 200 frames hold slips of 0.3 to 10 (RK4, dt = 0.05) | 3.6 s script, 201 frames, 0.56 MB |
| carbon_cycle | carbonate ion collapse, overshoot to 188, ringing back to 84 | peak `c` jumps from ~95 to ~180 at a deficit threshold near 100 | 15 s script, 201 frames, 0.11 MB |
| daisyworld | white and black daisy areas settling | temperature held near 22.5 °C for L in 0.7 to 1.2, bare planet warms steadily | 8 s script, 201 frames, 0.13 MB |
| earthquakes | OFC force field organized into patches | 30,000 sizes: CCDF straight over two decades, power-law exponent ~1.9, cutoff near grid size | 16 s script, 201 frames, 1.52 MB |
| energy_balance | temperature relaxing from 290 K to 286 K | three roots of dT/dt; runs split at the unstable climate 263 K | 7 s script, 101 frames, 0.05 MB |
| enso | regular El Niño / La Niña swings | delay embedding is one closed loop; spectral peak at period 16.8 with harmonics | 8 s script, 201 frames, 0.05 MB |
| geodynamo | Q, D, V with irregular polarity flips | two-lobe attractor; 84 polarity intervals spread from 30 to 800 | 34 s script, 241 frames, 0.10 MB |
| hasselmann | red-noise temperature anomaly | autocorrelation follows exp(-lam t); histogram matches variance sigma^2 / (2 lam) | 11 s script, 251 frames, 0.04 MB |
| lorenz_84 | chaotic jet and eddy series | Lyapunov spectrum 0.157, -0.004, -0.393 | 23 s script, 251 frames, 0.09 MB |
| lorenz_96 | three of twenty sites oscillating chaotically after the kick | space-time diagram of travelling waves; neighbour correlation peaks off zero lag | 17 s script, 201 frames, 0.07 MB |
| saltzman_glaciation | ice, carbon, and ocean in a self-sustained cycle | closed x-y loop; spectral peak at 102 kyr | 16 s script, 251 frames, 0.11 MB |
| soil_temperature | chaotic surface and damped deep temperature | orbit diagram: fixed point x* matches, period doubling to chaos, r = 3.8 chaotic | 9 s script, 201 frames, 0.05 MB |
| stommel | T and S settling in the thermal mode | 64 starts split between thermal (T > S) and haline (S > T) modes | 15 s script, 251 frames, 0.06 MB |
| tipping_cascade_network | network midway through the cascade: tipped elements bright around the first, the far end still dark | tipping time grows with graph distance from the first element; swept control tips one element alone past c = 1, all forty when coupled (from c = 0.9) | 0.5 s script, 181 frames, 0.09 MB |
| van_der_pol_ice_ages | sawtooth of slow build-up and fast terminations | limit cycle hugs the outer branches of the cubic nullcline | 12 s script, 201 frames, 0.06 MB |
| energy_return | extraction capacity rising, peaking, and falling as the reserve depletes | cost plus net energy equals gross energy; peaks of net energy per person (year 106), net energy (121), and gross energy (125) in that order | 0.4 s, 251 frames, 0.21 MB |
| sutera_fraedrich_tipping | temperature fluctuating near 291.6 K, then an abrupt fall at about 170 y to the cold root near -280 K (N-tipping) | B-tipping: T follows the warm branch in mu until the fold mu = 4 a2 / (c1 b2^2) = 0.761, then falls; R-tipping: rho = 0.16 tracks to 303 K, rho = 0.2 tips | 5.3 s, 151 frames, 0.14 MB |
| ramped_stochastic_fold | noisy state creeping up the lower branch with growing fluctuations, then jumping to the upper branch | residual variance against the noiseless run rises toward phi_c = 0.385; control at tipping 0.43, 0.52, 0.73 for rates 0.004, 0.02, 0.1, all past phi_c and growing with the rate | 24.4 s, 251 frames, 0.14 MB |
