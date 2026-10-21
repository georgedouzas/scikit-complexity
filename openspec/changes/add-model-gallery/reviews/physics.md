# Physics review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| harmonic_oscillator | cosine displacement of constant amplitude | matches cos t; energy drift below 2e-5 | 0.2 s, 201 frames, 62 KB |
| driven_damped_oscillator | displacement swelling to the resonant amplitude 5 | swept steady amplitudes lie on the resonance curve | 5.7 s, 226 frames, 77 KB |
| duffing | irregular hopping between the two wells | stroboscopic section fills Ueda's strange attractor | 20.6 s, 251 frames, 83 KB |
| van_der_pol | relaxation oscillation growing from rest (mu = 3) | three starts converge on one limit cycle | 0.5 s, 251 frames, 47 KB |
| stuart_landau | oscillation growing and saturating at sqrt(mu) | amplitude on the exact curve; phase turns at omega - beta mu | 0.2 s, 201 frames, 56 KB |
| adler_equation | phase difference locking at pi + arcsin(1/2) | slip rate follows sqrt(detuning^2 - eps^2) above eps | 7.7 s, 201 frames, 41 KB |
| forced_van_der_pol | response settling at the drive period | locking plateau of frequency ratio 1 around omega 0.95 | 27.3 s, 241 frames, 64 KB |
| henon_heiles | star orbit in the triangular potential (custom path view) | Poincare section at E = 1/8: islands in a chaotic sea | 35.6 s, 251 frames, 122 KB |
| billiards | Sinai table with criss-crossing path (custom table view) | separation of paths 1e-8 rad apart grows exponentially | 0.6 s, 251 frames, 110 KB |
| circular_billiard | star of chords around an empty caustic (custom table view) | angular momentum constant; radii never below the caustic | 0.3 s, 251 frames, 101 KB |
| bouncing_ball | ever shorter bounces of height and velocity | impact times on the geometric series to the Zeno time | 0.1 s, 191 frames, 48 KB |
| projectile | parabolic flight path (custom path view) | path on the parabola; drag shortens the range | 0.2 s, 145 frames, 68 KB |
| pendulum | large swing arc with rod and bob (custom view) | periods on 4 K(sin^2(theta0/2)) sqrt(l/g) | 5.3 s, 251 frames, 104 KB |
| double_pendulum | tangled trace of the lower bob (custom view) | separation of releases 1e-8 rad apart grows exponentially | 2.3 s, 241 frames, 264 KB |
| coupled_pendulums | beating displacements of the two bobs | left displacement follows the normal-mode beat envelope | 2.8 s, 251 frames, 196 KB |
| two_body_orbit | ellipse of eccentricity 0.6 | orbit on the Kepler ellipse; areal velocity constant | 6.0 s, 249 frames, 555 KB |
| n_body | three stars on the figure eight | distance to the initial configuration returns to ~0 every 6.3259 | 1.3 s, 212 frames, 209 KB |
| friedmann_expansion | a and a' with a' dipping then rising | a on the exact sinh^(2/3) solution; acceleration from a = 0.6 | 1.7 s, 201 frames, 57 KB |
| cooling_coffee | exponential cooling to the room temperature | excess temperature decays as exp(-k t) for three starts | 0.1 s, 121 frames, 33 KB |
| heat_diffusion | round bump spreading from the hot square | total heat constant; variance grows as var(0) + 2 D t | 1.1 s, 51 frames, 141 KB |
| standing_waves | space-time image of the third mode | snapshots share the nodes; antinode follows cos(omega t) | 0.7 s, 81 frames, 97 KB |
| swift_hohenberg | rolls with defects grown from noise | structure factor peaks at k = 1; wavelength 2 pi | 1.7 s, 41 frames, 462 KB |
| kuramoto | order parameter rising to about 0.98 at K = 4 | time-averaged r of 400 oscillators rises at K_c = 2 / (pi g(0)) | 12.2 s, 151 frames, 121 KB |
| winfree | order parameter settling as the phases lock | frequency spread vanishes on locking; mean frequency drops to 0 (death) near K = 0.8 | 2.5 s, 201 frames, 145 KB |
| ott_antonsen | sigmoid growth of coherence to sqrt(1/2) | growth on the exact logistic solution; final r on sqrt(1 - 2 Delta / K) | 0.6 s, 201 frames, 56 KB |
| mirollo_strogatz | five sawtooth charges merging into one | number of groups of 30 oscillators falls stepwise to 1 | 0.7 s, 201 frames, 52 KB |
| aggregate_averaging_synchronization | two chaotic maps on one curve after a transient | difference vanishes exactly in the window 1/4 < kappa < 3/4 | 0.2 s, 51 frames, 29 KB |
| coupled_rossler | chaotic peaks of the two oscillators in step | phase difference drifts at k = 0, stays bounded at k = 0.06 | 4.2 s, 251 frames, 160 KB |
| delayed_kuramoto | unwrapped phases settling on parallel lines | locked frequencies on the stable branch of Omega = 1 - K' sin(Omega tau) up to tau = 2 | 5.2 s, 251 frames, 646 KB |
| particle_in_a_well | anharmonic oscillation in the well | orbits follow energy levels; trapped below the separatrix, escaping above (v0 = 0.9) | 0.2 s, 251 frames, 63 KB |
| two_mass_spring_damper | quasi-periodic beating of the first mass | spectrum peaks at the normal modes 1 and sqrt(3) | 4.5 s, 251 frames, 201 KB |
| sitnikov | irregular bobbing of the light body (e = 0.2) | stroboscopic section: invariant curves near the centre, chaotic band outside | 19.0 s, 251 frames, 47 KB |
| restricted_three_body | precessing rosette around the Earth, Moon aside (custom path view) | path outside the Jacobi forbidden region; C drift below 2e-5 | 0.3 s, 251 frames, 121 KB |
| ising | up and down domains coarsening on a 256 x 256 torus at T = 2.0 | \|m\|(T) on 48 x 48 follows Onsager below T_c, rounded tail above | 5.4 s, 101 frames, 3.3 MB |
| potts | q = 5 domains of five colors after a quench | energy per site passes the exact e_c at T_c; smooth for q = 3, jump for q = 8 | 30.6 s, 101 frames, 4.2 MB |
| tasep | space-time image of jams drifting backwards on a ring | ring current matches the exact parallel-update formula for p = 0.5, 0.9 | 2.0 s, 251 frames, 3.1 MB |
| contact_process | sparse infected clusters just above lambda_c | steady density zero below 1.65, rising above, under mean field | 5.9 s, 251 frames, 3.7 MB |
| site_percolation | burnt spanning cluster at p = 0.6 | spanning probability curves for L = 16, 32, 64 cross near p_c | 30.9 s, 300 frames, 5.0 MB |
| forest_fire | fire fronts sweeping through older stands | tree density settles near 0.4, below the percolation threshold | 9.0 s, 300 frames, 11.3 MB |
| sandpile | stable heights 0-3 in the critical state | mean height plateaus at 2.1; size density follows s^-1.2 over two decades | 22.3 s, 300 frames, 1.1 MB |
| manna_sandpile | stable heights 0-1 in the critical state | density plateaus near 0.68; size density follows s^-1.27 | 15.3 s, 300 frames, 0.4 MB |
| brownian_motion | round Gaussian cloud of 500 particles | MSD follows the Ornstein-Uhlenbeck law from ballistic to diffusive | 0.6 s, 201 frames, 2.5 MB |
| optical_trap | elongated cloud in an anisotropic trap | position histograms match Gaussians of variance kT / k (k = 1.01, 0.26) | 0.2 s, 251 frames, 1.1 MB |
| stochastic_resonance | hops between wells locked to the forcing | response at the forcing frequency peaks at D = 0.12, near the matching 0.10 | 20.6 s, 251 frames, 0.06 MB |
| kramers_escape | six particles rattling in a well and hopping at random times | survival is exponential, slopes about 15% below the Kramers rate at barrier / D of 2.5-3 | 14.0 s, 201 frames, 0.06 MB |
| scaled_brownian_motion | paths of eight walkers fanning out then slowing | ensemble MSD matches 2 K ((t + t0)^alpha - t0^alpha) for alpha = 0.5, 1.5 | 0.6 s, 201 frames, 0.06 MB |
| multiplicative_noise | eight Itô particles lingering near the quiet wall | stationary histograms match 1 / sigma^2, 1 / sigma, flat for Itô, Stratonovich, anti-Itô | 5.6 s, 300 frames, 0.08 MB |
| brownian_near_wall | height sinking from 20 then wandering near the wall | anti-Itô heights match the barometric law with D0 / v = 5 | 30.0 s, 251 frames, 0.07 MB |
| diffusion_in_a_speckle_field | 300 particles clumped at hotspots | MSD from t = 10: free 4 D t, plateau with escapes at 30, flat plateau at 100 | 23.6 s, 241 frames, 0.8 MB |
| active_brownian_particles | 200 swimmers spread over the periodic square | MSD matches the exact ABP law for D_r = 1 and 0.1 | 0.9 s, 201 frames, 0.7 MB |
| active_ornstein_uhlenbeck_particles | 200 particles spread over the periodic square | MSD matches 4 D (t - tau (1 - exp(-t / tau))) for tau = 1 and 10 | 1.3 s, 201 frames, 1.0 MB |
| chiral_active_particles | 200 circling swimmers in the periodic square | MSD matches van Teeffelen-Löwen, oscillating for omega = 2, far below omega = 0 | 0.9 s, 201 frames, 0.7 MB |
| absorbing_birth_death | particle number fluctuating around the active density | histogram matches the quasi-stationary law; time-averaged density follows max(0, (kappa - nu) / lambda) | 0.8 s, 201 frames, 0.06 MB |
| aligning_active_brownian_particles | swimmers gathered in aligned clusters of one heading colour | polar order stays near 1 / sqrt(N) below the alignment threshold and rises above it | 5.9 s, 101 frames, 0.5 MB |
| annealed_transient_time_motion | dark slow walkers near the origin, bright fast ones spread out | ensemble MSD subdiffusive, t^0.5, approaching the predicted law | 0.6 s, 201 frames, 1.1 MB |
| asep | space-time image of particles streaming right at low density | flat profiles at alpha, 1 - gamma, 1/2 and currents matching the three phases | 0.6 s, 251 frames, 0.7 MB |
| biased_trap_walk | rows of crossed sites growing in bursts to very different lengths | mean displacement approaches t^alpha laws; displacements spread broadly | 8.0 s, 300 frames, 0.3 MB |
| binary_collision_flocking | particles of mostly one heading colour, partly ordered | polarization large at low noise, dropping near the threshold noise 0.64 | 8.0 s, 201 frames, 1.4 MB |
| colored_noise_langevin | velocity following the coloured noise smoothly with smaller amplitude | velocity variances follow the tau curve; noise correlation decays as exp(-t / tau) | 0.9 s, 300 frames, 0.07 MB |
| continuous_time_random_walk | cloud with a sharp peak at the origin | ensemble and time-averaged MSD follow the subdiffusive laws | 0.4 s, 201 frames, 0.2 MB |
| curie_weiss | space-time image turning to one colour with short-lived flips | magnetization follows the mean-field curve with a 1 / sqrt(N) tail above T_c | 2.8 s, 151 frames, 0.5 MB |
| dissipative_zero_range | binary tree on rings, root holding most quanta, outer rings empty | inverse temperature across levels as predicted for the two alpha regimes | 0.9 s, 201 frames, 0.7 MB |
| edwards_anderson | frozen two-colour pattern without ferromagnetic domains | energies follow the high-temperature curve above freezing | 2.4 s, 101 frames, 0.3 MB |
| fhp_lattice_gas | hexagonal cells with a ring of excess density around the centre | particle number and momentum conserved; disc excess spreads to the mean | 0.2 s, 121 frames, 0.7 MB |
| flashing_ratchet | positions climbing from well to well to the right | mean velocity peaks at an intermediate flashing rate | 8.5 s, 241 frames, 0.2 MB |
| fractional_brownian_motion | cloud of walkers spreading slowly | MSD follows 4 D t^0.6; DFA Hurst exponents spread around 0.3 | 1.1 s, 201 frames, 0.8 MB |
| graph_ising | honeycomb graph nearly one colour with a remaining domain | \|m\|(T) on a random 4-regular graph drops at the Bethe T_c = 2 / ln 2 | 2.1 s, 61 frames, 0.2 MB |
| hexagonal_forest_fire | burnt band bounded by hexagonal fire fronts | fire activity spreads over decades; fire on a graph burns nodes at graph distance t | 1.1 s, 201 frames, 1.4 MB |
| ideal_gas | molecules spread uniformly, slow and fast mixed | speeds match the 2D Maxwell-Boltzmann density | 4.1 s, 201 frames, 1.5 MB |
| levy_flight | weak: zoomed core of large overlapping discs, long jumps not visible | median distance slope 1 / mu = 0.83; jump lengths follow l^(-1 - mu) | 0.4 s, 201 frames, 0.8 MB |
| levy_walk | dense core with walkers far out | MSD follows t^1.5 at long times | 0.5 s, 201 frames, 2.1 MB |
| mattis_spin_glass | overlap rising to near 1 while magnetization stays near 0 | overlap matches Onsager's spontaneous magnetization | 1.5 s, 201 frames, 0.08 MB |
| quantum_heisenberg | weak: blurred vertical stripes, the magnon light cone is not visible | (no explicit should statement found) | 0.9 s, 161 frames, 0.1 MB |
| shaken_spring_block | space-time image of shaking and frozen phases | velocities fall to zero each cycle; correlation lengths linear in energy | 7.3 s, 287 frames, 0.4 MB |
| sherrington_kirkpatrick | space-time image of frozen columns of both colours | energies follow -J^2 / (2 T) above T_c; q grows below it | 4.7 s, 201 frames, 1.5 MB |
| tilted_periodic_potential | positions drifting right from well to well and spreading | mean velocity follows the exact force curve; positions drift at it | 1.5 s, 201 frames, 0.1 MB |
| trap_model | walkers coloured by trap depth, mostly deep | MSD grows as t^T (slope 0.5); mean depth grows as T ln t | 0.6 s, 201 frames, 0.5 MB |
| zero_range_process | one bright condensate column | fluid occupations follow the critical law; condensate holds about 125 particles | 0.4 s, 251 frames, 0.1 MB |
