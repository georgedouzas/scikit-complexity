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
