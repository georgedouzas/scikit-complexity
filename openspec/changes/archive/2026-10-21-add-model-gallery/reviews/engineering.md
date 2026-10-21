# Engineering review

| Model | Thumbnail shows | Analysis checks | Budget |
| --- | --- | --- | --- |
| chua | chaotic x, y, z of the double scroll | x-z projection winds around both scrolls; largest Lyapunov exponent 0.43 | 6 s script, 251 frames, 0.09 MB |
| pendulum_swing_up | swings growing until the pendulum flips over | energy rises in steps to omega^2 and stays; phase plane fills to the separatrix | 6 s script, 251 frames, 0.10 MB |
| pid_loop | output reaching the set point, dipping at the disturbance, recovering | PID returns to 1; P-only settles at 2/3, then lower after the disturbance | 5 s script, 201 frames, 0.09 MB |
| rlc_circuit | decaying ringing of q and its rate | matches the exact damped solution; resonance curve peaks at 1 / sqrt(L C) | 7 s script, 251 frames, 0.08 MB |
| thermostat | room warming to ~17.5 °C and swinging with the day | 13 leaks land on (setting + leak outside) / (1 + leak) | 5 s script, 193 frames, 0.06 MB |
| tunnel_diode_oscillator | voltage growing from 0.1 into a steady oscillation | four starts converge on one limit cycle of amplitude 1.16 (weak-nonlinearity estimate 1.15) | 5 s script, 251 frames, 0.07 MB |
| faraday_disk | disk spinning up while current and battery voltage settle at small negative values, the battery discharging | spin-up ends on a steady state; steady speed grows linearly with torque from 3 to 9, below the oscillations near 9.5 | 5 s script, 251 frames, 0.20 MB |
| impulsive_circuits | response kicked every 2 units settling into a periodic waveform | stroboscopic x and x' converge; last five periods coincide | 5 s script, 251 frames, 0.09 MB |
| memristor_oscillator | irregular oscillations of v, i, and w | v-i attractor; largest Lyapunov exponent 0.034 > 0 | 7 s script, 251 frames, 0.08 MB |
| mixed_mode_chua | small oscillations between large spikes | c = 0 small only, c = 0.00053 mixed, c = 0.00105 large only | 21 s script, 201 frames, 0.35 MB |
| power_grid | frequency deviations splitting to +10 and -10 with the added line | default ring synchronises (max deviation 0.00); added (0, 2) or doubled (0, 1) line does not (10.14) | 19 s script, 151 frames, 0.56 MB |
| regenerative_braking | speed falling as the capacitor voltage rises | stored energy reaches ~17% of the kinetic energy lost, peaking near the current reversal | 4 s script, 121 frames, 0.07 MB |
| vibration_energy_harvester | wheel and body oscillating with the road | voltage amplitude peaks near the two natural frequencies 7 and 74 rad/s | 18 s script, 251 frames, 0.16 MB |
| wheel_rail_contact | rail and sleeper dipping twice per bogie | two dips per bogie; peak deflection proportional to the load | 7 s script, 241 frames, 0.22 MB |
| delayed_car_following | speed swings growing from the leader to car 7 | amplitude grows along the chain for T = 1 and decays for T = 0.2, as omega < 2 lambda sin(omega T) predicts | 8 s script, 121 frames, 0.38 MB |
| intelligent_driver_model | 20 cars accelerating from rest to one common speed | final speeds for 10 to 60 cars match the IDM equilibrium | 7 s script, 201 frames, 0.16 MB |
| optimal_velocity | speeds fanning out into a stop-and-go wave | spread grows for headway 2 and decays for headway 4; late states loop around V(h) | 8 s script, 201 frames, 0.13 MB |
| lighthill_whitham_richards | density history with the shock edge moving upstream | shock position follows the Rankine-Hugoniot speed -0.1 | 8 s script, 51 frames, 0.03 MB |
| nagel_schreckenberg | space-time road with backward-leaning jam stripes | fundamental diagram: free flow (vmax - p) rho up to ~0.15, then falling flow | 7 s script, 201 frames, 7.95 MB |
| cascade_classifier | fine product growing much faster than the coarse | steady fine share matches Barsky's fractional extraction for k = 0.3 to 0.7 | 6 s script, 101 frames, 0.22 MB |
| satellite_transfer | distance swinging out while the energy rises and levels off | energy reaches -0.2 at 9.81 and stays; orbit widens around the planet | 5 s script, 201 frames, 0.08 MB |
| swarmtaxis | robot cluster with its illuminated side (yellow) facing the beacon | centre drifts from x = 19 to 40 in 3000 steps; spread stays ~2; ~20% illuminated | 10 s script, 201 frames, 0.43 MB |
| kerner_konhauser | density history with a bright travelling cluster | max density grows to 0.54 and saturates, min falls to 0.15; cluster keeps its shape | 13 s script, 251 frames, 1.43 MB |
| payne | density history: the bump splits into two waves crossing the ring | number of cars conserved exactly; bump height decays from 0.06 to 0.02 | 7 s script, 101 frames, 0.32 MB |
| ring_road_drivers | 30 cars as dots along the road, bunched where jams hold them (all one colour) | 10 cars reach the limit 40; 30 and 60 cars level off near 10 and 6 | 6 s script, 151 frames, 0.10 MB |
| bathtub | water level falling, then held when the faucet opens | level changes at faucet minus drain; flows balance after the faucet opens (budget from capture only) | 0.2 s run, 21 frames, 0.02 MB |
| food_relief | food in transit overshooting to ~1000 at the onset, then settling near 700 over the whole run | arrivals follow the Erlang response of the third-order delay; unmet need until enough food arrives | 6 s script, 121 frames, 0.10 MB |
| irrigation | reservoir rising, drawn down by the dry season, refilling and falling again over two years | release covers demand while the reservoir holds; pumping takes over; water stress only when both fall short | 6 s script, 147 frames, 0.22 MB |
| langevin_traffic | speeds of all cars fluctuating around 0.5 | flows follow rho u(rho); speed variance follows D1 rho / gamma at low density | 2.8 s run, 201 frames, 0.24 MB |
| rice_milling | paddy stock peaking at every harvest and running low before the next, over ten years | milling capped by capacity after harvests; grain conserved into rice, husk, and bran | 5 s script, 121 frames, 0.18 MB |
| solid_waste | landfill filling, then closing | landfilling rises until the acceptance table closes the landfill, then dumping | 0.2 s run, 81 frames, 0.13 MB |
| traffic_density_equation | density waves growing from the uniform flow | log Fourier amplitudes grow at the linear rate, fastest near q = 0.9 | 0.9 s run, 101 frames, 0.74 MB |
| charbonneau_traffic | space-time diagram of 100 cars whose lines run flat in jams that drift backwards while the platoon advances | speed-colored scatter of position against step shows dark jam bands moving against the traffic | 1.5 s, 201 frames, 0.18 MB |
| light_seeking_robots | x of 20 robots shuttling between the walls at -2 and 2 across the light at 0 | mean light intensity at 40 robots after t = 50 falls from ~0.17 for delays 0 and 1 to ~0.13 for delays 3 to 8; long delays reach the centre less often | 33 s script, 201 frames, 0.41 MB |
