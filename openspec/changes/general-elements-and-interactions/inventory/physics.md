# Audit: physics models against the general core of the brief

Scope: every element and interaction class in `src/skcomplex/models/physics` (common included). Analyses
(`MaxwellBoltzmannAnalysis`, `KineticTemperatureAnalysis`, `SusceptibilityAnalysis`, `CracklingNoiseAnalysis`) and agent
views (`FlockerView`, `TumblerView`, `WalkerView`) are out of scope. Module paths are relative to `models/physics/`.

Notation for "built from": `Rate(state, rate, noise)` = RateInteraction, `Grad(state, potential)` = GradientInteraction,
`Coup(state, f, reduce, over, ...)` = CouplingInteraction, `Trans` = TransitionInteraction, `Choice`, `React` =
ReactionInteraction, `Move` = MovementInteraction, `Xfer` = TransferInteraction. Names marked NEW are the general classes
proposed in section GAPS. `ord2` = state declared `StateSpec(order=2)`; `x/v` = first-order pair `RateLaw(x, v)`.

## How the LAMMPS backend recognises things today (constraint for the rebuild)

`engine/solvers/_lammps.py` + `physics/common/particles/_lammps.py`:

- Elements: `issubclass(element_type, ParticleElement)` (core); then reads states **by name** `'x'` and `'v'`
  (`element.states['x']`, `element.states['v']`) and the parameter **by name** `'m'` (atom mass per type).
- Pair potentials: `isinstance(interaction, PairPotentialInteraction)`; roles `'first'`/`'second'`, parameter `'cutoff'`;
  energy from `interaction._get_pair_energy(namespace, r)` -> Lepton string, or a native pair style looked up by class in
  `LAMMPS_PAIRS` (`LennardJonesInteraction -> lj/cut`).
- Fixes: `LAMMPS_FIXES` keyed by class, looked up through the MRO (`_native`): `ConstantForceInteraction -> addforce`
  (param `F`), `DampingInteraction -> viscous` (param `gamma`), `LangevinThermostatInteraction -> langevin` (params
  `temperature`, `gamma`, plus `fix.mass()` of role `'target'`).
- Everything else runs through `fix external`: `_check_external` accepts it only if every non-trivial compiled equation
  targets a state in `{element.states['v']}`; the callback multiplies the drift of `v` by the per-type mass `m` to get
  forces.

Consequences for a rebuild on `Rate`/`Coup` acting on order-2 positions:
1. Order-2 `x` compiles to `x`, `x_d1` with the intrinsic chain `RateLaw(x, x_d1)` (`engine/compile/passes/_orders.py`),
   and laws on `x` are retargeted to `x_d1`. LAMMPS must find the velocity as the companion `x_d1` of the position state
   (as `passes/_particles.py` and `_neighborhood_kernel.py` already do: `velocity_of` else `f'{name}_d1'`), not by the
   name `'v'`.
2. LAMMPS needs the inertia as a **named per-member parameter** (atom mass and force = m * acceleration). If inertia is
   only a divisor hidden inside a user's rate lambda, LAMMPS cannot set atom masses nor give pair styles the right
   force. So the general gradient/coupling classes must take an explicit divisor binding (see CORRECTIONS).
3. Pair styles must be recognised by "pair-binding coupling with a potential of the distance", e.g.
   `isinstance(i, CouplingInteraction) and i.potential is not None and i.over == 'radius'`, reading
   `i.potential(namespace, r)` instead of `_get_pair_energy`. Native tables keyed by class keep working because
   `_native` walks the MRO, so `LennardJonesInteraction`, `LangevinThermostatInteraction`, the constant-force and damping
   subclasses stay valid keys as long as they remain distinct (thin) classes.

## Rows grouped by mechanism

### M1. Minus the gradient of a potential of member states (explicit roles), inertial (ParticleElement x/v/m)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| GravityInteraction | common/particles/_gravity | PotentialInteraction -> RateLaw(v, -grad U/m) | 2-body potential -G m1 m2/r | Grad(state='x' ord2, potential=-G m_i m_j/|x_i-x_j|, divisor='m'), roles first/second | the energy |
| HarmonicPotentialInteraction | common/particles/_harmonic_potential | idem | U = k|x-c|^2/2 | Grad(x ord2, k|x-c|^2/2, divisor='m') | energy |
| SpringInteraction | common/particles/_spring | idem | U = k(|x1-x2|-l)^2/2 | Grad(x ord2, ..., roles first/second, divisor='m') | energy |
| MagnetInteraction | mechanics/_pendulums | idem | U = -g/sqrt(|x-c|^2+d^2) | Grad(x ord2, ...) | energy |
| ExponentialWellInteraction | mechanics/_wells | idem | U = k x^2 e^-x | Grad(x ord2, ...) | energy |
| DrivenAnchorInteraction | waves/_strings | idem | time-dependent anchor k|x-a(t)|^2/2 | Grad(x ord2, potential reads t) | energy; could subclass HarmonicPotential with moving centre |

### M2. Minus the gradient of a potential, overdamped / order-1 or own element ODE

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| OpticalTrapInteraction | soft_matter/_diffusion | RateLaw(x, on(t)·(-k(x-c(t)))/gamma) | overdamped harmonic trap, blinking | Grad(x ord1, sum k_d (x_d-c_d)^2/2, divisor='gamma') × on(t) | anisotropic k, moving centre, pulse_train gate |
| SpeckleFieldInteraction | soft_matter/_diffusion | RateLaw(x, s·grad I) | overdamped in -s·I | Grad(x ord1, -s·sum gaussians) | speckle intensity |
| TiltedWashboardInteraction | soft_matter/_ratchets | RateLaw(x, F/gamma) | U0 cos + tilt f | Grad(x ord1, -U0 cos(2πx/L) - f x, divisor='gamma') | potential |
| FlashingSawtoothInteraction | soft_matter/_ratchets | RateLaw(x, on(t)F(x)/gamma) | piecewise sawtooth, switched | Rate(x ord1, rate=on·F/gamma) (gradient of `mod`/`where` not safely differentiable) | force, switching series |
| OrderParameterElement | statistical_mechanics/_master_equations | RateLaw(phi, -Gamma dF/dphi) | Landau gradient flow | Grad(phi ord1, a(T-Tc)phi^2/2+b phi^4/4-h phi, divisor=1/Gamma) | free energy |
| DoubleWellElement | soft_matter/_brownian | RateLaw + NoiseLaw | double well + periodic drive + noise | Grad(x, x^4/4-x^2/2) + Rate(x, A cos(2πft), noise=sqrt(2D)) | none beyond parameters |
| HangingBallElement | mechanics/_mechanics | RateLaw(x ord2) | linearised pendulum | Grad(x ord2, g(x-rest)^2/(2L)) | none |
| OscillatorElement | mechanics/_mechanics | RateLaw(x ord2, (-kx - c x' + F cos)/m) | driven damped oscillator | Grad(x ord2, kx^2/2, divisor m) + Rate(x, -c x'/m) + Rate(x, F cos(ωt)/m) | none |
| DuffingElement | nonlinear_oscillators/_oscillators | RateLaw(x ord2) | Duffing | Grad(x ord2, αx^2/2+βx^4/4) + Rate(-δx') + Rate(γ cos ωt) | none |
| HenonHeilesElement | nonlinear_oscillators/_oscillators | RateLaw(x,y ord2) | 2-d polynomial potential | Grad((x,y) ord2, (x^2+y^2)/2 + x^2 y - y^3/3) | potential |
| SitnikovBodyElement | celestial_mechanics/_orbits | RateLaw(z ord2) | time-dependent central potential | Grad(z ord2, -1/sqrt(z^2+r(t)^2)) | r(t) |
| HarmonicForceInteraction | mechanics/_springs | RateLaw(v, F cos(ωt)/m) | time-periodic force | Rate(x ord2, F cos(ωt), divisor 'm') | none (or subclass of the constant-force Rate) |

### M3. Pair potential of the distance within a cutoff (PairInteraction binding)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| LennardJonesInteraction | common/particles/_lennard_jones | PairForce -> RateLaw(v1/v2, ∓U'(r) r̂/m) | LJ 12-6, LAMMPS `lj/cut` | Coup(x ord2, potential=4ε((σ/r)^12-(σ/r)^6), over='radius', cutoff, period, divisor='m') | energy, LAMMPS native key |
| SoftSphereInteraction | common/particles/_soft_sphere | idem | k(rc-r)^2/2 | Coup(x ord2, potential, over radius, divisor 'm') | energy |
| SoftRepulsionInteraction | active_matter/_collective | RateLaw(x1/x2, ±k(rc-r)r̂) on ord1 | SAME energy as SoftSphere, overdamped | Coup(x ord1, potential=k(rc-r)^2/2, over radius) | nothing: duplicate of SoftSphere on order 1 |
| PhoreticAttractionInteraction | active_matter/_collective | RateLaw(x1/x2, ∓v0R^2/r^2 r̂) ord1 | U = -v0 R^2/r | Coup(x ord1, potential=-v0R^2/r, over radius) | energy |

### M4. Coupling of a state to other members' states through a function (edges, pairs, mean field, explicit elements)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| PolarAlignmentInteraction | active_matter/_alignment | RateLaw(θ1, γ sin(θ2-θ1)), antisym | pair Kuramoto within radius | Coup(θ, f=sin(θj-θi), reduce sum, over radius, K=γ) | none |
| CuckerSmaleAlignmentInteraction | active_matter/_collective | RateLaw(v1/v2, K/N (1+|dx|^2)^-β (v2-v1)) | velocity alignment weighted by distance of x | Coup(x ord2 (second derivative), f(member_j, member_i)=w(|x_j-x_i|)(x'_j-x'_i), reduce mean, over all) | weight w |
| PhaseCouplingInteraction | synchronization/_kuramoto | RateLaw(θ, K sin(θj-θi-α)) | Kuramoto-Sakaguchi within radius on ring (period) | Coup(θ, sin(θj-θi-α), sum, over radius, period) | none |
| PhaseLagInteraction | synchronization/_kuramoto | RateLaw(θ, -K sin α) | removes the self term the pair binding omits | Rate(θ, -K sin α) — disappears if Coup has `include_self=True` | none (artefact) |
| MeanFieldCouplingInteraction | synchronization/_kuramoto | RateLaw(θ_t, K(<sin θ_s>cos(θ_t+α)-<cos θ_s>sin(θ_t+α))) | Kuramoto mean field, source group ≠ target group (chimera) | Coup(θ, sin(θj-θi-α), reduce mean, over all, source≠target groups) | the trig expansion is an O(N) lowering detail the reducer should do |
| DelayedCouplingInteraction | synchronization/_kuramoto | RateLaw(θ_t, K sin(θ_s(t-τ)-θ_t)) | delayed coupling between two named elements | Coup(θ, sin, lag/delay τ, over explicit pair of elements) | none |
| EdgeCouplingInteraction | synchronization/_networks | RateLaw(θ_s/θ_t, ±K w_e sin(θ_t-θ_s)) | Kuramoto on weighted network edges | Coup(θ, f reads edge state w, over edges) | none |
| DiffusiveCouplingInteraction | synchronization/_chaos | RateLaw(x2_i, k(x2_j-x2_i)) | diffusive coupling of one component between two elements | Coup(x2, difference, over explicit pair) | none |
| GlobalCouplingInteraction | synchronization/_amplitude_death | RateLaw(x,y; K(mean-own)) | mean-field diffusive coupling of two states | Coup(states=('x','y'), difference, reduce mean, over all) | none |
| WinfreeMeanFieldInteraction | synchronization/_pulses | RateLaw(θ, K <1+cos θ_j> (-sin θ_i)) | Winfree separable mean field | Coup(θ, f=P(θj)R(θi), reduce mean, over all) | P, R |
| SwarmalatorCouplingInteraction | synchronization/_swarmalators | RateLaw on x and θ of both ends, /N | joint position + phase pair coupling | Coup(states=('x','θ'), f(member_j, member_i)->{x: ..., θ: ...}, reduce mean, over all) | the swarmalator kernel |
| AggregateAveragingInteraction | synchronization/_chaos | UpdateLaw(x_i, f((1-κ)x_i+κx_j)) | coupled maps, map after mixing | Coup(x, form='update', combine=g∘mix, over explicit pair) | logistic map |
| LogisticNodeElement | synchronization/_networks | UpdateLaw(x, (1-ε)f(x)+ε mean_j f(x_j)) (element law with `neighbors`) | coupled map lattice on a network | Coup(x, form='update', f(x_j)=logistic, reduce mean, over edges, combine=(1-ε)f(x_i)+ε·reduced) | logistic map |
| PendulumSpringInteraction | mechanics/_pendulums | RateLaw(θ1,θ2) spring through sin/cos Jacobians | spring between two pendulum bobs in angle coordinates | NEW EulerLagrange(potential k(l2 sinθ2-l1 sinθ1)^2/2) or Grad with a metric | Jacobian |
| SupportSpringInteraction | mechanics/_pendulums | RateLaw(x, θ) via `_support_response` (mass-matrix inverse) | spring between clock supports | NEW EulerLagrange (needs the mass matrix of ClockPendulum) | Jacobian / inverse inertia |

### M5. Alignment update of headings within a radius (discrete time, Vicsek family)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| VicsekParticleElement | active_matter/_flocks | UpdateLaw(θ, atan2(Σsin,Σcos)+U), UpdateLaw(x, x+v0 e(θ)) | polar Vicsek | core AlignmentInteraction(symmetry='polar') + Rate/Update self-propulsion (NEW shared SelfPropulsion) | nothing: duplicates core AlignmentInteraction |
| CohesiveParticleElement | active_matter/_flocks | idem + attraction/repulsion zones | Grégoire-Chaté | AlignmentInteraction(repulsion=, attraction=) + self-propulsion | the piecewise attraction profile (`_attraction`) is not AlignmentInteraction's unit-vector zones: needs Coup(update, f, reduce circular-sum) |
| NematicParticleElement | active_matter/_flocks | idem, flipped sides | nematic Vicsek | AlignmentInteraction(symmetry='nematic') + self-propulsion | nothing |
| NonreciprocalMoverElement | active_matter/_nonreciprocal | UpdateLaw(θ, atan2 of species-weighted sums), UpdateLaw(x) | non-reciprocal Vicsek | Coup(θ, form='update', f(member_j, member_i)=J(species_i,species_j)·e(θ_j), reduce=circular sum, over radius) + self-propulsion | species weight matrix |

### M6. Drift (+noise) of one member's own states: rates, OU, relaxation, constant rates, limit cycles

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| LangevinThermostatInteraction | common/particles/_langevin_thermostat | OU -> RateLaw(v, -γ/m v), NoiseLaw(v, sqrt(2γT)/m) | Langevin bath; LAMMPS `fix langevin` | Rate(x ord2, rate=-γ x'/m, noise=sqrt(2γT)/m) i.e. OU on the companion x' | coefficients; LAMMPS key |
| CoffeeCupElement | heat/_cooling | RateLaw(T, -k(T-room)) | relaxation to a mean | OU(state T, θ=k, mean=room, σ=0) | none |
| OrnsteinUhlenbeckParticleElement | active_matter/_particles | RateLaw(x,u), Noise(x), RateLaw(u,-u/τ), Noise(u) | AOUP | Rate(x, u, noise √2D_t) + OU(u, θ=1/τ, σ=√2D/τ) | none |
| ColoredNoiseParticleElement | soft_matter/_colored_noise | RateLaw(v, -γv+ξ), OU(ξ) | colored-noise Langevin | Rate(v, -γv+ξ) + OU(ξ, 1/τ, √2D/τ) | none |
| ActiveParticleElement | active_matter/_particles | RateLaw(x, s e(θ)), Noise(x), RateLaw(θ, ω), Noise(θ) | active Brownian / chiral | Rate(x, s·(cosθ, sinθ), noise √2D_t) + Rate(θ, ω, noise √2D_r) | self-propulsion = shared field realization |
| DepotParticleElement | active_matter/_particles | x/v pair, RateLaw(v), Noise(v), RateLaw(e) | depot (energy) active Brownian | Rate(x ord2, (-γ+de)x', noise) + Rate(e, q-ce-de|x'|^2) | depot kinetics |
| BrownianParticleElement | soft_matter/_diffusion | NoiseLaw(x, sqrt(2T/γ)) | overdamped Brownian | Rate(x, 0, noise=sqrt(2T/γ)) | none (γ is the divisor reused by M2 interactions) |
| ScaledWalkerElement | soft_matter/_diffusion | NoiseLaw(x, time-dependent) | scaled Brownian motion | Rate(x, 0, noise(t)) | amplitude law |
| FractionalWalkerElement | soft_matter/_anomalous | FractionalNoiseLaw(x, √2D, H) | fractional Brownian motion | Rate(x, 0, noise=fractional(H)) — needs noise spec | none |
| ModulatedParticleElement | soft_matter/_diffusion | NoiseLaw(x, s(x), convention), BoundaryLaw reflection | multiplicative noise, Itô/anti-Itô, reflecting interval | Rate(x, 0, noise=s(x), convention) + space boundary | the reflecting box (could be ContinuousSpace boundary) |
| WallParticleElement | soft_matter/_brownian | RateLaw, NoiseLaw(convention), BoundaryLaw(|z|) | Brownian near a wall | Rate(z, drift, noise, convention) + reflecting boundary | hydrodynamic mobility |
| PhaseOscillatorElement | synchronization/_kuramoto | RateLaw(θ, ω) | free phase | Rate(θ, ω) (or element own law) | none |
| SitedOscillatorElement | synchronization/_kuramoto | RateLaw(θ, ω) + position x | free phase with a site | same as PhaseOscillator + position state | duplicate |
| WinfreeOscillatorElement | synchronization/_pulses | RateLaw(θ, ω) | free phase | same | duplicate |
| SwarmalatorElement | synchronization/_swarmalators | RateLaw(θ, ω) + position x | free phase with position | same | duplicate |
| AdlerPhaseElement | nonlinear_oscillators/_oscillators | RateLaw(φ, ν+ε sin φ) | Adler | Rate(φ, ...) | own law |
| RotatingBallElement | mechanics/_rotations | RateLaw(θ, H/(m r^2 sin^2β)) | constant angular rate | Rate(θ, const) | own law |
| StuartLandauElement | nonlinear_oscillators/_oscillators | RateLaw(x,y) (with twist β) | Hopf normal form | own law | Stuart-Landau: shared realization (duplicate below) |
| StuartLandauElement | synchronization/_amplitude_death | RateLaw(x,y) (with time scale τ) | Hopf normal form | own law | DUPLICATE of the above, different params |
| MeanFieldAmplitudeElement | synchronization/_kuramoto | RateLaw(x,y) | Ott-Antonsen reduction (Stuart-Landau form) | own law | could reuse shared Stuart-Landau with μ=K/2-Δ |
| VanDerPolElement | nonlinear_oscillators/_oscillators | RateLaw(x ord2) | VdP | own law | shared with Forced VdP |
| ForcedVanDerPolElement | nonlinear_oscillators/_oscillators | RateLaw(x ord2) | VdP + A cos ωt | VanDerPol + Rate(x, A cos ωt) | duplicate except forcing |
| RosslerOscillatorElement | synchronization/_chaos | RateLaw ×3 | Rössler | own law | own |
| DipoleChargeElement | celestial_mechanics/_dipoles | RateLaw(x,y,z ord2) velocity-dependent (Lorentz) | Störmer problem | NEW EulerLagrange (velocity-dependent potential q x'·A) or own | own |
| RotatingFrameBodyElement | celestial_mechanics/_orbits | RateLaw(x,y ord2) with Coriolis 2y', centrifugal, 2 attractors | CR3BP | NEW EulerLagrange(L in rotating frame) or Grad(effective potential) + Rate(Coriolis) | own |
| IntegrateAndFireOscillatorElement | synchronization/_pulses | RateLaw(x, S-γx), CrossingLaw(max(x)-1, reset all, kick) | Mirollo-Strogatz pulse coupling | Rate(x, S-γx) + NEW threshold-crossing reset (group-global CrossingLaw) | the fire/kick reset |
| FrictionalBlockElement | soft_matter/_friction | x/v pair, RateLaw(x, where(moving,v,0)), RateLaw(v, stick-slip), Noise(v) pulsed | Burridge-Knopoff chain | Coup(x, laplacian over lattice) + own stick-slip | stick-slip friction law |

### M7. Implicit Euler-Lagrange equations (EquationLaw on mass matrices)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| PendulumElement | mechanics/_mechanics | EquationLaw(m l θ'' = -m g sin θ) | pendulum | NEW EulerLagrange(θ ord2, L=ml²θ'²/2+mgl cosθ) or Grad(θ, -g/l cos θ) | none |
| DoublePendulumElement | mechanics/_pendulums | 2 EquationLaw | double pendulum | NEW EulerLagrange((θ1,θ2), L) | Lagrangian |
| CartPendulumElement | mechanics/_pendulums | 2 EquationLaw | cart-pole | NEW EulerLagrange((y,θ), L, generalized force F on y) | Lagrangian |
| ParametricPendulumElement | mechanics/_pendulums | EquationLaw | Kapitza/parametric | EulerLagrange(θ, L(t)) + Rate(-cθ') | Lagrangian |
| BallisticPendulumElement | mechanics/_pendulums | RateLaw(x ord2, 0), EquationLaw, CrossingLaw(x, impact reset) | projectile + pendulum + plastic impact | EulerLagrange(θ) + NEW Reflection/impact CrossingLaw | momentum-transfer reset |
| ClockPendulumElement | mechanics/_pendulums | 2 EquationLaw with escapement sign() | Huygens clock | EulerLagrange((θ,x), L, dissipation) | escapement |
| RotatingSwingElement | mechanics/_rotations | 2 EquationLaw | chair-o-plane | EulerLagrange((φ,θ), L, torque, friction) | Lagrangian |
| GyroscopeElement | mechanics/_rotations | 3 EquationLaw | gimballed gyroscope | EulerLagrange((α,β,γ), L, torque) | Lagrangian |
| ScaleFactorElement | cosmology/_expansion | EquationLaw(a^2 a'' = ...) | Friedmann acceleration | Grad(a ord2, potential -Ωm H0²/(2a) - ΩΛ H0² a²/2) (a'' = -dV/da) | none |

### M8. State-crossing events with reflection resets (CrossingLaw)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| BouncingBallElement | mechanics/_mechanics | RateLaw(h ord2, -g), CrossingLaw(h, h' -> -e h') | inelastic bounce on a floor | Rate(h ord2, -g) + NEW Reflection(h, surface=h, restitution=e) | none |
| ElasticCollisionInteraction | mechanics/_mechanics | CrossingLaw(x_r-x_l-d, swap x') | equal-mass 1-d collision | NEW Reflection over a pair (relative coordinate, equal masses) | velocity swap |
| DiscBilliardElement | mechanics/_mechanics | x,y,vx,vy pairs, RateLaw, CrossingLaw(circle, reflect) | circular billiard | (x ord2) + NEW Reflection(surface=|x|^2-R^2) | none |
| SinaiBilliardElement | nonlinear_oscillators/_billiards | RateLaw, 5 CrossingLaw | square + disc | (x ord2) + Reflection ×5 (or ContinuousSpace reflecting box + Reflection(disc)) | geometry |
| MushroomBilliardElement | nonlinear_oscillators/_billiards | RateLaw, 5 CrossingLaw with `where` guards | mushroom table | Reflection with piecewise surfaces | geometry |

### M9. Continuum fields (RateLaw/NoiseLaw with spatial operators, element-own PDEs on FieldElement)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| HeatElement | heat/_heat | RateLaw(u, D lap u) | diffusion | Rate(u, D·laplacian(u)) (NEW Diffusion subclass) | none |
| WaveStringElement | waves/_strings | u/v pair: RateLaw(u,v), RateLaw(v, c² lap u) | wave equation | Rate(u ord2, c² lap u) | none (should be ord2, not u/v) |
| WaveFunctionElement | waves/_quantum | RateLaw(u, -D lap w + V w), RateLaw(w, ...) | Schrödinger as real pair | Rate on complex ψ: i ψ' = Hψ, or NEW Generator on the field | potential |
| ChargedWaveFunctionElement | waves/_charged | RateLaw(u,w) with gradient(·), ramp(t) | gauge-coupled Schrödinger | own | gauge terms |
| KuramotoSivashinskyElement | fluids/_flows | RateLaw(u, -div(u²/2) - lap u - lap² u) | KS | Rate(u, linear operator) + NEW Transport/Conservation(u, flux u²/2) | own |
| VorticityElement | fluids/_flows | RateLaw(ω, -div(uω) + ν lap ω + f) | 2-d NS vorticity | NEW Transport(ω, velocity=∇⊥Δ⁻¹ω) + Diffusion + Rate(f) | Biot-Savart velocity |
| BoussinesqElement | fluids/_flows | RateLaw(ω), RateLaw(θ) | Rayleigh-Bénard | Transport + Diffusion + Rate(buoyancy) | buoyancy coupling |
| MarangoniLayerElement | fluids/_marangoni | RateLaw(u, (r-1)u - 2lap u - lap²u + g u² - u³) | Swift-Hohenberg with quadratic term | NEW Grad(u, functional F[u], conserved=False) | F (SH + cubic) |
| SwiftHohenbergElement | pattern_formation/_patterns | RateLaw(u, ...) | SH | Grad(u, F_SH[u]) | F |
| AllenCahnElement | pattern_formation/_ordering | RateLaw(φ, φ-φ³+κ lap φ) | non-conserved gradient flow | Grad(φ, ∫(φ⁴/4-φ²/2+κ|∇φ|²/2)) | F |
| CahnHilliardElement | pattern_formation/_ordering | RateLaw(φ, lap(δF/δφ)) | conserved gradient flow | Grad(φ, F, conserved=True) | F |
| ThermalCahnHilliardElement | pattern_formation/_ordering | + NoiseLaw(φ, √2T, conserved=True) | Model B | Grad(φ, F, conserved=True, noise=conserved √2T) | none (subclass of CH) |
| NonreciprocalMixtureElement | pattern_formation/_conserved | RateLaw(φ, lap(...)), RateLaw(ψ, lap(...)) | non-reciprocal CH | Grad(φ, F, conserved) + Rate(lap((χ±α)·other)) | non-reciprocal term (not a gradient) |
| ActiveScalarElement | active_matter/_model_b | RateLaw(φ, lap μ - div(ζ lap φ ∇φ)), conserved noise | Active Model B | Grad(φ, F, conserved) + Rate(active terms) + conserved noise | λ|∇φ|² and ζ terms |
| GinzburgLandauElement | pattern_formation/_patterns | RateLaw(u,v) | CGLE | Stuart-Landau(shared) + Diffusion with complex coefficient | none |
| ChemotaxisElement | active_matter/_chemotaxis | RateLaw(ρ, D lap ρ - χ div(ρ∇c)), RateLaw(c, ...) | Keller-Segel | Diffusion + NEW Transport(ρ, velocity=χ∇c) + Rate(c, ρ-kc) | none |
| PolarFluidElement | active_matter/_collective | RateLaw ρ, v; advection(); NoiseLaw | Toner-Tu | Transport(ρ, v) + Rate(v, ...) + Diffusion + noise | Toner-Tu terms |
| ActiveFieldElement | criticality/_absorbing | RateLaw, NoiseLaw(e√ρ), RateLaw(ν), BoundaryLaw clip ≥0 | DP Langevin (conserved energy) | Rate(ρ, ..., noise √ρ) + Diffusion + clip | reaction terms |

### M10. Hyperbolic conservation laws (FluxLaw, finite volume)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| BurgersElement | fluids/_flows | FluxLaw(u, u²/2) | Burgers | NEW Conservation(u, flux=u²/2) | flux |
| ShallowWaterElement | fluids/_flows | FluxLaw(h, q), FluxLaw(q, q²/h+gh²/2) | Saint-Venant | Conservation((h,q), flux) | flux |
| WaveNumberElement | fluids/_kinetics | FluxLaw(k, sqrt(gk)) | wave-action / ray kinetics | Conservation(k, ω(k)) | dispersion |
| CarlemanGasElement | fluids/_kinetics | FluxLaw ×2 + RateLaw collisions | discrete-velocity Boltzmann | Conservation(f±, ±c f±) + React(f+ + f+ <-> f- + f-, k=κ) | none |

### M11. Count jumps, hops, reservoirs (JumpLaw on integer states)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| ReactingParticlesElement | statistical_mechanics/_absorbing_processes | JumpLaw(+1, κn+ε[n<1]), JumpLaw(-1, νn+λn(n-1)/V) | birth-death with absorbing state | React(n -> 2n, κ; n -> 0, ν; 2n -> n, λ/V; ∅ -> n, ε·[n<1]) mode='jump' | spontaneous term with indicator |
| ZoneElement | statistical_mechanics/_master_equations | JumpLaw(+1, kp(Na-N)), JumpLaw(-1, k(1-p)N) | two-state exchange | React/Trans count form (Na-N <-> N) | none |
| ParticleSiteElement | statistical_mechanics/_driven_particles | none (holder of n) | occupancy holder | core CompartmentElement (state n) | duplicate |
| TreeNodeElement | statistical_mechanics/_driven_particles | none (n + layout x) | occupancy + position | CompartmentElement + position | layout |
| ZeroRangeHopInteraction | statistical_mechanics/_driven_particles | JumpLaw(p g(n1), {n1:-1, n2:+1}), reverse | unit hop on an explicit link | NEW/extended Xfer(n, amount=1, rate=g(n_source), jump form, over explicit pair, p/q directions) | g(n)=1+b/n |
| ZeroRangeHopsInteraction | statistical_mechanics/_driven_particles | JumpLaw with shift_member(n, ±1) | ZRP on a ring | Xfer(n, 1, rate=p·g(n_i), over lattice edges, directed p/q) | g |
| ExclusionHopsInteraction | statistical_mechanics/_driven_particles | JumpLaw(p n(1-n_right), shift) | ASEP bulk | Xfer(n, 1, rate=p n_i(1-n_j), over edges, directed) | exclusion factor |
| ReservoirInteraction | statistical_mechanics/_driven_particles | JumpLaw(+1 inflow·room), JumpLaw(-1 outflow) | boundary reservoirs | Xfer(Source -> site, rate α(1-n)), Xfer(site -> Sink, rate β[n>0]) — TransferLaw already has Source/Sink | none |
| EhrenfestUrnElement | statistical_mechanics/_urns | UpdateLaw(n, n+1-2·Bernoulli(n/N)) | Ehrenfest urn (discrete time) | Xfer between two urns, discrete-time, rate ∝ n | own (1-line) |

### M12. Renewal jumps and flights of continuous positions (JumpLaw with `waiting=` / per-step draws)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| ContinuousTimeWalkerElement | soft_matter/_anomalous | JumpLaw(waiting=Pareto, reset x+N(0,σ)) | CTRW | NEW Jump/Move(x, step=Gaussian(σ), waiting=Pareto(α, τ0)) compiled | none |
| LevyWalkerElement | soft_matter/_anomalous | RateLaw(x, v), JumpLaw(waiting=Pareto, reset v=speed·unit) | Lévy walk | Rate(x, v) + Jump(v, reset random direction, waiting Pareto) | none |
| TransientWalkerElement | soft_matter/_anomalous | NoiseLaw(x, √2D), JumpLaw(waiting=Exp(D^-γ), reset D~Beta) | annealed transient time | Rate(x, 0, noise √2D) + Jump(D, waiting, reset) | none |
| LevyFlierElement | soft_matter/_anomalous | UpdateLaw(x, x + Pareto·unit) | Lévy flight, discrete time | Move(x, step=Pareto length × random direction) compiled UpdateLaw | none |
| TrapWalkerElement | statistical_mechanics/_traps | JumpLaw(waiting=Exp(τ0 e^{E/T}), reset x~U, E~Exp) | Bouchaud trap | Jump(waiting, reset) | none |

### M13. Spin sampling and Glauber flips

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| IsingSpinElement | statistical_mechanics/_spins | SampleLaw(EnergySpec(-J s Σnbr - h s, NEAREST), MoveSpec flip, SamplerSpec) | lattice Metropolis/heat-bath/Wolff sweeps | NEW Sampling(state=spin, energy, move='flip', sampler, over NEAREST) | local energy |
| PottsSpinElement | statistical_mechanics/_spins | SampleLaw(... 'random-value', q values) | Potts | Sampling(move='random-value') | energy |
| XYSpinElement | statistical_mechanics/_spins | SampleLaw(... 'rotate', width) | XY, continuous moves | Sampling(move='rotate') | energy |
| BondSpinElement | statistical_mechanics/_spin_glasses | SampleLaw with per-bond J states, neighbors(reduce='each') | Edwards-Anderson | Sampling(energy reading ordered neighbour tuple and bond states) | bond layout |
| MattisSpinElement | statistical_mechanics/_spin_glasses | inherits BondSpin + pattern state | Mattis | BondSpin subclass | pattern |
| GraphIsingSpinElement | statistical_mechanics/_graph_spins | SampleLaw(AdjacencyNeighborhood, sampler 'colored') | Ising on graph | Sampling(over adjacency, schedule colored) | energy |
| FullyConnectedSpinElement | statistical_mechanics/_fully_connected_spins | JumpLaw(Glauber rate, {s:∓2}) with mean / total(weights=J_ij) | Curie-Weiss / SK continuous-time Glauber | Trans(spin, -1<->+1, rate=1/(1+e^{2 s h_i/T}), h_i = Coup reduce mean or weighted total) | local field |

### M14. Learning / choice updates

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| LearningAgentElement | criticality/_learning | UpdateLaw(Q, EMA of Σ_j J_ij σ_j), UpdateLaw(σ, logit draw) | Q-learning SK game | Choice(σ, options ±1, utility=Q, rule logit, intensity 2/T) + Coup(Q, form update, weights J, combine EMA) | none |
| LearnerElement | criticality/_evolutionary_learning | 5 UpdateLaw incl. min-fitness replacement | evolutionary SK game (Bak-Sneppen-like) | Choice(logit) + Coup(update, weights) + ExtremalLaw-like replacement | extremal replacement rule |

### M15. Cellular automata, whole-lattice drives, relaxation cascades (LocalRuleLaw / DriveLaw / RelaxationLaw)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| SandpileCellElement | criticality/_criticality | RelaxationLaw(Drive(drop grain), LocalRule(topple)) | BTW sandpile | core ThresholdInteraction(mode='redistribute', share=1, drive=_drop_grain) | nothing: duplicates core ThresholdInteraction |
| MannaCellElement | criticality/_criticality | RelaxationLaw(Drive, Drive(random topple)) | Manna stochastic sandpile | ThresholdInteraction with a random-share mode (missing) | random redistribution |
| ForestElement | criticality/_criticality | LocalRuleLaw(stochastic) | Drossel-Schwabl forest fire | NEW LocalRule interaction (CA) or Trans discrete-time (empty->tree p, tree->burning if nbr burning or f, burning->empty) | rule |
| SpreadingForestElement | criticality/_criticality | idem, triangular/graph neighbourhood | idem | same | rule |
| ContactSiteElement | statistical_mechanics/_processes | LocalRuleLaw(stochastic), 1-e^{-rate τ share} | discrete-time contact process | core SpreadingInteraction in a discrete-time form (missing) | none |
| PercolationSiteElement | statistical_mechanics/_processes | LocalRuleLaw(burn) | site-percolation burning | LocalRule CA | rule |
| Q2RSpinElement | statistical_mechanics/_processes | LocalRuleLaw((spin, phase), checkerboard) | Q2R reversible CA | LocalRule CA | rule |
| ExclusionSiteElement | statistical_mechanics/_processes | DriveLaw(whole-array hop) | discrete-time TASEP | Xfer(n, 1, rate p, over edges, discrete-time, exclusion) + Source/Sink | sequential update order |
| FHPSiteElement | fluids/_lattice_gases | 3 LocalRuleLaw (collide, propagate, count) | FHP lattice gas | LocalRule CA | collision table |
| TrapSiteElement | soft_matter/_traps | LocalRuleLaw along rows | discrete trap walk on a lattice | LocalRule CA | rule |
| MicrostateElement | statistical_mechanics/_master_equations | LocalRuleLaw(P, Glauber rates of U) | discrete master equation | Coup(P, form='update', f(member_j, member_i)=w_ji P_j - w_ij P_i, over lattice edges) | Glauber rate |

### M16. Agent rules (RuleLaw)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| TumblerElement | active_matter/_particles | RuleLaw(run_and_tumble) | run and tumble | Move(step=heading·speed) + Trans(θ, uniform redraw, rate) | none |
| BinaryCollisionsInteraction | active_matter/_collisions | RuleLaw(fly) with KDTree, substeps | Bertin binary collisions | own agent rule | collision-averaging rule |
| FlockerElement | active_matter/_collisions | none (θ holder, agent view) | holder | Element | none |
| WalkerElement | statistical_mechanics/_processes | RuleLaw(walk: stick next to stuck, else neighbour step) | DLA | core RandomWalkInteraction(step='neighbor') + stick condition | aggregation rule |

### M17. Quantum generators and dissipators (GeneratorLaw / DissipatorLaw), and Hilbert-space holders

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| XXZExchangeInteraction | statistical_mechanics/_quantum_spins | GeneratorLaw(J(XX+YY+ΔZZ)) per edge | Heisenberg exchange | NEW Generator(operator per edge) on NeighborInteraction | operator |
| ZeemanFieldInteraction | statistical_mechanics/_quantum_spins | GeneratorLaw(-h Σ Z) | uniform field | Generator(operator per member) | none |
| DephasingInteraction | statistical_mechanics/_quantum_spins | DissipatorLaw(Z_i, γ) per member | dephasing | NEW Dissipator(operator per member, rate) | none |
| LatticeHamiltonianInteraction | waves/_quantum | GeneratorLaw(-D L + V) | tight-binding | Generator(operator) | none |
| PositionDephasingInteraction | waves/_quantum | DissipatorLaw(site projector, γ) per site | position dephasing | Dissipator(operators=projectors) | none |
| SpinElement | common/quantum/_quantum | none (ψ ∈ C^n, hilbert) | spin-j holder | NEW core HilbertElement(psi, hilbert) | default `spin_space` |
| QubitElement | common/quantum/_quantum | none | qubit | HilbertElement | Pauli space |
| ModeElement | common/quantum/_quantum | none | bosonic mode | HilbertElement | `mode_space` |
| SiteElement | common/quantum/_quantum | none | particle on sites | HilbertElement | `site_space` |

### M18. Pure state holders (no laws)

| class | module | laws used | mechanism | built from | stays own |
|---|---|---|---|---|---|
| LinkElement | synchronization/_networks | none (edge weight w) | edge state | Element on edges | none |
| LogisticUnitElement | synchronization/_chaos | none (x) | map state | Element | none |

(ParticleSiteElement, TreeNodeElement, FlockerElement, quantum holders listed above.)

## 1. GAPS (mechanisms no general class of the brief covers)

1. **EulerLagrangeInteraction(states, lagrangian, dissipation=None, generalized_force=None)** — EquationLaw of
   `d/dt ∂L/∂q' - ∂L/∂q = Q - ∂R/∂q'` over ord2 states, derived symbolically (Sage `diff`, as `PotentialInteraction`
   already does). Needed by PendulumElement, DoublePendulumElement, CartPendulumElement, ParametricPendulumElement,
   BallisticPendulumElement, ClockPendulumElement, RotatingSwingElement, GyroscopeElement (all hand-written mass-matrix
   EquationLaws), PendulumSpringInteraction / SupportSpringInteraction (springs between generalized coordinates: today
   hand-coded Jacobians and an inverse inertia in `_support_response`), and velocity-dependent potentials that
   GradientInteraction cannot express: RotatingFrameBodyElement (Coriolis), DipoleChargeElement (magnetic). It is the
   strict generalisation of GradientInteraction (L = T - U); interactions would add terms to a shared Lagrangian of the
   system (sum of L contributions, then one EquationLaw per coordinate), which needs the compiler to sum Lagrangian
   contributions before differentiating.
2. **ReflectionInteraction(state, surface, restitution=1, over=None)** — CrossingLaw on `g(x)=0` with reset
   `x' <- x' - (1+e)(x'·∇g)∇g/|∇g|^2` on the companion derivative of an ord2 state. BouncingBallElement,
   DiscBilliardElement, SinaiBilliardElement, MushroomBilliardElement (piecewise surfaces via `where`),
   ElasticCollisionInteraction (pair form, relative coordinate). Today each writes its CrossingLaws and resets by hand,
   on x/v first-order pairs.
3. **SamplingInteraction(state, energy, move, sampler, over, schedule)** — lowers to SampleLaw. Ising, Potts, XY,
   BondSpin, Mattis, GraphIsing. The brief's ChoiceInteraction(rule='metropolis') lowers to a synchronous UpdateLaw for
   every member at once, which is the wrong dynamics for a local energy (all-sites-at-once Metropolis oscillates); it
   also cannot propose continuous moves (XY `rotate`) or cluster moves (Wolff).
4. **Field operators as general rates** (subclasses of RateInteraction, named by operator): `DiffusionInteraction(state,
   D)` (rate `D·laplacian`), `TransportInteraction(state, velocity)` (rate `-div(velocity·state)`; Keller-Segel,
   vorticity/Boussinesq advection, Toner-Tu, KS), and a **ConservationInteraction(state, flux, limiter, riemann)** that
   lowers to the current FluxLaw (Burgers, ShallowWater, WaveNumber, CarlemanGas). None is in the brief.
5. **Generator/Dissipator interactions** — `GeneratorInteraction(operator)` and `DissipatorInteraction(operator, rate)`
   per member / per edge / once, lowering to GeneratorLaw/DissipatorLaw: XXZExchange, Zeeman, Dephasing,
   LatticeHamiltonian, PositionDephasing all re-implement this with one-line laws. Plus a core state holder
   **HilbertElement(psi, hilbert)**: SpinElement/QubitElement/ModeElement/SiteElement are four copies differing only in
   the default HilbertSpec.
6. **Renewal / jump with drawn waiting times** — `JumpInteraction(state, reset, rate=None, waiting=None)` (JumpLaw with
   `waiting=` and `reset=`): ContinuousTimeWalker, LevyWalker, TransientWalker, TrapWalker. The brief's
   TransitionInteraction is only exponential-rate jumps of a discrete state between `from_`/`to`; MovementInteraction
   lowers to agent RuleLaws, not compiled jumps of a continuous position.
7. **Local stencil rule** — `LocalRuleInteraction(states, over, rule, reads, schedule, stochastic)` lowering to
   LocalRuleLaw: Forest, SpreadingForest, Percolation, Q2R, FHP, TrapSite. These are true CA rules on the ordered
   neighbourhood (not a reduction), so CouplingInteraction cannot express them (see CORRECTIONS for `reduce='each'`).
8. **Threshold-crossing reset of a group** — IntegrateAndFireOscillatorElement uses a group-global CrossingLaw on
   `max(x)-1` with a reset of every member (pulse coupling). Not covered by ThresholdInteraction (discrete cascades) nor
   by Reflection; a `CrossingInteraction(guard, reset)` general class would cover it and BallisticPendulum's impact.

## 2. BRIEF CORRECTIONS

**ParticleElement.** Today: core, physics-named, x/v first-order pair with `velocity_of='x'`, own law `RateLaw(x, v)`,
mass `m`; LAMMPS keys on `issubclass(ParticleElement)` and names `'x'`, `'v'`, `'m'`; VelocityVerlet pairs positions
with velocities from symbol-valued rates. Meanwhile half of physics already uses `StateSpec(order=2)` (oscillators,
pendulums, bouncing ball) while the other half uses hand x/v pairs (billiards, Depot, FrictionalBlock, WaveString u/v,
ParticleElement). Proposal: **remove ParticleElement from core**. The general notion is "an element with a vector
position state of order 2" — that is a StateSpec (`StateSpec(vector=True, position=True, order=2)`), not a class.
Keep a thin realization `physics/common/particles/ParticleElement(Element)` (x: position ord2, `m` parameter, no laws)
because social/crowds and biology/collective_behaviour also use it: either they import the physics one, or (better,
since three fields share it) the shared element lives in core under an operation name (e.g. `PointElement`: a point of a
continuous space whose position has order 2 and one positive weight parameter). `velocity_of` then becomes redundant
(companion `x_d1`), as `passes/_particles.py` and `_neighborhood_kernel.py` already fall back to `x_d1`.
LAMMPS must then (a) accept elements by declaration (one vector position state of order 2 and a scalar weight
parameter) instead of `issubclass`, (b) read velocity as the companion `x_d1`, (c) read mass from the divisor parameter
the interactions name, and (d) check external laws against `x_d1`.

**GradientInteraction(state, potential)** — insufficient:
- needs **`divisor=`** (name of a member parameter/state dividing minus the gradient: mass for ord2, friction γ for
  ord1, 1/Γ for Landau). Evidence: every overdamped force divides by `target.gamma` (OpticalTrap, Speckle, Washboard,
  Sawtooth) and every inertial one by `m`; LAMMPS needs this name to set atom masses and convert accelerations to
  forces (`forces = accel[v] * masses`). A divisor hidden in the user's lambda breaks LAMMPS.
- needs **roles** (one or several bound members, e.g. first/second for Gravity/Spring; target for the rest) since the
  potential is a function of several members' states.
- on fields it needs the **variational derivative** of a functional and **`conserved=True`** (`laplacian(δF/δφ)`) to
  cover AllenCahn, CahnHilliard, ThermalCahnHilliard, Swift-Hohenberg, Marangoni, Active Model B, NonreciprocalMixture.
- must admit time in the potential (DrivenAnchor, Sitnikov) and a multiplicative gate (OpticalTrap blink).

**CouplingInteraction(state, function, reduce, over, lag, delay, form)** — insufficient:
- `function(x_j, x_i)` of state values is too narrow: CuckerSmale weights by the distance of another state, Swarmalator
  couples two states jointly, NonreciprocalMover weights by `species`, EdgeCoupling reads the edge state `w`. Use
  **`function(member_j, member_i[, edge])` of namespaces**, returning one rate or a dict per state; `state` may be a
  tuple (GlobalCoupling couples (x, y)).
- `potential=` pair form must target the highest derivative of **any order**: SoftRepulsion and PhoreticAttraction are
  pair potentials on ord1 positions (SoftRepulsion is literally SoftSphere's energy), LJ/SoftSphere on ord2; same
  `divisor=` as GradientInteraction; must keep `cutoff`, `period` (minimum image), and expose `potential(namespace, r)`
  so LAMMPS can write `lj/cut` / Lepton pair styles (today via `_get_pair_energy`).
- `over` must include: explicit pairs of separate elements (DelayedCoupling, DiffusiveCoupling, AggregateAveraging,
  ZeroRangeHop), all-to-all mean field with **separate source and target groups** (Kuramoto chimera
  `MeanFieldCouplingInteraction` couples group 'first' from group 'second'), a dense weight matrix (SK, LearningAgent,
  FullyConnectedSpin use `total(·, weights=J)`), and `include_self` (PhaseLagInteraction exists only to add back the
  j=i term the pair binding omits).
- `reduce` must include a circular mean / vector sum of unit vectors (Vicsek family: `atan2(Σ sin, Σ cos)`), and
  `'each'` (ordered neighbour tuple, BondSpin) for stencils.
- `form='update'` needs a **combine** step, not additive: LogisticNode `x <- (1-ε)f(x_i) + ε·mean f(x_j)`,
  AggregateAveraging `x <- f((1-κ)x_i + κ x_j)`, LearningAgent `Q <- (1-α)Q + α·Σ J σ`.
- Mean-field reductions of `sin(θj - θi - α)` should be factorised by the reducer (MeanFieldCoupling hand-expands to
  `<sin>cos - <cos>sin` for O(N)).

**RateInteraction(state, rate, noise)** — `noise` must be a spec, not only an amplitude: `convention`
(Itô / Stratonovich / anti-Itô: ModulatedParticle, WallParticle), `hurst` (FractionalWalker -> FractionalNoiseLaw),
`conserved` (ThermalCahnHilliard, ActiveScalar), time/state-dependent amplitude (ScaledWalker, ActiveField `√ρ`). It
also needs a `divisor=` like Gradient (HarmonicForce divides by `m`; constant force / damping on ord2 positions; the
LangevinThermostat as OU on the companion `x'` with θ = γ/m, σ = √(2γT)/m). LAMMPS keys on the *classes*
ConstantForce/Damping/LangevinThermostat; they must survive as distinct thin subclasses (MRO lookup) with parameter
names `F`, `gamma`, `temperature` and role `target`.

**TransitionInteraction** — needs a discrete-time lowering (probability `1-exp(-rate·dt)`), used by ContactSiteElement
and Forest; continuous-time JumpLaw is not what those models run. Also `waiting=` (see GAP 6).

**ThresholdInteraction (core, not in brief list but used by criticality)** — SandpileCellElement duplicates it; Manna
needs a random-share mode (each unit to a random neighbour).

**TransferInteraction(source, target, amount)** — the brief lowers it to agent RuleLaws only. ZeroRangeHop(s),
ExclusionHops, Reservoir, discrete TASEP (ExclusionSiteElement), Ehrenfest need a **compiled count form**: unit JumpLaw
along (directed) lattice/network edges with rate `f(n_source, n_target)` and `p`/`q` directions, with core `Source`/
`Sink` ends for reservoirs (already in `laws/_transfer.py`), and a discrete-time form.

**MovementInteraction(step)** — agent-only lowering misses compiled flights: LevyFlier (UpdateLaw x + Pareto·unit), CTRW
(JumpLaw with waiting). Give it an equation lowering on a position state.

**ReactionInteraction** — fine for ReactingParticles/Zone/Carleman collisions, but needs an additive indicator
propensity (ε·[n<1]) — already possible with `propensity=`.

**ChoiceInteraction** — fine for LearningAgent (logit over ±1 from a utility state); not for lattice spins (GAP 3).

**Element-level neighbour reductions.** Vicsek, LogisticNode, BondSpin, Ising, FullyConnectedSpin put `neighbors(...)`
/ `mean(...)` inside *element* laws (reading `s.other`). If the brief's rule is "interactions couple, elements hold
own dynamics", these move to CouplingInteraction/Sampling; core AlignmentInteraction already implements Vicsek/Nematic
/Grégoire-Chaté and is unused by physics.

## 3. FIELD-SHARED (thin subclasses for `physics/common/<family>`)

- `physics/common/particles` (exists): keep Gravity, HarmonicPotential, Spring, LennardJones, SoftSphere,
  LangevinThermostat as thin subclasses of Gradient/Coupling(potential)/OU; **move in** SoftRepulsionInteraction
  (= SoftSphere on ord1, delete duplicate), PhoreticAttraction (inverse-distance pair potential), DrivenAnchor (=
  HarmonicPotential with time-dependent centre), ExponentialWell, Magnet, HarmonicForce (periodic force: subclass of
  the constant-force Rate), the ParticleElement realization (unless promoted to core), and the overdamped traps
  (OpticalTrap, TiltedWashboard, Speckle are Gradient on ord1 with divisor γ: "external potential" family). Also a
  BrownianParticleElement (Rate(x, 0, noise=√(2T/γ))) is used by diffusion, ratchets, traps: it is shared.
- `physics/common/oscillators` (new): **PhaseOscillatorElement** (θ' = ω) — four copies: PhaseOscillator,
  SitedOscillator, WinfreeOscillator, SwarmalatorElement; **StuartLandauElement** — two classes with the same name in
  nonlinear_oscillators and synchronization (+ MeanFieldAmplitudeElement and GinzburgLandau reaction part are the same
  normal form); **VanDerPolElement** with optional forcing (VanDerPol + ForcedVanDerPol); logistic map (`_logistic`
  defined twice: _chaos and _networks; LogisticUnitElement + LogisticNodeElement); the Kuramoto couplings
  (PhaseCoupling, MeanFieldCoupling, EdgeCoupling, PolarAlignment, Delayed) as thin `Coup(θ, sin(θj-θi-α))` subclasses
  differing only in `over`.
- `physics/common/active` (new): **self-propulsion** `x <- x + v0 e(θ)` / `x' = v0 e(θ)` shared by ActiveParticle,
  Vicsek, Cohesive, Nematic, NonreciprocalMover, Tumbler, BinaryCollisions (`_step`, `_heading`, `_polarization`
  observables are duplicated in _flocks, _alignment, _collisions, _nonreciprocal); the Vicsek family as thin
  AlignmentInteraction/Coupling subclasses.
- `physics/common/fields` (new): Diffusion/Transport operators and the Ginzburg-Landau-type free energies (Allen-Cahn,
  Cahn-Hilliard, Thermal CH, Active Model B, NonreciprocalMixture, Swift-Hohenberg, Marangoni share
  `φ^4`/`|∇φ|^2`/`(1+∇²)^2` functionals) as Gradient(conserved=...) subclasses.
- `physics/common/lattice_gases` or `statistical_mechanics` shared: exclusion/zero-range hop and reservoir transfers
  (ZeroRangeHop, ZeroRangeHops, ExclusionHops, Reservoir, ExclusionSite TASEP) as Transfer subclasses;
  ParticleSiteElement -> CompartmentElement.
- `physics/common/spins` (new): Ising/Potts/XY/Bond/Mattis/GraphIsing as Sampling subclasses with named energies; the
  `_square`, `_magnetization` observable and SPIN_IMAGE helpers are duplicated across _spins, _spin_glasses,
  _fully_connected_spins.
- `physics/common/quantum` (exists): reduce the four holders to thin subclasses of a core HilbertElement; add the
  per-member/per-edge Generator and Dissipator realizations (Zeeman, Dephasing, XXZ, LatticeHamiltonian,
  PositionDephasing).
- `physics/common/mechanics` (new, or in mechanics): billiard reflections (Sinai, Mushroom, Disc, BouncingBall) on
  ReflectionInteraction; pendulum Lagrangians on EulerLagrange.
- `physics/common/criticality`: the forest-fire `_burn` rule is duplicated (criticality `_burn` with growth/lightning,
  statistical_mechanics `_burn` percolation) — same CA family.

## 4. Field-flavoured core names and meanings (flag)

- **ParticleElement** — physics noun and x/v/m semantics; see CORRECTIONS.
- **PotentialInteraction**, **PairPotentialInteraction**, **PairForceInteraction** — "potential energy", "force",
  arguments `position`, `velocity`, `mass`: physics. -> GradientInteraction (+ divisor) and Coupling(potential=).
- **ConstantForceInteraction** (`F`, `velocity`, `mass`), **DampingInteraction** (`gamma`, `velocity`, `mass`) —
  physics; -> Rate subclasses (constant rate; linear rate in the first derivative). Keep thin physics subclasses for
  LAMMPS keys.
- **OrnsteinUhlenbeckInteraction** — named process, mathematically fine; but it targets a state of order 1 only
  (LangevinThermostat relies on the separate `v` state).
- **AlignmentInteraction** — "heading", "flock", `repulsion`/`attraction` zones: collective-motion flavoured; it is a
  circular-mean update coupling (Coup form='update', reduce=circular sum, over radius).
- **FluxLaw** ("physical flux of a conserved state") — the operation is a hyperbolic conservation law
  `u_t + div F(u) = 0`; rename by operation (e.g. ConservationLaw / DivergenceLaw).
- **DriveLaw** ("dropping a grain on a random site") — the operation is a whole-array (stochastic) map applied once per
  step; used by ExclusionSiteElement for TASEP, nothing to do with grains.
- **RelaxationLaw** (drive + relax until stable) — "slow drive / fast relaxation" is SOC jargon; the operation is
  iterate-to-fixed-point with an activity record.
- **EncounterLaw / DestructionLaw / ProliferationLaw** ("active particles", "candidate", "test" particle, "field") and
  **DensityElement** ("activity class") — kinetic theory of active particles vocabulary; operations are bilinear
  transition / loss / gain kernels on a distribution over discrete classes.
- **SampleLaw / EnergySpec** — "energy" is physics; the operation is MCMC with acceptance from a log-weight change.
- **GeneratorLaw** ("adds to the Hamiltonian", ħ = 1) and **DissipatorLaw** ("makes a quantum system open") — names are
  mathematical (Lindblad generator), descriptions quantum-specific; acceptable.
- **StateSpec(position=True, velocity_of=...)** — `velocity_of` is a physics concept made redundant by order-2 states.
- Not physics but same issue, for the record: ForagingInteraction, RegrowthInteraction, SpreadingInteraction,
  ImitationInteraction (field nouns).
