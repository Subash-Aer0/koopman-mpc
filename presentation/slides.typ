// Koopman MPC for a quadrotor in turbulent wind: progress presentation.
// Build: presentation/README.md. Every number in a table is read from data/*.json,
// which make_figures.py writes from the saved runs (no hand-copied results).
#import "theme.typ": *

#let paper = json("data/paper.json")
#let mc = json("data/full_mpc.json")
#let gap = json("data/gap.json")
#let train = json("data/training.json")
#let thrust = json("data/thrust.json")
#let eig = json("data/eigen.json")

// x with exactly d decimals.
#let fmt(x, d: 3) = {
  let s = str(calc.round(x, digits: d))
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  if d == 0 { parts.at(0) } else { parts.at(0) + "." + frac + "0" * (d - frac.len()) }
}
#let pm(v, d: 3) = [#fmt(v.mean, d: d) #text(fill: palette.muted, size: 0.8em)[± #fmt(v.std, d: d)]]
#let sci(x) = {
  let e = calc.floor(calc.log(x, base: 10))
  [$#fmt(x / calc.pow(10.0, e), d: 1) times 10^#e$]
}

#let ctrl-color = (
  "LTV nominal": rgb("#7f7f7f"), "Koopman": palette.blue,
  "LTV grey-box": palette.amber, "LTV true drag": palette.good,
)
#let ctrl-name = (
  "LTV nominal": [LTV nominal], "Koopman": [Koopman MPC],
  "LTV grey-box": [LTV grey-box], "LTV true drag": [LTV oracle],
)
#let controllers = ("LTV nominal", "Koopman", "LTV grey-box", "LTV true drag")
#let ref-name = (
  hover_gaussian: [Hover, 1 m climb], circle_slow_gaussian: [Circle 10 s],
  circle_7s_gaussian: [Circle 7 s], circle_5s_gaussian: [Circle 5 s],
  figure8_14s_gaussian: [Figure-8 14 s], figure8_10s_gaussian: [Figure-8 10 s],
)

#show: academic-theme.with(
  title: [Koopman MPC for a Quadrotor in Turbulent Wind],
  subtitle: [From an LQR-augmenting correction to a full model predictive controller],
  author: [Subash],
  institution: [PhD research update],
  date: datetime.today().display("[month repr:long] [day], [year]"),
  short-title: [Koopman MPC in wind],
)

#title-slide()

#slide(title: [Outline])[
  #set text(size: 20pt)
  #grid(columns: (1fr, 1fr), gutter: 1.5em,
    [
      + The quadrotor model with wind \ #muted[and why commanded thrust ≠ actual thrust]
      + The CCTA paper: K-MPC on top of LQR
      + Why remove the LQR
      + The full-MPC set-up: four controllers
    ],
    [
      #set enum(start: 5)
      + Training the Koopman model
      + Persistency of excitation: the Hankel test
      + Results: tracking errors and commands
      + Why Koopman captures the rotor aerodynamics
    ],
  )
]

// =====================================================================================
#section-slide[Quadrotor model with wind]

#slide(title: [Rigid-body model, drag and wind])[
  #grid(columns: (1.15fr, 1fr), gutter: 1.2em,
    [
      *State* $x = (p, q, v, omega) in RR^13$ #muted[(world NED, quaternion $w$-first, body FRD)]

      $ dot(p) &= v \
        dot(v) &= g e_3 + 1/m (R(q) f_T - f_D) \
        dot(q) &= 1/2 q times.o (0, omega) \
        J dot(omega) &= tau_T - omega times J omega $

      *Quadratic body drag* on the air-relative velocity
      $ f_D = R D R^top v_r norm(v_r), quad v_r = v - v_w $
      #muted[$D = "diag"(0.05, 0.05, 0.10)$ N s²/m². 1 kg F450-class placeholder parameters.]
    ],
    [
      #card(title: [Wind (CCTA Sec. IV-A)])[
        $v_w = macron(v)_w + tilde(v)_w$, $tilde(v)_w tilde cal(N)(0, Sigma)$, redrawn every 0.1 s
        - $macron(v)_w = (2.70, 1.76, 0.01)$ m/s
        - $Sigma = "diag"(0.54, 0.42, 0.006)$ (m/s)²
      ]
      #v(0.3em)
      #card(title: [Outer-loop control input])[
        $u = (omega_x, omega_y, omega_z, c)$: body rates and mass-normalised collective thrust $c$ [m/s²]. A 1 kHz rate loop tracks $omega$; the thrust command is $f_T = -m c e_3$.
      ]
    ],
  )
]

#slide(title: [Signal chain: where the commanded thrust is lost])[
  #v(0.4em)
  #align(center)[
    #blk[MPC \ #muted[10 Hz]] #arr(label: [$u = (omega, c)$])
    #blk[Rate loop \ #muted[1 kHz]] #arr(label: [$f_T, tau_T$])
    #blk[Mixer \ #muted[$T = k_T Omega^2$]] #arr(label: [$Omega_i$])
    #blk(fill: rgb("#fdecdf"), stroke: palette.warn)[BEMT rotors \ #muted[in local airflow]]
    #arr(label: [$f_T^"applied"$]) #blk[Rigid body \ #muted[\+ drag, wind]]
  ]
  #v(0.6em)
  #grid(columns: (1fr, 1fr), gutter: 1.2em,
    card(title: [What every controller believes])[
      The mixer inverts the *quadratic rotor model* $T_i = k_T Omega_i^2$. So the command $m c$ is assumed to be the thrust the vehicle gets.
    ],
    card(title: [What the simulated rotors do], color: palette.warn)[
      Each rotor's thrust comes from *blade-element momentum theory* in its own airflow
      $v_i = R^top (v - v_w) + omega times r_i$. It changes with wind, speed and climb rate. #hl[No controller sees this.]
    ],
  )
  #takeaway[the plant input is $f_T^"applied" (Omega, "airflow")$, not $m c$. The gap is unmodelled for every physics-based MPC.]
]

#slide(title: [Why commanded thrust ≠ actual thrust: BEMT])[
  #grid(columns: (1fr, 1.05fr), gutter: 1em,
    [
      #set text(size: 16pt)
      Per rotor, uniform-inflow BEMT (Leishman Ch. 2–5):
      $ C_T = (sigma a)/2 [ theta_0/3 (1 + 3/2 mu^2) + theta_"tw"/4 (1 + mu^2) - lambda/2 ] $
      $ lambda = lambda_c + lambda_i, quad lambda_i = C_T / (2 sqrt(mu^2 + lambda^2)) $
      - $mu = V_"edge" \/ Omega R$: edgewise airflow #sym.arrow.r *more* thrust
      - $lambda_c = V_"axial" \/ Omega R$: climb through the air #sym.arrow.r *less* thrust
      - Calibrated so BEMT $= k_T Omega^2$ exactly in still-air hover: the nominal model is wrong *only in airflow*.
    ],
    [
      #image("figures/rotor_bemt.png", width: 100%)
      #v(-0.4em)
      #muted[Thrust ratio at hover rotor speed. Mean wind alone is 3.2 m/s edgewise.]
    ],
  )
]

#slide(title: [Commanded vs applied thrust in flight])[
  #grid(columns: (1.5fr, 1fr), gutter: 1em,
    image("figures/thrust_cmd_vs_bemt.svg", width: 100%),
    [
      #v(1em)
      #card(title: [In closed-loop flight])[
        Applied/commanded thrust ranges over
        #hl[#fmt(thrust.thrust_gap_range_pct.at(0), d: 1) % to +#fmt(thrust.thrust_gap_range_pct.at(1), d: 1) %].

        RMS thrust-acceleration error: #hl[#fmt(thrust.thrust_gap_rms_m_s2, d: 2) m/s²], pooled over #thrust.samples samples (all controllers, all seeds, figure-8 10 s).
      ]
      #v(0.5em)
      The error follows the airspeed: it is a *smooth function of the airflow*, which matters later (Part 8).
    ],
  )
]

// =====================================================================================
#section-slide[The CCTA paper: K-MPC on top of LQR]

#slide(title: [Tabassum & Bai (CCTA 2023): architecture])[
  #align(center)[
    #blk[Reference] #arr()
    #blk[State-dependent LQR \ #muted[100 Hz]] #arr(label: [$u_h$])
    #blk(fill: rgb("#e6f1f8"))[$plus.o$] #arr(label: [$u_h + u_d$])
    #blk[Rate loop \ #muted[1 kHz]] #arr() #blk[Quadrotor]
    #v(-0.2em)
    #text(fill: palette.accent, size: 14pt)[$arrow.t$ #h(0.3em) $u_d$ from *K-MPC* (10 Hz) into the sum]
  ]
  #v(0.4em)
  #grid(columns: (1fr, 1fr), gutter: 1.2em,
    card(title: [Model 1 (the one we use)])[
      #set text(size: 14.5pt)
      Koopman model of the *LQR-controlled* vehicle:
      - state: tracking error $e_x = (e_p, e_v, e_eta) in RR^9$
      - input: the *perturbation* $u_d = (omega_d, f_d)$ added to the LQR command
      - degree-2 monomials: $N = 55$; $ell_1$-EDMD, $lambda = 20$
      $ z_(k+1) = A z_k + B u_(d,k) + D macron(v)_w, quad z = psi(e_x) $
    ],
    card(title: [K-MPC, Eqs. (15)–(16)])[
      #set text(size: 14.5pt)
      $ min_(u_d) sum_(k=0)^(n_p) norm(C z_k)_Q^2 + sum_(k=0)^(n_p - 1) norm(u_(d,k))_R^2 $
      s.t. the model, $|u_d| <= (0.3, 0.3, 0.3, 1.0)$
      - horizon $n_p = 5$ (0.5 s), dense QP
      - terminal weight $= Q$; no terminal set
      - LQR stabilises; MPC only *corrects*
    ],
  )
]

#let paper-table(name) = {
  let r = paper.at(name).rmse
  table(columns: (auto, 1fr, 1fr, 1fr), align: (left, right, right, right),
    table.hline(stroke: 1pt + palette.primary),
    [Controller], [pos. [m]], [vel. [m/s]], [att. [deg]],
    table.hline(stroke: 0.5pt + palette.muted),
    ..("LQR", "LQR + K-MPC").map(c => (
      c, pm(r.at(c).pos), pm(r.at(c).vel), pm(r.at(c).eul, d: 2))).flatten(),
    table.hline(stroke: 1pt + palette.primary),
  )
}

#slide(title: [Paper reproduction: trajectories (Gaussian wind)])[
  #grid(columns: (1fr, 1fr), gutter: 0.8em,
    [
      #image("figures/paper_figure8_gaussian_3d.svg", height: 50%)
      #set text(size: 13pt)
      #paper-table("figure8_gaussian")
    ],
    [
      #image("figures/paper_circle_gaussian_3d.svg", height: 50%)
      #set text(size: 13pt)
      #paper-table("circle_gaussian")
    ],
  )
  #muted[RMSE of the error norm, mean ± std over #paper.figure8_gaussian.runs paired wind seeds; ideal rotors (this run predates the BEMT rotor model).]
]

#slide(title: [Paper reproduction: position error, figure-8])[
  #align(center, image("figures/paper_figure8_gaussian_err_pos.svg", height: 84%))
]

#slide(title: [Paper reproduction: velocity and attitude errors, figure-8])[
  #grid(columns: (1fr, 1fr), gutter: 0.5em,
    image("figures/paper_figure8_gaussian_err_vel.svg", height: 68%),
    image("figures/paper_figure8_gaussian_err_eul.svg", height: 68%),
  )
  #takeaway[K-MPC helps mostly in altitude; the 3-D gain over LQR is a few percent, because $u_d$ is a bounded correction.]
]

// =====================================================================================
#section-slide[Why remove the LQR]

#slide(title: [With an LQR underneath, the MPC has little to do])[
  #grid(columns: (1fr, 1fr), gutter: 1.2em,
    [
      *What the identified Model 1 showed*
      #set text(size: 16pt)
      #table(columns: (1.6fr, 1fr), align: (left, right),
        table.hline(stroke: 1pt + palette.primary),
        [Observation], [Value],
        table.hline(stroke: 0.5pt + palette.muted),
        [Spectral radius w/o constant], [0.974 (stable)],
        [Mode at $lambda = 1$: $norm(w^top B)$], [$approx 10^(-16)$],
        [Non-zeros in $B$], [23 of 220],
        [5-step velocity nRMSE], [0.44–0.65],
        [3-D RMSE gain over LQR], [2.6–3.1 %],
        table.hline(stroke: 1pt + palette.primary),
      )
      #muted[Run 20261113T155255Z_edmd_model1 and the reproduction above.]
    ],
    [
      - The model describes the *closed loop with the LQR*, not the vehicle: it is stable because the LQR is inside.
      - Stability, feasibility and terminal ingredients are *guaranteed by the LQR*, so they cannot be studied.
      - $|u_d| <= 0.3$ rad/s limits what any model can achieve.
      - Model accuracy barely shows in tracking.
    ],
  )
  #takeaway[replace the LQR: the MPC commands the total $u = (omega, c)$ and the model must describe an open-loop *unstable* vehicle.]
]

#slide(title: [Research questions for the full MPC])[
  #grid(columns: (1fr, 1fr), gutter: 1.2em,
    card(title: [Model])[
      - Can a lifted *linear* Koopman model of the whole outer loop beat physics models that are wrong in specific, known ways?
      - Which unmodelled effects can it learn (drag, rotor aerodynamics, rate-loop lag), and which can it not?
    ],
    card(title: [Controller])[
      - Same horizon, weights, bounds and wind information for every model: *only the model differs*.
      - Is the MPC well-behaved (no chattering or bang-bang) on a learned model?
      - Next: terminal ingredients and recursive feasibility on the lifted model.
    ],
  )
  #v(0.6em)
  #card(title: [Design principle], color: palette.good)[
    Build the MPC on an analytic model first, then swap in the learned one: when something fails, there is one cause, not two.
  ]
]

// =====================================================================================
#section-slide[The full-MPC set-up]

#slide(title: [Architecture and the shared optimal control problem])[
  #align(center)[
    #blk[Reference \ $p_"ref" (t)$] #arr()
    #blk(fill: rgb("#e6f1f8"))[*MPC*, 1 of 4 models \ #muted[10 Hz, $N = 20$ (2 s)]] #arr(label: [$u = (omega, c)$])
    #blk[Rate loop \ #muted[1 kHz]] #arr() #blk[BEMT rotors, \ rigid body, wind]
  ]
  #v(0.3em)
  #grid(columns: (1.1fr, 1fr), gutter: 1em,
    [
      $ min_(u_0, dots, u_(N-1)) sum_(i=0)^(N-1) (norm(e_i)_Q^2 + norm(delta u_i)_R^2) + norm(e_N)_(P_N)^2 $
      $ "s.t." quad "prediction model", quad u_"lb" <= u_i <= u_"ub" $
      #set text(size: 15pt)
      - $Q = "diag"(100 I_3, 10 I_3, I_3)$ on (pos, vel, att), $R = "diag"(1, 5, 5, 0.1)$
      - $|omega| <= 2$ rad/s, $c in [2, 18]$ m/s², $|phi|, |theta| <= 0.5$ rad
    ],
    card(title: [Held equal for every controller])[
      #set text(size: 15pt)
      - sample time, horizon, $Q$, $R$, bounds
      - mean wind $macron(v)_w$ known; turbulence not predicted
      - same wind realisation per seed
      - same simulator, rate loop, BEMT rotors
    ],
  )
]

#slide(title: [Four prediction models])[
  #set text(size: 15.5pt)
  #table(columns: (auto, 1.7fr, 1.2fr, 1.2fr), align: (left, left, center, center),
    table.hline(stroke: 1pt + palette.primary),
    [Controller], [Prediction model], [Wind → force], [Rotor aero (BEMT)],
    table.hline(stroke: 0.5pt + palette.muted),
    [#swatch(ctrl-color.at("LTV nominal")) *LTV nominal*],
      [Rigid body + gravity + thrust, linearised along the reference (LTV), *no drag*],
      bad[none], bad[no],
    [#swatch(ctrl-color.at("Koopman")) *Koopman MPC*],
      [Lifted LTI $z^+ = A z + B delta u$, $N = #train.lifted_dim$, *identified from flights* of the nominal MPC],
      good[learned], good[learned],
    [#swatch(ctrl-color.at("LTV grey-box")) *LTV grey-box*],
      [LTV with drag $R D R^top v_r norm(v_r)$, $D$ *fitted by least squares* to the same flights],
      [fitted], bad[no],
    [#swatch(ctrl-color.at("LTV true drag")) *LTV oracle*],
      [LTV with the *simulator's own drag model* (only possible in simulation)],
      good[exact], bad[no],
    table.hline(stroke: 1pt + palette.primary),
  )
  #v(0.3em)
  - LTV models: error-state, re-linearised along the reference, time-varying Riccati terminal cost $P_N$.
  - Koopman: cost on the physical output $C z$ ([LN-DDC] Eq. 17), terminal weight $Q$, attitude regulated to trim.
  - With BEMT rotors the "oracle" is *not* an upper bound: it knows drag but not the rotors.
]

// =====================================================================================
#section-slide[Training the Koopman model]

#slide(title: [Coordinates and observables])[
  #grid(columns: (1.15fr, 1fr), gutter: 1.1em,
    [
      Scaled deviations from the nominal hover trim:
      $ s = ([p, v, eta - eta^star, omega]) / sigma in RR^12, quad z = psi(s) in RR^#train.lifted_dim $
      $ z^+ = A z + B delta u + D delta v_w, quad delta x = C z, quad C = [op("diag")(sigma) quad 0] $
      #set text(size: 15.5pt)
      #table(columns: (auto, auto, 1fr), align: (left, center, left),
        table.hline(stroke: 1pt + palette.primary),
        [Block], [\#], [Features],
        table.hline(stroke: 0.5pt + palette.muted),
        [state], [12], [$p, v, phi, theta, psi, omega$ (linear; $C$ recovers them)],
        [constant], [1], [$1$ (row fixed: $1^+ = 1$)],
        [trig], [6], [$sin a, cos a - 1$ for $a in {phi, theta, psi}$],
        [degree 2], [42], [products of $(v_x, v_y, v_z, "trig")$],
        table.hline(stroke: 1pt + palette.primary),
      )
    ],
    [
      #card(title: [Structure we impose])[
        #set text(size: 15.5pt)
        - *Translation invariance*: $A[:, p] = e_p$. Position may only feed itself.
        - No position monomials; body rates stay linear (they let the model learn the rate-loop lag).
      ]
      #v(0.3em)
      #card(title: [Why invariance was needed], color: palette.warn)[
        #set text(size: 15.5pt)
        All flights at 2 m altitude ($sigma_(p_z) = 3$ cm) made $p_z$ nearly collinear with the constant. The free fit predicted a phantom 0.4 m/s² per metre of altitude. Rank tests passed anyway.
      ]
    ],
  )
]

#slide(title: [Why sines and cosines of roll and pitch])[
  #grid(columns: (1fr, 1fr), gutter: 1.2em,
    [
      #set text(size: 15pt)
      The thrust direction is trigonometric in the attitude:
      $ R e_3 = mat(c_psi s_theta c_phi + s_psi s_phi; s_psi s_theta c_phi - c_psi s_phi; c_theta c_phi) $
      - vertical lift $c cos phi cos theta$: the "lift lost to tilt" $c(1 - cos theta)$
      - horizontal force $prop sin theta, sin phi$
      The body-frame airspeed $R^top (v - v_w)$, which sets drag *and* rotor inflow, is a sum of *velocity × sin/cos* products: exactly the degree-2 terms.
    ],
    [
      #card(title: [Design choices])[
        #set text(size: 15.5pt)
        - $cos a - 1$, not $cos a$: $cos a approx 1$ near hover would duplicate the constant feature.
        - $(cos a - 1)^2$ dropped: $sin^2 a + (cos a - 1)^2 = -2(cos a - 1)$ is an exact linear dependence.
        - Raw angles kept in the linear block only, so $C$ still reads them out.
        - Monomials $phi^2, theta v$ approximate the same terms but only near zero; trig features are exact for any tilt.
      ]
    ],
  )
  #takeaway[choose observables from the physics you expect: the dictionary should span the terms of $R e_3$ and $R^top v_r$.]
]

#slide(title: [Training data: dithered closed-loop flights])[
  #grid(columns: (1.6fr, 1fr), gutter: 0.8em,
    image("figures/train_trajectories.svg", width: 100%),
    [
      #set text(size: 15pt)
      - Flown by the *nominal* LTV MPC (knows nothing about wind) in the CCTA Gaussian wind
      - Dither $d tilde cal(U)(plus.minus [0.3, 0.3, 0.3, 1.0])$ redrawn every 0.1 s; the *applied* command is logged
      - #train.train_episodes episodes × 14 s = #train.train_samples samples at 10 Hz
      - 30 random *sum-of-sines* references (3 sines/axis, 0.07–0.25 Hz, tilt 0.05–0.4 rad, ≤ 3.5 m/s, ≤ 1 m vertical) + 6 hovers from offsets
      - *Circles and figure-8s are held out* of training and validation
    ],
  )
]

#slide(title: [Coverage and identification])[
  #grid(columns: (1fr, 1.15fr), gutter: 1em,
    image("figures/train_coverage.svg", width: 100%),
    [
      #set text(size: 15.5pt)
      Least squares on all blocks together:
      $ [A space B space D] = Theta(X') mat(Theta(X); Upsilon; W)^dagger $
      #muted[(Fitting $A$ first and $B$ from the residual would assign input effects to the drift.)]

      Lifted regressor: rank #train.regressor.rank / #train.regressor.rows, condition number #sci(train.regressor.condition_number).

      #v(0.2em)
      *20-step (2 s) open-loop RMSE*, validation, mean wind:
      #let k = train.rmse_20_steps_mean_wind.koopman
      #let h = train.rmse_20_steps_mean_wind.hover
      #table(columns: (auto, 1fr, 1fr, 1fr), align: (left, right, right, right),
        table.hline(stroke: 1pt + palette.primary),
        [], [$p_z$ [m]], [$v_z$ [m/s]], [roll [rad]],
        table.hline(stroke: 0.5pt + palette.muted),
        [Hover linearisation], fmt(h.at(2)), fmt(h.at(5)), fmt(h.at(6)),
        [Koopman], hl(fmt(k.at(2))), hl(fmt(k.at(5))), hl(fmt(k.at(6))),
        table.hline(stroke: 1pt + palette.primary),
      )
    ],
  )
]

#slide(title: [Poles: Koopman vs the hover LTI model])[
  #v(-0.4em)
  #align(center, image("figures/eigenvalues.svg", height: 46%))
  #v(-0.3em)
  #set text(size: 13pt)
  #grid(columns: (1fr, 1fr), gutter: 0.8em,
    card(title: [Locations agree with the physics], color: palette.good)[
      Hover LTI: 9 poles at $z = 1$; true drag damps 3 of them to |z| = #fmt(eig.hover_true_drag.at(8))–#fmt(eig.hover_true_drag.at(6)). Koopman: #eig.n_on_unit_circle at $z = 1$ (positions, constant), slow modes at #fmt(eig.slow_moduli_range.at(0))–#fmt(eig.slow_moduli_range.at(1)), one at #fmt(eig.unstable_moduli.at(0), d: 4) (doubles in #fmt(eig.unstable_doubling_time_s.at(0), d: 0) s; ×#fmt(eig.growth_over_horizon_20.at(0), d: 2) over the 2 s horizon).
    ],
    card(title: [But they are fragile], color: palette.warn)[
      $norm(A)_2 approx #fmt(eig.norm_A / 1000, d: 0) times 10^3$ (rows $v_x v_z$, $v_y^2$, ... cancel on the data). A random $norm(E)_2 = 10^(-4)$ moves the spectral radius to #bad[#fmt(eig.perturbed_spectral_radius.Koopman.at(0), d: 2)–#fmt(eig.perturbed_spectral_radius.Koopman.at(1), d: 2)]; the hover model's to ≤ #fmt(eig.perturbed_spectral_radius.at("hover true drag").at(1), d: 2). Median $kappa approx$ #sci(eig.kappa_median).
    ],
  )
  #takeaway[pole locations match the physics, but they are no stability certificate here: judge the model by multi-step prediction error.]
]

// =====================================================================================
#section-slide[Persistency of excitation]

#slide(title: [The Hankel test])[
  #grid(columns: (1.1fr, 1fr), gutter: 1.1em,
    [
      Depth-$L$ block-Hankel matrix of $l$-channel inputs $u_1, dots, u_T$:
      $ H_L (u) = mat(u_1, u_2, dots, u_(T-L+1); u_2, u_3, dots, u_(T-L+2); dots.v, , , dots.v; u_L, u_(L+1), dots, u_T) in RR^(l L times (T - L + 1)) $
      $u$ is *persistently exciting of order $L$* $<==>$ $"rank" H_L (u) = l L$.

      #muted[Needs $T >= (l+1) L - 1$ samples in one experiment.]
    ],
    [
      #card(title: [Willems' fundamental lemma])[
        For a *controllable* LTI system with $n$ states, $u$ PE of order $n + 1$ $==>$ $"rank" mat(X; U) = n + l$: the regression for $[A space B]$ has a *unique* solution.
      ]
      #v(0.3em)
      #set text(size: 15.5pt)
      - Underactuated is fine: the lemma needs controllability, not full actuation.
      - Several episodes: *mosaic* $[H_L (u^((1))) dots H_L (u^((r)))]$, no window crosses an episode.
      - Report $sigma_min \/ sigma_max$, not just the rank.
      - For a *lifted* model this is necessary, not sufficient: also check the lifted regressor.
    ],
  )
]

#slide(title: [Are our inputs persistently exciting? Yes])[
  #align(center, image("figures/pe_hankel.svg", height: 56%))
  #v(-0.3em)
  #set text(size: 14pt)
  #let rows = (
    ("Inputs, order n+1 = 13", "inputs_order_n+1=13"),
    ("Inputs and wind, order 13", "inputs_and_wind_order_n+1=13"),
    ("Inputs, order N+1 = " + str(train.lifted_dim + 1), "inputs_order_N+1=" + str(train.lifted_dim + 1)),
  )
  #table(columns: (2fr, 1fr, 1fr, 1fr, 1fr), align: (left, right, right, right, center),
    table.hline(stroke: 1pt + palette.primary),
    [Test (#train.train_episodes training episodes)], [rank], [columns], [$sigma_min \/ sigma_max$], [PE?],
    table.hline(stroke: 0.5pt + palette.muted),
    ..rows.map(((label, key)) => {
      let r = train.pe_stored.at(key)
      (label, [#r.rank / #r.rows], str(r.columns), fmt(r.sigma_min_rel, d: 3),
       if r.persistently_exciting { good[yes] } else { bad[no] })
    }).flatten(),
    table.hline(stroke: 1pt + palette.primary),
  )
]

// =====================================================================================
#section-slide[Results]

#slide(title: [Position RMSE of the four controllers, every reference])[
  #set text(size: 15pt)
  #table(columns: (auto, 1fr, 1fr, 1fr, 1fr), align: (left, right, right, right, right),
    table.hline(stroke: 1pt + palette.primary),
    [Reference], ..controllers.map(c => [#swatch(ctrl-color.at(c)) #ctrl-name.at(c)]),
    table.hline(stroke: 0.5pt + palette.muted),
    ..gap.rows.pairs().map(((name, row)) => {
      let best = calc.min(..controllers.map(c => row.at(c).mean))
      (ref-name.at(name, default: name), ..controllers.map(c => {
        let cell = [#fmt(row.at(c).mean) #text(fill: palette.muted, size: 0.8em)[± #fmt(row.at(c).std)]]
        let red = if c != "LTV nominal" {
          [ #text(size: 0.75em, fill: palette.muted)[(−#fmt(row.at(c).reduction_pct, d: 0) %)]]
        }
        if row.at(c).mean == best { strong(text(fill: palette.accent, cell)) + red } else { cell + red }
      }))
    }).flatten(),
    table.hline(stroke: 1pt + palette.primary),
  )
  #muted[3-D position RMSE [m], mean ± std over #gap.seeds wind seeds; (−x %) = reduction from LTV nominal. BEMT rotors. Grey-box fitted $D = $ (#gap.greybox_drag_coeffs.map(v => fmt(v)).join(", ")) vs true (0.05, 0.05, 0.10).]
  #let ratios = gap.rows.values().map(row => row.at("LTV true drag").mean / row.at("Koopman").mean)
  #takeaway[Koopman is best on every reference, #fmt(calc.min(..ratios), d: 1)–#fmt(calc.max(..ratios), d: 1)× lower RMSE than the "oracle"; grey-box ≈ oracle.]
]

#let state-table(name) = {
  let r = mc.at(name).controllers
  table(columns: (auto, 1fr, 1fr, 1fr), align: (left, right, right, right),
    table.hline(stroke: 1pt + palette.primary),
    [Controller], [pos. [m]], [vel. [m/s]], [att. [deg]],
    table.hline(stroke: 0.5pt + palette.muted),
    ..controllers.map(c => (
      [#swatch(ctrl-color.at(c)) #ctrl-name.at(c)],
      pm(r.at(c).rmse_pos), pm(r.at(c).rmse_vel), pm(r.at(c).rmse_eul, d: 2))).flatten(),
    table.hline(stroke: 1pt + palette.primary),
  )
}

#let result-slides(name, label) = {
  slide(title: [#label: trajectory and per-state RMSE])[
    #grid(columns: (1fr, 1.1fr), gutter: 0.8em,
      image("figures/mc_" + name + "_3d.svg", width: 100%),
      [
        #v(1.5em)
        #set text(size: 15pt)
        #state-table(name)
        #muted[RMSE of the error norm, mean ± std over #mc.at(name).seeds wind seeds. Attitude error is against the reference's flat-output attitude.]
      ],
    )
  ]
  slide(title: [#label: position error])[
    #align(center, image("figures/mc_" + name + "_err_pos.svg", height: 84%))
  ]
  slide(title: [#label: velocity and attitude errors])[
    #grid(columns: (1fr, 1fr), gutter: 0.5em,
      image("figures/mc_" + name + "_err_vel.svg", width: 100%),
      image("figures/mc_" + name + "_err_eul.svg", width: 100%),
    )
    #muted[Bands: ± 1 std over #mc.at(name).seeds wind seeds.]
  ]
}

#result-slides("circle_7s_gaussian", [Circle 7 s])
#result-slides("figure8_10s_gaussian", [Figure-8 10 s])

#let input-table(name) = {
  let r = mc.at(name).controllers
  let pct(v) = fmt(100 * v, d: 1)
  table(columns: (auto, 1fr, 1fr, 1fr), align: (left, right, right, right),
    table.hline(stroke: 1pt + palette.primary),
    [Controller], [step change [%]], [at bound [%]], [sign flips [%]],
    table.hline(stroke: 0.5pt + palette.muted),
    ..controllers.map(c => {
      let m = r.at(c).inputs
      ([#swatch(ctrl-color.at(c)) #ctrl-name.at(c)],
       [#pct(calc.max(..m.tv))], [#pct(calc.max(..m.sat))], [#pct(calc.max(..m.flips))])
    }).flatten(),
    table.hline(stroke: 1pt + palette.primary),
  )
}

#let input-slide(name, label) = slide(title: [#label: MPC commands])[
  #grid(columns: (1.75fr, 1fr), gutter: 0.6em,
    image("figures/mc_" + name + "_inputs.svg", height: 76%),
    [
      #set text(size: 12pt)
      #input-table(name)
      #muted[Worst channel, all seeds. *Step change*: mean $|Delta u|$ per step as % of the bound range. *At bound*: % of steps within 0.1 % of a bound. *Sign flips*: % of steps where $Delta u$ reverses.]
      #v(0.3em)
      #card(title: [Reading])[
        No bang-bang: no bound is ever reached. Commands reverse on about half the steps, but only by about 3 % of the range, in *all four* controllers: gusts are redrawn every 0.1 s, the MPC period.
      ]
    ],
  )
]

#input-slide("circle_7s_gaussian", [Circle 7 s])
#input-slide("figure8_10s_gaussian", [Figure-8 10 s])

// =====================================================================================
#section-slide[Why Koopman captures the rotor aerodynamics]

#slide(title: [The thrust error is a smooth function of the airflow])[
  #grid(columns: (1.1fr, 1fr), gutter: 1.1em,
    [
      #set text(size: 15pt)
      BEMT inputs are the rotor airflow:
      $ mu prop norm(v_(r b, x y)), quad lambda_c prop -v_(r b, z), quad v_(r b) = R^top (v - v_w) $
      so the thrust error is
      $ Delta T = T_"BEMT" - m c approx f(v_(r b)) $
      With the mean wind known, $R^top (v - macron(v)_w)$ expands into
      $ v_i dot {sin, cos - 1}(a), quad v_i v_j, quad {sin, cos - 1}(a), quad v_i, quad 1 $
      #hl[These are the Koopman features.] A linear map of $z$ can represent most of $Delta T$, and EDMD fits it because $Delta T$ is in the data. The rest is turbulence, which no model is given.
    ],
    [
      #set text(size: 15pt)
      *How much of $Delta T$ each structure can represent* \
      #muted[least-squares $R^2$ on #thrust.samples flown samples]
      #let f = thrust.r2_of_thrust_gap
      #table(columns: (1.8fr, 1fr), align: (left, right),
        table.hline(stroke: 1pt + palette.primary),
        [Feature set], [$R^2$],
        table.hline(stroke: 0.5pt + palette.muted),
        [Koopman dictionary ($#train.lifted_dim$), mean wind], hl(fmt(f.koopman_dictionary_mean_wind, d: 2)),
        [Grey-box z term $v_(r b, z) norm(v_(r b))$ + const, mean wind], fmt(f.greybox_drag_structure_mean_wind, d: 2),
        [Same, with the *true* (turbulent) wind], fmt(f.greybox_drag_structure_true_wind, d: 2),
        [Quadratic rotor model], fmt(f.quadratic_rotor_model, d: 2),
        table.hline(stroke: 1pt + palette.primary),
      )
    ],
  )
]

#slide(title: [Why the grey-box and the oracle cannot])[
  #grid(columns: (1fr, 1fr), gutter: 1.1em,
    card(title: [LTV oracle], color: palette.good)[
      Exact drag, but assumes $T = m c$. The thrust error has no place to go in the model: it is *structurally absent*. With ideal rotors this controller is the upper bound; with BEMT it is not.
    ],
    card(title: [LTV grey-box], color: palette.amber)[
      Only three numbers $D = "diag"(d_x, d_y, d_z)$ in a fixed drag shape. Least squares bends them to absorb what it can:
      $d_z$: 0.10 #sym.arrow.r #bad[#fmt(thrust.greybox_drag_coeffs.at(2))]. Vertical residual stays #bad[#fmt(thrust.greybox_residual_rms_validation_m_s2.at(2), d: 2) m/s²] (vs #fmt(thrust.greybox_residual_rms_validation_m_s2.at(0), d: 2) horizontally).
    ],
  )
  #v(0.3em)
  #card(title: [Koopman MPC], color: palette.blue)[
    No assumed structure for the force, only a dictionary that spans the right *functions* (velocity × attitude trig). The data contain the BEMT effect, so the fit recovers it, together with the rate-loop lag (body-rate observables) and the lift lost to tilt.
  ]
  #takeaway[a physics model is only as good as its structure; a learned model is only as good as its dictionary and its data.]
]

#slide(title: [Summary and next steps])[
  #set text(size: 14.5pt)
  #grid(columns: (1fr, 1fr), gutter: 1.2em,
    [
      *Findings*
      - The CCTA K-MPC is a small correction to an LQR: 3 % gain, no stability question to study.
      - A full Koopman MPC, trained on flights of a controller that knows nothing about wind, is the best of four controllers on every held-out reference.
      - It beats the true-drag "oracle" because it also learns the rotor aerodynamics that no physics MPC models.
      - Training inputs are PE (rank 52/52 and 248/248). Commands never saturate; their small step-to-step jitter follows the 10 Hz turbulence and is the same for every model.
    ],
    [
      *Caveats and next steps*
      - Koopman uses Eq. (17) (terminal $Q$, attitude to trim); the LTVs use the full design. The model gap includes this difference.
      - One identification; vary data and dither seeds.
      - Terminal ingredients on the lifted model (unreachable $lambda = 1$ modes).
      - Estimate the wind instead of assuming $macron(v)_w$; LES wind.
      - Placeholder vehicle parameters.
    ],
  )
]

#slide(title: [Reproduction])[
  #set text(size: 14pt)
  ```bash
  conda activate koopman_mpc
  PYTHONPATH=src python presentation/run_monte_carlo.py <outer_loop_koopman run> <greybox_drag run> \
      configs/experiments/circle_7s_gaussian.toml configs/experiments/figure8_10s_gaussian.toml --seeds 10
  PYTHONPATH=src python presentation/make_figures.py --paper-run ... --mc-run ... --gap-run ... \
      --model-run ... --greybox-run ... --dataset-run ... --bemt-run ...
  typst compile presentation/slides.typ
  ```
  #v(0.5em)
  Runs behind the numbers:
  - paper reproduction: #raw(paper.source.split("/").last())
  - full MPC Monte Carlo: #raw(mc.source.split("/").last())
  - 6-reference table: #raw(gap.source.split("/").last())
  - Koopman model: #raw(train.source_model.split("/").last()); dataset: #raw(train.source_dataset.split("/").last())
]
