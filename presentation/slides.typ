// Koopman MPC for a quadrotor in turbulent wind: course-project presentation (30–40 min).
// Flow follows report/koopman_mpc.tex. Build and themes: presentation/README.md.
// Every number in a results table is read from data/*.json, which make_figures.py
// writes from the saved runs (no hand-copied results).
#import "theme.typ": *

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
  "LQR": palette.vermillion,
  "K-Res-MPC": palette.purple,
  "LTV nominal": rgb("#7f7f7f"),
  "Koopman": palette.blue,
  "LTV grey-box": palette.amber,
  "LTV true drag": palette.good,
)
#let ctrl-name = (
  "LQR": [LQR],
  "K-Res-MPC": [LQR + K-Res],
  "LTV nominal": [LTV nominal],
  "Koopman": [Koopman MPC],
  "LTV grey-box": [LTV grey-box],
  "LTV true drag": [LTV oracle],
)
// The paper's architecture (LQR, LQR + K-Res-MPC) first, then the four full MPCs.
#let controllers = ("LQR", "K-Res-MPC", "LTV nominal", "Koopman", "LTV grey-box", "LTV true drag")
#let full-mpcs = controllers.slice(2)
#let ref-name = (
  hover_gaussian: [Hover, 1 m climb],
  circle_slow_gaussian: [Circle 10 s],
  circle_7s_gaussian: [Circle 7 s],
  circle_5s_gaussian: [Circle 5 s],
  figure8_14s_gaussian: [Figure-8 14 s],
  figure8_10s_gaussian: [Figure-8 10 s],
)

// Block diagram on a grid: `cells` is a flat list of (row 1 ..., row 2 ..., ...), `n` columns.
#let diagram(n, ..cells) = align(center, scale(125%, reflow: true, grid(
  columns: n, column-gutter: 0.25em, row-gutter: 0.35em, align: center + horizon, ..cells,
)))
#let up(label) = text(fill: palette.accent, size: 13pt)[#sym.arrow.t #h(0.2em) #label]
#let back(label) = text(fill: palette.muted, size: 12pt)[#sym.arrow.l.long #h(0.2em) #label]

#show: academic-theme.with(
  title: [Koopman MPC for a Quadrotor in Turbulent Wind],
  subtitle: [EE 6432 Course Project Presentation],
  author: [Subash K],
  institution: [AE23D203],
  date: datetime.today().display("[month repr:long] [day], [year]"),
  short-title: [Koopman MPC in wind],
)

#title-slide()

#slide(title: [Outline])[
  #set text(size: 20pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      + Quadrotor model in wind \ #muted[and why commanded thrust ≠ applied thrust]
      + Koopman models \ #muted[K-Res (CCTA paper) and K-Full (ours)]
      + Data and identifiability \ #muted[data generation, the Hankel condition]
    ],
    [
      #set enum(start: 4)
      + Koopman MPC \ #muted[terminal cost, no terminal set]
      + Results \ #muted[six controllers, same vehicle, same gusts]
    ],
  )
]

// =====================================================================================
#section-slide[Quadrotor model in wind]

#slide(title: [Rigid-body model, drag and wind])[
  #grid(
    columns: (1.15fr, 1fr),
    gutter: 1.2em,
    [
      *State* $x = (p, q, v, omega) in RR^13$ #muted[(world NED, body FRD, Hamilton quaternion)]
      $
              dot(p) & = v \
              dot(v) & = g e_3 + 1/m (R(q) f_T - f_D) \
              dot(q) & = 1/2 q times.o (0, omega) \
        J dot(omega) & = tau_T - omega times J omega
      $
      *Quadratic body drag* on the air-relative velocity
      $ f_D = R D R^top v_r norm(v_r), quad v_r = v - v_w $
      #muted[$D = "diag"(0.05, 0.05, 0.10) "Ns"^2"/m"^2$, $m = 1$ kg (placeholder vehicle).]
    ],
    [
      #card(title: [Wind (CCTA Sec. IV-A)])[
        $v_w = macron(v)_w + tilde(v)_w$, $tilde(v)_w tilde cal(N)(0, Sigma)$, redrawn every 0.1 s
        - $macron(v)_w = (2.70, 1.76, 0.01)$ m/s
        - $Sigma = "diag"(0.54, 0.42, 0.006)$ (m/s)²
      ]
      #v(0.3em)
      #card(title: [Outer-loop input])[
        $u = (omega_x, omega_y, omega_z, c)$: body rates and mass-normalised thrust $c$ [m/s²], tracked by a 1 kHz rate loop.
      ]
    ],
  )
]

#slide(title: [Commanded thrust ≠ applied thrust: BEMT rotors])[
  #align(center)[
    #blk[Controller] #arr(label: [$u = (omega, c)$])
    #blk[Rate loop \ #muted[1 kHz]] #arr(label: [$f_T, tau_T$])
    #blk[Mixer \ #muted[$T = k_T Omega^2$]] #arr(label: [$Omega_i$])
    #blk(fill: palette.light, stroke: palette.warn)[*BEMT rotors* \ #muted[in local airflow]]
    #arr(label: [$f_T^"applied"$]) #blk[Rigid body \ #muted[\+ drag, wind]]
  ]
  #v(0.2em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #set text(size: 14pt)
      The mixer assumes $T_i = k_T Omega_i^2$, with $k_T$ identified in still air.
      In wind the rotor thrust also depends on the airflow:
      $ T_i = f(Omega_i, V_infinity, v_i) $
      - *Blade element*: given induced velocity $v_i$, integrate lift and drag over the blade: $v_i arrow.r T_i$
      - *Momentum*: rotor as a disk, conserve mass and momentum: $T_i arrow.r v_i$
      - *BEMT*: solve both together for $v_i$, then $T_i$:
      $ f_"BET" (Omega_i, V_infinity, v_i) = 2 rho A v_i (V_infinity + v_i) $
    ],
    [
      #image("figures/thrust_cmd_vs_bemt.svg", width: 100%)
      #muted[Applied/commanded #fmt(thrust.thrust_gap_range_pct.at(0), d: 1) % to +#fmt(thrust.thrust_gap_range_pct.at(1), d: 1) %, RMS #fmt(thrust.thrust_gap_rms_m_s2, d: 2) m/s².]
    ],
  )
  #takeaway[the plant input is $f_T^"applied" (Omega, "airflow")$, not $m c$: a model has to learn this gap.]
]

// =====================================================================================
#section-slide[Koopman Operator]

#slide(title: [Koopman Theory])[
  #grid(
    columns: (1.2fr, 1fr),
    gutter: 1.2em,
    [

      Autonomous nonlinear system $x_(k+1) = F(x_k)$ can be lifted to a linear system in the space of observables \ $g: cal(G) arrow cal(G)$. \
      The Koopman operator acts on *observables*:
      $ g(x_(k+1)) = (cal(K) g)(x_k) = g(F(x_k)) $
      It is *linear* (in $g$), but infinite-dimensional.
      #v(0.3em)
      *EDMD* approximates it on a dictionary $psi = (psi_1, dots, psi_N)$:
      $ psi(x_(k+1)) approx A psi(x_k) + B u_k + D w_k, quad hat(x) = C psi(x) $
      #muted[$w$: measured disturbance (wind).]
    ],
    [
      #card(title: [The catch])[
        Exact only if $"span"{psi_j}$ is invariant. The *choice of dictionary* decides what the model can learn.
      ]
      #v(0.3em)
      #card(title: [])[
        The Koopman operator needs a map $F$ that takes a state to a next state. For a contolled system, $x_(k+1) = F(x_k, u_k)$ there is no such map. We can only assume an there is linear approximation in the space of observables.
      ]

    ],
  )
]

#slide(title: [K-Res (From reference paper): lifted model of the LQR tracking error])[
  #grid(
    columns: (1.2fr, 1fr),
    gutter: 1em,
    [
      #set text(size: 16pt)
      Error to the reference, standardised: $e = (e_p, e_v, e_eta) in RR^9$
      $ z^r_(k+1) = A_r z^r_k + B_r u_(d,k) + D_r v_(w,k), quad z^r = psi_r (e) in RR^55 $
      $ hat(e)_k = C_r z^r_k, quad C_r = [I_9 space 0] $
      #table(
        columns: (auto, auto, 1fr),
        align: (left, center, left),
        table.hline(stroke: 1pt + palette.primary),
        [Block], [\#], [Dictionary $psi_r$],
        table.hline(stroke: 0.5pt + palette.muted),
        [state], [9], [$e_1, dots, e_9$],
        [constant], [1], [$1$],
        [degree 2], [45], [all products $e_i e_j, i <= j$],
        table.hline(stroke: 1pt + palette.primary),
      )
    ],
    [
      #set text(size: 15pt)
      #card(title: [Variables])[
        - $e_p, e_v$: position / velocity error (m, m/s, NED)
        - $e_eta$: roll, pitch, yaw error (rad)
        - $u_d = (omega_d, c_d)$: *perturbation* added to the LQR command
        - $v_w$: wind; the MPC predicts with $macron(v)_w$
      ]
      #v(0.3em)
      #card(title: [Fit])[$ell_1$-EDMD, $lambda = 20$; constant row fixed ($1^+ = 1$).]
    ],
  )
]

#slide(title: [K-Res design: K-Res-MPC corrects an inner LQR])[
  #v(0.3em)
  #align(center, image("figures/kres_loop.svg", height: 52%))
  #v(0.6em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    card(title: [What it does])[
      #set text(size: 15pt)
      The LQR stabilises and tracks. The MPC adds a small *bounded correction* for the wind it learned from data.
    ],
    // card(title: [What it limits], color: palette.warn)[
    //   #set text(size: 15pt)
    //   The model describes the *LQR closed loop*, not the vehicle. Stability comes from the LQR, and $|u_d| <= 0.3$ rad/s caps what a better model can buy.
    // ],
  )
]

#slide(title: [K-Full : lifted model of the whole outer loop])[
  #grid(
    columns: (1.2fr, 1fr),
    gutter: 1em,
    [
      #set text(size: 15.5pt)
      $s = [p, v, eta, omega] in RR^12$
      $ z^f_(k+1) = A_f z^f_k + B_f delta u_k + D_f delta v_(w,k), quad z^f = psi_f (s) in RR^#train.lifted_dim $
      $delta hat(x)_k = C_f z^f_k, quad C_f = [I_9 space 0]$
      #table(
        columns: (auto, auto, 1fr),
        align: (left, center, left),
        table.hline(stroke: 1pt + palette.primary),
        [Block], [\#], [Dictionary $psi_f$],
        table.hline(stroke: 0.5pt + palette.muted),
        [state], [12], [$p, v, phi, theta, psi, omega$],
        [constant], [1], [$1$],
        [trig], [6], [$sin a, cos a - 1$, $a in {phi, theta, psi}$],
        [degree 2], [42], [products of $b = (v, sin a, cos a - 1)$],
        table.hline(stroke: 1pt + palette.primary),
      )
    ],
    [
      #set text(size: 14.5pt)
      #card(title: [Variables])[
        - $delta u = u - u^star$
        - $delta v_w = v_w - macron(v)_w$; MPC predicts $delta v_w = 0$
        - $omega$ in $s$: lets the model learn the rate-loop lag
      ]
      #v(0.3em)
      #card(title: [Why sin and cos?])[
        $R e_3$ and the body airspeed $R^top (v - v_w)$ are built from $v dot sin a$ and $v dot (cos a - 1)$: thrust direction, lift lost to tilt, drag and *rotor inflow* lie in the span.
      ]
    ],
  )
  // #muted[Fit: least squares, constant row fixed, *translation invariant* ($A_f [:, p] = e_p$).]
]

#slide(title: [K-Full design: the MPC is the whole controller])[
  #v(0.3em)
  #align(center, image("figures/kfull_loop.svg", height: 46%))
  #v(0.6em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    card(title: [What it does])[
      #set text(size: 15pt)
      No LQR: the MPC sends the *total* body-rate and thrust command. The lifted model covers the rate loop, rotors and drag together.
    ],
    card(title: [What changes from K-Res])[
      #set text(size: 15pt)
      The model describes the *vehicle*, not a closed loop, so a better model directly gives better tracking.
    ],
  )
]

// #slide(title: [K-Full design: the MPC is the whole controller])[
//   #v(0.2em)
//   #diagram(
//     7,
//     blk[Reference \ $p_"ref", v_"ref"$],
//     arr(),
//     blk(fill: palette.light, stroke: palette.blue)[*K-Full-MPC* #muted[10 Hz, $N = 20$] \ $z_0 = psi_f (s)$, convex QP],
//     arr(label: [$u = (omega, c)$]),
//     blk[Rate loop \ #muted[1 kHz]],
//     arr(),
//     blk(stroke: palette.warn)[Quadrotor \ #muted[BEMT, wind]],
//     [],
//     [],
//     grid.cell(colspan: 5, back[measured $x = (p, v, eta, omega)$]),
//   )
//   #v(0.4em)
//   #set text(size: 13.5pt)
//   #table(
//     columns: (auto, 1.6fr, auto, auto),
//     align: (left, left, center, center),
//     table.hline(stroke: 1pt + palette.primary),
//     [Controller (all compared)], [Prediction model], [Drag], [BEMT],
//     table.hline(stroke: 0.5pt + palette.muted),
//     [#swatch(ctrl-color.at("LQR")) LQR], [state-dependent LQR, no wind model], bad[no], bad[no],
//     [#swatch(ctrl-color.at("K-Res-MPC")) LQR + K-Res], [K-Res correction, $|u_d| <= 0.3$ rad/s], [partly], [partly],
//     [#swatch(ctrl-color.at("LTV nominal")) LTV nominal],
//     [rigid body, *no drag*: knows nothing about wind],
//     bad[no],
//     bad[no],
//     [#swatch(ctrl-color.at("Koopman")) *Koopman MPC*],
//     [K-Full, identified from nominal-MPC flights],
//     good[learned],
//     good[learned],
//     [#swatch(ctrl-color.at("LTV grey-box")) LTV grey-box],
//     [drag coefficients fitted to the same flights],
//     [fitted],
//     bad[no],
//     [#swatch(ctrl-color.at("LTV true drag")) LTV oracle],
//     [the simulator's own drag (simulation only)],
//     good[exact],
//     bad[no],
//     table.hline(stroke: 1pt + palette.primary),
//   )
//   #muted[Full MPCs share horizon, $Q$, $R$, bounds and the mean wind; only the model differs.]
// ]

// =====================================================================================
#section-slide[Data and identifiability]

#slide(title: [Data generation])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    card(title: [K-Res: LQR flies, $u_d$ is the input], color: palette.purple)[
      #set text(size: 14.5pt)
      - LQR tracks the paper's references: figure-8 and circle, 7 s, 4 repeats
      - $u_d tilde cal(U)(plus.minus [0.3, 0.3, 0.3, 1.0])$, redrawn every 0.1 s, *added* to the LQR command
      - Logged: *error states* $e = x - x_"ref"$, $u_d$, $v_w$
      - 8 episodes × 14 s at 10 Hz
      - $u_d$ is independent of the state: exciting by construction
    ],
    card(title: [K-Full: nominal LTV MPC flies, $u$ is the input], color: palette.blue)[
      #set text(size: 14.5pt)
      - MPC with LTV quadrotor model *without drag*
      - Same uniform dither added: $u = "clip"(u_"MPC" + d)$; the *applied* $u$ is logged
      - 30 random sum-of-sines references
      // (tilt ≤ 0.4 rad, ≤ 3.5 m/s) + 6 hovers
      - #train.train_episodes episodes × 14 s at 10 Hz
      - *Circles and figure-8s held out*: the tests are unseen trajectory families
    ],
  )
  //   #takeaway[the trajectories decide *which states* are visited; the dither decides whether the input's effect *can be separated* from the state's.]
]

#slide(title: [The base controller: LTV MPC with terminal ingredients])[
  #grid(
    columns: (1.25fr, 1fr),
    gutter: 1em,
    [
      #set text(size: 15pt)
      Error $e = x - x_("ref",k)$, $Delta = u - u_("ref",k)$, linearised *along the reference*:
      $ min_Delta sum_(i=1)^(N-1) e_i^top Q e_i + e_N^top P_(k+N) e_N + sum_(i=0)^(N-1) Delta_i^top R Delta_i $
      $ e_(i+1) = A_(k+i) e_i + B_(k+i) Delta_i + r_(k+i) $
      $ u_"lb" <= u_("ref",k+i) + Delta_i <= u_"ub", quad |phi|, |theta| <= 0.5 "rad" $
      #muted[$N = 20$ (2 s), $Q = "diag"(100 I_3, 10 I_3, I_3)$, $R = "diag"(1, 5, 5, 0.1)$.]
    ],
    [
      #set text(size: 14pt)
      #card(title: [Terminal cost])[
        $P_(k+N)$ from the backward *time-varying Riccati* recursion along the reference: the infinite-horizon cost-to-go. It makes a short horizon stabilising for an integrator plant.
      ]
      #v(0.2em)
      #card(title: [Terminal set: maximal admissible set])[
        $ cal(O)_infinity = {x : H A_"cl"^t x <= h, med forall t >= 0} $
        with $A_"cl" = A + B K_"LQR"$. Built for the *hover* MPC. Along a trajectory it would have to move, so the data controller uses the *cost only*.
      ]
    ],
  )
]

#slide(title: [When does the regression have a unique minimiser?])[
  #grid(
    columns: (1.1fr, 1fr),
    gutter: 1em,
    [
      #set text(size: 15pt)
      All snapshot pairs as columns: $Gamma_2 approx [A space B space D] Gamma_1$
      $ Gamma_1 = mat(psi(X); U; W) in RR^((N + l) times m) $
      Unique $[A space B space D] = Gamma_2 Gamma_1^dagger$ #h(0.3em) $<==>$ #h(0.3em) $"rank" Gamma_1 = N + l$

      *Necessary:* $m >= N + l$ columns. *Not sufficient:* a constant input adds columns, not rank.

      *Richness: Hankel of $[u; v_w]$* ($l = 7$ channels, depth $L$):
      $ H_L = mat(u_1, dots, u_(m-L+1); dots.v, , dots.v; u_L, dots, u_m) in RR^(l L times (m - L + 1)) $
      PE of order $L$ $<==>$ $"rank" H_L = l L$.
    ],
    [
      #set text(size: 14.5pt)
      #card(title: [Willems' lemma])[
        Controllable LTI with $n$ states, input PE of order $n + 1$ $==>$ $"rank" mat(X; U) = n + l$.
      ]
      #v(0.2em)
      #card(title: [For the lifted model])[
        Treat it as $N$ states: test order $N + 1$, then check $Gamma_1$ *directly*.
      ]
      #v(0.2em)
      #card(title: [Episodes: the mosaic])[
        $[H_L (u^((1))) dots H_L (u^((r)))]$; windows never cross an episode:
        $ m_i >= N + 1, quad sum_i (m_i - N) >= l (N + 1) $
      ]
    ],
  )
]

#slide(title: [The Hankel condition holds for both models])[
  #v(0.4em)
  #set text(size: 14pt)
  // Values: report Table 3 (koopman/excitation.py on the training data of both models;
  // wind mean-removed; depth N+1 with wind computed 2026-11-19, not stored by a run).
  #table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    table.hline(stroke: 1pt + palette.primary),
    [], table.cell(colspan: 3)[Hankel $H_(N+1)([u; v_w])$, mosaic], table.cell(colspan: 2)[Lifted regressor $Gamma_1$],
    table.hline(stroke: 0.5pt + palette.muted),
    [Model], [$m_i >= N+1$], [$sum_i (m_i - N) >= l(N+1)$], [$"rank" = l(N+1)$], [$m >= N + l$], [$"rank" = N + l$],
    table.hline(stroke: 0.5pt + palette.muted),
    [K-Res ($N = 55$, $l = 7$)], [$140 >= 56$], [$680 >= 392$], good[$392 = 392$], [$1112 >= 62$], good[$62 = 62$],
    [K-Full ($N = 61$, $l = 7$)], [$140 >= 62$], [$2844 >= 434$], good[$434 = 434$], [$5004 >= 68$], good[$68 = 68$],
    table.hline(stroke: 1pt + palette.primary),
  )
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    card(title: [Episodes pool their excitation], color: palette.good)[
      #set text(size: 14.5pt)
      One episode alone would need $(l+1)(N+1) - 1 = 447$ and $495$ samples; each has 140. Only the *mosaic* is PE. At order $n+1$ both are PE too (70/70, 91/91).
    ],
    // card(title: [Full rank is yes/no], color: palette.warn)[
    //   #set text(size: 14.5pt)
    //   K-Full's $Gamma_1$: $sigma_min \/ sigma_max = 1.4 times 10^(-8)$ (condition #sci(train.regressor.condition_number)) against 67 for K-Res. The cause is nearly dependent *features*, not the input.
    // ],
  )
]

// =====================================================================================
#section-slide[Koopman MPC]

#slide(title: [Koopman MPC formulations])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    card(title: [K-Full-MPC (replaces the LQR)], color: palette.blue)[
      #set text(size: 14.5pt)
      $
        min_(delta u) sum_(i=0)^(N-1) (norm(C_f z_i - r_i)_Q^2 + norm(delta u_i)_R^2) + #text(fill: palette.warn)[$norm(C_f z_N - r_N)_Q^2$]
      $
      $ z_(i+1) = A_f z_i + B_f delta u_i, quad z_0 = psi_f (s) $
      $ u_"lb" <= u^star + delta u_i <= u_"ub" $
      - $N = 20$; $r_i = (p_"ref", v_"ref", 0)$
      - wind only through the trim
    ],
    card(title: [K-Res-MPC (corrects the LQR)], color: palette.purple)[
      #set text(size: 14.5pt)
      $
        min_nu sum_(k=0)^(n_p - 1) (e_k^top Q e_k + nu_k^top R nu_k) + #text(fill: palette.warn)[$e_(n_p)^top Q e_(n_p)$]
      $
      $ z^r_(k+1) = A_r z^r_k + B_r nu_k + D_r macron(v)_w $
      $ |nu_k| <= (0.3, 0.3, 0.3, 1.0) $
      - $n_p = 5$; $nu_0$ is added to the LQR as $u_d$
    ],
  )
  #v(0.4em)
  #card(title: [Terminal ingredients, both models], color: palette.warn)[
    Terminal cost = the stage cost ($P = Q$ on the *physical* output $C z$). *No terminal constraint.*
  ]
]

// #slide(title: [Why P = Q and no terminal set in the lifted space])[
//   #grid(
//     columns: (1fr, 1fr),
//     gutter: 1em,
//     [
//       #set text(size: 14.5pt)
//       #card(title: [No Riccati $P$: $(A, B)$ is not stabilisable])[
//         The constant feature has the fixed row $A[c, :] = e_c^top$, $B[c, :] = 0$. So $w = e_c$ is a left eigenvector at $lambda = 1$ with $w^top B = 0$: by PBH an *unreachable* marginal mode, and the DARE has *no stabilising solution*.
//       ]
//       #v(0.2em)
//       $P = Q$ on $C z$ needs no stabilisability. With $N = 20$ (2 s) the plan has nearly converged at $i = N$, so the terminal term matters little.
//     ],
//     [
//       #set text(size: 14.5pt)
//       #card(title: [No terminal set: $z$ is not free])[
//         - Reachable $z$ lie on the $n$-dim manifold ${psi(x)} subset RR^N$: a set in $RR^N$ is mostly points no vehicle state produces
//         - An invariant set needs a terminal law, which does not exist (left)
//         - $z_N = psi(x_"ref")$ would pin every monomial: easily infeasible
//       ]
//       #v(0.2em)
//       #card(title: [Fragile eigenvalues], color: palette.warn)[
//         $norm(A_f)_2 approx #fmt(eig.norm_A / 1000, d: 0) times 10^3$; a random $norm(E)_2 = 10^(-4)$ moves the spectral radius from #fmt(eig.spectral_radius, d: 3) to #fmt(eig.perturbed_spectral_radius.Koopman.at(0), d: 2)–#fmt(eig.perturbed_spectral_radius.Koopman.at(1), d: 2).
//       ]
//     ],
//   )
//   #takeaway[K-Res-MPC inherits stability from the LQR; K-Full-MPC's stability is empirical. Terminal ingredients on a stabilisable subspace are the next step.]
// ]

// =====================================================================================
#section-slide[Results]

// #slide(title: [The paper's architecture, re-run with BEMT rotors])[
//   #grid(
//     columns: (1fr, 1fr),
//     gutter: 1.1em,
//     card(title: [What was redone])[
//       #set text(size: 15pt)
//       - K-Res training data *re-flown with BEMT rotors*: same recipe and seed (LQR + uniform $u_d$ on the 7 s figure-8 and circle).
//       - K-Res *refitted* with the same config ($N = 55$, $lambda = 20$); inputs PE as before (40/40, 70/70, 224/224).
//       - LQR (100 Hz) and LQR + K-Res-MPC (10 Hz, $n_p = 5$) flown on the six full-MPC references.
//     ],
//     card(title: [Paired with the full MPCs], color: palette.good)[
//       #set text(size: 15pt)
//       - Same wind seeds; the merge *checks* that wind, reference and time grid are identical, and stops otherwise.
//       - Table: seeds 0–4 (as the full-MPC table); time series: seeds 0–9.
//       - The paper's own results used ideal rotors and its 7 s references, so they are not comparable with these.
//     ],
//   )
//   #takeaway[now all six controllers fly the same vehicle (BEMT rotors) through the same gusts.]
// ]

#slide(title: [Position RMSE of all six controllers, every reference])[
  #set text(size: 12.5pt)
  #table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right, right, right),
    table.hline(stroke: 1pt + palette.primary),
    [Reference], ..controllers.map(c => [#swatch(ctrl-color.at(c)) #ctrl-name.at(c)]),
    table.hline(stroke: 0.5pt + palette.muted),
    ..gap
      .rows
      .pairs()
      .map(((name, row)) => {
        let best = calc.min(..controllers.map(c => row.at(c).mean))
        (
          ref-name.at(name, default: name),
          ..controllers.map(c => {
            let cell = [#fmt(row.at(c).mean) #text(fill: palette.muted, size: 0.8em)[± #fmt(row.at(c).std)]]
            if row.at(c).mean == best { strong(text(fill: palette.accent, cell)) } else { cell }
          }),
        )
      })
      .flatten(),
    table.hline(stroke: 1pt + palette.primary),
  )
  #muted[3-D position RMSE [m], mean ± std over #gap.seeds paired wind seeds; best per row in colour. BEMT rotors. Grey-box fitted $D =$ (#gap.greybox_drag_coeffs.map(v => fmt(v)).join(", ")) vs true (0.05, 0.05, 0.10).]
  #let ratios = gap.rows.values().map(row => row.at("LTV true drag").mean / row.at("Koopman").mean)
  #let kres = gap.rows.values().map(row => 100 * (1 - row.at("K-Res-MPC").mean / row.at("LQR").mean))
  #let n-best = gap.rows.values().filter(row => controllers.all(c => row.at("Koopman").mean <= row.at(c).mean)).len()
  #takeaway[Koopman is best on #n-best of #gap.rows.len() references (#fmt(calc.min(..ratios), d: 1)–#fmt(calc.max(..ratios), d: 1)× below the "oracle"). K-Res-MPC improves the LQR by only #fmt(calc.min(..kres), d: 0)–#fmt(calc.max(..kres), d: 0) %.]
]

#let state-table(name) = {
  let r = mc.at(name).controllers
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    table.hline(stroke: 1pt + palette.primary),
    [Controller], [pos. [m]], [vel. [m/s]], [att. [deg]],
    table.hline(stroke: 0.5pt + palette.muted),
    ..controllers
      .map(c => (
        [#swatch(ctrl-color.at(c)) #ctrl-name.at(c)],
        pm(r.at(c).rmse_pos),
        pm(r.at(c).rmse_vel),
        pm(r.at(c).rmse_eul, d: 2),
      ))
      .flatten(),
    table.hline(stroke: 1pt + palette.primary),
  )
}

#let result-slides(name, label) = {
  slide(title: [#label: trajectory and per-state RMSE])[
    #grid(
      columns: (1fr, 1.1fr),
      gutter: 0.8em,
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
    #grid(
      columns: (1fr, 1fr),
      gutter: 0.5em,
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
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    table.hline(stroke: 1pt + palette.primary),
    [Controller], [step change [%]], [at bound [%]], [sign flips [%]],
    table.hline(stroke: 0.5pt + palette.muted),
    ..controllers
      .map(c => {
        let m = r.at(c).inputs
        (
          [#swatch(ctrl-color.at(c)) #ctrl-name.at(c)],
          [#pct(calc.max(..m.tv))],
          [#pct(calc.max(..m.sat))],
          [#pct(calc.max(..m.flips))],
        )
      })
      .flatten(),
    table.hline(stroke: 1pt + palette.primary),
  )
}

#let input-slide(name, label) = slide(title: [#label: commands])[
  #align(center, image("figures/mc_" + name + "_inputs.svg", height: 60%))
  // #grid(
  //   columns: (1.15fr, 1fr),
  //   gutter: 0.8em,
  //   [
  //     #set text(size: 11pt)
  //     #input-table(name)
  //     #muted[Worst channel, all seeds. *Step change*: mean $|Delta u|$ per step, % of the bound range. *At bound*: % of steps within 0.1 % of a bound. *Sign flips*: % of steps where $Delta u$ reverses.]
  //   ],
  //   [
  //     #set text(size: 12pt)
  //     #card(title: [Reading])[
  //       No bang-bang: no controller reaches a bound. All six reverse their command on about half the steps, by 3–4 % of the range: gusts are redrawn every 0.1 s. LQR runs at 100 Hz, so its column shows the command sampled at 10 Hz.
  //     ]
  //   ],
  // )
]

#input-slide("circle_7s_gaussian", [Circle 7 s])
#input-slide("figure8_10s_gaussian", [Figure-8 10 s])

// =====================================================================================
// #slide(title: [Why Koopman wins: it can represent the rotor aerodynamics])[
//   #grid(
//     columns: (1fr, 1fr),
//     gutter: 1em,
//     [
//       #set text(size: 14.5pt)
//       BEMT depends on the rotor airflow $v_(r b) = R^top (v - v_w)$, so the thrust error is $Delta T approx f(v_(r b))$. With $macron(v)_w$ known, $R^top (v - macron(v)_w)$ expands into $v_i dot {sin, cos - 1}$, $v_i v_j$, ...: #hl[the K-Full dictionary.]

//       #muted[How much of $Delta T$ each structure can represent (least-squares $R^2$, #thrust.samples flown samples):]
//       #let f = thrust.r2_of_thrust_gap
//       #table(
//         columns: (1.8fr, 1fr),
//         align: (left, right),
//         table.hline(stroke: 1pt + palette.primary),
//         [Feature set], [$R^2$],
//         table.hline(stroke: 0.5pt + palette.muted),
//         [Koopman dictionary ($#train.lifted_dim$)], hl(fmt(f.koopman_dictionary_mean_wind, d: 2)),
//         [Grey-box $z$ drag term + const], fmt(f.greybox_drag_structure_mean_wind, d: 2),
//         [Quadratic rotor model (oracle)], fmt(f.quadratic_rotor_model, d: 2),
//         table.hline(stroke: 1pt + palette.primary),
//       )
//     ],
//     [
//       #set text(size: 14.5pt)
//       #card(title: [LTV oracle], color: palette.good)[
//         Exact drag, but assumes $T = m c$: the thrust error has *no place* in its model.
//       ]
//       #v(0.2em)
//       #card(title: [LTV grey-box], color: palette.amber)[
//         Three drag numbers in a fixed shape; least squares bends $d_z$: 0.10 #sym.arrow.r #bad[#fmt(thrust.greybox_drag_coeffs.at(2))]; vertical residual #bad[#fmt(thrust.greybox_residual_rms_validation_m_s2.at(2), d: 2) m/s²].
//       ]
//       #v(0.2em)
//       #card(title: [Koopman MPC], color: palette.blue)[
//         No assumed force structure, only a dictionary spanning the right *functions*; the data contain the BEMT effect, so the fit recovers it.
//       ]
//     ],
//   )
// ]

#slide(title: [Summary and next steps])[
  #set text(size: 14.5pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.2em,
    [
      *Findings*
      - BEMT rotors make applied thrust differ from the command by up to about 30 %; no physics MPC models it.
      - The paper's LQR + K-Res-MPC gains only 2–5 % over LQR: a bounded correction on a model of the LQR closed loop.
      - A full Koopman MPC, trained on flights of a controller that knows nothing about wind, is *best on all six held-out references*, 1.5–2.8× below the true-drag oracle on circles and figure-8s.
      - Training data satisfy the Hankel condition (mosaic PE, $Gamma_1$ full rank); commands never saturate.
    ],
    [
      *Caveats and next steps*
      - K-Full-MPC: Designing terminal ingredients for lifted states.
      - Estimate the wind instead of assuming $macron(v)_w$ using a disturbance observer or a Kalman filter.
    ],
  )
]
