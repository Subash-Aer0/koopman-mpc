# Design Report: Full MPC and Koopman MPC for a Quadrotor in Wind

This report records how the project scope was formed, every design assumption,
the derivations behind the implementation, the problems that came up and how
they were fixed, and the results. It is written to be learned from, so each
decision states *why*, names the alternatives that were rejected, and points to
the code and the run that backs it.

**Companion documents.**
- [koopman_mpc_study_guide.md](koopman_mpc_study_guide.md): the theory (MPC
  stability, terminal ingredients, Koopman operator, EDMD, Koopman with
  control). This report cites its sections as **[SG §x]**.
- [mpc_guide.md](mpc_guide.md): implementation guide for the paper's
  LQR-augmented K-MPC (the starting point).
- Course notes in `learning_materials/`: **[LN-MPC]** (stabilisation, LQR,
  MPC), **[LN-DDC]** (DMDc, Koopman with control, SINDy, DeePC).

**How the numbers in this report are labelled.**
- *Run*: produced by a script in this repository; the run directory under
  `results/runs/` is named, and its `metadata.json` records git state, seed
  and configs.
- *Analysis*: a one-off calculation done during design (scratch script, not in
  the repository). Reproduce it before quoting it.
- *Derived*: follows from the equations shown.

---

## Contents

1. [Where we started](#1-where-we-started)
2. [How the scope was formed](#2-how-the-scope-was-formed)
3. [Assumptions](#3-assumptions)
4. [The plant model and its hover linearisation](#4-the-plant-model-and-its-hover-linearisation)
5. [Baseline: error-state linear MPC](#5-baseline-error-state-linear-mpc)
6. [Koopman model of the outer loop](#6-koopman-model-of-the-outer-loop)
7. [Koopman MPC and the comparison](#7-koopman-mpc-and-the-comparison)
8. [Issues log](#8-issues-log)
9. [Limitations and threats to validity](#9-limitations-and-threats-to-validity)
10. [Next steps](#10-next-steps)
11. [Reproduction and file map](#11-reproduction-and-file-map)
12. [What to understand from this project](#12-what-to-understand-from-this-project)

---

## 1. Where we started

### 1.1 The paper's architecture

The reference paper (Tabassum and Bai, CCTA 2023, "Data-Driven Koopman-based
Modeling and Control of Quadcopter in a Turbulent Wind Field") uses three loops:

```
reference ─► state-dependent LQR (100 Hz) ─► (+) ─► body-rate loop (1 kHz) ─► quadrotor
                                              ▲
                              K-MPC correction u_d (10 Hz)
```

The Koopman model describes the **tracking error of the LQR-controlled vehicle**,
and the MPC only adds a small bounded correction $u_d$ to the LQR command.
That is what was implemented first in this repository (`controllers/kmpc.py`,
`scripts/run_kmpc.py`).

### 1.2 What the initial analysis showed

A spectral analysis of the identified model (run
`20261113T155255Z_edmd_model1`, 55 features, λ = 20) showed:

| Observation | Value | Meaning |
|---|---|---|
| Spectral radius | 1.000 | only the constant feature |
| Without the constant | 0.974 | stable: the plant includes the LQR |
| Controllability of the λ = 1 mode | $\|w^\top B\| \approx 10^{-16}$ | uncontrollable; a Riccati equation on the full model fails |
| Nonlinear → state coupling | 0.29 vs 2.60 for the linear block | the lifting contributes about 10% |
| Non-zeros in $B$ | 23 of 220 | the small perturbation input barely needs nonlinear terms |
| 5-step velocity nRMSE | 0.44–0.65 | weak at the MPC horizon |

**Conclusion.** With an LQR underneath, the MPC is a feed-forward correction.
Stability comes from the inner loop, the QP is always feasible, and terminal
ingredients are optional. The questions the project wanted to study
(stability, terminal ingredients, the value of lifting) have no real answer in
that setting.

### 1.3 The research questions

From the first discussion:

- **Koopman with control:** what the eigenvalues mean; whether stability of the
  identified model matters; whether eigenfunctions have physical meaning;
  whether a bilinear model or SINDy is needed.
- **MPC:** why the paper uses no terminal ingredients; how stability and
  feasibility are ensured; what a terminal cost and terminal set would change;
  how to design them.

The theory answers are in the study guide. This report covers what was
*built* to study them.

---

## 2. How the scope was formed

Each row below is a decision taken in discussion, in the order it was taken.

| # | Decision | Why | Rejected alternative, and why |
|---|---|---|---|
| D1 | **MPC replaces the LQR** as the outer-loop controller | Makes stability, terminal ingredients and model accuracy matter: the model now has to describe an unstable open-loop vehicle | Keep the LQR: the MPC stays a correction, nothing to analyse (§1.2) |
| D2 | **Keep the body-rate loop**; the MPC commands body rates and collective thrust | Time-scale separation: the rate loop (10–20 s⁻¹) is much faster than position control. Removing it adds work without adding insight | MPC on motor commands: a much larger model, attitude dynamics at kHz rates |
| D3 | **Build the MPC on an analytic model first**, then swap in Koopman | Separation of concerns: debug the controller on a model you trust, so a later failure has one cause | Start with Koopman: controller and model errors would be confounded |
| D4 | **References: hover setpoints and level circles, constant yaw**; speed is the dial | Keeps a hover linearisation valid (tilt ≤ 0.3 rad), so the MPC can be LTI; first MPC implementation | Figure-8 (0.51 rad tilt) and squares (corners need unbounded acceleration): deferred |
| D5 | **LTI model, linearised at the hover trim** | One fixed QP, one Riccati solution, one terminal set, clean proofs; same model class as linear Koopman | LTV along the reference / re-linearise every step: more accurate, harder theory; future work |
| D6 | **Error-state coordinates** with a model-consistent reference | Tracking becomes regulation; the regulation proofs apply | Plant coordinates with error only in the cost: equivalent for LTI with a consistent reference, used later for Koopman (D9) |
| D7 | **Hold yaw constant**, not "low yaw rate" | The hover linearisation is taken at one yaw; any accumulated yaw deviation breaks it | Heading-following circle: bilinear in the input, needs a bilinear model |
| D8 | **Mean wind known to the MPC** (trim and drag in the model); a no-wind configuration as ablation | The Koopman model will learn a wind term; both must have the same information so the comparison tests the model, not the information | No wind in the baseline: the Koopman MPC would win by knowing more |
| D9 | **For the Koopman comparison, both controllers solve [LN-DDC] Eq. (17)**: cost on $Cz$, terminal weight $Q$, input bounds only | The course formulation; terminal ingredients for a lifted model raise a stabilisability problem (§6.9) that is a separate study | Use the full baseline design for Koopman too: would require solving that problem first |
| D10 | **Check input persistency of excitation (Hankel rank)** before fitting | Willems' lemma and the course's Question 1 | Trust the residual: closed-loop data can give a perfect residual and a wrong model ([EX-DMDc]) |
| D11 | **Keep existing code intact**; add new modules only | The paper reproduction must keep working | Refactor `simulate()` around a controller protocol: cleaner, but changes public interfaces; deferred |

Two framing points shaped the Koopman part:

- **"If the model is known, why Koopman?"** In simulation the plant is known,
  so Koopman's value is limited to (a) effects the nominal model leaves out,
  (b) one linear model valid over a region instead of re-linearising, and
  (c) a convex QP. This project's simulator already contains effects the
  baseline ignores: rate-loop lag, quadratic drag, trigonometric thrust
  projection. Knowing the truth is then an advantage: the identified model can
  be checked against it.
- **Course versus PhD.** For the course project, the claim is "Koopman
  recovers more than the textbook linearisation". For a PhD, the competitor
  would be a grey-box model (physics structure with fitted coefficients), and
  the wind would have to be estimated, not assumed.

---

## 3. Assumptions

Each assumption, where it is used, and what would break it.

| # | Assumption | Used in | Breaks when |
|---|---|---|---|
| A1 | **Ideal body-rate tracking**: commanded rates are achieved instantly | Baseline model (§4) | Always slightly false: the rate loop is a first-order lag of 0.05 s (roll, pitch) and 0.1 s (yaw), comparable to the 0.1 s sample time. The Koopman model learns it (§6.8) |
| A2 | **Small tilt** (below about 0.3 rad) | Hover linearisation | Fast circles: the 5 s circle needs 0.31 rad and the physics MPC degrades sharply (§7.4) |
| A3 | **Constant yaw** | Hover linearisation; reference generation | Any heading change; checked at start-up by `run_hover_mpc.py` |
| A4 | **Mean wind known and constant**; turbulence not predicted | Trim, both MPCs | Real flight: the wind must be estimated (offset-free MPC) |
| A5 | **Quadratic body drag** $f_D = R D_d R^\top v_r \|v_r\|$ with $v_r = v - v_w$ | Simulator and baseline model | Real rotors have additional drag effects (blade flapping, induced drag) |
| A6 | **Vehicle parameters are placeholders** (1 kg, F450-class) | Everything | Quantitative claims about a real vehicle |
| A7 | **10 Hz MPC with zero-order hold** | Both MPCs, the Koopman sample time | The LQR runs at 100 Hz; the MPC is at a rate disadvantage |
| A8 | **Solve time is negligible** (no delay compensation) | Simulation | Measured: below 1 ms against a 100 ms budget |
| A9 | **Euler angles away from gimbal lock** | State definition | Pitch near ±90°; far from our flights |
| A10 | **Nominal guarantees only** | Terminal ingredients (§5.6) | Turbulence and model error: guarantees become empirical |

---

## 4. The plant model and its hover linearisation

Code: [`src/controllers/hover_model.py`](../src/controllers/hover_model.py),
tests: [`tests/test_hover_model.py`](../tests/test_hover_model.py).

### 4.1 State, input, frames

$$x = [p,\ v,\ \eta] \in \mathbb{R}^9,\quad \eta = (\phi, \theta, \psi),\qquad u = [\omega_x,\ \omega_y,\ \omega_z,\ c] \in \mathbb{R}^4,\qquad v_w \in \mathbb{R}^3.$$

- $p$, $v$, $v_w$: world NED (z down), m and m/s.
- $\eta$: ZYX Euler angles, the convention of `Quaternion.to_euler`; checked
  against the quaternion rotation matrix in a test.
- $\omega$: body FRD, rad/s. $c$: mass-normalised collective thrust along
  $-z_{\text{body}}$, m/s².

**Why Euler angles and not quaternions or rotation matrices.** A quaternion
has 4 numbers for 3 degrees of freedom. Linearised, its norm direction is
uncontrollable, which breaks Riccati equations and terminal sets. A rotation
matrix has 9 numbers for 3 degrees of freedom. Three Euler angles are minimal,
and they match the Koopman error state. The simulator keeps its quaternion; the
MPC converts when it measures.

### 4.2 Nonlinear model

$$\dot p = v,\qquad \dot v = g\,e_3 - c\,R(\eta)\,e_3 - \tfrac1m f_D\bigl(R,\ v - v_w\bigr),\qquad \dot\eta = W(\phi,\theta)\,\omega,$$

$$R\,e_3 = \begin{bmatrix}c_\psi s_\theta c_\phi + s_\psi s_\phi\\ s_\psi s_\theta c_\phi - c_\psi s_\phi\\ c_\theta c_\phi\end{bmatrix},\qquad
W = \begin{bmatrix}1 & s_\phi t_\theta & c_\phi t_\theta\\ 0 & c_\phi & -s_\phi\\ 0 & s_\phi/c_\theta & c_\phi/c_\theta\end{bmatrix}.$$

The drag law is the simulator's own (`dynamics.drag.drag_force`), so the
baseline and the simulator share one definition (A5).

### 4.3 Two levels of "linearisation"

| Level | Operation | Approximation? |
|---|---|---|
| 1 | Taylor expansion about the **trim**: $\delta x = x - x^\star$ | **Yes**: higher-order terms dropped |
| 2 | Shift to the **reference**: $e = \delta x - \delta x_{\text{ref}}(t)$ | **No**: exact for a linear model, by superposition |

Level 2 works only because the model is linear. For a nonlinear model,
subtracting a reference does not give error dynamics independent of the
reference; you would have to linearise again along the reference (LTV).

### 4.4 Trim without wind

Set every derivative to zero:
1. $\dot p = 0 \Rightarrow v^\star = 0$.
2. $\dot\eta = 0$, and $W$ is invertible $\Rightarrow \omega^\star = 0$.
3. $\dot v = 0$ with zero airspeed (no drag) $\Rightarrow c^\star R(\eta^\star)e_3 = g\,e_3$. Taking lengths, $c^\star = g$; taking directions, $R e_3 = e_3$, so $\phi^\star = \theta^\star = 0$.

$p^\star$ and $\psi^\star$ are free: these are symmetries (translation and
heading).

### 4.5 Trim in mean wind

At rest, the air passes at $-\bar v_w$:

$$c^\star b_3 = g\,e_3 - \tfrac1m f_D^\star,\qquad b_3 = R(\eta^\star)e_3.$$

The length gives $c^\star$, the direction gives the tilt (with yaw removed,
$b_3 = [s_\theta c_\phi,\ -s_\phi,\ c_\theta c_\phi]$). Drag is computed in body
axes, so it depends on $\eta^\star$; a fixed-point iteration converges in a
few steps.

*Run* (test `test_trim_in_wind_is_an_equilibrium_and_leans_into_the_wind`), for
$\bar v_w = (2.70, 1.76, 0.01)$ m/s:

| | Roll | Pitch | Thrust |
|---|---:|---:|---:|
| No wind | 0 | 0 | 9.810 m/s² |
| Mean wind | −0.029 rad | 0.044 rad | 9.855 m/s² |

Hand check: drag $\approx 0.05 \times 3.22 \times (2.70, 1.76) = (0.435, 0.283)$ N,
so $\theta^\star \approx 0.435/9.855$. The exact thrust is slightly above
$\sqrt{g^2 + |a_D|^2} = 9.824$, because the tilted body exposes its $z$ face
($D_d = 0.10$) to the wind.

### 4.6 Jacobians at level hover, by hand

Block structure, from which variables appear in which equation:

$$A_c = \begin{bmatrix}0 & I & 0\\ 0 & \partial\dot v/\partial v & \partial\dot v/\partial\eta\\ 0 & 0 & \partial\dot\eta/\partial\eta\end{bmatrix},\qquad
B_c = \begin{bmatrix}0 & 0\\ 0 & \partial\dot v/\partial c\\ \partial\dot\eta/\partial\omega & 0\end{bmatrix}.$$

Two shortcuts do most of the work:
- **A term multiplied by a zero trim value vanishes.** $\partial(W\omega)/\partial\eta$ is evaluated at $\omega^\star = 0$, so it is zero.
- **A quadratic has zero slope at zero.** Drag $\propto v\|v\|$, so without wind $\partial\dot v/\partial v = 0$: drag disappears from the linear model.

The one real calculation, $\partial(R e_3)/\partial\eta$ at $\phi = \theta = 0$, gives
$$\frac{\partial\dot v}{\partial\eta} = -g\begin{bmatrix}s_\psi & c_\psi & 0\\ -c_\psi & s_\psi & 0\\ 0 & 0 & 0\end{bmatrix}.$$

At $\psi^\star = 0$:
$$\dot{\delta v}_x = -g\,\delta\theta,\qquad \dot{\delta v}_y = g\,\delta\phi,\qquad \dot{\delta v}_z = -\delta c,\qquad \dot{\delta\eta} = \delta\omega.$$

Sign check: positive pitch (nose up) tilts the thrust backward and accelerates
the vehicle south ($-x$). Tested, including that at $\psi = 90°$ roll drives
north instead.

**Structure.** Four decoupled chains: north (rate → pitch → $v_x$ → $p_x$), east
(the same through roll), down (thrust → $v_z$ → $p_z$), yaw (a single
integrator). All eigenvalues are 0. This is underactuation made visible: 4
inputs reach 9 states through chains. Controllability index 3 (*Analysis*), so
$[B\ AB\ A^2B]$ already has rank 9.

### 4.7 What changes in mean wind

| Block | Change |
|---|---|
| $\partial\dot v/\partial v$ | **Drag damping** $-\tfrac1m R^\star D_d(\|\bar v_{rb}\|I + \bar v_{rb}\bar v_{rb}^\top/\|\bar v_{rb}\|)R^{\star\top}$, about $2D\|\bar v\|$ along the wind and $D\|\bar v\|$ across it: rates 0.16, 0.31, 0.34 s⁻¹ |
| $D_c$ (wind input) | exactly $-\partial\dot v/\partial v$: wind acts as the negative of the vehicle's own velocity |
| $\partial\dot v/\partial\eta$ | at a tilted trim: yaw couples into horizontal acceleration (0.28, −0.44), tilt couples into vertical (−0.56, 0.87) |
| $\partial\dot v/\partial c$ | $-b_3$: small horizontal components (−0.044, −0.029) |
| $\partial\dot\eta/\partial\omega$ | $W(\eta^\star)$: off-diagonal entries around 0.03 |

**Implementation choice.** The hand derivation is for understanding; the code
computes Jacobians by central finite differences of the nonlinear model
(step $10^{-6}$), and tests compare the two. If they disagreed, a sign or a
frame would be wrong. One subtlety appeared: at zero airspeed the central
difference of $v\|v\|$ returns the step itself (about $2\times10^{-7}$), so the
test tolerance is $10^{-6}$, with a comment saying why.

### 4.8 Exact discretisation

Zero-order hold on the inputs and the wind:
$$\exp\!\left(\begin{bmatrix}A_c & B_c & D_c\\ 0 & 0 & 0\end{bmatrix}T_s\right) = \begin{bmatrix}A & B & D\\ 0 & I & 0\end{bmatrix}.$$

For the north chain, $A_c$ is nilpotent ($A_c^3 = 0$), so the series stops
after three terms:
$$A = \begin{bmatrix}1 & T & -\tfrac g2 T^2\\ 0 & 1 & -gT\\ 0 & 0 & 1\end{bmatrix},\qquad B = \begin{bmatrix}-\tfrac g6 T^3\\ -\tfrac g2 T^2\\ T\end{bmatrix}.$$

The general routine must reproduce these to round-off; a test checks it,
along with the down chain and yaw.

---

## 5. Baseline: error-state linear MPC

Code: [`src/controllers/linear_mpc.py`](../src/controllers/linear_mpc.py),
[`src/controllers/terminal_set.py`](../src/controllers/terminal_set.py),
[`src/simulation/outer_loop.py`](../src/simulation/outer_loop.py);
config: [`configs/controllers/hover_mpc.toml`](../configs/controllers/hover_mpc.toml);
tests: [`tests/test_linear_mpc.py`](../tests/test_linear_mpc.py).

### 5.1 Why a terminal cost matters on this vehicle

The hover model is all integrators, so nothing is stable without control.
*Analysis* (unconstrained, $Q$ and $R$ from §5.4): closed-loop spectral radius
of the MPC law against horizon:

| $N$ | $Q_f = 0$ | $Q_f = Q$ | $Q_f = P$ (Riccati) |
|---:|---:|---:|---:|
| 1 | 1.0000 | **1.0119** | 0.9563 |
| 2 | **1.0119** | **1.0037** | 0.9563 |
| 5 | 0.9921 | 0.9902 | 0.9563 |
| 20 | 0.9696 | 0.9686 | 0.9563 |
| 40 | 0.9588 | 0.9586 | 0.9563 |

Bold entries are unstable. With $Q_f = P$, unconstrained MPC *is* LQR at every
horizon, which became the first correctness test.

### 5.2 The reference: model-consistent by construction

The LQR's reference comes from differential flatness of the *nonlinear*
vehicle, assuming level hover. The trim in wind is tilted, so regulating to
that reference would make the MPC fight to stay level. Instead the MPC builds
its own reference from $p_{\text{ref}}(t)$ by **inverting the linear model**,
with yaw fixed:

$$\delta\dot v = A_{vv}\,\delta v + A_{v\phi}\,\delta\phi + A_{v\theta}\,\delta\theta + B_{vc}\,\delta c
\ \Rightarrow\ \begin{bmatrix}\delta\phi\\ \delta\theta\\ \delta c\end{bmatrix} = M^{-1}\left(a_{\text{ref}} - A_{vv}v_{\text{ref}}\right),\quad M = [A_{v\phi}\ \ A_{v\theta}\ \ B_{vc}].$$

Differentiating once gives the angle rates from the jerk,
$M^{-1}(j_{\text{ref}} - A_{vv}a_{\text{ref}})$, and then
$\delta\omega = W(\eta^\star)^{-1}\dot{\delta\eta}$.

Because the reference is a trajectory of the model, the error dynamics are
exactly LTI: $e^+ = Ae + B\Delta$. The inversion is continuous-time and
sampled with a zero-order hold, so the discrete residual is small but not zero
(tested: below 5% of the per-step motion of the reference).

Two tests check the physics. In still air, the circle's tilt equals
$r\Omega^2/g$ exactly. In wind it is about 10% larger, because the drag
damping term needs extra tilt. That second result first looked like a failing
test; it was correct physics (§8, issue I3).

### 5.3 From MPC to a QP

$$\min_{\Delta}\ \sum_{i=1}^{N-1} e_i^\top Q e_i + e_N^\top P e_N + \sum_{i=0}^{N-1}\Delta_i^\top R\,\Delta_i
\quad\text{s.t.}\quad e_{i+1} = Ae_i + B\Delta_i.$$

Predictions are affine in the inputs, $E = \Phi e_0 + \Gamma U$, so the cost
becomes $\tfrac12 U^\top HU + f(e_0)^\top U$ with $H = 2(\Gamma^\top\bar Q\Gamma + \bar R)$
fixed offline and $f = 2\Gamma^\top\bar Q\Phi e_0$ [LN-MPC §3.3]. The existing
`qp.condense` builds this (its `P` argument is the terminal weight), and DAQP
solves it.

### 5.4 Constraints and weights

- **Input bounds on the total command**:
  $u_{\text{lb}} \le u^\star + \delta u_{\text{ref},i} + \Delta_i \le u_{\text{ub}}$.
  Time-varying in $\Delta$ because the reference uses some input itself.
- **Body-rate bound.** The LQR bounds $\|\omega\| \le 2$ rad/s, a ball, which is
  not a linear constraint. The QP uses the largest box inside it,
  $|\omega_i| \le 2/\sqrt3 = 1.155$ rad/s. This is conservative: the MPC has
  less rate authority than the LQR.
- **Thrust**: $c \in [2, 18]$ m/s².
- **Tilt**: $|\phi|, |\theta| \le 0.5$ rad along the horizon (total attitude).
- **Weights**: $Q = \operatorname{diag}(100, 100, 100, 10, 10, 10, 1, 1, 1)$,
  $R = \operatorname{diag}(1, 5, 5, 0.1)$, the LQR's weights mapped to the error
  state (same as `kmpc.toml`).
- **Rate and horizon**: $T_s = 0.1$ s (the Koopman sample time), $N = 20$ (2 s).

### 5.5 Terminal modes

| Mode | Terminal cost | Terminal set | Guarantees (nominal) |
|---|---|---|---|
| `polytope` | $P$ (Riccati) | $\mathcal{O}_\infty$ of $A + BK$ | recursive feasibility and stability |
| `equality` | irrelevant | $\{0\}$ | recursive feasibility and stability |
| `cost_only` | $P$ | none | stability on a region; feasibility automatic with input bounds only |

**Why the equality constraint makes $P$ irrelevant.** If every feasible plan
ends at $e_N = 0$, then $e_N^\top P e_N = 0$. A terminal set of $\{0\}$ is a
*complete* set of terminal ingredients (admissible, invariant, and the cost
decreases trivially), but it gives a tiny feasible region and is fragile.

**Comparison of terminal sets** (*Analysis*, no-wind hover model, the limits
above). Largest single-axis error contained in the set:

| Error | $\{0\}$ | Ellipsoid $e^\top Pe \le \alpha$ | Polytope $\mathcal{O}_\infty$ |
|---|---:|---:|---:|
| north position | 0 | 0.054 m | 0.379 m |
| down position | 0 | 0.067 m | 0.480 m |
| north velocity | 0 | 0.144 m/s | 0.522 m/s |

Largest pure north offset from which the problem is feasible:

| $N$ | equality | polytope | cost only |
|---:|---:|---:|---:|
| 5 | 0.04 m | 1.6 m | always |
| 10 | 0.34 m | 3.6 m | always |
| 20 | 2.8 m | 9.6 m | always |

### 5.6 Computing $\mathcal{O}_\infty$ and making one set valid at all times

**Algorithm** (Gilbert and Tan, 1991; [SG §5.6]). With constraints
$Hx \le h$ (state rows, and input rows $\pm K$):

1. Start with $\{Hx \le h\}$.
2. At step $t$, consider the rows $H(A + BK)^t$.
3. For each new row, solve an LP: maximise it over the current set. If no new
   row can exceed its bound, stop: the set is invariant. Otherwise add the
   rows and repeat.

For the no-wind hover model with the constraints of §5.4 it stops after 7
steps with 84 inequalities (*Analysis*). In the runs (trim in mean wind,
bounds tightened for the reference) the set has 96 inequalities
(`summary.json`, `terminal_set_rows`). Tests:
the lecture example ($x^+ = 1.5x + u$, $K = -1$, $|u| \le 1$) gives exactly
$[-1, 1]$, and the hover set is verified to be invariant and admissible by
solving one LP per row.

**Tightening for tracking.** The input and tilt bounds on $e$ shift with the
reference. To use one fixed set at every time, the set is computed with bounds
shrunk by the reference's own use of input and tilt over the whole run (the
reference is sampled at construction). Any time's true bounds contain the
tightened ones, so the set is valid everywhere, which is what the proof needs.
If the reference itself uses all the margin, construction fails with a clear
error instead of building an empty set.

### 5.7 Fallback

If a solve is infeasible (only possible in `equality` or `polytope` mode),
the step uses the `cost_only` solution, which is always feasible because only
input bounds remain. Every fallback is logged and counted. That is the "no
silent failure" rule: infeasibility is a measured result, not something
hidden.

### 5.8 Simulator for a full outer-loop controller

`closed_loop.simulate` hard-wires the LQR. Rather than change it (D11),
`simulate_outer_loop` repeats the same loop with any controller that has a
`period` and `__call__(t, x)`. It produces the same `SimulationLog`, so all
existing plots and metrics work. The cost is about 50 duplicated lines; the
cleaner fix (a controller protocol used by `simulate`) is listed in §10.

### 5.9 Baseline results

*Runs* `20261116T183014Z`/`183105Z` (all modes, N = 20),
`183237Z`/`183319Z` (N = 5), `183220Z`/`183258Z` (wind ablation). One wind
seed each. Position RMSE in m:

| Controller | Hover (1 m climb) | Circle, 10 s |
|---|---:|---:|
| LQR, 100 Hz, no wind knowledge | 0.178 | 0.109 |
| MPC, any terminal mode, N = 20, mean wind known | 0.156 | 0.021 |
| MPC, no wind knowledge | 0.189 | 0.132 |
| MPC equality, N = 5 | 0.156 (65/150 fallbacks) | 0.031 (86/200 fallbacks) |
| MPC polytope or cost_only, N = 5 | 0.156 | 0.021 |

**What each row teaches:**
1. **The entire gain over the LQR is wind knowledge.** With the same
   information, the MPC is slightly worse than the LQR: 10 Hz against 100 Hz,
   a smaller rate box, and the LQR re-linearises at the current state.
2. **At N = 20 the terminal mode doesn't matter.** The cost-only plan already
   ends with $\|e_N\| = 0.0085$, inside $\mathcal{O}_\infty$, so neither
   constraint binds and the first move (the only one applied) is the same.
3. **At N = 5 the equality set breaks.** About 43% of steps are infeasible,
   and the commands chatter between the bounds as the controller switches
   between the equality and fallback solutions.
4. **With a Riccati terminal cost, N = 5 equals N = 20**, as the theory
   predicts (§5.1).

---

## 6. Koopman model of the outer loop

Code: [`src/koopman/selective_lifting.py`](../src/koopman/selective_lifting.py),
[`excitation.py`](../src/koopman/excitation.py),
[`outer_model.py`](../src/koopman/outer_model.py),
[`structured_edmd.py`](../src/koopman/structured_edmd.py),
[`src/datagen/outer_loop.py`](../src/datagen/outer_loop.py);
scripts `collect_outer_loop_data.py`, `identify_outer_loop_model.py`;
tests: [`tests/test_outer_koopman.py`](../tests/test_outer_koopman.py).

### 6.1 Why a new model was needed

The paper's model maps the *LQR perturbation* to the *tracking error*, with
the LQR inside the plant. A controller that replaces the LQR needs the map
from the **total command** to the **vehicle state**. New data, new
coordinates.

### 6.2 What linear Koopman could and could not capture

Decided before building, so the results could be read against a prediction:

| Effect missing from the baseline | Linear Koopman? |
|---|---|
| Drag varying with airspeed (mean wind fixed) | Yes: a nonlinearity in the state only |
| Rate-loop lag | Yes, if measured body rates are in the lifted state |
| Lift lost to tilt ($1 - \cos\theta$) | Yes, through attitude monomials |
| Drag interacting with turbulence ($v\cdot\delta v_w$) | No: wind enters through a constant $D$ (bilinear) |
| Yaw changing the input direction | No: $\tfrac{d}{dt}(\theta\cos\psi) = \omega_y\cos\psi - \dots$ is bilinear in the input |

The last two are [LN-DDC]'s Question 2: a constant $B$ cannot represent an
input whose effect depends on the state.

### 6.3 Coordinates

$$s = \frac{[p,\ v,\ \eta - \eta^\star,\ \omega]}{\sigma}\ \in \mathbb{R}^{12},\qquad z = \psi(s),$$
$$z^+ = Az + B\,\delta u + D\,\delta v_w,\qquad \delta x_{\text{phys}} = Cz,\quad C = [\operatorname{diag}(\sigma_{1..9})\ \ 0].$$

- **Deviations from the same trim as the baseline.** Both models share the
  operating point (and therefore the mean-wind information); only the
  dynamics around it are learned.
- **Scaled but not shifted.** With $\sigma$ only, $z = 0$ at the trim (apart
  from the constant feature), so the model's origin is the physical
  equilibrium. Shifting by the data mean would move it.
- **Body rates as observables.** They are measured on any real vehicle (gyros)
  and let the model represent the rate-loop lag (A1). They are not costed.
- **Inputs and wind in physical units**, so $B$ and $D$ act on what the MPC
  commands.

### 6.4 The dictionary: selective monomials

$$\psi(s) = \bigl(s,\ 1,\ \text{monomials of degree 2 in } (v_x, v_y, v_z, \phi, \theta, \psi)\bigr),\qquad 12 + 1 + 21 = 34 \text{ features}.$$

- **No position monomials.** The dynamics don't depend on position, so
  monomials in it could only fit noise.
- **Velocity monomials** for drag; **attitude monomials** for the thrust
  direction. Body rates stay linear.
- 34 features instead of 55 (full degree 2 in 9 states) or 91 (in 12): less
  data needed, fewer spurious slow modes.
- Same contract as `MonomialLifting` (state first, so $C = [I\ 0]$; known
  constant index), so the existing `build_snapshots` and `fit_edmd` work.

### 6.5 Data collection: dithered closed-loop flights

**The identifiability problem.** If the controller computes $u$ from $x$,
the regressor rows for $u$ are a function of the rows for $x$, and the fit
recovers only the closed loop, with a perfect residual ([EX-DMDc]; *Analysis*:
LQR data without dither gave rank 55 of 59 and an infinite condition number).

**The fix: dither.** Flights are flown by the baseline MPC (it already holds
commands at the model's 10 Hz) plus uniform noise,
$u = \operatorname{clip}(u_{\text{MPC}} + d)$ with $d \sim U(\pm[0.3, 0.3, 0.3, 1.0])$,
redrawn every 0.1 s. The **applied** command is logged, not the MPC's.

**Episodes** (`configs/datagen/outer_loop.toml`), 14 s each, Gaussian wind,
independent wind and dither streams per episode:

| Split | Circles (period × radius) | Hover (random start up to ±1, ±1, ±0.5 m) |
|---|---|---|
| Train | {5, 6, 8, 10, 12, 14} s × {1.5, 2.5} m = 12 | 6 |
| Validation | {5, 7, 10} s × {2.0} m = 3 | 2 |

Validation radii lie *between* the training radii, so the comparison
references are interpolated, not memorised. Peak tilt in the data reaches
0.50 rad (the r = 2.5 m, 5 s circle). *Run* `20261116T185409Z_outer_loop_data`:
18 training episodes, 2502 snapshot pairs.

**Alternatives considered.** Short open-loop episodes from random initial
states gave 7–30× better conditioning on the linear model (*Analysis*), but a
long open-loop flight of an integrator plant drifts away (one 20 s run
reached 696 m). Dithered closed-loop flight gives realistic trajectories, and
its excitation is checked rather than assumed.

### 6.6 Persistency of excitation: the Hankel test

For the inputs $u_1,\dots,u_T$ ($l$ channels), the depth-$L$ block-Hankel
matrix
$$H_L(u) = \begin{bmatrix}u_1 & u_2 & \cdots & u_{T-L+1}\\ \vdots & & & \vdots\\ u_L & u_{L+1} & \cdots & u_T\end{bmatrix} \in \mathbb{R}^{lL \times (T-L+1)}$$
has full row rank exactly when the input is PE of order $L$. **Willems'
lemma**: a controllable linear system with $n$ states and an input PE of
order $n + 1$ gives $\operatorname{rank}[X; U] = n + l$, so the regression for
$[A\ B]$ has a unique solution [LN-DDC §2.2.3].

**Why this doesn't conflict with the quadrotor being underactuated.** The lemma
needs *controllability*, not full actuation. The hover model is controllable
(index 3), because inputs reach positions through the dynamics.

Implementation (`koopman/excitation.py`):
- episodes are combined in a **mosaic Hankel matrix**
  $[H_L(u^{(1)}) \cdots H_L(u^{(r)})]$, with no window crossing an episode
  boundary (collective PE, [LN-DDC §4.5]);
- the rank uses a relative singular-value threshold, and $\sigma_{\min}/\sigma_{\max}$
  is reported, because rank is a yes/no answer and $\sigma_{\min}$ says how
  close the data is to losing it.

Signals and their Hankel rank at order 10, 4 channels (*Analysis*, matches the
tests): uniform random 40/40; binary PRBS 40/40; one sine per channel 8/40
(each sinusoid adds 2); three sines per channel 24/40; constant 1/40.

*Run* `20261116T190931Z_outer_loop_koopman`:

| Test | Rank | σmin/σmax | Result |
|---|---:|---:|---|
| Inputs, order $n + 1 = 13$ | 52/52 | 0.134 | PE |
| Inputs and wind, order 13 | 91/91 | 0.060 | PE |
| Inputs, order $N + 1 = 35$ (lifted dimension) | 140/140 | 0.042 | PE |
| Lifted regressor $[\psi(X); U; W]$ | 41/41 | condition number 6.4e3 | full rank |

**The Hankel test is necessary, not sufficient, for a lifted model.** Willems'
lemma is a linear-systems result. The rank of the *lifted regressor* is what
decides uniqueness for EDMD, so both are reported. §6.10 shows a case where
both passed and the fit was still wrong in one direction.

### 6.7 Fitting

The regression
$[A\ B\ D] = \Theta(X')\,[\Theta(X); \Upsilon; W]^+$, fitting all blocks
together ([LN-DDC]: fitting $A$ first and $B$ from the residual would assign
part of the input effect to the drift). The constant feature's row is fixed to
$e_c^\top$, so a "1" cannot decay over the horizon.

**Regularisation.** λ ∈ {0, 1, 10} was configured; the existing Lasso solver
did not converge for λ > 0 on this regressor (issue I5), so plain least
squares is used. With 61 snapshots per regressor row, it is well posed.

### 6.8 Validation: what the model actually learned

Open-loop prediction from every start of the validation flights, lifting once
and iterating without re-lifting (as the MPC does), with only the mean wind
known (*Run* `20261116T190931Z`):

**One step ahead (RMSE):**

| | roll | pitch | yaw | $v_z$ |
|---|---:|---:|---:|---:|
| Hover model | 0.0162 rad | 0.0140 rad | 0.0144 rad | 0.0272 m/s |
| Koopman | 0.0014 rad | 0.0015 rad | 0.0030 rad | 0.0053 m/s |

The attitude error is about 10× smaller: that is the rate-loop lag (A1),
learned through the body-rate observables.

**Twenty steps (2 s) ahead, per validation flight, vertical position:**

| Flight | Peak tilt | Hover model | Koopman |
|---|---:|---:|---:|
| circle 10 s | 0.17 | 0.100 m | 0.078 m |
| circle 7 s | 0.26 | 0.304 m | 0.100 m |
| circle 5 s | 0.41 | 0.908 m | 0.127 m |
| hover 0 | 0.12 | 0.033 m | 0.066 m |
| hover 1 | 0.19 | 0.041 m | 0.027 m |

The hover model's vertical error grows with tilt: that is lift lost to tilt,
$c(1 - \cos\theta)$, which the linearisation drops. Koopman represents it
through roll² and pitch². In near-hover flight the exact linearisation can be
as good or better (hover 0), as expected for a data-driven model near the
trim point.

Averaged over all states, the Koopman 20-step error is 0.27 of the hover
model's.

### 6.9 Spectrum and controllability

From `identification.json` (`slowest_modes`):

- **One spurious unstable real mode**, $|\lambda| = 1.034$ (doubling time
  about 2 s), dominated by roll, pitch and $v_y^2$, with good input authority
  ($\|w^\top B\| = 0.83$).
- **Several $\lambda = 1$ modes the input cannot reach** ($\|w^\top B\| = 0$):
  the constant feature, and the identity structure imposed on position.
- Then lightly damped modes around 0.994 (15 s time constant).

**Why this matters later.** Equation (17) needs neither stability nor
stabilisability, because it has no Riccati terminal cost. Terminal ingredients
computed on the full lifted model would fail: unreachable modes at
$\lambda = 1$ make the pair not stabilisable. That is [SG §9.4] in practice,
and it is the first problem to solve in the terminal-ingredient step.

### 6.10 The altitude identifiability problem, and the structural fix

**Symptom.** The first Koopman MPC (model `20261116T185727Z`) held a steady
0.10 m north offset in hover, worse than the physics MPC.

**Diagnosis.** Predicted acceleration at rest, at the trim attitude:

| Altitude | north accel. predicted |
|---|---:|
| 2 m (training altitude) | 0.03 m/s² |
| 1 m (hover test) | 0.43 m/s² |
| 0 m | 0.84 m/s² |

Fitted coupling: $\partial\dot v_x/\partial p_z \approx 0.40$ m/s² per metre,
and the constant column carried 0.84 m/s². They cancel at 2 m.

**Root cause.** Every training flight was centred at 2 m altitude; the
altitude standard deviation in the data is **0.033 m** (against 1.18 m for
north position). The altitude feature was then nearly collinear with the
constant feature. The regression split a bias between them arbitrarily: a
combination that is exactly right at 2 m and wrong anywhere else.

- The rank and PE tests passed: the collinearity is near, not exact, so the
  rank was full, but that one direction was poorly determined.
- Validation could not catch it: the validation flights were also at 2 m.

This is [LN-DDC]'s Question 1 (can the data separate the effects?), in a form
the global tests don't flag.

**Fix: impose the structure physics guarantees** (`koopman/structured_edmd.py`).
Translation invariance means position may only feed itself:
$$A[:, p_i] = e_{p_i},\qquad \text{fit the remaining columns of } [A\ B\ D].$$
In practice: subtract the known position contribution from the targets, then
run least squares on the other regressor rows. The constant row stays fixed.
A test builds data at a single altitude and checks that the invariant fit
recovers the true $A$ and $B$ and has exactly zero position coupling.

**After the fix:** predicted drift at rest is identical at every altitude
(0.03 m/s², a small disagreement with the physics trim). Validation at 2 m got
slightly worse (score 0.266 against 0.251), because the unconstrained fit had
been using altitude as a proxy that happened to work there. Hover RMSE went
from 0.185 m to 0.156 m (§7.4).

**Two other possible fixes, not chosen:**
- **More data:** vary the altitude in training. Fixes the symptom for the
  altitudes flown; leaves the structure unimposed.
- **Drop position from the lifted state:** not possible, since the MPC costs
  position.

The lesson generalises: **use physics structure where you have it, and learn
only what you don't know.**

---

## 7. Koopman MPC and the comparison

Code: [`src/controllers/output_mpc.py`](../src/controllers/output_mpc.py),
`scripts/compare_koopman_mpc.py`.

### 7.1 One formulation for both models

$$\min\ \sum_{i=0}^{N-1}\left(\|Cz_i - r_i\|_Q^2 + \|\delta u_i\|_R^2\right) + \|Cz_N - r_N\|_Q^2
\quad\text{s.t.}\ z_{i+1} = Az_i + B\,\delta u_i,\ z_0 = \psi(x),\ u_{\text{lb}} \le u^\star + \delta u_i \le u_{\text{ub}}.$$

- The cost is on the **physical output** $Cz$; the lifted state is only for
  prediction ([LN-DDC] Eq. (17)).
- Terminal weight $Q$, no terminal set, input bounds only.
- $\delta v_w = 0$ in prediction: the mean wind is in the trim, turbulence is
  not predicted.
- **One class, `OutputMPC`, runs both models**: physics ($z = x - x^\star$,
  $C = I$) and Koopman ($z = \psi(s)$, $C = [\operatorname{diag}\sigma\ 0]$).
  This guarantees the two controllers differ *only* in $(A, B, C, \psi)$. A
  test builds a Koopman model whose lifting is the identity and checks it gives
  the same command as the physics controller, to $10^{-8}$.

### 7.2 Reference

$r(t) = [p_{\text{ref}}(t),\ v_{\text{ref}}(t),\ 0,\ 0,\ 0]$: position and
velocity, which don't depend on any model, with attitude regulated to the trim.
There is no model-based attitude or input feedforward, so neither model is
favoured.

The cost: on a circle the controller is penalised for the tilt and rates it
needs, which adds a small lag. It affects both controllers equally. The
"MPC full" column (§7.4) shows what the physics model reaches with the full
design.

### 7.3 Fairness checklist

| Held equal | Value |
|---|---|
| Sample time, horizon | 0.1 s, N = 20 |
| $Q$, $R$, input bounds | `hover_mpc.toml` |
| Terminal weight | $Q$ |
| Wind information | mean wind via the shared trim; turbulence not predicted |
| Wind realisations | same seed → same draws for every controller |
| Simulator, rate loop | identical |

### 7.4 Results

*Run* `20261116T190947Z_koopman_vs_physics_mpc`, 5 wind seeds. 3-D position
RMSE, mean ± std:

| Reference | Peak tilt | LQR | MPC physics | **MPC Koopman** | MPC full |
|---|---:|---:|---:|---:|---:|
| Hover, 1 m climb | 0 | 0.179 ± 0.001 | 0.157 ± 0.000 | **0.156 ± 0.001** | 0.157 ± 0.000 |
| Circle, 10 s | 0.08 rad | 0.112 ± 0.003 | 0.025 ± 0.003 | **0.018 ± 0.002** | 0.024 ± 0.003 |
| Circle, 7 s | 0.16 rad | 0.131 ± 0.003 | 0.039 ± 0.003 | **0.018 ± 0.002** | 0.037 ± 0.003 |
| Circle, 5 s | 0.31 rad | 0.152 ± 0.003 | 0.107 ± 0.003 | **0.022 ± 0.004** | 0.091 ± 0.003 |

Koopman beats the physics MPC in **5/5 seeds in every row** (paired by seed).
Per axis on the 5 s circle: physics (0.086, 0.056, 0.030) m against Koopman
(0.019, 0.011, 0.002) m. The vertical axis shows the largest relative gain,
consistent with lift lost to tilt (§6.8).

**Reading the speed sweep:**
- The physics MPC degrades steeply as tilt grows; the Koopman MPC stays almost
  flat (0.018 → 0.022 m).
- "MPC full" (Riccati terminal cost, model-consistent attitude and input
  reference) recovers only a little of the gap (0.107 → 0.091 m). The gap
  comes from the **model**, not the formulation.
- In hover, the RMSE is dominated by the initial 1 m climb, which every
  controller flies at the thrust limit.

**The first comparison** (model `20261116T185727Z`, before the fix of §6.10):
the same circle results, but hover 0.185 m, worse than the physics MPC in all
5 seeds. That run (`20261116T190007Z`) is kept for the record.

---

## 8. Issues log

| # | Issue | Symptom | Root cause | Fix | Lesson |
|---|---|---|---|---|---|
| I1 | Jacobian test failed at $2\times10^{-7}$ | Drag entries not exactly 0 at zero airspeed | Central difference of $v\|v\|$ returns the step $h$ | Tolerance $10^{-6}$ with a comment | Know the error of your numerical method before setting test tolerances |
| I2 | All three terminal modes identical at N = 20 | Same RMSE to 4 digits | Not a bug: the cost-only plan already ends at $\|e_N\| = 0.0085$; the constraints don't bind | Verified by computing $e_N$ and re-running at N = 5 | Terminal ingredients matter at short horizons; check whether a constraint is active before concluding |
| I3 | Circle tilt test off by 11% | 0.090 rad against $r\Omega^2/g = 0.080$ | Not a bug: drag damping at 1.26 m/s needs about 0.009 rad more tilt | Test run in still air; the wind case is physics | A "failed" test can be a correct model; work out the number by hand |
| I4 | Large gain over the LQR | Circle 0.021 m against 0.109 m | Information, not MPC: the MPC knew the mean wind | No-wind ablation configuration | Compare controllers with equal information |
| I5 | Lasso did not converge (λ = 1, 10) | `RuntimeError` after 100,000 sweeps | Iteration cap of the existing coordinate-descent solver on a larger regressor | Failure recorded in the report; λ = 0 used; `edmd.py` unchanged | Report failed options; don't hide them |
| I6 | Koopman MPC offset in hover | Steady 0.1 m north error | Altitude nearly collinear with the constant feature (§6.10) | Translation-invariant fit | Passing global rank tests doesn't mean every direction is identified; impose known structure |
| I7 | Equality-mode chattering at N = 5 | Rates jumping between the bounds | Infeasible on 43% of steps; switching between the two solutions | Measured and logged as a result | The fragility of a terminal set with no interior, made visible |
| I8 | Rate disadvantage of the MPC | MPC without wind knowledge slightly worse than LQR | 10 Hz against 100 Hz, smaller rate box | Not fixed: reported | Separate the effect of the controller from the effect of its update rate |

---

## 9. Limitations and threats to validity

- **Interpolation, not extrapolation.** Training circles cover periods 5–14 s
  at radii 1.5 and 2.5 m; the test circles (r = 2 m) are inside that range.
  Faster circles, other altitudes beyond what invariance covers, and figure-8
  paths are untested.
- **One dataset, one identification.** Results over wind seeds are paired, but
  the model was fitted once. A second dataset (different dither and wind
  streams) would show how much the result depends on the particular fit.
- **Shared trim.** The Koopman model uses the physics trim as its operating
  point. Without it the model would have to learn the offsets through the
  constant feature, which works but is not tested here.
- **Simulated plant, placeholder parameters** (A6). The "unmodelled effects"
  are the simulator's own (rate lag, quadratic drag, trigonometry), not
  real-world aerodynamics.
- **Mean wind assumed known** (A4) by both controllers.
- **No terminal ingredients in the Koopman comparison**, so no stability claim
  for the Koopman MPC; everything there is empirical.
- **Equation-(17) reference has no feedforward**, which costs both controllers
  a little on circles. The "MPC full" column bounds that effect for the physics
  model.

---

## 10. Next steps

In order of dependence:

1. **Terminal ingredients for the Koopman MPC.** Start from the per-mode PBH
   report. Options [SG §9.4]: (A) ingredients from the physical linear model
   imposed through $C$; (B) Riccati equation on the stabilisable part of the
   lifted model, after removing the constant and the unreachable $\lambda = 1$
   modes; (C) long horizon, no set. Apply the same choice to the physics MPC.
2. **Horizon ablation on the Koopman MPC** with the Riccati terminal cost:
   does it allow a shorter horizon, as for the physics model (§5.9)?
3. **Yaw-following circle**: where linear Koopman should fail (§6.2), and the
   motivating case for a bilinear model $z^+ = Az + B_0u + \sum_i u_iB_iz$,
   solved with one QP per step by linearising about the previous plan.
4. **Robustness of the identification**: vary altitude in the data; a second
   dataset; the unconstrained fit with altitude varied, to confirm the
   diagnosis of §6.10 from the data side.
5. **Wind estimation** (offset-free MPC with a disturbance observer) to remove
   assumption A4.
6. **Refactor** `closed_loop.simulate` around a controller protocol so the LQR
   and the MPCs share one loop (removes the duplication of §5.8).
7. **Grey-box competitor** (for the PhD framing): physics structure with
   drag and lag parameters fitted to the same data.

---

## 11. Reproduction and file map

Use `conda activate koopman_mpc` and run from the repository root.

```bash
PYTHONPATH=src python scripts/run_hover_mpc.py configs/experiments/circle_slow_gaussian.toml configs/controllers/hover_mpc.toml --terminal all
```
```bash
PYTHONPATH=src python scripts/run_hover_mpc.py configs/experiments/hover_gaussian.toml configs/controllers/hover_mpc.toml configs/controllers/hover_mpc_no_wind.toml --terminal polytope
```
```bash
PYTHONPATH=src python scripts/collect_outer_loop_data.py configs/datagen/outer_loop.toml
```
```bash
PYTHONPATH=src python scripts/identify_outer_loop_model.py configs/identify/outer_loop.toml results/runs/<time>_outer_loop_data
```
```bash
PYTHONPATH=src python scripts/compare_koopman_mpc.py results/runs/<time>_outer_loop_koopman configs/controllers/hover_mpc.toml configs/experiments/hover_gaussian.toml configs/experiments/circle_slow_gaussian.toml configs/experiments/circle_7s_gaussian.toml configs/experiments/circle_5s_gaussian.toml --seeds 5
```

| Layer | File | Role |
|---|---|---|
| Model | `src/controllers/hover_model.py` | 9-state model, trim, Jacobians, discretisation |
| Controller | `src/controllers/linear_mpc.py` | error-state MPC, 3 terminal modes, reference inversion, fallback |
| Controller | `src/controllers/terminal_set.py` | $\mathcal{O}_\infty$ (Gilbert–Tan) |
| Controller | `src/controllers/output_mpc.py` | Eq. (17) MPC shared by physics and Koopman |
| Koopman | `src/koopman/selective_lifting.py` | dictionary without position monomials |
| Koopman | `src/koopman/excitation.py` | Hankel / PE test |
| Koopman | `src/koopman/structured_edmd.py` | translation-invariant least squares |
| Koopman | `src/koopman/outer_model.py` | model container, lifting of measurements, horizon RMSE |
| Data | `src/datagen/outer_loop.py` | dithered controller, dataset format |
| Simulation | `src/simulation/outer_loop.py` | closed loop with a full outer-loop controller |
| Analysis | `src/analysis/mpc_plots.py` | commands, solver, top view, sweep |
| Glue | `src/experiment/hover_mpc_config.py` | config loader, controller factory |
| Configs | `configs/controllers/hover_mpc*.toml`, `configs/datagen/outer_loop.toml`, `configs/identify/outer_loop.toml`, `configs/experiments/{hover_gaussian,circle_slow_gaussian,circle_7s_gaussian,circle_5s_gaussian}.toml` | all parameters |
| Tests | `tests/test_hover_model.py`, `tests/test_linear_mpc.py`, `tests/test_outer_koopman.py` | 34 tests; full suite 296 |

Runs referenced in this report:

| Run | Content |
|---|---|
| `20261116T183014Z`, `183105Z` | baseline, all terminal modes, N = 20 (hover, circle) |
| `20261116T183220Z`, `183258Z` | baseline with and without wind knowledge |
| `20261116T183237Z`, `183319Z` | baseline, all terminal modes, N = 5 |
| `20261116T185409Z_outer_loop_data` | identification dataset |
| `20261116T185727Z_outer_loop_koopman` | first Koopman model (altitude problem) |
| `20261116T190931Z_outer_loop_koopman` | translation-invariant Koopman model |
| `20261116T190007Z_koopman_vs_physics_mpc` | comparison with the first model |
| `20261116T190947Z_koopman_vs_physics_mpc` | final comparison |

---

## 12. What to understand from this project

1. **Why the LQR-augmented MPC hides everything interesting.** A stable inner
   loop plus a bounded correction makes stability and feasibility trivial.
   You can only study MPC once MPC carries the stabilising burden.
2. **Linearisation and change of coordinates are different operations.** The
   Taylor step about the trim is the only approximation; shifting to the
   reference is exact for a linear model.
3. **The terminal cost buys horizon.** On integrator dynamics, $Q_f = Q$
   with a short horizon is unstable; $Q_f = P$ makes the horizon nearly
   irrelevant (§5.1, §5.9).
4. **A terminal *set* is about feasibility, and its size matters.** $\{0\}$ is
   complete in theory and fragile in practice; $\mathcal{O}_\infty$ is large
   and stays a QP.
5. **Equal information before equal comparison.** The baseline's whole
   advantage over the LQR was knowing the mean wind (§5.9).
6. **Identifiability has two levels.** The Hankel and rank tests check that
   the input is rich enough overall. A single direction (altitude) can still
   be unidentified. Physics structure is the cheapest fix (§6.10).
7. **Koopman's value is in what the nominal model leaves out.** Here: rate
   lag (10× better one-step attitude prediction) and lift lost to tilt (7×
   better vertical prediction on the fastest circle). Near the trim point,
   the exact linearisation is as good or better.
8. **Linear Koopman with control has a structural limit.** Effects where the
   input's influence depends on the state (yaw-dependent thrust direction,
   drag–turbulence coupling) need a bilinear model.

**Exercises that use this code:**
- Reproduce the closed-form discretisation of §4.8 by hand, then run
  `test_zoh_matches_closed_form_integrator_chains`.
- Run the Gilbert–Tan iteration by hand for the lecture example; compare with
  `test_terminal_set_of_the_lecture_example_is_unit_interval`.
- Find the smallest horizon at which the polytope terminal constraint becomes
  active on the slow circle.
- In `identification.json`, find the unstable mode and the unreachable
  $\lambda = 1$ modes, and explain why a Riccati equation on the full lifted
  model would fail.
- Fit the unconstrained model on data with altitude varied, and check
  whether the phantom force of §6.10 disappears without the structural fix.
