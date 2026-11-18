# Koopman MPC: from derivation to implementation

A study and implementation guide for K-MPC / K-SMPC in this repo
(CCTA paper "Data-Driven Koopman-based Modeling and Control of Quadcopter in a
Turbulent Wind Field", Sec. III-C, Eqs. (12)–(19)).

Notation follows the paper and the stubs in `src/controllers/kmpc.py`,
`src/controllers/ksmpc.py` and `src/controllers/qp.py`:

| Symbol | Meaning | Size (this repo) |
|---|---|---|
| $\psi_k$ | lifted error state, $\psi(x_k)$ with $x_k$ the (standardized) tracking error | $N = 55$ |
| $z_k,\ \zeta_k$ | deterministic / stochastic part of $\psi_k$, Eq. (12a) | $N$ |
| $\nu_k$ | deterministic input, the perturbation $u_d = [\omega_{xd}, \omega_{yd}, \omega_{zd}, f_d]$ added to the LQR command | $m = 4$ |
| $\bar v_w,\ \tilde v_{w,k}$ | mean / fluctuating wind | $3$ |
| $A, B, D, C$ | Koopman model, Eq. (5); $C = [I_n\ 0]$ | $N{\times}N$, $N{\times}m$, $N{\times}3$, $n{\times}N$ |
| $n_p$ | prediction horizon (paper: 5) | |
| $\Delta t$ | MPC / Koopman sample time, `dt_augment` | 0.1 s |

---

## 0. Short answers first

**Does the paper use linear MPC?** Yes. Sec. III-C says that because the
identified model is linear, a linear MPC can be designed. The model is linear in the
*lifted* state, so the controller is a linear MPC on $z \in \mathbb{R}^{55}$.
The nonlinearity is all in the lifting $\psi(\cdot)$, which is evaluated once per
control step at the measured state. It is never inside the optimization.

**Is it a QP?** Yes, a convex QP:

- quadratic cost with $Q \succeq 0$ and $R \succ 0$;
- linear (affine, because of $D\bar v_w$) equality dynamics;
- box constraints on $\nu$ and on $z$, which are linear inequalities.

When $R \succ 0$, the condensed Hessian is positive definite, so the QP is strictly
convex with a unique minimizer. Following Korda & Mezić (2018) [25], the paper solves it
in **dense (condensed) form**. The K-SMPC variant is the same QP with tighter
bounds, so it is also a QP.

**Terminal ingredients in the paper?** As far as I can read Eqs. (15)–(16) (the
PDF text extraction is garbled there, so check this against the PDF), the cost
weights the terminal state with the same $Q$. There is no terminal set and no
Riccati or Lyapunov terminal cost. That means **the paper gives no formal closed-loop
stability or recursive-feasibility guarantee for the MPC itself**. In practice it
works because the MPC sits *on top of* an LQR that already stabilizes the vehicle
(see §3.4). This is a gap you can fill in your thesis.

---

## 1. Learning path (in order)

Do these on a toy system first (double integrator, $\Delta t = 0.1$). Only then move
to the 55-dim Koopman model. Each step has a test you can write.

1. **Finite-horizon unconstrained LQ, batch form.** Derive the stacked prediction
   $Z = \Phi z_0 + \Gamma V$ and the minimizer $V^\* = -H^{-1} f$.
   *Test:* with $n_p \to \infty$ (or terminal $P$ = DARE solution), the first block
   of $V^\*$ equals $-K_{\text{LQR}} z_0$.
2. **Dynamic programming / Riccati recursion** for the same problem. *Test:* it
   gives the same $V^\*$ as the batch form. This shows you why the sparse
   form and the condensed form are the same problem.
3. **Add box constraints → QP.** Solve with a QP solver. *Test:* when no
   constraint is active, the QP solution equals $-H^{-1}f$.
4. **Receding horizon.** Apply only $\nu_0$, measure, re-solve. Learn why the
   closed loop is *not* the open-loop plan.
5. **Terminal ingredients** (§3): stability and recursive feasibility.
6. **Disturbances:** a known mean (affine term), then offset-free MPC with
   estimated disturbance, then robust/tube and stochastic (chance-constrained) MPC.
7. **Lifted (Koopman) version:** the same machinery, with the caveats of §5.

Recommended reading, in order of usefulness:

- Rawlings, Mayne & Diehl, *Model Predictive Control: Theory, Computation, and
  Design*, 2nd ed. (2017). Ch. 1 (LQ and batch form), Ch. 2 (stability, terminal
  ingredients), Ch. 3 (robust/tube), Ch. 8 (numerics).
- Borrelli, Bemporad & Morari, *Predictive Control for Linear and Hybrid Systems*
  (2017). Ch. 8–12 for condensed QP, invariant sets and MPC feasibility.
- Mayne, Rawlings, Rao & Scokaert, "Constrained model predictive control:
  Stability and optimality," *Automatica* 36(6), 2000. The canonical terminal
  ingredient conditions.
- Korda & Mezić, "Linear predictors for nonlinear dynamical systems: Koopman
  operator meets model predictive control," *Automatica* 93, 2018. The paper's [25];
  read the part on the dense form and why its cost does not depend on $N$.
- Pannocchia & Rawlings, "Disturbance models for offset-free model-predictive
  control," *AIChE J.*, 2003.
- Mesbah, "Stochastic model predictive control: An overview and perspectives for
  future research," *IEEE Control Systems Magazine*, 2016. Background for K-SMPC.

---

## 2. Formulation

### 2.1 The optimal control problem (K-MPC, Eqs. (15)–(16))

At each MPC step, with measured error $x_0$:

$$
\begin{aligned}
\min_{\nu_0,\dots,\nu_{n_p-1}}\quad & \underbrace{(Cz_{n_p})^\top P\,(Cz_{n_p})}_{\text{terminal, paper: } P=Q}
\;+\; \sum_{k=0}^{n_p-1} (Cz_k)^\top Q\,(Cz_k) + \nu_k^\top R\,\nu_k \\
\text{s.t.}\quad & z_{k+1} = A z_k + B\nu_k + D\bar v_w,\qquad z_0 = \psi(x_0) \\
& u_{lb} \le \nu_k \le u_{ub},\qquad k = 0,\dots,n_p-1 \\
& h_{lb} \le E z_k \le h_{ub},\qquad k = 1,\dots,n_p
\end{aligned}
$$

Then apply $u_d = \nu_0^\*$ (added to the LQR command) and hold it for
$\Delta t = 0.1$ s, i.e. 10 LQR steps at 100 Hz.

Notes on choices the paper leaves implicit:

- **Cost on $Cz$, not $z$.** You penalize the physical error, not the monomials.
  $C^\top Q C$ is then only PSD in lifted space. That is fine because $R \succ 0$
  keeps the condensed Hessian PD.
- **The state-constraint selector $E$.** The stub writes $h_{lb} \le z_k \le h_{ub}$
  on the full lifted state. Constraining monomials like $x_1 x_2$ or the constant
  $1$ makes no physical sense. Use $E = C$, or a subset of rows of $C$, e.g.
  position error only. Decide this explicitly and write it in the docstring.
- **$z_0$ is not a decision variable.** Its cost term is a constant, so drop it
  from the QP. It does not change the minimizer, and it removes an off-by-one source.
- **The constraint at $k=0$** cannot be enforced because $z_0$ is measured. Start
  state constraints at $k=1$.

### 2.2 Condensing (dense form), step by step

Unroll the dynamics:

$$
z_k = A^k z_0 + \sum_{j=0}^{k-1} A^{k-1-j} B\,\nu_j + \sum_{j=0}^{k-1} A^{j} D\bar v_w .
$$

Stack $Z = [z_1; \dots; z_{n_p}] \in \mathbb{R}^{N n_p}$ and
$V = [\nu_0; \dots; \nu_{n_p-1}] \in \mathbb{R}^{m n_p}$:

$$
Z = \Phi z_0 + \Gamma V + \Lambda \bar v_w,
\quad
\Phi = \begin{bmatrix} A \\ A^2 \\ \vdots \\ A^{n_p}\end{bmatrix},
\quad
\Gamma = \begin{bmatrix}
B & 0 & \cdots & 0\\
AB & B & \cdots & 0\\
\vdots & & \ddots & \\
A^{n_p-1}B & \cdots & AB & B
\end{bmatrix},
\quad
\Lambda = \begin{bmatrix} D \\ (A+I)D \\ \vdots \\ \sum_{j=0}^{n_p-1}A^jD\end{bmatrix}.
$$

Block-diagonal weights: $\bar C = I_{n_p}\otimes C$,
$\bar Q = \text{blkdiag}(Q,\dots,Q,P)$, $\bar R = I_{n_p}\otimes R$.
Substituting $Z$ and collecting terms:

$$
J(V) = V^\top \underbrace{\left(\Gamma^\top \bar C^\top \bar Q \bar C\,\Gamma + \bar R\right)}_{H/2} V
+ 2\,\underbrace{\left(\Phi z_0 + \Lambda\bar v_w\right)^\top \bar C^\top \bar Q \bar C\,\Gamma}_{f^\top/2} V + \text{const.}
$$

So the QP in standard form is

$$
\min_V\ \tfrac12 V^\top H V + f^\top V
\quad\text{s.t.}\quad
\underbrace{\begin{bmatrix} \bar E\Gamma \\ -\bar E\Gamma \end{bmatrix}}_{G} V \le
\underbrace{\begin{bmatrix} \bar h_{ub} - \bar E(\Phi z_0 + \Lambda\bar v_w) \\ -\bar h_{lb} + \bar E(\Phi z_0 + \Lambda\bar v_w)\end{bmatrix}}_{w(z_0)},
\quad \bar u_{lb} \le V \le \bar u_{ub}.
$$

**Design point (separation of concerns):** $H$, $\Gamma$, $\Phi$, $\Lambda$ and
$G$ depend only on the model and the weights. Build them **once** in the
constructor. Per step, only $f(z_0)$ and $w(z_0)$ change, and both are affine in
$z_0$. That is the natural split between `condense()` (pure, offline) and
`KoopmanMPC.__call__` (online).

### 2.3 Dense vs sparse: why dense here

| | Dense (condensed) | Sparse |
|---|---|---|
| Decision vars | $m n_p = 20$ | $N n_p + m n_p = 295$ |
| Hessian | $20\times20$ dense | $295\times295$ sparse, block-banded |
| Cost grows with | $n_p^2$ in vars, independent of $N$ online | $N n_p$ |
| Numerics | powers $A^k$ appear explicitly (§5.2) | no powers; better conditioned |

Korda & Mezić's main practical point: after condensing, the online cost does
**not** depend on the lifted dimension $N$. With $N=55$ and $n_p=5$, dense is
clearly right. Use sparse only if you go to long horizons ($n_p \gtrsim 30$)
or if $A$ has eigenvalues near or above 1.

### 2.4 K-SMPC: constraint tightening (Eqs. (17)–(19))

Propagate the covariance of the stochastic part, starting from
$\Sigma_{\zeta,0} = 0$:

$$
\Sigma_{\zeta,k+1} = A\Sigma_{\zeta,k}A^\top + D\Sigma_{v_w}D^\top,
\qquad
\gamma_k = \sqrt{2\,\text{diag}(\Sigma_{\zeta,k})}\ \text{erf}^{-1}(\beta),
$$

and replace the bounds with $h_{lb}+\gamma_k \le Ez_k \le h_{ub}-\gamma_k$.

- $\gamma_k$ does not depend on $z_0$, so it is **offline**. The QP keeps
  the same structure and only $w$ changes. K-SMPC therefore costs nothing extra online.
- Use $\text{diag}(E\,\Sigma_{\zeta,k}E^\top)$ so the tightening lives in the same
  coordinates as the constraint. It must also be in the *standardized* units if the
  model is standardized (§5.1).
- $\sqrt2\,\sigma\,\text{erf}^{-1}(\beta)$ is the **two-sided** bound:
  $P(|\zeta_i| \le \gamma) = \beta$. With $\beta = 0.99$ that is about $2.58\sigma$. A
  one-sided bound would be $\sigma\sqrt2\,\text{erf}^{-1}(2\beta-1)$, about
  $2.33\sigma$. Know which one you are using. Also note that the paper's guarantee
  holds per constraint and per step, not jointly over the horizon.
- The paper treats $\tilde v_w$ as i.i.d. Gaussian. LES turbulence is
  time-correlated, so the real $\Sigma_\zeta$ grows faster than this recursion
  predicts. This is a known limitation and a possible thesis contribution
  (e.g., a colored-noise / shaping-filter wind state).
- `scipy.special.erfinv` is available, so no new dependency is needed.

---

## 3. Terminal ingredients

### 3.1 What they are for

Truncating at $n_p$ means the controller is short-sighted. Terminal
ingredients make the finite-horizon cost behave like the infinite-horizon cost.
You need three pieces:

1. **Terminal cost** $V_f(z) = z^\top P z$.
2. **Terminal set** $\mathbb{X}_f$, imposed as the constraint $z_{n_p} \in \mathbb{X}_f$.
3. **Local controller** $\kappa_f(z)$, typically $-Kz$, which is never applied but is used in the proof.

Standard sufficient conditions (Mayne et al. 2000; Rawlings–Mayne–Diehl Ch. 2):

- (A1) $\mathbb{X}_f$ is **positively invariant** under $z^+ = Az + B\kappa_f(z)$.
- (A2) Inside $\mathbb{X}_f$, $\kappa_f$ satisfies the input constraints and the state constraints.
- (A3) $V_f$ is a **local control Lyapunov function**:
  $V_f(Az+B\kappa_f(z)) - V_f(z) \le -\ell(z,\kappa_f(z))$ for all $z\in\mathbb{X}_f$.

Under (A1)–(A3), you get **recursive feasibility** (a feasible problem now
implies a feasible problem at the next step, by the shifted solution plus
$\kappa_f$). You also get **asymptotic stability**, because the optimal cost is a
Lyapunov function. Each step of this proof is a good exercise to write out once by hand.

Usual choice: $K$, $P$ from the DARE on $(A, B, C^\top QC, R)$, and $\mathbb{X}_f$ = the
maximal positively invariant set of $A-BK$ inside the constraints (a polytope),
or a sublevel set $\{z^\top P z \le \alpha\}$ (an ellipsoid).

### 3.2 Why the textbook recipe breaks for a Koopman model

- **Uncontrollable modes.** The lifting contains the constant monomial
  $\psi_j = 1$. Ideally its row of $A$ is $e_j^\top$ and its row of $B$ is $0$, which
  is an eigenvalue at exactly 1 that the input cannot move. Then $(A,B)$ is **not
  stabilizable**, and the DARE fails or returns garbage. l1-EDMD gives you
  only approximately this row, so check the spectrum of $A$ first.
- **The lifted state lives on a manifold.** $\psi(x)$ cannot be an arbitrary point in
  $\mathbb{R}^{55}$, because $x_1^2$ is tied to $x_1$. Invariant sets computed in
  $\mathbb{R}^{N}$ ignore this, so they are conservative and expensive to compute
  (polytopes in 55-D are hard).
- **Equilibrium shift.** With $D\bar v_w \ne 0$, the origin is not an
  equilibrium of the model. The terminal ingredients must be centred at
  $z_{ss}$, the solution of $(I-A)z_{ss} = B\nu_{ss} + D\bar v_w$. That is a
  target-calculation problem.

### 3.3 What you can do instead

The MPC here is an **augmentation of an already-stable loop**. The "plant" the
Koopman model describes is *quadrotor + LQR*. So ideally $\rho(A_{\text{phys}}) < 1$
on the physical modes, and $\nu = 0$ (do nothing, let LQR work) is always input-feasible.
That gives a clean choice:

- **$\kappa_f = 0$** and **$P$ from the discrete Lyapunov equation**
  $A_s^\top P A_s - P + C_s^\top Q C_s = 0$ on the stable subspace $A_s$ (drop or
  handle the constant mode separately). Use `scipy.linalg.solve_discrete_lyapunov`.
  Then $V_f$ is the exact infinite-horizon cost of "doing nothing afterwards", and
  (A3) holds with equality.
- **Terminal set:** with $\kappa_f = 0$, the input constraints hold trivially. For the
  state constraints, use a sublevel set of $P$ inside the box, or skip the terminal set
  and use soft state constraints (§5.6), and state honestly that you get stability
  but not hard feasibility guarantees.

Suggested ablation for the thesis: compare (a) the paper's $P = Q$, (b) the Lyapunov $P$
with $\kappa_f=0$, and (c) the DARE $P$ on a reduced stabilizable model, each with
$n_p \in \{3,5,10\}$. If (b) lets you shorten $n_p$ with the same performance,
that is a real result.

### 3.4 Why the paper works without them

There are three reasons. The inner LQR stabilizes the plant. $u_d$ is a small, bounded
perturbation. And $n_p = 5$ at 10 Hz is short enough that the model is still accurate.
This is "MPC as a feed-forward correction", not "MPC as the stabilizing controller".
Say this explicitly when you write up the comparison.

---

## 4. Choosing $n_p$ and $n_c$

### 4.1 Prediction horizon $n_p$

The horizon must satisfy three constraints:

1. **Long enough for the dynamics.** $n_p\Delta t$ should cover the dominant
   closed-loop time constant you want to influence. Compute it from the model:
   $\tau_i = -\Delta t / \ln|\lambda_i(A)|$. Ignore modes with $|\lambda| \approx 1$
   (constant) and modes with very small $|\lambda|$ (they are gone in one step).
2. **Short enough for the model.** A Koopman/EDMD predictor degrades with
   steps. Plot the **multi-step prediction RMSE vs $k$** on held-out data. You
   already validate at `horizon = 5` in `configs/identify/model1.toml`. Pick
   $n_p$ before the error curve takes off. Planning past the point where the model is
   accurate only adds noise.
3. **Fits the compute budget.** You have one MPC period ($100$ ms) minus the lifting
   and the loop overhead. With 20 decision variables this is far from tight in
   Python, but it will matter if you go embedded.

Also consider the **wind correlation time**. If the mean-wind assumption is only
valid for about 0.5–1 s, a longer horizon optimizes against a disturbance model that
is wrong.

Procedure: make $n_p$ a config parameter. Sweep $\{3, 5, 8, 10, 15\}$ over the
**same** Monte Carlo wind seeds, and plot RMS error and solve time vs $n_p$. Report
where the curve flattens.

### 4.2 Control horizon $n_c \le n_p$

Optimize only $\nu_0..\nu_{n_c-1}$ and fix the rest. Two common choices:

- $\nu_k = 0$ for $k \ge n_c$ ("hand back to LQR"). This fits the augmentation
  structure and the $\kappa_f=0$ terminal design.
- $\nu_k = \nu_{n_c-1}$ (move blocking, hold last). This is better against a
  persistent wind bias.

Implementation: $V = T\,V_c$ with a fixed selection matrix $T$. Substitute
it into $H$, $f$ and $G$ ($H_c = T^\top H T$, etc.). Nothing else changes. The paper uses
$n_c = n_p$. With 20 variables there is no computational reason to reduce
$n_c$, so treat it as a regularization knob, not a speed knob.

---

## 5. Numerical implementation caveats

These are ordered roughly by how likely they are to bite you in this repo.

### 5.1 Scaling consistency (most likely bug)

`src/koopman/preprocessing.py` standardizes the error state before lifting.
The model's $A, B, D, C$ then act on **standardized** coordinates, and possibly on
scaled inputs too. Every quantity entering the MPC must be in the same units:

- $x_0$: standardize **before** $\psi(\cdot)$, with the training mean and std (a
  frozen transform, never refit online).
- $Q$, $R$: if the paper's $Q$, $R$ are in physical units, transform them,
  e.g. $Q_s = S_x Q S_x$ with $S_x = \text{diag}(\sigma_x)$.
- $h_{lb}, h_{ub}$, $u_{lb}, u_{ub}$, $\bar v_w$, $\Sigma_{v_w}$: all scaled.
- $\nu_0^\*$: un-scale before adding to the LQR command.

Put this in **one** adapter object with a test that a round trip gives the identity.
The MPC core should not know scaling exists (dependency inversion: the core
depends on "a linear model in some coordinates", not on your preprocessing).

### 5.2 Conditioning of the condensed problem

- $\Gamma$ contains $A^{n_p-1}B$. If $\rho(A) > 1$ (EDMD can return slightly
  unstable spurious modes), these entries blow up and $H$ becomes ill-conditioned.
  **Check `max(abs(eigvals(A)))` at model load and log it.** At $n_p=5$ it is
  usually harmless, but at $n_p=20$ it is not.
- Log `np.linalg.cond(H)`. Above about $10^8$, expect solver tolerance problems.
- Symmetrize: `H = 0.5 * (H + H.T)` after construction. Round-off makes it
  slightly asymmetric, and some solvers reject that or silently use one triangle.
- Assert $R \succ 0$ (Cholesky succeeds). If $R$ is only PSD, $H$ can be singular.
- Compute powers by repeated multiplication (`Ak = A @ Ak`), not
  `matrix_power` in a loop from scratch. Never invert $A$.

### 5.3 Model structure checks

- **Constant monomial row.** Check that $A[j,:] \approx e_j^\top$ and $B[j,:] \approx 0$ for
  the constant feature. If l1-EDMD shrinks $A[j,j]$ away from 1, the
  "constant" decays in prediction, which biases every monomial term. Consider
  fixing that row analytically after the fit.
- **Unit test:** lifting then $C$ returns $x$. This already exists via the
  `ψ(x)[:n] == x` contract. Keep it.
- **Lifted prediction drift:** after one step, $z_1 \neq \psi(Cz_1)$ in general. The
  MPC predicts in lifted space without re-lifting, which is standard (Korda–Mezić).
  Re-lifting would make the problem nonlinear.

### 5.4 Solver choice (this needs a dependency decision)

The `koopman_mpc` env has only SciPy, and SciPy has **no dedicated QP solver**.
`scipy.optimize.minimize(method="SLSQP" | "trust-constr")` will run, but it is slow,
tolerance-sensitive and not built for real-time use. I don't recommend it beyond a first
sanity check. Options:

| Solver | Type | Fit here |
|---|---|---|
| OSQP | ADMM, sparse, warm-start | Robust, popular, C core. Moderate accuracy (tune `eps_abs`). |
| DAQP / qpOASES / quadprog | Active-set, dense | Best fit for a small dense QP (20 vars). Exact solutions, fast warm starts. |
| HPIPM / acados | Structure-exploiting IPM | Overkill now, relevant for C++/embedded later. |
| `qpsolvers` | Wrapper over many | Makes the solver choice a config parameter. |

Wrap the solver behind `solve_qp(H, f, G, w, lb, ub) -> (V, status)` so the
MPC never imports a solver directly (dependency inversion). Per CLAUDE.md,
adding OSQP or DAQP is a dependency change. **Decide it before implementation.**

### 5.5 Warm start and shifting

Initialize each solve with the shifted previous solution
$[\nu_1^\*, \dots, \nu_{n_p-1}^\*, \nu_{n_p-1}^\*]$ (or $0$ appended). Active-set
solvers also benefit from the previous active set. Without warm starting, ADMM
solvers can need many more iterations.

### 5.6 Infeasibility (no silent failure)

Hard state constraints plus a disturbance can make the QP infeasible. With K-SMPC this
happens whenever $\gamma_k$ approaches $(h_{ub}-h_{lb})/2$. Handle it explicitly:

- **Soft state constraints:** add a slack $s \ge 0$:
  $h_{lb} - s \le Ez_k \le h_{ub} + s$, with cost $\rho\|s\|_1 + \rho_2\|s\|^2$.
  An exact-penalty $\rho$ (larger than the largest Lagrange multiplier) means the
  slack stays zero whenever the hard problem is feasible.
- **Always check solver status.** If it is not solved, fall back to $\nu_0 = 0$ (pure LQR),
  **log it with the timestamp**, and count fallbacks in the metrics. A plot
  that hides fallbacks is not a valid MPC result.
- Check offline that $\gamma_k < (h_{ub}-h_{lb})/2$ for all $k$, and raise at
  construction if not.
- Input bounds apply to the **perturbation** $u_d$, not to the total command
  $u_{LQR}+u_d$. The total can still saturate the rotors. Decide whether the MPC
  should know the LQR's current command (time-varying bounds $u_{lb}-u_{LQR}$).

### 5.7 Timing, indexing and sample-rate mismatches

- **Off-by-one errors** are the most common MPC bug. Write down once which index the
  cost, the constraint and the terminal term apply to, and test it on a 1-D system
  with $n_p=1$ by hand.
- **Multi-rate:** MPC at 10 Hz, LQR at 100 Hz, sim at 1 kHz. The Koopman model
  was identified with $u_d$ held for 0.1 s (ZOH), so the MPC must hold it the
  same way. Any mismatch breaks the model.
- **Computation delay:** in simulation the solve is instantaneous. On hardware,
  $\nu_0$ arrives up to 100 ms late. The standard fix is to predict $z_1$ with the
  previous input and optimize from there.
- **Mean wind $\bar v_w$:** the paper assumes it is known and constant. In practice it
  is estimated, and a biased $\bar v_w$ becomes a steady tracking offset. This is
  the entry point to offset-free MPC (disturbance observer).

### 5.8 Solver tolerances vs problem scale

Solver tolerances are absolute. If the standardized $\nu$ is around $10^{-2}$ and
`eps_abs = 1e-3`, the solution is mostly noise. Scale the problem (§5.1) so the
variables are $\mathcal O(1)$, and verify constraint satisfaction *a posteriori*
on the returned $V$ with your own tolerance.

### 5.9 Tests to write (each one can fail for the reason it tests)

1. Condensed prediction $\Phi z_0 + \Gamma V + \Lambda\bar v_w$ equals the loop
   rollout of `KoopmanModel.predict` for random $z_0, V$.
2. Unconstrained solution equals $-H^{-1}f$, and with a DARE terminal $P$ on a
   stabilizable toy system, $\nu_0 = -K_{\text{LQR}}z_0$.
3. Very loose bounds give the same answer as the unconstrained problem. Tight input bounds give an
   output exactly on the bound.
4. Zero error and zero wind give $\nu = 0$.
5. Tightening: $\gamma_0 = 0$, $\gamma_k$ is non-decreasing in $k$ when $A$ is stable,
   $\gamma$ scales by $\sqrt{c}$ when $\Sigma_{v_w}$ scales by $c$, and $\Sigma_{\zeta,k}$ stays
   symmetric PSD.
6. Infeasible bounds are reported as infeasible (no silent wrong answer), and the fallback is logged.

---

## 6. Proposed implementation plan for this repo (for your go-ahead)

No code has been written yet. This is the order I'd propose, one small PR each:

1. **Dependency decision:** add a QP solver (my recommendation: DAQP or OSQP,
   behind `qpsolvers` or a thin wrapper).
2. `src/controllers/qp.py`: `condense(A, B, D, C, Q, R, P, E, horizon)` as a pure
   function returning a frozen dataclass of $H, \Phi_f, \Lambda_f, G, \dots$, and
   `solve_qp(...)`. Add `tests/test_qp.py` with tests 1–4 above.
3. `src/controllers/kmpc.py`: `KoopmanMPC` holds the condensed matrices and
   the lifting/scaling adapter. `__call__(error_state) -> u_d` with fallback and logging.
4. `src/controllers/ksmpc.py`: `tightening()` (pure) plus a subclass or
   composition that swaps the bounds. Test 5.
5. `configs/controllers/kmpc.toml`: $Q$, $R$, $n_p$, $n_c$, bounds, $\beta$,
   terminal mode (`paper` | `lyapunov`), soft-constraint weight, solver name.
6. Hook into `src/simulation/closed_loop.py` at `dt_augment`, then Monte Carlo
   comparison.

Open interface question to settle first: should the MPC receive the raw error
state and own the lifting and scaling, or receive $\psi_0$ already lifted? I
lean toward the first, through an injected adapter, so the closed-loop sim
only ever deals in physical units.

---

## 7. After the MPC vs LQR comparison: towards ship landing

### 7.1 Make the comparison credible first

- Use the **same wind seeds** for every controller (paired comparison), report mean ±
  CI over Monte Carlo runs, and use a paired test, not just averages.
- Metrics: RMS and max tracking error per axis, constraint violation rate,
  $\|u_d\|$ effort, QP solve time (mean / p99), infeasibility and fallback count.
- Include ablations: Model 1 vs 2, K-MPC vs K-SMPC, $n_p$ sweep, terminal choice.
  The interesting result is *why* MPC helps (preview of the mean wind? constraint
  handling?), not only *that* it helps.

### 7.2 What changes for ship landing, and why MPC (not LQR) is the right tool

Ship landing has the features that make MPC worth its cost: **preview,
constraints and a terminal condition**. LQR handles none of them natively.

1. **Reference frame and error state.** The target is a moving deck. Define the
   error in the **ship (deck) frame**: relative position, velocity and attitude to the
   pad. Document the frames (world NED vs ship body, and how the deck's roll, pitch and heave
   enter). The Koopman model must then be identified on relative-frame errors, or
   you need to show that frame changes preserve the lifted linearity.
2. **Time-varying reference with preview.** Deck motion is predictable over a
   few seconds (wave-induced 6-DOF motion; ship motion from RAOs with a
   JONSWAP or Pierson–Moskowitz sea spectrum). Feed the predicted deck
   trajectory into the MPC as a reference sequence. Only the $f$ vector changes,
   so the QP structure is identical. This is the biggest structural advantage over LQR.
3. **Terminal ingredients become physical.** Touchdown constraints include relative vertical
   speed below a limit, attitude matched to the deck, position inside the pad footprint,
   and rotor clearance. These form a **terminal set at touchdown**. Look at
   *shrinking-horizon* or *time-to-go* MPC, where the horizon ends at the landing
   instant, and at landing in the "quiescent period" of deck motion.
4. **Ship airwake replaces open-ocean turbulence.** The wake behind the
   superstructure is non-stationary, spatially varying and strongly sheared (see the
   SFS2 "Simple Frigate Shape" benchmark used in airwake CFD). The constant
   $\bar v_w$ assumption breaks. Options:
   - an **online disturbance estimate** (an augmented-state Kalman filter on the lifted
     model, i.e. offset-free MPC);
   - a **position-dependent wind map** $\bar v_w(p_{\text{rel}})$ from CFD/LES,
     linearized along the planned approach;
   - **phase-scheduled Koopman models** (approach, station-keeping over the deck,
     descent), each fitted separately, with switching or blending.
5. **Uncertainty in two places.** Wind *and* deck-motion prediction error both go
   into the chance constraints. K-SMPC already has the structure; add the deck
   prediction covariance to $\Sigma_\zeta$. Temporal correlation matters even more here.
6. **Safety layer.** Consider tube MPC (robust bounds instead of probabilistic
   ones) for the final metres, or a control barrier function filter on top. In
   ship landing, a single violation means a crash, so $\beta = 0.99$ per step may not be enough.
7. **Online model adaptation.** Recursive (windowed) EDMD to track airwake changes
   with ship speed and heading. This is a nice thesis chapter, but keep it separate
   from the controller (single responsibility: identification vs control).

### 7.3 Suggested research sequence

1. K-MPC with deck-motion **preview** in calm air (isolates the preview benefit).
2. Add the **airwake** with constant-mean K-MPC, then with an online disturbance estimate.
3. Add **terminal touchdown constraints** with shrinking horizon.
4. K-SMPC with **combined wind and deck uncertainty**. Monte Carlo over sea states and
   wind-over-deck angles.
5. Then hardware-oriented work: C++ port of the QP (qpOASES/OSQP/HPIPM), timing,
   and delay compensation.

Keep the algorithm core free of simulator and ROS IO throughout. The same
`KoopmanMPC` class should run in the Python Monte Carlo and behind a ROS node.

---

## 8. What to practice or understand next

- Derive §2.2 by hand for $n_p=2$, then check it numerically against
  `KoopmanModel.predict` rollouts. This builds a habit: every derived matrix gets
  a rollout test.
- Prove recursive feasibility and stability under (A1)–(A3) on paper. It is half a
  page, and it is the core MPC idea that reviewers will ask about.
- Compute the eigenvalues of your fitted $A$ and the multi-step RMSE curve. Use them to
  justify $n_p$ instead of copying the paper's 5.
- Learn the KKT conditions of a QP. They explain warm starting, active sets, why the
  exact-penalty slack works, and what the solver's dual variables tell you.
