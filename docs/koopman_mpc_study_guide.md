# Koopman with Control and MPC: A Study Guide for the Quadrotor Project

This guide teaches the theory you need to build and analyse the project in its new form:
**MPC as the only outer-loop controller, on a Koopman model of the open-loop quadrotor**,
with a proper study of terminal cost and terminal constraints.

It is built on the four course documents in [`learning_materials/`](../learning_materials):

| Short name | File | What it covers |
|---|---|---|
| **[LN-MPC]** | `Lecture_Notes_Control_Methods.pdf` | Stabilisation, LQR, MPC, terminal ingredients, proofs |
| **[LN-DDC]** | `Data_Driven_Control.pdf` | DMDc, Koopman with control, SINDYc, DeePC, the "two questions" |
| **[EX-DMDc]** | `DMDc_Example.pdf` | Closed-loop data gives a wrong model with zero residual |
| **[EX-KoopC]** | `KoopmanC_Notes.pdf` | When the linear lifted predictor is exact, bilinear, or neither |

Where the guide goes beyond those notes, it says so and cites a source (§13).
[`docs/mpc_guide.md`](mpc_guide.md) remains the implementation guide for the paper's
LQR-augmented controller; this guide is about understanding and about the new design.

**Status of the numbers in this guide.** Three kinds appear, and they are labelled:

- *Measured on your model*: computed from `results/runs/20261113T155255Z_edmd_model1`.
- *Computed for this guide*: small calculations on the linearised hover model or toy
  systems. You should reproduce them; they are your first unit tests.
- *Predictions*: what the theory says you should see in experiments not yet run.

---

## 0. Map of the subject

The project has one chain of reasoning. Every section below is one link.

1. A linear model $z^+ = Az + Bu$ makes MPC a **convex QP** (§3).
2. A QP solved repeatedly is a feedback law. It is stable and stays feasible only if the
   finite horizon is closed off with a **terminal cost and terminal set** (§4, §5).
3. A quadrotor is nonlinear. **Koopman lifting** replaces the nonlinear state map by a
   linear map on functions of the state (§6), which EDMD approximates from data (§7).
4. Adding an input raises **two separate questions**: can the data separate drift from
   actuation (§8.2), and can a constant $B$ represent the actuation at all (§8.3)?
5. Putting the lifted model inside MPC keeps the QP, but the guarantees of step 2 are
   about the *model*. What survives for the *plant* must be argued and tested (§9).
6. The quadrotor tells you where each issue bites (§10), and the experiments (§11) are
   designed so that each one answers one question.

### 0.1 Notation

The course notes, the paper and your code use different symbols. This guide uses the
course notation.

| Quantity | This guide / course | Paper and code |
|---|---|---|
| Physical state, dimension | $x \in \mathbb{R}^n$ | $e_x$ (error state), $n = 9$ |
| Input, dimension | $u \in \mathbb{R}^l$ | $u_d$, $\nu$; `m` in code |
| Dictionary (lifting) | $\theta(x)$ | $\psi(x)$, `MonomialLifting` |
| Lifted state, dimension | $z = \theta(x) \in \mathbb{R}^p$ | $\psi$; `N = 55` in code |
| Lifted model | $z^+ = Az + Bu + Dw$, $x = Cz$ | `KoopmanModel(A, B, D, C)` |
| Snapshot matrices | $\Theta(X)$, $\Theta(X')$, $\Upsilon$ | $\Gamma_1$, $\Gamma_2$ in `edmd.py` |
| Number of snapshots | $m$ | `K` in `edmd.py` |
| Prediction horizon | $N$ | $n_p$, `horizon` |
| Terminal weight, terminal set | $Q_f$, $\mathbb{X}_f$ | `P` in `qp.condense` |
| Feedback law | $u = Kx$, $K = -(R + B^\top PB)^{-1}B^\top PA$ | same sign: `u = u_ref + K (x − x_ref)` in `lqr.py` |
| Sample time | $\Delta t$ | `dt_augment = 0.1` s |

Two traps: the course writes $u = Kx$ with closed loop $A + BK$ in [LN-MPC] but
$u = -K_f x$ in [LN-DDC]; and $N$ is the horizon here but the lifted dimension in your code.

---

# Part I. Linear foundations

## 1. Stability, controllability and the tests you will actually run

### 1.1 Stability of $x_{k+1} = Ax_k$

- **Schur stable**: every eigenvalue satisfies $|\lambda_i| < 1$. Then $x_k \to 0$.
- **Marginal**: some $|\lambda_i| = 1$. Bounded only if those eigenvalues have no Jordan
  block. A double integrator has $\lambda = 1$ twice in one Jordan block and grows linearly.
- **Lyapunov characterisation** [LN-MPC §1.2]: $A$ is Schur if and only if for every
  $Q \succ 0$ there is $P \succ 0$ with $A^\top PA - P = -Q$. Then $V(x) = x^\top Px$
  decreases along trajectories by exactly $x^\top Qx$ per step.

The Lyapunov view matters more than the eigenvalue view, because MPC stability proofs are
Lyapunov proofs with the optimal cost as $V$.

**Eigenvalues do not describe the transient.** If $A$ is non-normal, $\|A^k\|$ can grow
before it decays. *Measured on your model:* spectral radius 0.974 on the non-constant
part, yet $\|A^5\|_2 = 1.9$. Over an MPC horizon of 5 steps the model can amplify.

### 1.2 From discrete eigenvalue to physical meaning

With sample time $\Delta t$, a discrete eigenvalue $\lambda$ corresponds to a
continuous-time rate $s = \ln(\lambda)/\Delta t$:

- time constant $\tau = -1/\operatorname{Re}(s) = -\Delta t / \ln|\lambda|$,
- oscillation frequency $f = |\operatorname{Im}(s)|/2\pi = |\arg\lambda| / (2\pi\Delta t)$.

Example: $\lambda = 0.974$ at $\Delta t = 0.1$ s gives $\tau = 3.8$ s.
Anything with $|\arg\lambda|$ near $\pi$ is at the Nyquist frequency and is usually an artefact.

### 1.3 Controllability, stabilisability, and the PBH test

- $(A, B)$ is **controllable** if the input can move the state anywhere:
  $\operatorname{rank}[B\ AB \cdots A^{n-1}B] = n$.
- $(A, B)$ is **stabilisable** if every mode the input cannot move is already stable.
  This is the assumption behind LQR [LN-MPC Assumption 1.1].
- **PBH test, per mode**: let $w_i^\top A = \lambda_i w_i^\top$ be a left eigenvector.
  Mode $i$ is uncontrollable if and only if $w_i^\top B = 0$.

The PBH test is the one to use on a lifted model. A rank test on a $55 \times 220$
controllability matrix is numerically meaningless; $\|w_i^\top B\|$ per mode tells you
*which* mode is unreachable and how badly.

*Measured on your model:* the mode at $\lambda = 1$ (the constant feature) has
$\|w^\top B\| \approx 10^{-16}$. It is uncontrollable and not stable, so the lifted pair is
not stabilisable, and `solve_discrete_are` fails on it.

- **Detectability** of $(Q^{1/2}, A)$: every mode the cost cannot see is stable. LQR needs
  it for $P$ to be unique and stabilising [LN-MPC "Important assumption"]. In a lifted
  model the cost is $z^\top C^\top QCz$, which sees only the physical states, so lifted
  modes that do not feed back into $x$ are invisible. They must be stable.

## 2. LQR: the infinite-horizon answer that MPC borrows

### 2.1 The result

For $\min \sum_{k=0}^\infty x_k^\top Qx_k + u_k^\top Ru_k$ with $Q \succeq 0$, $R \succ 0$
[LN-MPC §2.2]:

$$P = A^\top PA + Q - A^\top PB(R + B^\top PB)^{-1}B^\top PA \quad\text{(DARE)},$$
$$u_k = Kx_k,\qquad K = -(R + B^\top PB)^{-1}B^\top PA,\qquad \text{optimal cost} = x_0^\top Px_0.$$

### 2.2 How it is derived, and why the derivation matters

The notes derive it by **completing the square**: add and subtract the stage cost in
$x_{k+1}^\top P_{k+1}x_{k+1} - x_k^\top P_kx_k$, choose $P_k$ to cancel the state-only
bracket, and what is left is $\|u_k - Kx_k\|^2$ in a weighted norm, which is zero for the
optimal input. Telescoping the sum gives the cost $x_0^\top P_0x_0$.

Two things to keep from this:

- **$P$ is a cost-to-go.** $x^\top Px$ is exactly the cost of running the LQR law forever
  from $x$. This sentence is the whole reason $Q_f = P$ works in MPC.
- **The finite-horizon version is a backward recursion** (DRDE, [LN-MPC (2.7)]) started at
  $P_N = Q_f$. So *unconstrained* MPC with horizon $N$ is a linear feedback $u = K_Nx$
  where $K_N$ comes from running that recursion $N$ times. You can therefore study
  unconstrained MPC stability without solving a single QP (§4.4 does this).

### 2.3 The identity that makes MPC proofs work

With $P$, $K$ from the DARE [LN-MPC Prop. 3.2]:

$$(A + BK)^\top P(A + BK) - P = -(Q + K^\top RK).$$

Read it as: under the LQR law, the cost-to-go drops by exactly the stage cost each step.
This is a Lyapunov equation for the closed loop, and it is the "terminal decrease
condition" used in §5.

---

# Part II. Model predictive control

## 3. From LQR to a quadratic program

### 3.1 Why MPC exists

LQR ignores constraints. Saturating the LQR input is not optimal and has no guarantee
[LN-MPC §3]. The constrained infinite-horizon problem has no Riccati solution, so MPC
truncates it to $N$ steps, applies the first input, and re-solves.

### 3.2 The finite-horizon problem

$$\begin{aligned}
V_N(x_k) = \min_{u_{k|k},\dots,u_{k+N-1|k}}\ & \sum_{i=0}^{N-1}\left[x_{k+i|k}^\top Qx_{k+i|k} + u_{k+i|k}^\top Ru_{k+i|k}\right] + x_{k+N|k}^\top Q_fx_{k+N|k} \\
\text{s.t. } & x_{k+i+1|k} = Ax_{k+i|k} + Bu_{k+i|k},\quad x_{k|k} = x_k, \\
& x_{k+i|k} \in \mathbb{X},\ u_{k+i|k} \in \mathbb{U},\quad i = 0,\dots,N-1, \\
& x_{k+N|k} \in \mathbb{X}_f.
\end{aligned}$$

$x_{k+i|k}$ is a *prediction made at time $k$*, not the state the plant will have.
Keeping this distinction clear is half of understanding MPC.

### 3.3 Condensing into a QP

Because the model is linear, predictions are affine in the input sequence [LN-MPC §3.3]:

$$X_k = \Phi x_k + \Gamma U_k,$$

with $\Phi$ stacking powers of $A$ and $\Gamma$ the block lower-triangular matrix of
$A^{i-1-j}B$. Substituting:

$$J_k = \tfrac12 U_k^\top HU_k + f(x_k)^\top U_k + \text{const},\qquad
H = 2(\Gamma^\top\bar Q\Gamma + \bar R),\quad f(x_k) = 2\Gamma^\top\bar Q\Phi x_k.$$

- $H \succ 0$ because $R \succ 0$, so the QP is **strictly convex** with a unique solution.
- $H$ and the constraint matrix are computed **once, offline**. Only $f(x_k)$ and the
  constraint right-hand side change online.
- All constraints become $GU_k \le g - Sx_k$.

Your `qp.condense` does this. Its `P` argument is $Q_f$.

**Numerical caveat (beyond the notes).** Condensing multiplies by powers of $A$. For an
unstable $A$ and long $N$, $\Gamma$ becomes badly conditioned. With open-loop integrators
($\lambda = 1$) and $N \le 30$ this is mild. If you ever see solver trouble, the fix is
*pre-stabilisation*: optimise over $c_i$ in $u_i = Kx_i + c_i$, so predictions use the
stable $A + BK$.

## 4. Why a terminal cost is needed

### 4.1 The scalar example [LN-MPC §3.1]

System $x^+ = 1.5x + u$, $|u| \le 1$, $|x| \le 3$, $Q = 1$, $R = 2$, $N = 1$.
The DARE gives $P = 4$, $K = -1$.

- $Q_f = 0$, $x_0 = 1.5$: the cost is $x_0^2 + 2u_0^2$. Its minimiser is $u_0 = 0$, so
  $x_1 = 2.25$. The controller watches the state grow, because nothing in the cost
  mentions $x_1$.
- $Q_f = P = 4$: the minimiser is $u_0 = \operatorname{sat}(-x_0) = -1$, so $x_1 = 1.25$.

Same plant, same horizon. The terminal cost changed the answer because it prices
everything the horizon cuts off.

### 4.2 The principle

$Q_f$ is an estimate of the cost-to-go beyond the horizon.

- $Q_f = 0$ or $Q_f = Q$ says "the future after $N$ is free or costs one more stage".
  For an unstable plant this is badly wrong, because the tail cost is large.
- $Q_f = P$ says "after $N$ steps, the LQR law takes over, and here is its exact cost".
  For an unconstrained problem this makes MPC *identical* to LQR for every $N \ge 1$.

### 4.3 Stable plants are different

If $A$ is Schur and $u = 0$ is admissible, the tail can be closed with the policy "do
nothing": $Q_f$ solves the Lyapunov equation $A^\top Q_fA - Q_f = -Q$. The terminal
controller is $K_f = 0$. With input constraints only, the terminal set can be the whole
space (Rawlings and Muske 1993).

This is the situation of your current LQR-augmented setup, and it is why the paper works
without terminal ingredients. It stops being true the moment the LQR is removed.

### 4.4 What this predicts for the quadrotor

*Computed for this guide.* Linearised hover model of §10.2 (9 states, ideal rate
tracking), exact discretisation at $\Delta t = 0.1$ s, $Q = \operatorname{diag}(100,100,100,10,10,10,1,1,1)$,
$R = \operatorname{diag}(1,5,5,0.1)$, **no constraints**. The table gives the closed-loop
spectral radius $\rho(A + BK_N)$ of the MPC feedback, obtained from $N$ steps of the
Riccati recursion.

| $N$ | $Q_f = 0$ | $Q_f = Q$ | $Q_f = P$ (DARE) |
|---:|---:|---:|---:|
| 1 | 1.0000 | **1.0119** | 0.9563 |
| 2 | **1.0119** | **1.0037** | 0.9563 |
| 3 | **1.0037** | 0.9941 | 0.9563 |
| 5 | 0.9921 | 0.9902 | 0.9563 |
| 10 | 0.9831 | 0.9814 | 0.9563 |
| 20 | 0.9696 | 0.9686 | 0.9563 |
| 40 | 0.9588 | 0.9586 | 0.9563 |

How to read it:

- **Bold entries are unstable.** With $Q_f = Q$ and $N \le 2$ the loop diverges, with a
  doubling time of about 6 s for $\rho = 1.0119$.
- With the paper's choice ($Q_f = Q$, $N = 5$) the loop is stable but slow:
  $\rho = 0.990$ is a time constant of 10 s, against 2.2 s for LQR ($\rho = 0.956$).
- The $Q_f = 0$ column is the $Q_f = Q$ column shifted by one row, because one Riccati
  step from zero gives $Q$.
- $Q_f = P$ gives the LQR closed loop at every horizon.
- Convergence in $N$ is slow: you need $N \approx 40$ (4 s of prediction) to match what
  the terminal cost gives at $N = 1$.

This is the cleanest result available to you: **a terminal cost is worth about 35 steps
of horizon on this vehicle.** Reproducing this table is experiment E1 (§11).

## 5. Why a terminal set is needed, and how to prove everything

### 5.1 A cost is a preference, a constraint is a restriction

Continue the scalar example with $Q_f = 4$, $x_0 = 2.5$ [LN-MPC §3.1]:

- Step 1: $u_0 = -1$, $x_1 = 2.75$. Legal.
- Step 2: the best legal input is $u_1 = -1$, giving $x_2 = 3.125 > 3$. The state
  constraint is violated, and no input could have prevented it.

The terminal cost made the controller try. It could not make the problem solvable.
From $x_0 = 2.5$ the constraint violation was already unavoidable.

Add $\mathbb{X}_f = [-1, 1]$ as a hard constraint on the last predicted state. Now the QP
at $x_0 = 2.5$ is **infeasible at the first solve**. That is the correct answer: it
reports, two steps early, that this state cannot be controlled within the limits.

### 5.2 The three terminal ingredients

1. Terminal cost $V_f(x) = x^\top Q_fx$.
2. Terminal set $\mathbb{X}_f \subseteq \mathbb{X}$.
3. Terminal controller $u = K_fx$. It is **never applied**. It exists only so that the
   proofs have an input to append.

They must satisfy, for every $x \in \mathbb{X}_f$ [LN-MPC Assumption 3.1, Prop. 3.2]:

- **(A1) admissibility**: $K_fx \in \mathbb{U}$;
- **(A2) invariance**: $(A + BK_f)x \in \mathbb{X}_f$;
- **(A3) decrease**: $V_f((A + BK_f)x) - V_f(x) \le -\left(x^\top Qx + x^\top K_f^\top RK_fx\right)$.

$K_f = K$ and $Q_f = P$ from the DARE satisfy (A3) with equality (§2.3). (A1) and (A2)
are what the *set* must be built to satisfy (§5.6).

### 5.3 Recursive feasibility: shift and pad

**Claim** [LN-MPC Prop. 3.3]: if the QP is feasible at time $k$, it is feasible at $k+1$.

**Proof idea.** You do not need to solve tomorrow's QP, only to exhibit one feasible plan
for it. Take today's optimal plan, drop its first move, and append the terminal controller
at the end:

$$\tilde U_{k+1} = \left(u^\star_{k+1|k},\ \dots,\ u^\star_{k+N-1|k},\ K_fz\right),\qquad z = x^\star_{k+N|k} \in \mathbb{X}_f.$$

- The reused moves were feasible yesterday, and (in the nominal case) the plant is exactly
  where the plan said it would be.
- The appended move is legal by (A1), and it lands in $\mathbb{X}_f$ by (A2).

So the terminal constraint on a state that is never reached is a **certificate for the
next solve**. This is the answer to "why constrain a prediction I will throw away?".

### 5.4 Stability: the optimal cost as a Lyapunov function

**Claim** [LN-MPC Prop. 3.4]: $V_N(x_{k+1}) - V_N(x_k) \le -(x_k^\top Qx_k + u_k^\top Ru_k)$.

**Proof idea.** Cost out the shifted plan $\tilde U_{k+1}$:

- it keeps stages $1,\dots,N-1$ of the old plan,
- it adds one new stage cost at $z$ and a new terminal cost at $(A + BK_f)z$,
- by (A3), those two new terms sum to at most the old terminal cost $z^\top Q_fz$.

So $J(\tilde U_{k+1}) \le V_N(x_k) - \text{(first stage cost)}$. The true optimum at $k+1$
can only be lower. Summing over time, the stage costs are summable, so they go to zero,
and with $Q \succ 0$ so does $x_k$.

The three roles are now clear:

| Ingredient | Used in feasibility proof | Used in stability proof |
|---|---|---|
| $K_f$ | supplies the appended move | supplies the appended move |
| $\mathbb{X}_f$ | makes the appended move legal and the end point valid | guarantees (A3) applies at $z$ |
| $Q_f$ | not used | cancels the two new cost terms |

### 5.5 What the theorem does and does not say

It says: **nominal** model (plant equals model, no disturbance), initial state in the
feasible set $\mathbb{X}_N$, then feasibility forever and convergence to the origin.

It does not say:

- anything about a plant that differs from the model (§9.3),
- anything about disturbances such as wind,
- that the closed loop is stable from *every* state. The **region of attraction is
  $\mathbb{X}_N$**, the set of states from which an $N$-step plan into $\mathbb{X}_f$ exists
  [LN-MPC Def. 3.1].

$\mathbb{X}_N$ grows with $N$ and with the size of $\mathbb{X}_f$. This is the real
trade-off of a terminal constraint:

> A small terminal set with a short horizon gives a small region of attraction. You pay
> for the guarantee with either horizon length or a larger, better-designed set.

### 5.6 How to design the terminal set

Three constructions, in increasing quality.

**(a) Terminal equality, $\mathbb{X}_f = \{0\}$.** Trivially invariant. Requires reaching
the origin in $N$ steps, so $\mathbb{X}_N$ is tiny for short horizons. Useful only as a
baseline.

**(b) Ellipsoid, $\mathbb{X}_f = \{x : x^\top Px \le \alpha\}$.**
Invariant under $A + BK$ for any $\alpha$, because $x^\top Px$ decreases (§2.3). Choose
the largest $\alpha$ for which every constraint holds inside. For a constraint
$|c^\top x| \le b$, the maximum of $c^\top x$ over the ellipsoid is $\sqrt{\alpha\, c^\top P^{-1}c}$, so

$$\alpha = \min_i \frac{b_i^2}{c_i^\top P^{-1}c_i},$$

taken over state constraints ($c_i$ = rows of the constraint matrix) *and* input
constraints ($c_i$ = rows of $K$). Cost: $x_N^\top Px_N \le \alpha$ is a quadratic
constraint, so the QP becomes a QCQP.

*Computed for this guide*, hover model, limits $|\phi|,|\theta| \le 0.5$ rad,
$|\omega_i| \le 2$ rad/s, $|\delta c| \le 7.81$ m/s²: $\alpha = 6.35$, limited by the roll-rate
command. The resulting set allows a position error of only **0.125 m** and a velocity
error of 0.47 m/s. With $N = 5$ you could only start from states that reach that small
ball in 0.5 s. The ellipsoid is easy but very conservative here, because $P$ weights
position heavily.

**(c) Maximal admissible set $\mathcal{O}_\infty$ (Gilbert and Tan 1991; beyond the notes).**
The largest set satisfying (A1) and (A2):

$$\mathcal{O}_\infty = \{x : H_c(A + BK)^tx \le h_c\ \text{for all } t \ge 0\},$$

where $H_cx \le h_c$ collects the state constraints and the input constraints written as
$H_uKx \le h_u$. Algorithm:

1. Start with the constraints at $t = 0$.
2. Add the rows $H_c(A + BK)^{t+1}x \le h_c$.
3. For each new row, solve an LP: maximise the row over the current set. If no new row can
   be violated, stop. Otherwise set $t \leftarrow t + 1$ and repeat.

It terminates in finitely many steps when $A + BK$ is Schur and the constraint set is
bounded. The result is a **polytope**, so the terminal constraint is linear and the
problem stays a QP. In 9 dimensions this is practical with `scipy.optimize.linprog`. In 55
dimensions it is not, which shapes the design in §9.4.

### 5.7 Alternatives to a terminal set

- **Long horizon, no terminal constraint.** For large enough $N$ the optimal plan ends
  inside $\mathbb{X}_f$ anyway, so the constraint is inactive and can be dropped
  (Limon et al. 2006; Grüne and Pannek). You keep stability on a sublevel set of $V_N$.
  The table in §4.4 shows how large $N$ must be here.
- **Soft constraints.** Replace $x \in \mathbb{X}$ by $x \in \mathbb{X} \oplus \epsilon$ with a
  penalty on $\epsilon$. The QP is always feasible, at the price of permitted violations.
  This is the standard practical choice when a disturbance can push the state out.

### 5.8 Three topics the notes leave out, which the quadrotor needs

**Tracking and steady-state offset.** The theory regulates to the origin. With a constant
wind force the origin is not an equilibrium for $u = 0$. You need a target $(x_s, u_s)$
with $x_s = Ax_s + Bu_s + Dw$ and to regulate the deviation from it; all terminal
ingredients are centred at $x_s$. If the wind estimate is wrong you get a steady offset,
and the cure is a disturbance estimator (offset-free MPC).

**Inherent robustness.** Nominal MPC with a continuous value function tolerates small
disturbances and model errors: the state converges to a neighbourhood of the origin whose
size scales with the error (input-to-state stability). Hard state constraints can destroy
this, because a small disturbance can make the QP infeasible. This is a second reason to
soften state constraints and keep input constraints hard.

**Constraint tightening and tubes.** To *guarantee* constraint satisfaction under bounded
disturbances, plan with constraints shrunk by the reachable set of the error (tube MPC).
For Gaussian wind there is no bounded set, so guarantees become probabilistic (chance
constraints, as in the paper's K-SMPC).

---

# Part III. Koopman operator theory

## 6. The operator

### 6.1 Definition

For an autonomous system $x_{k+1} = F(x_k)$, an **observable** is any scalar function
$g(x)$. The Koopman operator advances observables instead of states:

$$(\mathcal{K}g)(x) = g(F(x)).$$

- $\mathcal{K}$ is **linear**: $\mathcal{K}(ag + bh) = a\,\mathcal{K}g + b\,\mathcal{K}h$, for any $F$.
- $\mathcal{K}$ is **infinite-dimensional**: it acts on a function space.

Nothing has been approximated yet. Nonlinearity has been traded for dimension.

### 6.2 Eigenfunctions

$\varphi$ is an eigenfunction with eigenvalue $\lambda$ if $\varphi(F(x)) = \lambda\varphi(x)$.
Along any trajectory, $\varphi(x_k) = \lambda^k\varphi(x_0)$. An eigenfunction is a
coordinate in which the nonlinear system evolves exactly like a scalar linear one.

Properties you should be able to derive:

- **Linear systems.** If $F(x) = Ax$ and $w^\top A = \lambda w^\top$, then
  $\varphi(x) = w^\top x$ is an eigenfunction. *Left* eigenvectors give eigenfunctions.
- **Products.** If $\varphi_1, \varphi_2$ are eigenfunctions with $\lambda_1, \lambda_2$, then
  $\varphi_1\varphi_2$ is one with $\lambda_1\lambda_2$, because
  $(\varphi_1\varphi_2)(F(x)) = \varphi_1(F(x))\,\varphi_2(F(x))$.
- **Constant.** $\varphi \equiv 1$ is always an eigenfunction with $\lambda = 1$.
- **Level sets.** $\{\varphi = 0\}$ is invariant. For $|\lambda| < 1$ eigenfunctions, the
  joint zero set contains the attractor. For $\lambda = 1$, $\varphi$ is a conserved quantity.

**Consequence for your dictionary.** For a linear system with eigenvalues
$\lambda_1,\dots,\lambda_n$, the degree-2 monomial dictionary spans an *exactly invariant*
subspace with spectrum

$$\{1\}\ \cup\ \{\lambda_i\}\ \cup\ \{\lambda_i\lambda_j\}_{i \le j},$$

which is $1 + 9 + 45 = 55$ values for $n = 9$. *Measured on your model:* the slowest linear
eigenvalue is 0.9746; its square is 0.9498; the model has an eigenvalue 0.9506 whose
eigenfunction is dominated by yaw². So the lifted spectrum is mostly the *product lattice*
of the linear spectrum. It contains little independent information, and it tells you the
identified dynamics are close to linear.

### 6.3 Koopman mode decomposition

If the state can be written in eigenfunctions, $x = \sum_i v_i\varphi_i(x)$, then

$$x_k = \sum_i v_i\,\lambda_i^k\,\varphi_i(x_0).$$

- $\lambda_i$: how each pattern evolves in time (decay, oscillation).
- $\varphi_i(x_0)$: how strongly the initial condition excites it.
- $v_i$ (**Koopman mode**): how it appears in the state.

For a matrix approximation $A$ with $x = Cz$: $\varphi_i(x) = w_i^\top\theta(x)$ from left
eigenvectors, and $v_i = Cr_i$ from right eigenvectors $r_i$, normalised so $w_i^\top r_j = \delta_{ij}$.

### 6.4 Invariant subspaces and closure

A finite matrix can represent $\mathcal{K}$ exactly only on a **finite-dimensional invariant
subspace**: a span of observables that $\mathcal{K}$ maps into itself. This is "closure".

*Closes* [EX-KoopC Step 1, LN-DDC Example 3]:
$\dot x_1 = \mu x_1$, $\dot x_2 = \lambda(x_2 - x_1^2)$ with $\theta = (x_1, x_2, x_1^2)$.
Differentiating, $\frac{d}{dt}x_1^2 = 2\mu x_1^2$, and nothing outside the span appears.

*Never closes* [EX-KoopC Step 3]: forced Van der Pol. The term $-10x_1^2x_2$ raises the
degree by two each time it is differentiated. Truncating at degree $d$ leaves degree
$d + 2$ terms unaccounted for. The residual has a **floor** that no amount of data lowers.

The lesson the course draws:

> A residual at machine precision certifies exactness. A residual with a floor is a
> verdict on the dictionary, not on the data or the fit.

Requiring the state itself to be in the span adds a further restriction (beyond the
notes; Brunton et al. 2016): a finite invariant subspace that contains $x$ cannot
represent a system with more than one isolated fixed point. A quadrotor about hover has
one equilibrium of interest, so this is not an obstacle here, but it explains why
"global linearisation" claims should be read carefully.

## 7. EDMD: the finite approximation

### 7.1 The regression

Choose a dictionary $\theta$, collect pairs $(x_j, x_j')$ with $x_j' = F(x_j)$, and solve

$$\min_A \|\Theta(X') - A\,\Theta(X)\|_F,\qquad A = \Theta(X')\,\Theta(X)^+.$$

With $\theta(x) = x$ this is DMD.

### 7.2 What $A$ is

$A$ is the **least-squares projection** of $\mathcal{K}$ onto the span of the dictionary, with
respect to the distribution of the data. Three consequences:

- If the span is invariant, $A$ is exact (given informative data).
- If not, $A$ is the best one-step predictor *on average over the sampled region*. It is
  not a Taylor expansion at a point. This is why a lifted model can beat linearisation
  over a region: [EX-KoopC] reports a factor of about 15 over 100 steps for Van der Pol.
- The answer **depends on where the data were sampled**. A model fitted on small tracking
  errors under LQR says nothing about large attitude excursions.

### 7.3 One-step error versus multi-step error

EDMD minimises the one-step residual $r_j = \theta(x_j') - A\theta(x_j)$. MPC uses
$N$-step predictions made by iterating $A$ *without re-lifting*. The prediction error obeys

$$e_{i+1} = Ae_i + r_i\quad\Rightarrow\quad \|e_N\| \le \sum_{j=0}^{N-1}\|A^{N-1-j}\|\,\|r_j\|.$$

Small one-step residuals can still give poor horizon predictions if $\|A^k\|$ is large or
the residuals are correlated. *Measured on your model:* one-step velocity nRMSE 0.26 to
0.32, five-step 0.44 to 0.65. Always report the error at the horizon the MPC uses.

A diagnostic worth implementing: compare three rollouts over the horizon.

1. **Lifted rollout**: $z_{i+1} = Az_i + Bu_i$ (what MPC uses).
2. **Re-lifted rollout**: take only the state rows, $x_{i+1} = C(A\theta(x_i) + Bu_i)$,
   and re-evaluate $\theta$ each step.
3. **Linear rollout**: DMDc on the same data.

The gap between 1 and 2 is closure error. The gap between 2 and 3 is what the nonlinear
features buy.

### 7.4 What to trust in the spectrum

- **Spectral pollution.** Eigenvalues of $A$ need not approximate any true Koopman
  eigenvalue when the subspace is not invariant.
- **Regularisation bias.** The $\ell_1$ penalty shrinks coefficients, which generally moves
  eigenvalues toward the origin, so modes look more damped than they are. Compare against
  your unregularised `edmd_model1_ls` run before quoting any damping.
- **A test per eigenpair** (in the spirit of residual DMD, Colbrook and Townsend):
  on validation data compute
  $$\varepsilon_i = \frac{\|\varphi_i(X') - \lambda_i\varphi_i(X) - w_i^\top B\Upsilon\|}{\|\varphi_i(X')\|}.$$
  Report an eigenfunction as physical only if $\varepsilon_i$ is small and the pair survives a
  change of $\lambda$ and of dataset.

### 7.5 Practical conditioning

- **Scale the state** so that monomials have comparable size. Your `Standardizer` does this.
- For the new project, **scale but do not shift**: use $x/\sigma$, not $(x - \mu)/\sigma$.
  Then $\theta(0) = 0$ for every non-constant feature, the model's origin is the physical
  equilibrium, and the terminal ingredients are centred where they should be. Your current
  K-MPC needs the output reference $r = -\mu/\sigma$ only because of the shift.
- **The constant feature.** It represents affine offsets (gravity trim, mean wind). It is an
  uncontrollable mode at $\lambda = 1$. If you work in deviation coordinates about a trim
  point and feed the wind through $D$, you can drop it, and the model becomes
  stabilisable. If you keep it, remove it before solving any Riccati equation.

---

# Part IV. Koopman with control

## 8. Inputs change the problem

### 8.1 There is no single Koopman operator

$\mathcal{K}g = g \circ F$ needs an autonomous map. With an input there is a **family**
$\{\mathcal{K}_u\}$, one operator per constant input value [LN-DDC §3.1]. To get a usable
model one must *assume* how the family depends on $u$. The simplest assumption is additive:

$$\theta(x_{k+1}) \approx A\,\theta(x_k) + Bu_k.$$

Korda and Mezić justify this form by treating the whole future input sequence as part of
an enlarged autonomous state and restricting to observables that are linear in the input
[LN-DDC Remark 2]. That justifies the *structure* of the regression. It does not make the
model exact.

The regression is EDMD with the input appended [LN-DDC (15)]:

$$[A\ \ B] = \Theta(X')\begin{bmatrix}\Theta(X) \\ \Upsilon\end{bmatrix}^+.$$

State and input must be regressed **together**. Fitting $A$ first and $B$ from the residual
assigns to the drift whatever part of the input effect correlates with the state.

Your `edmd.py` does this with a third block for the wind, $[A\ B\ D]$.

### 8.2 Question 1: can the data separate drift from actuation?

**The rank condition** [LN-DDC Prop. 1]. The minimiser is unique if and only if the
regressor has full row rank:

$$\operatorname{rank}\begin{bmatrix}\Theta(X) \\ \Upsilon\end{bmatrix} = p + l.$$

- It is a property of the **data**, computable before fitting.
- It needs $m \ge p + l$ snapshots, but sample count is necessary, not sufficient.
- Watch the **smallest singular value**, not the integer rank.

**Closed-loop data fail it** [EX-DMDc, LN-DDC §2.2.2]. If $u_k = K_fx_k$ during collection,
the input rows are a linear combination of the state rows. The fit reproduces the data to
machine precision, recovers $A + BK_f$ exactly, and is wrong about $A$ and $B$ separately
by $O(1)$. The residual gives no warning.

*Computed for this guide*, scalar integrator $x^+ = x + 0.1u$ with $\theta = (x, x^2)$ and
data collected under $u = -x$: smallest singular value 0, residual $10^{-15}$, fitted
$\hat A_{11} = 0.45$, $\hat B_1 = -0.45$ against the truth $1$ and $0.1$. Yet
$\hat A_{11} - \hat B_1 = 0.9$, which is the true closed loop.

**The fix is dither**: $u_k = K_fx_k + d\,\eta_k$. A tiny dither restores rank; accuracy
tracks $\sigma_{\min}$ [EX-DMDc table].

**How this applies to you.**

- *Current setup.* You identify the map from the perturbation $u_d$ to the tracking error,
  with the LQR inside the plant. $u_d$ is random and independent of the state, so the
  regression is well posed. *Measured:* rank 62 of 62, condition number 67.
- *New setup.* You want the open-loop plant, so the regressor must contain the **total**
  command $u = u_{\text{LQR}}(x) + u_d$. If the LQR runs during collection, $u$ is mostly a
  function of the state and the dither is a small part of it. Expect a much worse
  condition number. Check it before fitting anything.

**In lifted coordinates the requirement is stronger** [LN-DDC §4.5]. A large dictionary
does not supply its own excitation. Every dictionary direction must be excited, and the
data must cover the joint state–input region where the model will be used, in amplitude as
well as frequency. If a bilinear model includes $u \otimes \theta(x)$, an input taking only
two values makes columns collinear, as in the binary-PRBS row of the SINDy table.

### 8.3 Question 2: can a constant $B$ represent the actuation?

For a control-affine system $\dot x = f(x) + G(x)u$, differentiate the lift [LN-DDC (19)]:

$$\dot z = D\theta(x)f(x) + D\theta(x)G(x)\,u.$$

**Proposition** [LN-DDC Prop. 2]. The lifted dynamics are exactly $\dot z = Az + Bu$ if and
only if

$$D\theta(x)f(x) = A\theta(x)\quad\text{and}\quad D\theta(x)G(x) = B\ \text{ for all } x.$$

The first is ordinary closure. The second is new and restrictive: **the lifted input
matrix must not depend on the state**. Even with constant $G$, any nonlinear feature has a
state-dependent gradient, so the condition generically fails.

> Lifting makes the drift linear at the cost of making the input channel nonlinear.

*Computed for this guide*, the integrator again, with random open-loop data. Exactly,
$(x^2)^+ = x^2 + 0.2\,xu + 0.01u^2$.

| Model | Relative residual | Row fitted for $x^2$ |
|---|---:|---|
| Linear, $[\theta; u]$ | 9.0% | $x^2 \leftarrow 1.005\,x^2 + 0.001\,u$ |
| Bilinear, adds $u\theta$ | 0.5% | $x^2 \leftarrow 1.006\,x^2 + 0.1998\,xu$ |

The linear fit gives the input **no** influence on $x^2$, because $xu$ averages to zero.
The bilinear fit recovers the coefficient $0.2$. The remaining 0.5% is the $u^2$ term.

**The bilinear model** [LN-DDC §3.6]:

$$z_{k+1} = Az_k + B_0u_k + \sum_{i=1}^{l}u_{i,k}B_iz_k.$$

- Fitted by the same least squares with extra regressor rows $u \otimes \theta(x)$.
- It is the natural exact class for control-affine systems (Otto and Rowley 2021).
- It costs the convexity of MPC, because predicted state multiplies decision variable.

So the course's trade-off: *keep the linear form and be wrong, or fix the model and lose
the QP.* §9.5 shows the loss is smaller than it sounds.

### 8.4 Spectral analysis with inputs

With inputs and a disturbance, each eigenfunction obeys

$$\varphi_i(x_{k+1}) \approx \lambda_i\varphi_i(x_k) + (w_i^\top B)u_k + (w_i^\top D)w_k.$$

Three numbers per mode, and this is the analysis that makes a Koopman model more than a
black box:

| Quantity | Meaning | Use |
|---|---|---|
| $\lambda_i$ | decay time and frequency of the mode | compare with the analytic linearisation |
| $\|w_i^\top B\|$ | how well the input reaches the mode (PBH) | find unreachable modes; explain Riccati failures |
| $\|w_i^\top D\|$ | how strongly wind excites the mode | say *which* motion the wind drives |

*Measured on your model* (closed loop with LQR): the slowest mode is yaw
($\tau = 3.8$ s); the lateral pair at 0.28 Hz has the largest wind gain (0.09).

Caveats:

- $A$ is a **predictor**, not the Koopman operator of any single autonomous system. Its
  eigenvalues describe the unforced ($u = 0$, $w = 0$) model. Whether that matches the
  unforced plant depends on Question 2.
- With a bilinear model, "eigenvalues of $A$" are those of the drift only; the input
  changes the effective dynamics to $A + \sum_iu_iB_i$.

### 8.5 SINDy with control, and how it relates

SINDYc fits $\dot x = \Xi\,\Theta(x, u)$ sparsely in the original coordinates [LN-DDC §4].

- The cross terms $x \otimes u$ in the library play exactly the role of the bilinear $B_i$.
  Leaving them out is the same assumption as a constant $B$.
- The model is nonlinear, so the controller is nonlinear MPC.
- Your $\ell_1$-EDMD on monomials is already close to a discrete-time SINDy: the *state rows*
  of $[A\ B]$ are a sparse polynomial model of $x^+$. The difference is how predictions are
  made. SINDy re-evaluates the library each step (re-lifted rollout of §7.3). EDMD iterates
  the lifted state.

You do not need SINDy as a separate method. The re-lifted rollout gives you the SINDy-style
prediction for free and makes a useful comparison.

---

# Part V. Koopman MPC

## 9. Putting the lifted model in the loop

### 9.1 The problem

$$\min_{u_0,\dots,u_{N-1}}\ \sum_{i=0}^{N-1}\left(\|Cz_i - r_i\|_Q^2 + \|u_i\|_R^2\right) + \|Cz_N - r_N\|_{Q_f}^2
\quad\text{s.t. } z_{i+1} = Az_i + Bu_i,\ z_0 = \theta(x),\ u_i \in \mathbb{U}.$$

- The only nonlinear operation is the lift $z_0 = \theta(x)$ of the measurement.
- Cost and constraints on the physical state are linear in $z$ through $C$.
- After condensing, the QP has $lN$ variables. **The dictionary size $p$ does not appear
  online** [LN-DDC (18)]. This is Korda and Mezić's practical point.

### 9.2 What "linear MPC for a nonlinear plant" really means

The prediction is linear in $z$, but the *feedback law* $u = \kappa(\theta(x))$ is nonlinear
in $x$. For example, lifted LQR gives $u = K\theta(x) = K_1x_1 + K_2x_2 + K_3x_1^2$
[LN-DDC §3.5].

There is a subtlety that matters for all guarantees. Predicted lifted states $z_i$ for
$i \ge 1$ are **not** on the manifold $\{\theta(x)\}$ in general: the predicted value of the
"$x_1^2$" coordinate need not equal the square of the predicted "$x_1$" coordinate. The
model is linear on all of $\mathbb{R}^p$, but physically meaningful only near that manifold.

### 9.3 Which guarantees survive

The theorems of §5 hold for the system $z^+ = Az + Bu$. Applied to the plant:

- **Recursive feasibility** needs the plant to land where the model predicted. With model
  error it does not, so the shifted plan may be infeasible. Input constraints are
  unaffected; state and terminal constraints can fail.
- **Stability** needs the cost decrease to dominate the effect of prediction error. If the
  one-step model error is bounded by something proportional to the size of the state and
  input, stability can still be proven (the "proportional error bound" in the survey of
  Strässer et al. 2025). A constant error floor only gives convergence to a neighbourhood.

What you can honestly claim in the project:

1. A proof for the nominal lifted (or linearised) model.
2. A measured prediction error at the horizon.
3. Empirical closed-loop evidence: feasibility counts, region of attraction, convergence.

Stating exactly this, rather than "the controller is stable", is what a careful reader
looks for.

### 9.4 Terminal ingredients in lifted space

The textbook recipe (DARE on the model) runs into four problems.

1. **Stabilisability.** The constant mode is uncontrollable at $\lambda = 1$. For an
   open-loop vehicle, other lifted modes sit near $|\lambda| = 1$ as well (products of
   integrator modes), and a constant $B$ gives the input little or no authority over them
   (§8.3 table). Check every mode with $|\lambda_i| \ge 1$ using PBH.
2. **Detectability.** The cost sees only $Cz$.
3. **Dimension.** Invariant polytopes in $\mathbb{R}^{55}$ are impractical.
4. **Manifold.** A set invariant in $\mathbb{R}^p$ says little about the plant.

Three designs, from most defensible to most ambitious:

- **Design A: terminal ingredients on the physical linear model.** Compute $K$, $P$ and
  $\mathcal{O}_\infty$ for the 9-state linearisation (analytic, or the state block of the
  identified model). Impose them through $C$: terminal cost $\|Cz_N\|_P^2$, terminal
  constraint $H_fCz_N \le h_f$. The guarantee is exact for the linear model and local for
  the plant, because near the origin the lifted and linear predictions agree. This is the
  recommended design.
- **Design B: DARE on a reduced lifted model.** Remove the constant, then remove or
  truncate modes that fail PBH, and solve the DARE on what remains. More faithful to the
  lifted dynamics, but every truncation must be justified.
- **Design C: no terminal set, long horizon.** Use $Q_f$ from Design A and choose $N$ from
  the sweep of §4.4.

Whichever you use, test (A3) numerically on data: sample states in $\mathbb{X}_f$, apply the
terminal controller to the *plant simulator*, and check that $V_f$ decreases by at least the
stage cost. That check turns an assumption into a result.

### 9.5 MPC with a bilinear model

The problem is non-convex, but it has structure. Two standard options (beyond the notes;
Folkestad and Burdick 2021; Bruder et al. 2021):

- **Frozen input matrix.** Over the horizon use $B(z_0) = B_0 + [B_1z_0\ \cdots\ B_lz_0]$,
  evaluated at the current lifted state. Still one QP per step. Accurate when the state
  changes little over the horizon.
- **Sequential QP (real-time iteration).** Linearise about the previous solution
  $(\bar z_i, \bar u_i)$:
  $$z_{i+1} \approx \Big(A + \sum_j\bar u_{i,j}B_j\Big)z_i + \Big(B_0 + [B_1\bar z_i \cdots B_l\bar z_i]\Big)u_i - \sum_j\bar u_{i,j}B_j\bar z_i.$$
  This is a time-varying linear model, hence a QP. One iteration per sample is usually enough.

So the real cost of bilinearity is an online re-condensation, not a general nonlinear
solver.

---

# Part VI. The quadrotor

## 10. Where each issue appears on your vehicle

### 10.1 The model the MPC sees

Keep the body-rate loop. Outer-loop state and input:

- $x = [p,\ v,\ \phi,\ \theta,\ \psi]$: position and velocity in world NED, roll/pitch/yaw.
- $u = [\omega_x,\ \omega_y,\ \omega_z,\ c]$: body-rate commands (FRD) and mass-normalised thrust.

With ideal rate tracking:

$$\dot p = v,\qquad \dot v = g\,e_3 - c\,R(\phi,\theta,\psi)\,e_3 - \tfrac1m f_D(v - v_w),\qquad
\begin{bmatrix}\dot\phi \\ \dot\theta \\ \dot\psi\end{bmatrix} = W(\phi,\theta)\,\omega,$$

$$R\,e_3 = \begin{bmatrix}c_\psi s_\theta c_\phi + s_\psi s_\phi \\ s_\psi s_\theta c_\phi - c_\psi s_\phi \\ c_\theta c_\phi\end{bmatrix},\qquad
W = \begin{bmatrix}1 & s_\phi t_\theta & c_\phi t_\theta \\ 0 & c_\phi & -s_\phi \\ 0 & s_\phi/c_\theta & c_\phi/c_\theta\end{bmatrix}.$$

It is control-affine, $\dot x = f(x) + G(x)u$, with

$$G(x) = \begin{bmatrix}0 & 0 \\ 0 & -R\,e_3 \\ W(\phi,\theta) & 0\end{bmatrix}.$$

**$G$ depends on the state.** By the proposition of §8.3 the linear predictor cannot be
exact, already for the un-lifted state rows.

**A modelling decision you must make.** The rate loop has gains 20, 20, 10 s⁻¹, so the
true body rate lags its command with time constants 0.05, 0.05, 0.1 s. At
$\Delta t = 0.1$ s this is not negligible, especially in yaw. The 9-state model is then not
Markov. Options: add the measured body rate to the state (12 states), or accept the error
and quantify it. The numbers in §4.4 and §5.6 assume ideal rate tracking.

### 10.2 Hover linearisation

Trim: $c = g$, zero attitude, zero velocity. With $\delta c = c - g$:

$$\dot p = v,\quad \dot v_x = -g\,\theta,\quad \dot v_y = g\,\phi,\quad \dot v_z = -\delta c,\quad
\dot\phi = \omega_x,\ \dot\theta = \omega_y,\ \dot\psi = \omega_z.$$

- Horizontal axes are **triple integrators** (rate → angle → velocity → position).
- Vertical is a double integrator, yaw a single integrator.
- All nine continuous eigenvalues are 0, so all nine discrete eigenvalues are 1, in
  Jordan blocks. The model is controllable.

This is why the open-loop case is interesting: nothing is stable, so the terminal cost
carries the whole burden of stability (§4.4).

The translational equation matches `sdlqr_model.py` ($\dot v = g\,e_3 - c\,R(q)\,e_3$); the
Euler-angle form here is my derivation from it. NED with FRD body is where sign errors
live, so test it. The unit test: a positive pitch must accelerate the vehicle toward $-x$.

### 10.3 Where the nonlinearity is, and what it implies

| Source | Term | Kind | Consequence |
|---|---|---|---|
| Thrust direction | $-c\,R\,e_3$ | input × function of attitude | bilinear: $\delta c \times \{\theta, \phi\}$ |
| Thrust direction | $-g\,R\,e_3$ | drift, trigonometric in attitude | needs attitude monomials; never closes exactly |
| Euler kinematics | $W(\phi,\theta)\,\omega$ | input × function of attitude | bilinear: $\omega \times \{\phi, \theta\}$ |
| Drag | $R D R^\top(v - v_w)\|v - v_w\|$ | quadratic in relative velocity | wind enters nonlinearly; $Dw$ is an approximation |

Sizes, which tell you when each matters:

- $\dot v_x = -(g + \delta c)\,\theta + \dots$. The bilinear part relative to the linear
  part is $\delta c / g$. For gentle flight it is a few percent. For aggressive vertical
  manoeuvres it is first order.
- $W - I$ is second order in the angles. Negligible below about 0.3 rad.
- In your current setup $u_d$ is bounded by 0.3 rad/s and 1 m/s², so bilinear terms are
  small. *Measured:* only 23 of 220 entries of $B$ are non-zero.

**Prediction:** bilinear terms will matter in the open-loop model in proportion to how
hard you excite thrust and attitude together. Design the data to show this (E5).

### 10.4 Using symmetry to choose the dictionary

- **Translation invariance.** The dynamics do not depend on $p$ (for a uniform wind
  field). So $p$ should appear only linearly. Monomials such as $p_xp_y$ can only fit noise.
- **Nonlinearity lives in attitude and velocity.** A physics-informed dictionary is
  $x$ plus monomials in $(\phi, \theta, \psi)$ and, for drag, in $v$.
- Degree-2 monomials in those 6 variables give 21 features, so $p = 30$ instead of 55.
- Yaw: keep it near zero, or expect that the $x$–$y$ coupling needs yaw features.

A smaller dictionary needs less data, has fewer spurious near-unit modes, and is easier to
defend. Comparing it against the full degree-2 dictionary is a cheap, informative experiment.

### 10.5 Data collection for the open-loop model

Requirements from §8.2: the input must not be a function of the state; every feature must
be excited; the data must cover the region the MPC will visit.

| Option | How | Strength | Weakness |
|---|---|---|---|
| **Short random episodes** | sample $x_0$ in the operating box, apply random held inputs for 5–10 steps | no feedback correlation; uniform coverage; best conditioning | only possible in simulation; say so |
| **LQR + large dither** | fly references, log total $u$ | realistic trajectories | $u$ mostly a function of $x$; check $\sigma_{\min}$ |
| **Mixed** | both, concatenated | coverage plus realism | two datasets to justify |

The course notes point out that snapshot triples need not come from one trajectory
[LN-DDC §3.4] and that several short experiments can jointly provide excitation
[LN-DDC §4.5]. Short episodes are the principled choice in simulation.

Checks to report for every dataset:

1. Rank, smallest singular value and condition number of the full regressor.
2. Snapshots per regressor row. *Measured now:* 1112 for 62, about 18. Aim far higher.
3. Histogram of visited states against the constraint box.
4. Number of distinct input values (more than the highest power of $u$ in the model).

### 10.6 Wind

- With quadratic drag, the wind force depends on $(v - v_w)$ nonlinearly. A constant
  $D$ is a linearisation about the conditions in the data.
- For regulation with mean wind $\bar w$: compute the target $(z_s, u_s)$ from
  $z_s = Az_s + Bu_s + D\bar w$ and centre cost and terminal ingredients there (§5.8).
- Turbulence is a disturbance the nominal theory does not cover. Report its effect as
  ultimate bound on the error, not as a stability failure.

For the course project, a sensible order is: no wind, then constant wind, then turbulence.

## 11. The experiments

Each experiment has a question, a prediction from the theory, and the plot that answers it.

| # | Question | What to vary | Prediction | Evidence |
|---|---|---|---|---|
| E1 | Does the terminal cost matter? | $Q_f \in \{0, Q, P\}$, $N$ from 1 to 40, unconstrained, linear model | table of §4.4 | $\rho(A + BK_N)$ vs $N$; then closed-loop runs on the simulator |
| E2 | Does it still matter on the nonlinear plant? | same, on the simulator | same ordering; instability thresholds shift | settling time and cost vs $N$ |
| E3 | What does the terminal set buy and cost? | none / ellipsoid / $\mathcal{O}_\infty$, with a tilt or position constraint | without: violations; ellipsoid: small feasible region; $\mathcal{O}_\infty$: larger | infeasible-step count; constraint violations |
| E4 | How big is the region of attraction? | grid of initial conditions, $N$, terminal set | grows with $N$ and with set size | feasible/converged map in a 2-D slice |
| E5 | Does lifting help? | DMDc, EDMDc, bilinear EDMDc; dictionary size | gain grows with excitation amplitude | horizon prediction error; closed-loop cost |
| E6 | Was the data informative? | LQR+dither vs short random episodes; dither size | closed-loop data: low residual, poor model | $\sigma_{\min}$, parameter error vs linearisation |
| E7 | Does the spectrum mean anything? | compare identified eigenvalues with the analytic model | linear block near 1; lifted spectrum = product lattice | eigenvalue plot; $\varepsilon_i$ per mode |

Order of work, and the principle behind it (separation of concerns: change one thing at a
time, so every failure has one possible cause):

1. Analytic linearised model + MPC + terminal ingredients (E1–E4). No learning involved.
2. Identification pipeline on open-loop data (E6, E7), validated against the linearisation.
3. Swap the learned model into the same MPC (E5).
4. Wind as an extension.

Keep the LQR-augmented results as the reference case. The contrast "MPC as a correction"
versus "MPC as the stabiliser" is itself a finding.

### 11.1 Pitfalls checklist

- [ ] Regressor uses the **total** applied input, not the dither.
- [ ] Rank and $\sigma_{\min}$ reported before any fit is trusted.
- [ ] State scaled but not shifted; $\theta(0) = 0$; origin is the trim point.
- [ ] Constant mode removed before any Riccati or Lyapunov solve.
- [ ] PBH checked for every mode with $|\lambda| \ge 1$.
- [ ] Prediction error reported at horizon $N$, not only one step.
- [ ] Terminal ingredients centred at the target, in the same coordinates as the cost.
- [ ] $Q_f$ and $\mathbb{X}_f$ expressed in the model's scaled coordinates (transform with the scaling matrix).
- [ ] Input constraints on the total command, including the trim thrust.
- [ ] Claims separated into: proven for the model, measured on the plant.

---

# Part VII. Your current model as a worked example

## 12. Reading `20261113T155255Z_edmd_model1`

This is the closed-loop (quadrotor + LQR) error model, $p = 55$, $\lambda_{\ell_1} = 20$.
All values are *measured on your model*.

| Observation | Value | Interpretation, with section |
|---|---|---|
| Spectral radius | 1.000 | constant feature, exactly (§6.2) |
| Without the constant | 0.974 | stable, as a closed loop must be (§4.3) |
| PBH of the $\lambda = 1$ mode | $10^{-16}$ | uncontrollable; DARE fails on the full model (§1.3) |
| Lyapunov $Q_f$, constant removed | exists; trace 13× that of $C^\top QC$ | the tail ignored by $N = 5$ is large (§4.2) |
| $\|A^5\|_2$ | 1.9 | non-normal transient growth over the horizon (§1.1) |
| Eigenvalue 0.9506, yaw²-dominated | $\approx 0.9746^2$ | product lattice (§6.2) |
| Linear-block eigenvalues | 0.975, 0.931 ± 0.049j, … | LQR closed-loop poles at 10 Hz |
| Norm of nonlinear→state block | 0.29 vs 2.60 linear | lifting contributes about 10% |
| Non-zeros in $B$ | 23 of 220 | small bounded input: bilinear terms negligible (§10.3) |
| Regressor | rank 62, condition 67 | Question 1 passes (§8.2) |
| Snapshots | 1112 for 62 rows | thin; likely limits accuracy |
| Horizon nRMSE, velocity | 0.44–0.65 | the weak point; diagnose with §7.3 rollouts |

What it says overall: the identification is well posed, the model is nearly linear, and its
spectrum is the LQR's. Its main weakness is velocity prediction at the horizon, for which
the candidates are too little data, wind entering through quadratic drag, and the unmodelled
rate-loop lag. The residual tests of §7.3 and §8.3 separate these.

---

# Part VIII. Consolidation

## 13. Sources

Course material, in `learning_materials/`: [LN-MPC], [LN-DDC], [EX-DMDc], [EX-KoopC].

Beyond the course, in the order you will need them:

- Rawlings, Mayne, Diehl, *Model Predictive Control: Theory, Computation, and Design*,
  2nd ed. Ch. 1–2 for stability; Ch. 3 for robustness.
- Borrelli, Bemporad, Morari, *Predictive Control for Linear and Hybrid Systems* (the
  course's reference [1]). Invariant sets and their computation.
- Mayne, Rawlings, Rao, Scokaert, "Constrained model predictive control: stability and
  optimality", *Automatica* 2000. The terminal-ingredient framework.
- Gilbert and Tan, "Linear systems with state and control constraints: the theory and
  application of maximal output admissible sets", *IEEE TAC* 1991.
- Rawlings and Muske, "The stability of constrained receding horizon control",
  *IEEE TAC* 1993. Stable plants, input constraints.
- Limon, Alamo, Salas, Camacho, "On the stability of constrained MPC without terminal
  constraint", *IEEE TAC* 2006.
- Korda and Mezić, "Linear predictors for nonlinear dynamical systems: Koopman operator
  meets model predictive control", *Automatica* 2018.
- Proctor, Brunton, Kutz, "Dynamic mode decomposition with control" (2016) and
  "Generalizing Koopman theory to allow for inputs and control" (2018), *SIAM J. Appl. Dyn. Syst.*
- Otto and Rowley, "Koopman operators for estimation and control of dynamical systems",
  *Annu. Rev. Control Robot. Auton. Syst.* 2021. Why bilinear.
- Brunton, Brunton, Proctor, Kutz, "Koopman invariant subspaces and finite linear
  representations of nonlinear dynamical systems for control", *PLoS ONE* 2016.
- Bruder, Fu, Vasudevan, "Advantages of bilinear Koopman realizations for the modeling and
  control of systems with unknown dynamics", *IEEE RA-L* 2021.
- Folkestad and Burdick, "Koopman NMPC: Koopman-based learning and nonlinear model
  predictive control of control-affine systems", *ICRA* 2021. Bilinear Koopman on a quadrotor.
- Strässer, Schaller, Worthmann, Berberich, Allgöwer, "An overview of Koopman-based
  control: from error bounds to closed-loop guarantees", *Annu. Rev. Control* 2025.
- Colbrook and Townsend, residual DMD, *Comm. Pure Appl. Math.* 2024. Trusting spectra.

I have cited these from memory of the literature; check volume and page details before
putting them in a report.

## 14. Exercises

Do them on paper first, then in code. Each one becomes a unit test.

**Linear and MPC**

1. For $x^+ = 1.5x + u$, $Q = 1$, $R = 2$: solve the DARE by hand. (*$P = 4$, $K = -1$.*)
2. Same system, $N = 1$, $Q_f = 0$: show $u_0^\star = 0$. With $Q_f = 4$: show
   $u_0^\star = \operatorname{sat}(-x_0)$.
3. Show $\mathbb{X}_f = [-1, 1]$ satisfies (A1) and (A2) for $K_f = -1$, $|u| \le 1$.
   What is the largest valid interval? (*$[-1, 1]$: the input limit binds first.*)
4. For that system with $N = 2$ and $\mathbb{X}_f = [-1, 1]$, find the feasible set
   $\mathbb{X}_2$. (*Work backwards: $x_1 \in [-4/3, 4/3]$, then $x_0 \in [-14/9, 14/9]$.*)
5. Prove the identity of §2.3 from the DARE.
6. Write out the cost of the shifted plan and show where (A3) is used.
7. Derive $\alpha = \min_ib_i^2 / (c_i^\top P^{-1}c_i)$ using Cauchy–Schwarz in the $P$-norm.
8. Reproduce the table of §4.4. Then repeat it with $R$ ten times larger and explain the change.

**Koopman**

9. Show that $\varphi_1\varphi_2$ is an eigenfunction with eigenvalue $\lambda_1\lambda_2$.
10. For $\dot x_1 = \mu x_1$, $\dot x_2 = \lambda(x_2 - x_1^2)$, write the $3 \times 3$ generator
    in $\theta = (x_1, x_2, x_1^2)$ and find its eigenvalues. (*$\mu$, $\lambda$, $2\mu$.*)
11. Move the input to the $x_1$ equation and show $D\theta\,G$ is not constant.
12. For $x^+ = x + \Delta t\,u$, expand $(x^+)^2$ and identify the linear, bilinear and
    unmodelled terms. Reproduce the table of §8.3.
13. Take data from that system under $u = -x$. Compute the singular values of the regressor
    and show that only $\hat A - \hat B$ is determined.
14. For the quadrotor of §10.1, compute $D\theta\,G$ for $\theta = (x, \theta^2)$ and list
    which entries depend on the state.

**On your model**

15. Compute $\varepsilon_i$ of §7.4 for the ten slowest modes on the validation split.
16. Compare the eigenvalues of the $\ell_1$ and least-squares fits. Which way did they move?
17. Implement the three rollouts of §7.3 and plot error against prediction step.

## 15. What to understand next

- **First**: §5.3 and §5.4. Be able to reproduce both proofs without notes and say which
  ingredient each step uses.
- **Second**: §8.2 and §8.3 as two independent failure modes, each with its own test.
- **Third**: §9.3. Being precise about what is proven for the model and what is measured on
  the plant is the difference between a project that reports results and one that
  understands them.
