# Speaker notes: Koopman MPC for a quadrotor in turbulent wind

Target: about 35 min plus questions. 33 slides; section slides take seconds.
Time budget: Part 1 = 5 min, Part 2 = 9 min, Part 3 = 7 min, Part 4 = 4 min, Results = 9 min, close = 2 min.

**The story in one line:** the wind does not just push the vehicle; it changes what the rotors
deliver. A full Koopman MPC can learn that effect from data, while models with a fixed physical
structure cannot.

---

## Part 1: Quadrotor model in wind (about 5 min)

**4. Rigid-body model** (1.5 min)
- Standard 13-state model. The only wind term is drag on the *air-relative* velocity.
- The wind is the CCTA paper's: about 3.2 m/s mean plus Gaussian gusts redrawn every 0.1 s.
- The controller commands body rates and collective thrust; a fast rate loop does the rest.

**5. BEMT** (2 min)
- Walk the chain left to right. Every controller assumes T = k_T Ω².
- Under that assumption, BEMT: in edgewise flow (wind across the disk) the rotor gives *more* thrust; when climbing through the air it gives *less*.
- It is calibrated to be exact in calm hover, so the mismatch appears only in airflow, the condition we study.

**6. Applied thrust in flight** (1.5 min)
- In real flight the gap reaches about +30 %, and it tracks the airspeed.
- Take-away: this is a smooth function of the state and the wind. Remember it; it is why Koopman wins at the end.

*Likely question: "Is your BEMT the paper's?"* No. Ours is the classical uniform-inflow BEMT. The paper uses a hybrid BEMT with rotor-speed adjustment; the qualitative effect is the same.

## Part 2: Koopman models (about 9 min)

**8. Koopman in one slide** (1.5 min)
- Lift the state into observables, where the dynamics are approximately linear. EDMD is least squares on snapshot pairs.
- For MPC this matters because the nonlinearity sits in ψ, outside the QP.
- The catch: everything depends on the dictionary.

**9. K-Res** (2 min)
- The paper's Model 1. The state is the *tracking error* of the LQR-controlled vehicle; the input is the *perturbation* u_d.
- Dictionary: all monomials up to degree 2, 55 features. Fitted with ℓ1-EDMD.

**10. K-Res design** (1.5 min)
- The LQR flies; the MPC only nudges, within ±0.3 rad/s.
- Main point: the model describes the closed loop *with* the LQR, so stability comes from the LQR and the gain is capped.

**11. K-Full** (2.5 min)
- Now the input is the *total* command, and the model describes the open-loop vehicle.
- Dictionary: state, constant, sin/cos of the angles, and degree-2 products of velocity with trig terms (61 features).
- Why trig: the thrust direction and the body-frame airspeed are made of exactly these products, so drag and rotor inflow lie in the span.
- Translation invariance: position may only feed itself (it fixed a real identifiability bug).

**12. K-Full design** (1.5 min)
- The MPC is the whole controller. The table lists all six controllers you will see in the results.
- Stress the "oracle": it knows the true drag but not the rotors.

## Part 3: Data and identifiability (about 7 min)

**14. Data generation** (2 min)
- K-Res: LQR plus random u_d; we log error states.
- K-Full: a nominal LTV MPC that knows nothing about wind, plus dither. Circles and figure-8s are held out.
- Point to the take-away: trajectories give coverage, dither gives identifiability.

**15. Base controller** (2 min)
- Why LTV: it must command the total input, know nothing about wind, and track well at large tilt.
- Terminal cost: a time-varying Riccati P, i.e. the infinite-horizon cost-to-go, which keeps a short horizon stabilising.
- The maximal admissible set O∞ is implemented for the hover MPC. Along a trajectory it would have to move, so the data controller uses the cost only. **Say this plainly; it is a known limitation, not a hidden one.**

**16. Unique minimiser** (2 min)
- The fit is unique iff Γ₁ has full row rank N + l.
- Counting is necessary but not enough: a constant input adds columns, not rank.
- PE via the Hankel matrix; Willems' lemma gives the guarantee for linear models only, so we also check Γ₁ directly.
- Mosaic: windows never cross episodes, but episodes pool their excitation.

**17. Hankel condition verified** (1 min)
- Every check passes for both models.
- Highlight: one episode alone would need about 450–500 samples, but each has only 140. Only the pooled mosaic is PE.
- Honest flag: K-Full's Γ₁ is near-singular because of its *features*, not its input.

## Part 4: Koopman MPC (about 4 min)

**19. Formulations** (1.5 min)
- Both use terminal cost equal to the stage cost (highlighted), no terminal constraint, and input bounds only, so the QP is always feasible.

**20. Why** (2.5 min)
- No Riccati P: the constant feature is an unreachable λ = 1 mode, so the DARE has no stabilising solution. Show the three-line PBH argument.
- No terminal set: lifted states live on a manifold, and there is no terminal law to make a set invariant.
- K-Res inherits stability from the LQR; K-Full's stability is empirical. That is the next research step.

## Results (about 9 min)

**22. Re-run with BEMT** (1 min): all six controllers fly the same vehicle through the same gusts, and the pairing is checked.

**23. Six-controller table** (2.5 min), the key slide.
- Koopman is best on every reference: 2.4–2.8× below the oracle on circles and figure-8s, and about 5× below the paper's LQR + K-Res.
- K-Res improves the LQR on every seed, but only by 2–5 %.
- The grey-box equals the oracle.

**24–29. Circle and figure-8** (about 4 min; spend time on 24, 25, 27, 28)
- 3-D view: the others drift with the wind; Koopman stays on the reference.
- Position error: the others *rise* (negative down error) when the vehicle tilts and speeds up, which is where BEMT over-delivers thrust.
- Attitude error is about 5° for everyone, an artefact of the drag-free reference attitude. Don't rank on it.

**30–31. Commands** (1 min)
- No bounds reached, no bang-bang.
- Jitter of 3–4 % of the range at the gust rate, the same for all controllers.

## Close (about 2 min)

**32. Why Koopman wins:** R² of 0.76 against 0.05. The oracle has no place for the thrust error, and the grey-box can only bend its drag coefficients.

**33. Summary:** read the findings, then two caveats (terminal ingredients, near-singular Γ₁) as next steps.

---

## Likely questions, short answers
- **"Isn't the oracle an upper bound?"** Only with ideal rotors. Under BEMT it misses the rotor aerodynamics.
- **"Is the comparison fair to Koopman?"** It is slightly *unfavourable* to Koopman: it gets terminal Q and attitude regulated to trim, while the LTVs get a Riccati P and a model-consistent reference.
- **"Does Model 2 (f_res) fix K-Res?"** We checked with a motor model. f_res barely correlates with the thrust error (r = 0.13), because most of the BEMT error occurs at constant rotor speed.
- **"Are the eigenvalues meaningful?"** Their locations match the physics, but they are fragile (‖A‖ ≈ 3×10⁴). Judge the model by prediction error, not poles.
- **"Why 10 Hz?"** It is the paper's MPC rate; the 1 kHz rate loop handles attitude.
