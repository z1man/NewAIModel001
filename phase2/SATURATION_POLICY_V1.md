# Saturation Management Policy v1

**Scope:** u_max=1.0, disturbance_scale around OP20/OP5. Derived from Phase 2 dynamic reference shaping experiments.

## ✅ Policy
1) **Step-like reference**
   - Default: **LPF τ=1.0s** → stable, sat <2%, finite recovery.
   - If faster response needed: **Rate limit dr=0.2** → sat ≈ 4–5%, RMSE lower.

2) **Ramp reference**
   - If ramp is already slow enough (e.g., OP20/OP5 conditions), **no shaping required**.

3) **Sine tracking**
   - Shaping alone cannot rescue saturation. Must reduce **(A, f)** into reachable region.
   - Use feasibility frontier to pick safe (A, f); shaping is only a helper.

## Evidence
- Step: LPF τ=1.0 → sat 1.65%, RMSE 0.0462, recovery 1.36s
- Step: Rate 0.2 → sat 4.30%, RMSE 0.0385, recovery 0.00s
- Ramp: sat already <3%, shaping no effect
- Sine: sat remains high for large A/f; requires amplitude/frequency reduction

## Notes
- Ensure r_shaped differs from r (non‑zero max|r-r_shaped|) for shaping to matter.
- If saturation exceeds ~20%, reduce disturbance or reference amplitude before shaping.
