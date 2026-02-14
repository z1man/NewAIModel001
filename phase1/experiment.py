#!/usr/bin/env python3
"""
Phase 1: Minimum Viable Cerebellum (MVC)
- 1D mass with unknown friction
- Baseline controller uses nominal model
- Residual learner compensates model error
"""
import math
import random

# System parameters
DT = 0.01
STEPS = 4000

# True system (unknown to controller)
TRUE_MASS = 1.0
TRUE_FRICTION = 0.35  # unknown friction coefficient

# Controller assumed model
MODEL_MASS = 1.0
MODEL_FRICTION = 0.0  # missing friction

# Control target
TARGET = 1.0

# Baseline PD gains
KP = 18.0
KD = 6.0

# Residual learner (simple linear reg on velocity)
# correction = w * v
w = 0.0
LR = 0.001


def simulate(use_residual=False):
    global w
    w = 0.0
    x = 0.0
    v = 0.0
    total_error = 0.0

    for t in range(STEPS):
        # Baseline PD control
        error = TARGET - x
        u = KP * error - KD * v

        # Residual correction
        if use_residual:
            u += w * v

        # True dynamics
        # x'' = (u - friction*v)/m
        a = (u - TRUE_FRICTION * v) / TRUE_MASS
        v = v + a * DT
        x = x + v * DT

        # Online residual learning (local error signal)
        # We estimate error in acceleration due to friction mismatch
        # a_model = (u - MODEL_FRICTION*v)/MODEL_MASS
        a_model = (u - MODEL_FRICTION * v) / MODEL_MASS
        a_error = a - a_model

        # Learn w to reduce a_error (simple delta rule)
        if use_residual:
            w += LR * a_error * v

        total_error += abs(error)

    return total_error / STEPS


if __name__ == "__main__":
    baseline = simulate(use_residual=False)
    learned = simulate(use_residual=True)

    print("Baseline avg error:", round(baseline, 5))
    print("With residual learner avg error:", round(learned, 5))
    if learned < baseline:
        print("✅ Residual learner improved tracking")
    else:
        print("⚠️ No improvement; adjust gains or LR")
