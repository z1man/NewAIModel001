"""Policies for Phase 6."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import Phase6Config
from .env_runner import EnvParams, EnvState, clamp
from .teacher import ActionGrid, action_to_params, apply_action


@dataclass
class ClassifierModel:
    w: np.ndarray
    b: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        x_n = (x - self.x_mean) / self.x_std
        return x_n @ self.w + self.b

    def predict_action(self, x: np.ndarray) -> int:
        logits = self.predict_logits(x)
        return int(np.argmax(logits))


def cg_v1_update(
    r: float, state: EnvState, config: Phase6Config, sat_hist: list
) -> Tuple[float, EnvState]:
    tau = 0.5
    alpha = config.dt / tau
    sat_hi = 0.2
    sat_lo = 0.05
    down = 0.9
    up = 1.02

    sat_rate = (1 - alpha) * (sat_hist[-1] if sat_hist else 0.0) + alpha * (
        1.0 if (len(sat_hist) > 0 and sat_hist[-1] == 1) else 0.0
    )
    dr_max = getattr(state, "dr_max", 1.0)
    setattr(state, "a_scale", config.a_max)
    if sat_rate > sat_hi:
        dr_max *= down
    elif sat_rate < sat_lo:
        dr_max *= up
    dr_max = clamp(dr_max, config.dr_min, config.dr_cap)
    setattr(state, "dr_max", dr_max)
    dr = clamp(r - state.r_feasible, -dr_max * config.dt, dr_max * config.dt)
    state.r_feasible = state.r_feasible + dr
    return state.r_feasible, state


def baseline_policy(
    r: float, state: EnvState, dist_scale: float, config: Phase6Config, params: EnvParams
) -> Tuple[float, EnvState]:
    setattr(state, "dr_max", config.dr_cap)
    setattr(state, "a_scale", config.a_max)
    state.r_feasible = r
    return r, state


def cg_v1_policy(
    r: float, state: EnvState, dist_scale: float, config: Phase6Config, params: EnvParams
) -> Tuple[float, EnvState]:
    if not hasattr(state, "sat_hist"):
        setattr(state, "sat_hist", [])
    sat_hist = getattr(state, "sat_hist")
    r_use, state = cg_v1_update(r, state, config, sat_hist)
    return r_use, state


def student_policy(
    r: float,
    state: EnvState,
    dist_scale: float,
    config: Phase6Config,
    params: EnvParams,
    model: ClassifierModel,
    action_grid: ActionGrid,
) -> Tuple[float, EnvState]:
    feat = np.array(
        [r, state.x, state.v, state.r_feasible, dist_scale, state.sat_prev], dtype=float
    )
    action_idx = model.predict_action(feat)
    dr_max, a_scale = action_to_params(action_grid, action_idx)
    r_use, next_state = apply_action(r, state, dr_max, a_scale, config)
    return r_use, next_state


def safety_filter(
    r: float,
    state: EnvState,
    config: Phase6Config,
) -> Tuple[float, float, bool]:
    e = r - state.x
    u_cmd = config.kp * e - config.kd * state.v
    sat_pred = abs(u_cmd) > config.u_max
    if not hasattr(state, "safety_hist"):
        setattr(state, "safety_hist", [])
    hist = getattr(state, "safety_hist")
    hist.append(1.0 if sat_pred else 0.0)
    if len(hist) > config.safety_window:
        hist.pop(0)
    sat_rate = float(np.mean(hist)) if hist else 0.0
    hold = getattr(state, "safety_hold", 0)
    if sat_rate > config.safety_sat_rate_hi:
        hold = config.safety_hold_steps
    hold = max(0, hold - 1)
    setattr(state, "safety_hold", hold)
    return sat_rate, hold, sat_pred


def safe_policy(
    base_policy_fn,
    r: float,
    state: EnvState,
    dist_scale: float,
    config: Phase6Config,
    params: EnvParams,
) -> Tuple[float, EnvState]:
    r_use, state = base_policy_fn(r, state, dist_scale, config, params)
    sat_rate, hold, _ = safety_filter(r_use, state, config)
    if hold > 0 or sat_rate > config.safety_sat_rate_hi:
        dr_max = max(config.dr_min, getattr(state, "dr_max", config.dr_cap) * config.safety_dr_scale)
        a_scale = max(config.a_min, getattr(state, "a_scale", config.a_max) * config.safety_a_scale)
        r_use, state = apply_action(r, state, dr_max, a_scale, config)
    return r_use, state


def adaptive_student_policy(
    r: float,
    state: EnvState,
    dist_scale: float,
    config: Phase6Config,
    params: EnvParams,
    model: ClassifierModel,
    action_grid: ActionGrid,
) -> Tuple[float, EnvState]:
    if not hasattr(state, "adapt_a_scale"):
        setattr(state, "adapt_a_scale", 1.0)
    base_r_use, state = student_policy(r, state, dist_scale, config, params, model, action_grid)
    adapt_scale = getattr(state, "adapt_a_scale")
    r_use = clamp(base_r_use * adapt_scale, -1.0, 1.0)
    err = abs(r_use - state.x)
    if err > config.adapt_target_err:
        adapt_scale = max(config.adapt_a_min, adapt_scale * (1.0 - config.adapt_lr))
    else:
        adapt_scale = min(config.adapt_a_max, adapt_scale * (1.0 + config.adapt_lr))
    setattr(state, "adapt_a_scale", adapt_scale)
    return r_use, state


def train_classifier(
    X: np.ndarray,
    y_idx: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    seed: int = 10,
) -> Tuple[ClassifierModel, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X = X[idx]
    y_idx = y_idx[idx]

    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    X_train, y_train = X[:n_train], y_idx[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y_idx[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y_idx[n_train + n_val :]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std
    X_test_n = (X_test - x_mean) / x_std

    n_features = X_train.shape[1]
    n_classes = int(np.max(y_idx)) + 1
    w = np.zeros((n_features, n_classes))
    b = np.zeros(n_classes)

    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def loss_and_grad(Xn: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        logits = Xn @ w + b
        probs = softmax(logits)
        y_onehot = np.eye(n_classes)[y]
        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))
        loss += l2 * np.sum(w**2)
        grad_logits = (probs - y_onehot) / len(Xn)
        grad_w = Xn.T @ grad_logits + 2.0 * l2 * w
        grad_b = np.sum(grad_logits, axis=0)
        return loss, grad_w, grad_b

    for _ in range(epochs):
        loss, grad_w, grad_b = loss_and_grad(X_train_n, y_train)
        if not math.isfinite(loss):
            break
        grad_norm = float(np.linalg.norm(grad_w))
        if grad_norm > 5.0:
            grad_w = grad_w * (5.0 / grad_norm)
        grad_b = np.clip(grad_b, -5.0, 5.0)
        w -= lr * grad_w
        b -= lr * grad_b

    def accuracy(Xn: np.ndarray, y: np.ndarray) -> float:
        preds = np.argmax(Xn @ w + b, axis=1)
        return float(np.mean(preds == y))

    metrics = {
        "train_acc": accuracy(X_train_n, y_train),
        "val_acc": accuracy(X_val_n, y_val),
        "test_acc": accuracy(X_test_n, y_test),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    model = ClassifierModel(w=w, b=b, x_mean=x_mean, x_std=x_std)
    return model, metrics
