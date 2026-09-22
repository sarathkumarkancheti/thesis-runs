#!/usr/bin/env python3
"""reset_shooting.py -- the supervisor's multi-window shooting WITH RESET
(notebook 'multi-window-shooting-with-reset.ipynb'), run on the npy datasets.

    python reset_shooting.py --data_dir <data> --out_dir runs_prof/reset
    python reset_shooting.py --data_dir <data> --out_dir runs_prof/reset --anchor_epochs 0

WHAT IS KEPT FROM THE NOTEBOOK (unchanged)
  state        [T, log10(Y + 1e-10)], min-max scaled to [-1, 1] (train cases only)
  target       symlog(physical dy/dt, eps = 1), min-max scaled to [-1, 1]
  network      (state, P) -> 3 x 256 SiLU -> Linear -> Tanh, Xavier init
  ODE          dz/dt in PHYSICAL time: decode tanh output -> inverse symlog ->
               chain rule. No log-time, no time input, no clock.
  split        seed 42, 70 / 15 / 15   (identical to the chemnode split)
  pretrain     MSE on the scaled derivative, Adam, lr 1e-4, batch 256, 2000 ep
  windows      sequential windows, overlap, last short window merged
  loss         Huber(delta 1): 1 x trajectory + 2 x endpoint + 0.5 x derivative
               evaluated at the PREDICTED states
  rejection    a window whose solve fails, is non-finite, or whose loss exceeds
               max_window_loss is dropped and counted (TrainRej / ValRej)
  solver       dopri5; stage 0 rtol 1e-5 atol 1e-7 max 2000 steps, max loss 1e6;
               later stages rtol 1e-3 atol 1e-5 max 1000 steps, max loss 20
  schedule     (4,1,50,1e-4) (10,3,200,5e-5) (15,3,200,1e-5) (20,3,200,2e-6)
  optimiser    Adam, constant lr per stage, grad clip 3, batch 32 windows,
               validate every 5 epochs, best-val checkpoint per stage

WHAT IS DIFFERENT, AND WHY
  * Windows are integrated as a BATCH instead of one odeint call per window
    (the notebook's loop would take hours per epoch on 300 x 550 points).
    Same maths: each data interval [t_k, t_k+1] of every window is integrated
    with dopri5 over s in [0, 1] with dz/ds = (t_k+1 - t_k) f(z). The error
    norm is the MAX over windows of each window's RMS error, so every window
    gets the accuracy it would get alone. If the batched solve fails, that
    batch is re-solved window by window, so rejection stays per window.
    max_num_steps therefore applies per data interval, not per window.
  * Integrating interval by interval also avoids float32 time collisions:
    the data has sample gaps down to 1e-14 s, which vanish at t ~ 1e-3 s in
    float32. Only the (float64-computed) interval lengths enter the solve.
  * Y is clipped at 0 before the log (the solver leaves ~ -1e-20 values).
  * ADDED: an 'anchor' stage -- one window per case covering the whole
    trajectory from the true t = 0 state (free rollout), same loss and
    rejection. --anchor_epochs 0 turns it off.
  * ADDED: after every stage, two evaluations on the validation cases:
      reset   T RMSE over all windows, each started from the true state
              (what the notebook reports)
      free    ONE integration from t = 0 to t_end per case (no reset):
              T RMSE, |T(t_end) error|, ignition-time error (half-rise, dex),
              mean |dlog10 Y| where true Y > 1e-6, |dY| of major species
    and at the end the same on the test cases.
  * RESUME: stages that finished (their _final.pt exists) are loaded, not
    retrained; an interrupted stage continues from <stage>_latest.pt (model,
    optimiser, best-so-far, RNG state, saved every epoch). The evaluation
    lines of finished stages and of the final model are stored in
    rollout_cache.json with the checkpoint's modification time, so after a
    restart they are printed from the cache instead of being recomputed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

SCHEDULE = [  # (window_length, overlap, epochs, lr, name)
    (4, 1, 50, 1e-4, "stage_0_warmup_window_5"),
    (10, 3, 200, 5e-5, "stage_1_multi_window_10"),
    (15, 3, 200, 1e-5, "stage_2_multi_window_15"),
    (20, 3, 200, 2e-6, "stage_3_multi_window_20"),
]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def symlog(x, eps):
    return np.sign(x) * np.log10(1.0 + np.abs(x) / eps)


def load_data(a, dev):
    d = Path(a.data_dir)
    st = np.load(d / "sampled_states_physical.npy").astype(np.float64)
    dv = np.load(d / "sampled_derivatives_physical.npy").astype(np.float64)
    t = np.load(d / "time_points_physical.npy").astype(np.float64)
    m = dict(np.load(d / "case_metadata.npz", allow_pickle=True))
    P = np.asarray(m["pressures"], dtype=np.float64)
    names = [str(s) for s in m["state_columns"]]
    C, K, D = st.shape
    assert (np.diff(t, axis=1) > 0).all(), "time must be strictly increasing"

    Y = np.clip(st[..., 1:], 0.0, None)
    if a.species == "log":
        L = np.log10(Y + a.y_eps)
        dL = dv[..., 1:] / (math.log(10.0) * (Y + a.y_eps))
    else:
        L, dL = Y, dv[..., 1:]
    s_raw = np.concatenate([st[..., :1], L], -1)
    d_raw = symlog(np.concatenate([dv[..., :1], dL], -1), a.deriv_symlog_eps)

    # split exactly as the notebook (and chemnode): seed 42, 70/15/15
    rng = np.random.default_rng(42)
    perm = rng.permutation(np.arange(C))
    n_tr, n_va = int(round(0.70 * C)), int(round(0.15 * C))
    tr, va, te = (np.sort(perm[:n_tr]), np.sort(perm[n_tr:n_tr + n_va]),
                  np.sort(perm[n_tr + n_va:]))

    smin, smax = s_raw[tr].reshape(-1, D).min(0), s_raw[tr].reshape(-1, D).max(0)
    srange = smax - smin
    srange[srange == 0] = 1.0
    dmin, dmax = d_raw[tr].reshape(-1, D).min(0), d_raw[tr].reshape(-1, D).max(0)
    drange = dmax - dmin
    drange[drange == 0] = 1.0
    pmin, pmax = P[tr].min(), P[tr].max()
    prange = pmax - pmin if pmax > pmin else 1.0

    f32 = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    return dict(
        C=C, K=K, D=D, names=names, t=t, raw=st, P=P, tr=tr, va=va, te=te,
        z=f32(2 * (s_raw - smin) / srange - 1),
        dz=f32(2 * (d_raw - dmin) / drange - 1),
        dt=f32(np.diff(t, axis=1)),                # float64 diff, then cast
        cond=f32(2 * (P - pmin) / prange - 1)[:, None],
        smin=smin, srange=srange, dmin=dmin, drange=drange,
        t_ign=np.asarray(m.get("ignition_times", np.full(C, np.nan)), float),
    )


def unscale(ds, z, a):
    """(…, D) scaled state -> T [K], Y (mass fraction)."""
    y = 0.5 * (z + 1.0) * ds["srange"] + ds["smin"]
    T = y[..., 0]
    Y = 10.0 ** np.minimum(y[..., 1:], 1.0) - a.y_eps if a.species == "log" \
        else y[..., 1:]
    return T, np.clip(Y, 0.0, None)


# ----------------------------------------------------------------------------
# model (notebook cells 14 and 21)
# ----------------------------------------------------------------------------
class DerivNet(nn.Module):
    def __init__(self, D, cond_dim=1, hidden=256, layers=3):
        super().__init__()
        mods, i = [], D + cond_dim
        for _ in range(layers):
            mods += [nn.Linear(i, hidden), nn.SiLU()]
            i = hidden
        mods += [nn.Linear(hidden, D), nn.Tanh()]
        self.network = nn.Sequential(*mods)
        for l in self.network:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, z, cond):
        if cond.dim() < z.dim() or cond.shape[0] != z.shape[0]:
            cond = cond.expand(*z.shape[:-1], cond.shape[-1])
        return self.network(torch.cat([z, cond], -1))


class ODEFunc(nn.Module):
    """dz/dt in physical time from the net's scaled, symlog-compressed output."""

    def __init__(self, net, ds, eps):
        super().__init__()
        self.vector_field = net
        dev = ds["z"].device
        self.register_buffer("srange", torch.tensor(ds["srange"], dtype=torch.float32, device=dev))
        self.register_buffer("dmin", torch.tensor(ds["dmin"], dtype=torch.float32, device=dev))
        self.register_buffer("drange", torch.tensor(ds["drange"], dtype=torch.float32, device=dev))
        self.eps = eps

    def dzdt(self, z, cond):
        o = self.vector_field(z, cond)
        s = 0.5 * (o + 1.0) * self.drange + self.dmin
        phys = torch.sign(s) * self.eps * (torch.pow(10.0, s.abs()) - 1.0)
        return 2.0 * phys / self.srange


class _Interval(nn.Module):
    """dz/ds = dt * f(z) on s in [0, 1]; dt is per row."""

    def __init__(self, ode, dt, cond):
        super().__init__()
        self.ode, self.dt, self.cond = ode, dt, cond

    def forward(self, s, z):
        return self.dt * self.ode.dzdt(z, self.cond)


# Dormand-Prince 5(4) tableau -- the same method as torchdiffeq's 'dopri5'
_A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
      [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
_B5 = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]
_B4 = [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
_E = [b5 - b4 for b5, b4 in zip(_B5, _B4)]


def _rms(x):
    return x.pow(2).mean(-1).sqrt()


def dopri5_rows(f, y0, rtol, atol, max_steps):
    """Integrate dy/ds = f(y, rows) from s = 0 to s = 1 for every row of y0
    (B, D) with its OWN adaptive step -- one stiff row no longer forces small
    steps (or a failure) on the whole batch, exactly as if each row had its
    own odeint call. Step control as torchdiffeq: RMS error norm, safety 0.9,
    factor limits 0.2 / 10, order 5. A row that needs more than max_steps
    steps, or whose error becomes non-finite, is FROZEN at its last accepted
    state and flagged failed (never NaN inside the graph, so it cannot poison
    the gradients of the other rows).
    Returns (y1 (B, D), failed (B,) bool)."""
    B, dev = y0.shape[0], y0.device
    y = y0
    s = torch.zeros(B, device=dev, dtype=torch.float64)
    nstep = torch.zeros(B, device=dev, dtype=torch.long)
    failed = torch.zeros(B, dtype=torch.bool, device=dev)
    allr = torch.arange(B, device=dev)
    k1 = f(y, allr)
    # initial step (Hairer, Norsett & Wanner II.4), per row
    with torch.no_grad():
        sc = atol + rtol * y.abs()
        d0, d1 = _rms(y / sc), _rms(k1 / sc)
        h0 = torch.where((d0 < 1e-5) | (d1 < 1e-5), torch.full_like(d0, 1e-6), 0.01 * d0 / d1)
        k2 = f(y + h0[:, None] * k1, allr)
        d2 = _rms((k2 - k1) / sc) / h0
        dm = torch.maximum(d1, d2)
        h1 = torch.where(dm <= 1e-15, torch.maximum(torch.full_like(h0, 1e-6), h0 * 1e-3),
                         (0.01 / dm.clamp_min(1e-30)) ** 0.2)
        h = torch.minimum(100 * h0, h1).double().clamp(max=1.0)
        bad = ~torch.isfinite(h) | (h <= 0)
        failed |= bad
        h = torch.where(bad, torch.ones_like(h), h)
    act = allr[~failed]
    while act.numel():
        hr = torch.minimum(h[act], 1.0 - s[act])
        hf = hr.float()[:, None]
        ya, k = y[act], [k1[act]]
        for i in range(1, 7):
            yi = ya + hf * sum(c * kj for c, kj in zip(_A[i], k) if c != 0.0)
            k.append(f(yi, act))
        y5 = yi                                   # stage 7 point = 5th-order solution
        err = hf * sum(c * kj for c, kj in zip(_E, k) if c != 0.0)
        with torch.no_grad():
            tol = atol + rtol * torch.maximum(ya.abs(), y5.abs())
            ratio = _rms(err / tol)
            fin = torch.isfinite(ratio) & torch.isfinite(y5).all(-1)
            acc = fin & (ratio <= 1.0)
            fac = torch.where(ratio == 0, torch.full_like(ratio, 10.0),
                              (0.9 * ratio.clamp_min(1e-30) ** -0.2).clamp(0.2, 10.0))
            fac = torch.where(acc, fac, fac.clamp(max=1.0))
            nstep[act] += 1
        ia = act[acc]
        if ia.numel():
            y = y.index_copy(0, ia, y5[acc])
            k1 = k1.index_copy(0, ia, k[6][acc])  # FSAL
            s[ia] += hr[acc]
        with torch.no_grad():
            h[act] = (hr * fac.double()).clamp(min=1e-300)
            failed[act[~fin]] = True
            failed |= (nstep >= max_steps) & (s < 1.0 - 1e-12)
            act = act[(s[act] < 1.0 - 1e-12) & ~failed[act]]
    return y, failed


def integrate(ode, z0, dt, cond, rtol, atol, max_steps):
    """z0 (B, D), dt (B, L-1), cond (B, 1) -> (pred (B, L, D), failed (B,)).
    Each data interval: dz/ds = dt_k f(z) over s in [0, 1], per-row dopri5.
    A row that fails in one interval stays frozen for the rest."""
    out, z = [z0], z0
    failed = torch.zeros(z0.shape[0], dtype=torch.bool, device=z0.device)
    for k in range(dt.shape[1]):
        dtk = dt[:, k:k + 1]
        live = (~failed).nonzero().flatten()
        if live.numel():
            dl, cl = dtk[live], cond[live]
            zl, fl = dopri5_rows(lambda y, r: dl[r] * ode.dzdt(y, cl[r]),
                                 z[live], rtol, atol, max_steps)
            z = z.index_copy(0, live, zl)
            failed = failed.index_copy(0, live, fl)
        out.append(z)
    return torch.stack(out, 1), failed


# ----------------------------------------------------------------------------
# windows (notebook cell 9)
# ----------------------------------------------------------------------------
def build_sequential_windows(number_points, window_length, overlap_points=3,
                             min_last_window_ratio=0.5):
    """Verbatim from the notebook. (start, end) with end EXCLUSIVE."""
    windows = []
    start = 0
    while start < number_points - 1:
        end = min(start + window_length, number_points)
        if end == number_points:
            last_window_size = end - start
            if last_window_size < min_last_window_ratio * window_length and len(windows) > 0:
                prev_start, prev_end = windows[-1]
                windows[-1] = (prev_start, end)
                break
        windows.append((start, end))
        if end == number_points:
            break
        start = end - overlap_points
    return windows


def build_windows(cases, K, L, overlap):
    L_eff = min(L, K)
    ov = min(overlap, L_eff - 1)
    return [(c, s, e) for c in cases for (s, e) in build_sequential_windows(K, L_eff, ov)]


# ----------------------------------------------------------------------------
# loss on a list of windows (notebook cell 24, batched)
# ----------------------------------------------------------------------------
def huber_rows(p, q, delta):
    return F.huber_loss(p, q, delta=delta, reduction="none").reshape(p.shape[0], -1).mean(1)


def windows_loss(ode, ds, wins, sv, a):
    """Per-window loss terms for windows of equal length. Returns dict of (B,)
    tensors (NaN where rejected by the solver) and the prediction."""
    dev = ds["z"].device
    Lw = wins[0][2] - wins[0][1]
    assert all(w[2] - w[1] == Lw for w in wins), "windows of one call share a length"
    c = torch.tensor([w[0] for w in wins], device=dev)[:, None]          # (B, 1)
    k = (torch.tensor([w[1] for w in wins], device=dev)[:, None]
         + torch.arange(Lw, device=dev)[None])                            # (B, Lw)
    zt = ds["z"][c, k]
    dzt = ds["dz"][c, k]
    dt = ds["dt"][c, k[:, :-1]]
    cond = ds["cond"][c[:, 0]]
    pred, failed = integrate(ode, zt[:, 0], dt, cond, sv["rtol"], sv["atol"],
                             sv["max_steps"])
    B, Lw, D = pred.shape
    traj = huber_rows(pred, zt, a.huber_delta)
    end = huber_rows(pred[:, -1], zt[:, -1], a.huber_delta)
    q = pred if a.derivative_loss_target == "predicted" else zt
    dp = ode.vector_field(q.reshape(-1, D), cond.repeat_interleave(Lw, 0)).reshape(B, Lw, D)
    der = huber_rows(dp, dzt, a.huber_delta)
    total = a.w_traj * traj + a.w_end * end + a.w_deriv * der
    nan = torch.full_like(total, float("nan"))
    total = torch.where(failed, nan, total)          # solver failure -> rejected
    Tsq = (0.5 * (pred[..., 0] - zt[..., 0]) * float(ds["srange"][0])).pow(2).mean(1)
    return {"total": total, "traj": traj, "end": end, "deriv": der, "Tsq": Tsq}


def run_windows(ode, ds, wins, sv, a, max_loss, train=False, opt=None, bs=32):
    """One pass over windows. Batches of `bs`, split by window length inside
    a batch (the merged last window of a case is longer)."""
    tot = dict(total=0.0, traj=0.0, end=0.0, deriv=0.0, Tsq=0.0)
    n_ok = n_rej = 0
    gsum = 0.0
    for b0 in range(0, len(wins), bs):
        batch = wins[b0:b0 + bs]
        groups = {}
        for w in batch:
            groups.setdefault(w[2] - w[1], []).append(w)
        losses = []
        for g in groups.values():
            with torch.set_grad_enabled(train):
                r = windows_loss(ode, ds, g, sv, a)
            if r is None:
                n_rej += len(g)
                continue
            ok = torch.isfinite(r["total"]) & (r["total"] <= max_loss)
            n_rej += int((~ok).sum())
            n_ok += int(ok.sum())
            if ok.any():
                for k in tot:
                    tot[k] += float(r[k][ok].detach().sum())
                losses.append(r["total"][ok])
        if train and losses:
            loss = torch.cat(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(ode.parameters(), a.max_grad_norm)
            if torch.isfinite(gn):
                gsum += float(gn)
                opt.step()
    m = {k: (v / n_ok if n_ok else float("inf")) for k, v in tot.items()}
    m["T_rmse"] = math.sqrt(m["Tsq"]) if n_ok else float("inf")
    m.update(valid=n_ok, rej=n_rej, grad=gsum / max(1, n_ok))
    return m


# ----------------------------------------------------------------------------
# free rollout from t = 0 (no reset)
# ----------------------------------------------------------------------------
@torch.no_grad()
def free_rollout(ode, ds, idx, a, bs=None):
    bs = bs or a.eval_batch
    rows, failed = [], 0
    sv = dict(rtol=a.eval_rtol, atol=a.eval_atol, max_steps=a.eval_max_steps)
    for b0 in range(0, len(idx), bs):
        cs = list(idx[b0:b0 + bs])
        c = torch.tensor(cs, device=ds["z"].device)
        p, fl = integrate(ode, ds["z"][c, 0], ds["dt"][c], ds["cond"][c], **_sv(sv))
        failed += int(fl.sum())
        preds, pc = [p[~fl]], [[ci for ci, f_ in zip(cs, fl.tolist()) if not f_]]
        for p, cl in zip(preds, pc):
            p = p.cpu().numpy()
            for j, ci in enumerate(cl):
                if not np.isfinite(p[j]).all():
                    failed += 1
                    continue
                rows.append(case_metrics(ds, ci, p[j], a))
    return summarize(rows, failed, len(idx))


def _sv(sv):
    return dict(rtol=sv["rtol"], atol=sv["atol"], max_steps=sv["max_steps"])


def case_metrics(ds, c, zp, a):
    Tp, Yp = unscale(ds, zp, a)
    Tt, Yt = ds["raw"][c, :, 0], np.clip(ds["raw"][c, :, 1:], 0, None)
    t = ds["t"][c]
    thr = Tt[0] + 0.5 * (Tt.max() - Tt[0])

    def cross(T):
        k = np.flatnonzero(T >= thr)
        return t[k[0]] if k.size else np.nan

    ti_t, ti_p = cross(Tt), cross(Tp)
    m6 = Yt > 1e-6
    lp = np.log10(np.maximum(Yp, 1e-12))   # same floor as the chemnode metrics
    maj = Yt.max(0) > 1e-3
    return dict(
        T_rmse=float(np.sqrt(np.mean((Tp - Tt) ** 2))),
        T_end=float(abs(Tp[-1] - Tt[-1])),
        ign_dex=float(abs(np.log10(ti_p / ti_t))) if np.isfinite(ti_p) and ti_t > 0 else np.nan,
        logY=float(np.mean(np.abs(lp[m6] - np.log10(Yt[m6])))) if m6.any() else np.nan,
        majors=float(100 * np.mean(np.abs(Yp[:, maj] - Yt[:, maj]))),
        Ymax=float(Yp.max()),
    )


def summarize(rows, failed, n):
    if not rows:
        return dict(n=n, ok=0, failed=failed)
    g = lambda k, f=np.nanmean: float(f([r[k] for r in rows]))  # noqa: E731
    return dict(n=n, ok=len(rows), failed=failed,
                T_rmse=g("T_rmse"), T_rmse_med=g("T_rmse", np.nanmedian),
                T_end_med=g("T_end", np.nanmedian),
                ign_dex_med=g("ign_dex", np.nanmedian),
                ign_missed=int(sum(np.isnan(r["ign_dex"]) for r in rows)),
                logY=g("logY"), majors=g("majors"), Ymax=g("Ymax", np.max))


def fmt_free(m):
    if not m.get("ok"):
        return f"free rollout from t=0: all {m['n']} cases failed"
    return (f"free rollout from t=0: {m['ok']}/{m['n']} ok ({m['failed']} failed) | "
            f"T RMSE {m['T_rmse']:.1f} K (median {m['T_rmse_med']:.1f}) | "
            f"|T(t_end)| med {m['T_end_med']:.1f} K | t_ign err med "
            f"{m['ign_dex_med']:.3f} dex, {m['ign_missed']} missed | "
            f"|dlog10 Y|(Y>1e-6) {m['logY']:.3f} | |dY| majors {m['majors']:.2f} mass-% | "
            f"max Y {m['Ymax']:.3g}")


# ----------------------------------------------------------------------------
# training (notebook cell 26)
# ----------------------------------------------------------------------------
def solver_settings(name, a):
    if name.startswith("stage_0"):
        return dict(rtol=1e-5, atol=1e-7, max_steps=a.max_steps), 1e6
    return dict(rtol=a.rtol, atol=a.atol, max_steps=a.max_steps), a.max_window_loss


def train_stage(ode, ds, a, L, overlap, epochs, lr, name, bs, log, out):
    fin = out / f"{name}_final.pt"
    if fin.exists():
        ode.load_state_dict(torch.load(fin, map_location=ds["z"].device,
                                       weights_only=False)["model_state"])
        log(f"[{name}] already complete -- loaded {fin.name}")
        return
    sv, max_loss = solver_settings(name, a)
    trw = build_windows(ds["tr"], ds["K"], L, overlap)
    vaw = build_windows(ds["va"], ds["K"], L, overlap)
    log("=" * 72)
    log(f"{name}: window length {L}, overlap {overlap}, {epochs} epochs, lr {lr:g}, "
        f"batch {bs} | rtol {sv['rtol']:g} atol {sv['atol']:g} max_steps "
        f"{sv['max_steps']} (per interval) max_window_loss {max_loss:g}")
    log(f"train cases / windows {len(ds['tr'])} / {len(trw)}   "
        f"valid cases / windows {len(ds['va'])} / {len(vaw)}")
    log("=" * 72)
    opt = torch.optim.Adam(ode.parameters(), lr=lr)
    best, best_state = float("inf"), copy.deepcopy(ode.state_dict())
    vm = dict(total=float("inf"), valid=0, rej=0, T_rmse=float("inf"))
    latest = out / f"{name}_latest.pt"
    start = 1
    if latest.exists():                        # interrupted mid-stage: continue
        ck = torch.load(latest, map_location=ds["z"].device, weights_only=False)
        ode.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt"])
        best, best_state, vm = ck["best"], ck["best_state"], ck["vm"]
        np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"])
        start = ck["epoch"] + 1
        log(f"[{name}] resuming from {latest.name} at epoch {start}")
    improved = False            # new best val since the last printed line -> '*'
    for ep in range(start, epochs + 1):
        t0 = time.time()
        ode.train()
        perm = np.random.permutation(len(trw))
        m = run_windows(ode, ds, [trw[i] for i in perm], sv, a, max_loss,
                        train=True, opt=opt, bs=bs)
        if ep % a.validate_every == 0 or ep == epochs:
            ode.eval()
            vm = run_windows(ode, ds, vaw, sv, a, max_loss, bs=a.eval_batch)
            if vm["total"] < best:
                best, best_state = vm["total"], copy.deepcopy(ode.state_dict())
                improved = True
                torch.save({"epoch": ep, "model_state": best_state, "val_loss": best},
                           out / f"{name}_best.pt")
        if ep == 1 or ep % a.print_every == 0 or ep == epochs:
            log(f"Epoch {ep:4d}/{epochs:4d} | Train={m['total']:.6e} | Val={vm['total']:.6e} | "
                f"Traj={m['traj']:.6e} | End={m['end']:.6e} | Deriv={m['deriv']:.6e} | "
                f"Grad={m['grad']:.3e} | TrainValid={m['valid']:5d} TrainRej={m['rej']:5d} | "
                f"ValValid={vm['valid']:5d} ValRej={vm['rej']:5d} | LR={lr:.2e} | "
                f"BestVal={best:.6e} | {time.time() - t0:.1f}s" + ("  *" if improved else ""))
            improved = False
        seen = m["valid"] + m["rej"]
        if seen and m["rej"] / seen > 0.5:
            log(f"  WARNING: {m['rej']}/{seen} training windows rejected this epoch (>50%)")
        torch.save({"epoch": ep, "model_state": ode.state_dict(), "opt": opt.state_dict(),
                    "best": best, "best_state": best_state, "vm": vm,
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()},
                   latest)
    ode.load_state_dict(best_state)
    torch.save({"model_state": best_state, "val_loss": best, "stage_name": name}, fin)
    latest.unlink(missing_ok=True)
    log(f"[{name}] best validation loss {best:.6e}")


def _cache(out):
    p = out / "rollout_cache.json"
    try:
        return p, json.loads(p.read_text())
    except (OSError, ValueError):
        return p, {}


def cached_report(out, key, ckpt, log, compute):
    """Print the evaluation lines for `key`. If `ckpt` is unchanged since they
    were computed (same mtime) they come from rollout_cache.json instead of
    being recomputed -- a restart does not re-evaluate finished stages."""
    p, cache = _cache(out)
    mt = ckpt.stat().st_mtime if ckpt.exists() else None
    e = cache.get(key)
    if mt is not None and e and abs(e.get("mtime", -1) - mt) < 1e-3:
        for line in e["lines"]:
            log(line + "   (cached)")
        return
    lines = compute()
    for line in lines:
        log(line)
    if mt is not None:
        cache[key] = {"mtime": mt, "lines": lines}
        p.write_text(json.dumps(cache, indent=1))


def stage_report(ode, ds, a, L, overlap, name, log, out):
    def compute():
        ode.eval()
        sv, max_loss = solver_settings(name, a)
        vaw = build_windows(ds["va"], ds["K"], L, overlap)
        r = run_windows(ode, ds, vaw, sv, a, max_loss, bs=a.eval_batch)
        lines = [f"  -> val with reset (window {min(L, ds['K'])}): T RMSE "
                 f"{r['T_rmse']:.1f} K over {r['valid']} windows, {r['rej']} rejected"]
        if not a.no_free_eval:
            lines.append("  -> val " + fmt_free(free_rollout(ode, ds, ds["va"], a)))
        return lines
    key = f"{name}|free={not a.no_free_eval}"
    cached_report(out, key, out / f"{name}_final.pt", log, compute)


# ----------------------------------------------------------------------------
def pretrain(net, ds, a, log, out):
    p = out / "pretrain.pt"
    if p.exists():
        net.load_state_dict(torch.load(p, map_location=ds["z"].device))
        log("pretrain: already complete -- loaded pretrain.pt")
        return
    D = ds["D"]
    flat = lambda idx: (ds["z"][idx].reshape(-1, D),  # noqa: E731
                        ds["cond"][idx].repeat_interleave(ds["K"], 0),
                        ds["dz"][idx].reshape(-1, D))
    (xs, xc, y), (vs, vc, vy) = flat(ds["tr"]), flat(ds["va"])
    opt = torch.optim.Adam(net.parameters(), lr=a.pretrain_lr)
    best, best_state = float("inf"), copy.deepcopy(net.state_dict())
    n = xs.shape[0]
    for ep in range(1, a.pretrain_epochs + 1):
        net.train()
        perm = torch.randperm(n, device=xs.device)
        for i in range(0, n, a.pretrain_batch):
            j = perm[i:i + a.pretrain_batch]
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(net(xs[j], xc[j]), y[j])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            tl = F.mse_loss(net(xs, xc), y).item()
            vl = F.mse_loss(net(vs, vc), vy).item()
        if vl < best:
            best, best_state = vl, copy.deepcopy(net.state_dict())
        if ep == 1 or ep % 100 == 0:
            log(f"Pretrain epoch {ep:4d} | train MSE = {tl:.6e} | val MSE = {vl:.6e}")
        if vl < 1e-6:
            log(f"Pretraining converged at epoch {ep}.")
            break
    net.load_state_dict(best_state)
    torch.save(best_state, p)


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="runs_prof/reset")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--species", default="log", choices=("log", "linear"),
                    help="notebook: log10(Y + 1e-10). 'linear' = raw Y, min-max scaled")
    ap.add_argument("--y_eps", type=float, default=1e-10)
    ap.add_argument("--deriv_symlog_eps", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--pretrain_epochs", type=int, default=2000)
    ap.add_argument("--pretrain_lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=256,
                    help="windows per optimizer step (notebook: 32). Larger = fewer "
                         "solver calls per epoch, but also fewer optimizer steps")
    ap.add_argument("--pretrain_batch", type=int, default=2048,
                    help="points per pretraining step (notebook: 256)")
    ap.add_argument("--eval_batch", type=int, default=512,
                    help="windows / cases per solve in validation and evaluation "
                         "(no gradients, so this only affects speed)")
    ap.add_argument("--epochs_scale", type=float, default=1.0,
                    help="multiply every stage's epochs (e.g. 0.25 for a quick run)")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--max_window_loss", type=float, default=20.0)
    ap.add_argument("--max_grad_norm", type=float, default=3.0)
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--w_traj", type=float, default=1.0)
    ap.add_argument("--w_end", type=float, default=2.0)
    ap.add_argument("--w_deriv", type=float, default=0.5)
    ap.add_argument("--derivative_loss_target", default="predicted",
                    choices=("predicted", "true"))
    ap.add_argument("--validate_every", type=int, default=5)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--anchor_epochs", type=int, default=50,
                    help="ANCHOR stage, run after stage 3 (ON by default): every "
                         "case's whole trajectory from the true t=0 state as ONE "
                         "window, same loss and rejection (TrainRej/ValRej). 0 = off")
    ap.add_argument("--anchor_lr", type=float, default=1e-6)
    ap.add_argument("--anchor_batch", type=int, default=32)
    ap.add_argument("--eval_rtol", type=float, default=1e-5)
    ap.add_argument("--eval_atol", type=float, default=1e-7)
    ap.add_argument("--eval_max_steps", type=int, default=5000)
    ap.add_argument("--no_free_eval", action="store_true",
                    help="skip the free-rollout evaluation after each stage")
    a = ap.parse_args(argv)

    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if a.device == "auto" else a.device)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    logf = open(out / "train.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    ds = load_data(a, dev)
    log(f"device {dev} | {ds['C']} cases x {ds['K']} points, state dim {ds['D']} "
        f"({ds['names'][0]} + {ds['D'] - 1} species) | species '{a.species}'")
    log(f"split (seed 42): train {len(ds['tr'])} val {len(ds['va'])} test {len(ds['te'])}")
    log(f"time: physical seconds, t in [0, {ds['t'].max():.3g}] s, smallest sample "
        f"gap {np.diff(ds['t'], axis=1).min():.2e} s")
    net = DerivNet(ds["D"], 1, a.hidden, a.layers).to(dev)
    ode = ODEFunc(net, ds, a.deriv_symlog_eps).to(dev)
    log(f"parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.3f} M")

    pretrain(net, ds, a, log, out)
    stages = [(L, ov, max(1, int(round(ep * a.epochs_scale))), lr, nm)
              for L, ov, ep, lr, nm in SCHEDULE]
    if a.anchor_epochs > 0:
        stages.append((ds["K"], 1, max(1, int(round(a.anchor_epochs * a.epochs_scale))),
                       a.anchor_lr, "anchor_full_trajectory"))
    log("stages: " + " -> ".join(f"{nm} (L={min(L, ds['K'])}, {ep} ep)"
                                 for L, ov, ep, lr, nm in stages))
    for L, ov, ep, lr, nm in stages:
        bs = a.anchor_batch if nm.startswith("anchor") else a.batch_size
        train_stage(ode, ds, a, L, ov, ep, lr, nm, bs, log, out)
        stage_report(ode, ds, a, L, ov, nm, log, out)

    ode.eval()
    log("\n=== final model (last stage: " + stages[-1][4] + ") ===")
    L, ov = SCHEDULE[-1][0], SCHEDULE[-1][1]

    def compute_final():
        lines = []
        for split, idx in (("val", ds["va"]), ("test", ds["te"])):
            sv, ml = solver_settings("final", a)
            r = run_windows(ode, ds, build_windows(idx, ds["K"], L, ov), sv, a, ml, bs=a.eval_batch)
            lines.append(f"{split}: with reset (window {L}): T RMSE {r['T_rmse']:.1f} K, "
                         f"{r['valid']} windows, {r['rej']} rejected")
            lines.append(f"{split}: " + fmt_free(free_rollout(ode, ds, idx, a)))
        return lines
    cached_report(out, "final", out / f"{stages[-1][4]}_final.pt", log, compute_final)
    torch.save({"model_state": ode.state_dict(), "args": vars(a)}, out / "final.pt")


if __name__ == "__main__":
    main()
