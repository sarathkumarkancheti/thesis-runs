#!/usr/bin/env python3
"""full_trajectory.py -- the professor's Optuna-script NODE (optuna_search v2),
run on the npy datasets: every case integrated from its t = 0 state over the
whole trajectory in one go, loss = MSE on the normalized states.

    python full_trajectory.py --data_dir <data> --out_dir runs_prof/full
    python full_trajectory.py --data_dir <data> --out_dir runs_prof/full --grid uniform --n_uniform 80

WHAT IS KEPT FROM THE SCRIPT (unchanged)
  network      [Linear -> BatchNorm1d -> act] x n_layers -> Linear (no output
               activation). Defaults = the v1 sweet spot the script quotes:
               hidden 128, 4 layers, SiLU, weight decay 0, lr 7e-4
  BatchNorm    in eval mode during integration; running statistics refreshed
               from all training states at the start and every 20 epochs
  time         t normalized linearly to [0, 1] (t / t_max); the network learns
               dz/dtau directly -- no log-time, no time input, no derivatives
  training     one optimizer step per epoch on the mean loss over all training
               cases; AdamW; grad clip 1.0; ReduceLROnPlateau(0.5, patience 50,
               min 1e-6) on the validation loss
  validation   every 10 epochs; early stop after 80 epochs without improvement
  solver       dopri5, rtol 1e-5, atol 1e-7

WHAT IS DIFFERENT, AND WHY
  * State: the script loads a ready-made 'normalized_data.npy' whose recipe is
    not in the script. Here: [T, log10(Y + 1e-10)] min-max scaled to [-1, 1]
    from the training cases -- the same as the supervisor's notebook, so the
    two codes differ only in the method. --species linear uses raw Y instead.
  * Conditioning: the script's data had no pressure input. This data spans
    49-73 bar, so P (min-max scaled) is appended to the input. --no_cond
    removes it.
  * Grid: the script used a uniform grid of 40-80 points. Default here is the
    data's own 550 sample times (they resolve ignition); --grid uniform
    resamples each case onto n_uniform equally spaced times in [0, t_end]
    (linear interpolation of the scaled state), as the script's data was.
    Evaluation is ALWAYS on the data's own times, so both are comparable.
  * One optimizer step per epoch is kept, but the loss is accumulated over
    mini-batches of cases (--case_batch) because 300 full trajectories do not
    fit in memory at once. Gradient = the same mean over valid cases.
  * The script abandoned the whole trial when any case failed. Here a failed
    or non-finite case is rejected and counted (TrainRej / ValRej), exactly
    like the notebook's windows; the epoch continues with the others.
  * Integration interval by interval (dz/ds = dtau_k f(z) over s in [0, 1]),
    batched over cases with a per-case error norm -- the same maths as one
    odeint call per case, without float32 time collisions (sample gaps go
    down to 1e-14 s). --max_steps caps each interval (the script had no cap;
    a stiff field would otherwise hang).
  * ADDED: the same free-rollout metrics as reset_shooting.py, on val every
    --eval_every epochs and on val and test at the end.
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
from torchdiffeq import odeint, odeint_adjoint

ACT = {"silu": nn.SiLU, "relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.2)}


# ----------------------------------------------------------------------------
def load_data(a, dev):
    d = Path(a.data_dir)
    st = np.load(d / "sampled_states_physical.npy").astype(np.float64)
    t = np.load(d / "time_points_physical.npy").astype(np.float64)
    m = dict(np.load(d / "case_metadata.npz", allow_pickle=True))
    P = np.asarray(m["pressures"], dtype=np.float64)
    names = [str(s) for s in m["state_columns"]]
    C, K, D = st.shape
    assert (np.diff(t, axis=1) > 0).all(), "time must be strictly increasing"
    Y = np.clip(st[..., 1:], 0.0, None)
    L = np.log10(Y + a.y_eps) if a.species == "log" else Y
    s_raw = np.concatenate([st[..., :1], L], -1)

    rng = np.random.default_rng(42)                    # same split as the notebook
    perm = rng.permutation(np.arange(C))
    n_tr, n_va = int(round(0.70 * C)), int(round(0.15 * C))
    tr, va, te = (np.sort(perm[:n_tr]), np.sort(perm[n_tr:n_tr + n_va]),
                  np.sort(perm[n_tr + n_va:]))
    smin = s_raw[tr].reshape(-1, D).min(0)
    srange = s_raw[tr].reshape(-1, D).max(0) - smin
    srange[srange == 0] = 1.0
    z = 2 * (s_raw - smin) / srange - 1
    t_scale = float(t[tr].max())                       # tau = t / t_max in [0, 1]
    pmin, pmax = P[tr].min(), P[tr].max()
    prange = pmax - pmin if pmax > pmin else 1.0

    # training grid
    if a.grid == "uniform":
        tg = np.stack([np.linspace(0.0, t[c, -1], a.n_uniform) for c in range(C)])
        zg = np.stack([np.stack([np.interp(tg[c], t[c], z[c, :, j]) for j in range(D)], -1)
                       for c in range(C)])
    else:
        tg, zg = t, z
    f32 = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    return dict(C=C, K=K, D=D, names=names, t=t, raw=st, tr=tr, va=va, te=te,
                smin=smin, srange=srange, t_scale=t_scale,
                z=f32(z), dtau=f32(np.diff(t, axis=1) / t_scale),
                zg=f32(zg), dtau_g=f32(np.diff(tg, axis=1) / t_scale),
                cond=f32(2 * (P - pmin) / prange - 1)[:, None])


def unscale(ds, z, a):
    y = 0.5 * (z + 1.0) * ds["srange"] + ds["smin"]
    Y = 10.0 ** np.minimum(y[..., 1:], 1.0) - a.y_eps if a.species == "log" else y[..., 1:]
    return y[..., 0], np.clip(Y, 0.0, None)


# ----------------------------------------------------------------------------
class ODEFunc(nn.Module):
    """dz/dtau = MLP(z[, P]) with BatchNorm after every hidden Linear."""

    def __init__(self, D, cond_dim, hidden, n_layers, act):
        super().__init__()
        mods, i = [], D + cond_dim
        for _ in range(n_layers):
            mods += [nn.Linear(i, hidden), nn.BatchNorm1d(hidden), ACT[act]()]
            i = hidden
        mods.append(nn.Linear(hidden, D))
        self.net = nn.Sequential(*mods)
        self.cond_dim = cond_dim

    def inp(self, z, cond):
        if not self.cond_dim:
            return z
        return torch.cat([z, cond.expand(z.shape[0], -1) if cond.shape[0] != z.shape[0]
                          else cond], -1)

    def f(self, z, cond):
        return self.net(self.inp(z, cond))


def set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()


@torch.no_grad()
def warmup_bn_stats(model, ds):
    """Refresh BN running stats from all training states (script: every 20 ep)."""
    model.train()
    D = ds["D"]
    x = ds["zg"][ds["tr"]].reshape(-1, D)
    c = ds["cond"][ds["tr"]].repeat_interleave(ds["zg"].shape[1], 0)
    for i in range(0, x.shape[0], 2048):
        model.net(model.inp(x[i:i + 2048], c[i:i + 2048]))


class _Interval(nn.Module):
    def __init__(self, ode, dtau, cond):
        super().__init__()
        self.ode, self.dtau, self.cond = ode, dtau, cond

    def forward(self, s, z):
        return self.dtau * self.ode.f(z, self.cond)


def integrate(ode, z0, dtau, cond, a, rtol=None, atol=None, max_steps=None, adjoint=False):
    B = z0.shape[0]
    s01 = torch.tensor([0.0, 1.0], device=z0.device)

    def norm(x):
        return x.reshape(B, -1).pow(2).mean(1).sqrt().max()

    opts = {"max_num_steps": max_steps or a.max_steps, "norm": norm}
    out, z = [z0], z0
    for k in range(dtau.shape[1]):
        fn = _Interval(ode, dtau[:, k:k + 1], cond)
        if adjoint:
            z = odeint_adjoint(fn, z, s01, method="dopri5", rtol=rtol or a.rtol,
                               atol=atol or a.atol, options=opts,
                               adjoint_params=tuple(ode.parameters()),
                               adjoint_options={"norm": "seminorm"})[-1]
        else:
            z = odeint(fn, z, s01, method="dopri5", rtol=rtol or a.rtol,
                       atol=atol or a.atol, options=opts)[-1]
        out.append(z)
    return torch.stack(out, 1)


def solve_cases(ode, ds, cases, a, grid="train", **kw):
    """Integrate cases; on a batch failure fall back to one case at a time.
    Returns list of (case, prediction or None)."""
    zs, dts = (ds["zg"], ds["dtau_g"]) if grid == "train" else (ds["z"], ds["dtau"])
    c = torch.tensor(list(cases), device=zs.device)
    try:
        p = integrate(ode, zs[c, 0], dts[c], ds["cond"][c], a, **kw)
        return [(ci, p[j]) for j, ci in enumerate(cases)]
    except (AssertionError, RuntimeError):
        if len(cases) == 1:
            return [(cases[0], None)]
        out = []
        for ci in cases:
            out += solve_cases(ode, ds, [ci], a, grid, **kw)
        return out


# ----------------------------------------------------------------------------
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
        T_rmse=float(np.sqrt(np.mean((Tp - Tt) ** 2))), T_end=float(abs(Tp[-1] - Tt[-1])),
        ign_dex=float(abs(np.log10(ti_p / ti_t))) if np.isfinite(ti_p) and ti_t > 0 else np.nan,
        logY=float(np.mean(np.abs(lp[m6] - np.log10(Yt[m6])))) if m6.any() else np.nan,
        majors=float(100 * np.mean(np.abs(Yp[:, maj] - Yt[:, maj]))), Ymax=float(Yp.max()))


@torch.no_grad()
def free_rollout(ode, ds, idx, a):
    ode.eval()
    rows, failed = [], 0
    for b0 in range(0, len(idx), a.case_batch):
        for ci, p in solve_cases(ode, ds, list(idx[b0:b0 + a.case_batch]), a, grid="data",
                                 rtol=a.eval_rtol, atol=a.eval_atol,
                                 max_steps=a.eval_max_steps):
            if p is None or not torch.isfinite(p).all():
                failed += 1
                continue
            rows.append(case_metrics(ds, ci, p.cpu().numpy(), a))
    if not rows:
        return f"free rollout from t=0: all {len(idx)} cases failed"
    g = lambda k, f=np.nanmean: float(f([r[k] for r in rows]))  # noqa: E731
    return (f"free rollout from t=0: {len(rows)}/{len(idx)} ok ({failed} failed) | "
            f"T RMSE {g('T_rmse'):.1f} K (median {g('T_rmse', np.nanmedian):.1f}) | "
            f"|T(t_end)| med {g('T_end', np.nanmedian):.1f} K | t_ign err med "
            f"{g('ign_dex', np.nanmedian):.3f} dex, "
            f"{int(sum(np.isnan(r['ign_dex']) for r in rows))} missed | "
            f"|dlog10 Y|(Y>1e-6) {g('logY'):.3f} | |dY| majors {g('majors'):.2f} mass-% | "
            f"max Y {g('Ymax', np.max):.3g}")


# ----------------------------------------------------------------------------
def epoch_pass(ode, ds, idx, a, train):
    """Mean MSE over valid cases; with train=True accumulates the gradient of
    that mean (one optimizer step is taken by the caller)."""
    n_ok = n_rej = 0
    tot = 0.0
    losses = []
    for b0 in range(0, len(idx), a.case_batch):
        cases = list(idx[b0:b0 + a.case_batch])
        with torch.set_grad_enabled(train):
            res = solve_cases(ode, ds, cases, a, grid="train",
                              adjoint=a.adjoint and train)
            batch = []
            for ci, p in res:
                if p is None or not torch.isfinite(p).all():
                    n_rej += 1
                    continue
                l = ((p - ds["zg"][ci]) ** 2).mean()
                if not torch.isfinite(l):
                    n_rej += 1
                    continue
                batch.append(l)
                n_ok += 1
                tot += float(l.detach())
            if train and batch:
                # d(mean over all valid)/dw; rescaled below once n_ok is known
                torch.stack(batch).sum().div(len(idx)).backward()
    if train and n_ok:
        for p in ode.parameters():
            if p.grad is not None:
                p.grad.mul_(len(idx) / n_ok)
    return (tot / n_ok if n_ok else float("nan")), n_ok, n_rej


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="runs_prof/full")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--species", default="log", choices=("log", "linear"))
    ap.add_argument("--y_eps", type=float, default=1e-10)
    ap.add_argument("--no_cond", action="store_true")
    ap.add_argument("--grid", default="data", choices=("data", "uniform"))
    ap.add_argument("--n_uniform", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--activation", default="silu", choices=tuple(ACT))
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--val_every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--bn_warmup_every", type=int, default=20)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-7)
    ap.add_argument("--max_steps", type=int, default=2000,
                    help="dopri5 step cap per data interval; a case that needs "
                         "more is rejected")
    ap.add_argument("--case_batch", type=int, default=16)
    ap.add_argument("--adjoint", action="store_true",
                    help="adjoint backprop: O(1) memory in the number of solver "
                         "steps, slower. Use if full trajectories run out of memory")
    ap.add_argument("--eval_every", type=int, default=50,
                    help="free-rollout metrics on val every N epochs (0 = only at end)")
    ap.add_argument("--eval_rtol", type=float, default=1e-5)
    ap.add_argument("--eval_atol", type=float, default=1e-7)
    ap.add_argument("--eval_max_steps", type=int, default=5000)
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
    log(f"device {dev} | {ds['C']} cases x {ds['K']} points, state dim {ds['D']} | "
        f"species '{a.species}' | conditioning {'none' if a.no_cond else 'P'}")
    log(f"split (seed 42): train {len(ds['tr'])} val {len(ds['va'])} test {len(ds['te'])}")
    log(f"time: tau = t / {ds['t_scale']:.3g} s in [0, 1]; training grid '{a.grid}' "
        f"({ds['zg'].shape[1]} points per case); evaluation on the data's own times")
    ode = ODEFunc(ds["D"], 0 if a.no_cond else 1, a.hidden, a.layers, a.activation).to(dev)
    log(f"parameters: {sum(p.numel() for p in ode.parameters()) / 1e6:.3f} M | "
        f"h{a.hidden} L{a.layers} {a.activation} wd {a.weight_decay:g} lr {a.lr:g}")

    ck = out / "last.pt"
    opt = torch.optim.AdamW(ode.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=50, min_lr=1e-6)
    start, best, best_state, no_imp = 0, float("inf"), None, 0
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        ode.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        start, best, no_imp = s["epoch"], s["best"], s["no_imp"]
        best_state = s["best_state"]
        log(f"resumed at epoch {start}")
    else:
        warmup_bn_stats(ode, ds)

    vl, vok, vrej = float("nan"), 0, 0
    for ep in range(start, a.epochs):
        t0 = time.time()
        if ep > 0 and ep % a.bn_warmup_every == 0:
            warmup_bn_stats(ode, ds)
        ode.train()
        set_bn_eval(ode)
        opt.zero_grad(set_to_none=True)
        tl, tok, trej = epoch_pass(ode, ds, ds["tr"], a, train=True)
        if tok:
            gn = torch.nn.utils.clip_grad_norm_(ode.parameters(), a.max_grad_norm)
            if torch.isfinite(gn):
                opt.step()
        if (ep + 1) % a.val_every == 0 or ep == 0:
            ode.eval()
            with torch.no_grad():
                vl, vok, vrej = epoch_pass(ode, ds, ds["va"], a, train=False)
            if vok:
                sched.step(vl)
                if vl < best:
                    best, best_state, no_imp = vl, copy.deepcopy(ode.state_dict()), 0
                    torch.save(best_state, out / "best.pt")
                else:
                    no_imp += a.val_every
        log(f"Epoch {ep + 1:4d}/{a.epochs} | Train={tl:.6e} | Val={vl:.6e} | "
            f"TrainValid={tok:4d} TrainRej={trej:4d} | ValValid={vok:4d} ValRej={vrej:4d} | "
            f"LR={opt.param_groups[0]['lr']:.2e} | BestVal={best:.6e} | "
            f"{time.time() - t0:.1f}s")
        if a.eval_every and (ep + 1) % a.eval_every == 0:
            log("  -> val " + free_rollout(ode, ds, ds["va"], a))
        torch.save({"model": ode.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep + 1, "best": best,
                    "best_state": best_state, "no_imp": no_imp}, ck)
        if no_imp >= a.patience:
            log(f"early stop at epoch {ep + 1}")
            break

    if best_state is not None:
        ode.load_state_dict(best_state)
    log("\n=== best model ===")
    for split, idx in (("val", ds["va"]), ("test", ds["te"])):
        log(f"{split}: " + free_rollout(ode, ds, idx, a))
    torch.save({"model_state": ode.state_dict(), "args": vars(a)}, out / "final.pt")


if __name__ == "__main__":
    main()
