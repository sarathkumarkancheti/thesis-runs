#!/usr/bin/env python3
"""clamp_eval.py -- the existing checkpoints of a run, rolled out from t = 0
with and without the hard state clamp. No training, no GPU needed.

    python clamp_eval.py --run_dir runs/phys_tend20 --data_dir DATA
    python clamp_eval.py --run_dir runs/phys_tend20 --data_dir DATA --split test

For every checkpoint found (pretrain, each stage, anchor) it prints the same
continuous-rollout numbers as the '->' lines of train.log, once with the
clamp OFF (exactly what the log reported) and once ON (|z| <= --clamp after
every RK4 sub-step, i.e. y_floor <= Y <= 1 and T inside the data range):

  T RMSE mean / median   over the cases of the split, in K
  T(t_end) err median    |T_pred - T_true| at the last sample, in K
  |dY| majors            mean |Y_pred - Y_true| over species with max Y > 1e-3,
                         in mass-%
  max log10 Y            > 0 is unphysical; > 20 is a blow-up
  at clamp               cases whose rollout touched the bound -- with the
                         clamp on these would have left the physical range
  non-finite             cases whose rollout produced NaN/inf

Integration: RK4 with cfg.rk4_substeps (2), the stepper of the log lines.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chemnode_train as ct                                    # noqa: E402
from chemnode_deploy import build_cfg, load_model              # noqa: E402


def checkpoints(run: Path):
    out = []
    if (run / "pretrain.pt").exists():
        out.append(("pretrain", run / "pretrain.pt"))
    st = sorted(run.glob("stage*_L*/done.pt"),
                key=lambda p: int(p.parent.name.split("_")[0][5:]))
    out += [(p.parent.name, p) for p in st]
    if (run / "anchor" / "done.pt").exists():
        out.append(("anchor", run / "anchor" / "done.pt"))
    return out


@torch.no_grad()
def evaluate(model, ds, idx, dev, sub, clamp):
    model.ode.state_clamp = clamp
    model.ode.rk4_tol = None                 # plain RK4, as the log lines
    Tr, Te, Am, Ym = [], [], [], []
    n_hit = n_bad = 0
    for c in idx:
        u = torch.tensor(ds.u[c][None].astype(np.float32), device=dev)
        z0 = torch.tensor(ds.z[c][None, 0].astype(np.float32), device=dev)
        cond = torch.tensor(ds.cond[c][None], dtype=torch.float32, device=dev)
        pr = model.rollout(z0, u, cond, substeps=sub)[0].cpu().numpy()
        if not np.isfinite(pr).all():
            n_bad += 1
            continue
        # "at the bound": any channel at the TOP of its range (Y = 1, T max)
        # or T at the bottom. The species floor (z = -1, Y = 1e-12) is where
        # the data itself sits before ignition, so it does not count.
        lim = (clamp if clamp is not None else 1.0) - 1e-6
        if (pr.max() >= lim + (0 if clamp is not None else 1e-3)) or \
                (pr[:, 0].min() <= -lim - (0 if clamp is not None else 1e-3)):
            n_hit += 1
        rp, rt = ds.from_z(pr), ds.from_z(ds.z[c])
        Tr.append(np.sqrt(np.mean((rp[:, 0] - rt[:, 0]) ** 2)))
        Te.append(abs(rp[-1, 0] - rt[-1, 0]))
        yf = float(getattr(ds.cfg, "y_floor", 1e-12))
        Yp = np.clip(10.0 ** np.minimum(rp[:, 1:], 1.0) - yf, 0.0, None)
        Yt = np.clip(10.0 ** rt[:, 1:] - yf, 0.0, None)
        maj = Yt.max(0) > 1e-3
        Am.append(100.0 * np.mean(np.abs(Yp[:, maj] - Yt[:, maj])))
        Ym.append(rp[:, 1:].max())
    f = lambda v, g=np.mean: float(g(v)) if v else float("nan")  # noqa: E731
    return dict(Tm=f(Tr), Tmed=f(Tr, np.median), Tend=f(Te, np.median),
                Am=f(Am), Ym=f(Ym, np.max), hit=n_hit, bad=n_bad, n=len(idx))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--clamp", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None,
                    help="comma-separated checkpoint names, e.g. stage2_L32,stage4_L128")
    a = ap.parse_args(argv)

    run = Path(a.run_dir)
    cfg = build_cfg(run, a.data_dir, legacy=False, verbose=False)
    ds = ct.ChemDataset(cfg)
    dev = torch.device(a.device)
    idx = ds.val_idx if a.split == "val" else ds.test_idx
    sub = int(getattr(cfg, "rk4_substeps", 2))
    cks = checkpoints(run)
    if a.only:
        keep = set(a.only.split(","))
        cks = [c for c in cks if c[0] in keep]
    if not cks:
        raise SystemExit(f"no checkpoints in {run}")
    print(f"{run}: {len(idx)} {a.split} cases, RK4 x{sub}, clamp |z| <= {a.clamp}")
    hdr = (f"{'checkpoint':<14}{'clamp':>6}{'T RMSE mean':>13}{'median':>9}"
           f"{'T(t_end) med':>14}{'|dY| maj %':>12}{'max log10Y':>12}"
           f"{'at clamp':>10}{'non-finite':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, ck in cks:
        model = load_model(ck, ds, cfg, dev)
        model.eval()
        for cl in (None, a.clamp):
            r = evaluate(model, ds, idx, dev, sub, cl)
            print(f"{name:<14}{'on' if cl else 'off':>6}{r['Tm']:13.1f}"
                  f"{r['Tmed']:9.1f}{r['Tend']:14.1f}{r['Am']:12.2f}"
                  f"{r['Ym']:+12.2f}{r['hit']:>6}/{r['n']:<3}{r['bad']:>7}/{r['n']}")
        print()
    print("'at clamp' with clamp off = cases that left |z| <= 1 somewhere;\n"
          "with clamp on = cases held at the bound (the true data never is).")


if __name__ == "__main__":
    main()
