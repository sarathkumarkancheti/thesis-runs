#!/usr/bin/env python3
"""rollout_diagnose.py -- WHERE does the continuous rollout leave the data?

    python rollout_diagnose.py --run_dir runs/X --data_dir DATA
    python rollout_diagnose.py --run_dir runs/X --data_dir DATA --ckpt runs/X/stage0_L8/done.pt

The training log only prints "val continuous rollout ... DIVERGED" with a mean
and a max over all validation cases. This rolls out every validation case from
t = 0 exactly as that line does, and for each one reports

  T RMSE, max |z|       how bad it gets (|z| <= 1 is the physical range)
  exit                  the first sample where any channel leaves |z| > 1.1:
                        time, t / t_ign, phase, and WHICH channel left first
  q at exit             (physical rate space) the network's compressed rate at
                        that point, against the +-1.05 bound: |q| near the bound
                        means the decoder saturated -- the exponential decode
                        turned a bounded output into an extreme rate
  jump                  the largest single-step change in z along the rollout
                        and where it happened

and, for the diverging cases, re-integrates with an ADAPTIVE solver (dopri5):

  dopri5 stays bounded  -> the field is fine, RK4's fixed step is too coarse
                           there (a step-size / stiffness problem)
  dopri5 also diverges  -> the field itself drives the state away (a modelling
                           problem: the network extrapolates badly off the data)
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

PH = ("induction", "ignition", "post")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no_adaptive", action="store_true")
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--mech", default=None,
                    help="mechanism file; with --adaptive_all, splits the "
                         "end-temperature error into energy violation and "
                         "composition error (Cantera, constant H and P)")
    ap.add_argument("--adaptive_all", action="store_true",
                    help="also integrate EVERY validation case with dopri5 and "
                         "compare its T RMSE with RK4's, case by case: tells "
                         "how much of the reported error is the integrator")
    a = ap.parse_args(argv)

    run = Path(a.run_dir)
    ck = Path(a.ckpt) if a.ckpt else run / "final.pt"
    if not ck.exists():
        cands = sorted(run.glob("*/done.pt"), key=lambda p: p.stat().st_mtime) \
            + sorted(run.glob("*/latest.pt"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit(f"no checkpoint in {run}")
        ck = cands[-1]
    print(f"checkpoint: {ck}")
    cfg = build_cfg(run, a.data_dir, legacy=False, verbose=False)
    ds = ct.ChemDataset(cfg)
    dev = torch.device(a.device)
    model = load_model(ck, ds, cfg, dev)
    model.eval()
    sub = int(getattr(cfg, "rk4_substeps", 2))
    tig = ds._t_ign_for_phases()
    names = ds.state_columns
    physical = bool(getattr(model.ode, "physical", False))

    rows = []
    for c in ds.val_idx:
        u = torch.tensor(ds.u[c][None, :].astype(np.float32), device=dev)
        z0 = torch.tensor(ds.z[c][None, 0].astype(np.float32), device=dev)
        cond = torch.tensor(ds.cond[c][None, :], dtype=torch.float32, device=dev)
        with torch.no_grad():
            pr = model.rollout(z0, u, cond, substeps=sub)[0].cpu().numpy()
        zt = ds.z[c]
        raw_p, raw_t = ds.from_z(pr), ds.from_z(zt)
        fin = np.isfinite(pr).all(1)
        Trmse = float(np.sqrt(np.nanmean((raw_p[fin, 0] - raw_t[fin, 0]) ** 2))) \
            if fin.any() else np.inf
        zmax = float(np.nanmax(np.abs(pr))) if fin.any() else np.inf
        out = (np.abs(pr) > 1.1) | ~np.isfinite(pr)
        k_exit = int(np.argmax(out.any(1))) if out.any() else -1
        ch = int(np.argmax(out[k_exit])) if k_exit >= 0 else -1
        ph = lambda k: PH[0 if ds.t[c, k] < 0.5 * tig[c] else  # noqa: E731
                           (1 if ds.t[c, k] <= 3 * tig[c] else 2)]
        dz = np.abs(np.diff(pr, axis=0)).max(1)
        k_jump = int(np.nanargmax(np.where(np.isfinite(dz), dz, -1)))
        qx = np.nan
        if physical and k_exit > 0:
            with torch.no_grad():
                zz = torch.tensor(pr[k_exit - 1][None], dtype=torch.float32,
                                  device=dev)
                q = model.ode.net(model._pad(zz),
                                  u[:, k_exit - 1:k_exit].reshape(1, 1), cond)
                qx = float(q.abs().max())
        rows.append(dict(c=c, T0=ds.raw[c, 0, 0], Trmse=Trmse, zmax=zmax,
                         k_exit=k_exit, t_exit=ds.t[c, k_exit] if k_exit >= 0 else np.nan,
                         r_exit=ds.t[c, k_exit] / tig[c] if k_exit >= 0 else np.nan,
                         ph_exit=ph(k_exit) if k_exit >= 0 else "-",
                         ch=names[ch] if ch >= 0 else "-", q=qx,
                         jump=float(dz[k_jump]), k_jump=k_jump + 1,
                         ph_jump=ph(k_jump + 1)))

    bad = [r for r in rows if r["zmax"] > 1.1]
    good = [r for r in rows if r["zmax"] <= 1.1]
    print(f"\n{len(rows)} validation cases: {len(bad)} leave the physical range "
          f"(|z| > 1.1), {len(good)} stay inside")
    if good:
        print(f"  T RMSE over the {len(good)} bounded cases: mean "
              f"{np.mean([r['Trmse'] for r in good]):.1f} K, median "
              f"{np.median([r['Trmse'] for r in good]):.1f} K")
    if bad:
        print(f"  T RMSE over the {len(bad)} escaping cases: mean "
              f"{np.mean([min(r['Trmse'], 1e9) for r in bad]):.3g} K")
        print("\n  escaping cases, earliest exit first:")
        print(f"  {'case':>5}{'T0':>7}{'T RMSE':>10}{'max|z|':>9}  exit: "
              f"{'sample':>6}{'t [s]':>11}{'t/t_ign':>9} {'phase':<10}{'channel':<8}"
              + (f"{'|q|':>6}" if physical else "") + "   largest step")
        for r in sorted(bad, key=lambda r: r["r_exit"])[:a.show]:
            print(f"  {r['c']:>5}{r['T0']:>7.0f}{min(r['Trmse'], 9e9):>10.3g}"
                  f"{min(r['zmax'], 9e9):>9.3g}        {r['k_exit']:>6}"
                  f"{r['t_exit']:>11.3e}{r['r_exit']:>9.3g} {r['ph_exit']:<10}"
                  f"{r['ch']:<8}" + (f"{r['q']:>6.3f}" if physical else "")
                  + f"   {r['jump']:.3g} at {r['k_jump']} ({r['ph_jump']})")
        from collections import Counter
        print("\n  exit phase: " + str(dict(Counter(r["ph_exit"] for r in bad)))
              + "   first channel out: "
              + str(dict(Counter(r["ch"] for r in bad).most_common(5))))
        if physical:
            qs = np.array([r["q"] for r in bad if np.isfinite(r["q"])])
            if qs.size:
                print(f"  |q| just before exit: median {np.median(qs):.3f}, "
                      f"{int((qs > 0.95).sum())}/{qs.size} within 10 % of the "
                      f"1.05 bound (decoder saturated)")
        T0b = np.array([r["T0"] for r in bad])
        T0a = np.array([r["T0"] for r in rows])
        print(f"  T0 of escaping cases: {T0b.min():.0f}..{T0b.max():.0f} K "
              f"(all val cases {T0a.min():.0f}..{T0a.max():.0f} K)")

    def dopri(c, rtol=1e-5, atol=1e-7):
        from torchdiffeq import odeint
        u = torch.tensor(ds.u[c].astype(np.float32), device=dev)
        cond = torch.tensor(ds.cond[c][None, :], dtype=torch.float32, device=dev)
        z0 = torch.tensor(ds.z[c][None, 0].astype(np.float32), device=dev)
        model.ode.set_cond(cond)
        with torch.no_grad():
            z1 = model.anchor(z0, cond, u[1:2].reshape(1, 1)) \
                if model.init is not None else z0
            zz = odeint(model.ode, model._pad(z1), u[1:], method="dopri5",
                        rtol=rtol, atol=atol,
                        options={"max_num_steps": 20000})[:, 0]
        zz = zz[:, :model.d_phys].cpu().numpy()
        return np.concatenate([ds.z[c][None, 0], zz], 0)

    gas = None
    if a.mech:
        import warnings
        import cantera as cant
        warnings.filterwarnings("ignore")
        gas = cant.Solution(a.mech)
        P_case = np.asarray(ds.pressures, dtype=np.float64) * 1e5

    def full_Y(raw_row):
        """kept channels (log10(Y+floor)) -> Y over all mechanism species."""
        Yall = np.zeros(len(ds.species_names_all))
        kept = np.where(ds.keep_mask[1:])[0]
        Yall[kept] = np.clip(10.0 ** raw_row[1:] - cfg.y_floor, 0.0, None)
        drop = np.where(~ds.keep_mask[1:])[0]
        Yall[drop] = np.clip(10.0 ** ds.const_values[1:][drop] - cfg.y_floor,
                             0.0, None)
        return Yall / Yall.sum()

    MAJ = [sp for sp in ("CH4", "O2", "H2", "CO", "CO2", "H2O")
           if sp in ds.species_names_all]
    if gas is not None:
        EL = ["C", "H", "O"]
        Emat = np.array([[gas.n_atoms(sp, e) * cant.Element(e).weight
                          / gas.molecular_weights[i] for e in EL]
                         for i, sp in enumerate(gas.species_names)])

    def species_split(c, rp, rt, k):
        """Predicted vs true mass fractions of the majors at sample k, and the
        element drift of the prediction relative to its own t = 0 state."""
        Yp, Yt = full_Y(rp[k]), full_Y(rt[k])
        maj = {sp: (Yp[gas.species_index(sp)], Yt[gas.species_index(sp)])
               for sp in MAJ}
        Z0 = full_Y(rt[0]) @ Emat
        dZ = (Yp @ Emat - Z0) / Z0                 # relative drift C, H, O
        return maj, dZ

    def energy_split(c, rp, rt, k):
        """T_pred - T_true at sample k = energy violation + composition error."""
        Y0 = np.clip(ds.states_phys[c, 0, 1:], 0.0, None)
        gas.TPY = ds.states_phys[c, 0, 0], P_case[c], Y0 / Y0.sum()
        h0 = gas.enthalpy_mass
        gas.HPY = h0, P_case[c], full_Y(rp[k])
        T_h = gas.T                      # T the predicted Y implies at h0
        return rp[k, 0] - T_h, T_h - rt[k, 0]

    if a.adaptive_all:
        import time
        print("\n  every validation case, RK4 (as trained/logged) vs adaptive "
              "dopri5 on the SAME field:")
        t0 = time.time()
        pairs = []
        species_rows = {}
        for r in rows:
            c = r["c"]
            try:
                zz = dopri(c)
                rp, rt = ds.from_z(zz), ds.from_z(ds.z[c])
                Td = float(np.sqrt(np.mean((rp[:, 0] - rt[:, 0]) ** 2)))
                # ignition time = half-rise crossing of T, predicted vs true
                def half(Tk):
                    thr = Tk[0] + 0.5 * (Tk.max() - Tk[0])
                    i = int(np.argmax(Tk >= thr))
                    return ds.t[c, i] if Tk.max() - Tk[0] > 100 else np.nan
                ti_p, ti_t = half(rp[:, 0]), half(rt[:, 0])
                dig = float(np.log10(ti_p / ti_t)) if ti_p > 0 and ti_t > 0 \
                    else np.nan
                Tend_err = float(rp[-1, 0] - rt[-1, 0])
                ev = cv = np.nan
                maj, dZ = None, None
                if gas is not None:
                    try:
                        ev, cv = energy_split(c, rp, rt, len(rp) - 1)
                        maj, dZ = species_split(c, rp, rt, len(rp) - 1)
                    except Exception:                        # noqa: BLE001
                        pass
                species_rows[c] = (maj, dZ)
                Ld = float(np.mean(np.abs(rp[:, 1:] - rt[:, 1:])[rt[:, 1:] > -6]))
                zm = float(np.abs(zz).max())
            except Exception:                                    # noqa: BLE001
                Td, Ld, zm, dig, Tend_err, ev, cv = (np.nan,) * 7
            pr_rk = None
            pairs.append((c, r["T0"], r["Trmse"], Td, zm, Ld, dig, Tend_err,
                          ev, cv))
        P_ = np.array([(p[2], p[3]) for p in pairs], dtype=np.float64)
        ok = np.isfinite(P_).all(1) & (P_[:, 0] < 1e8)
        print(f"  ({time.time() - t0:.0f} s)   over {ok.sum()} cases:  T RMSE "
              f"RK4 mean {P_[ok, 0].mean():.1f} / median {np.median(P_[ok, 0]):.1f} K"
              f"   dopri5 mean {P_[ok, 1].mean():.1f} / median "
              f"{np.median(P_[ok, 1]):.1f} K")
        ld = np.array([p[5] for p in pairs], dtype=np.float64)
        print(f"  dopri5 species error |dlog10 Y| where true Y > 1e-6: mean "
              f"{np.nanmean(ld):.3f} dec;  dopri5 cases leaving |z|>1.1: "
              f"{int(np.nansum(np.array([p[4] for p in pairs]) > 1.1))}")
        worst = sorted(pairs, key=lambda p: -np.nan_to_num(p[3]))[:8]
        print("  worst cases under dopri5 (i.e. genuine model error):")
        for c, T0, Tr, Td, zm, Ld, dig, Te, ev, cv in worst:
            print(f"    case {c:>4}  T0 {T0:6.0f} K   RK4 {min(Tr, 9e9):8.1f} K   "
                  f"dopri5 {Td:8.1f} K   max|z| {zm:.3g}   ignition "
                  f"{'late' if dig > 0 else 'early'} by {abs(dig):.2f} dec "
                  f"(x{10 ** abs(dig):.2g})   T(t_end) error {Te:+.0f} K"
                  + (f" = energy {ev:+.0f} + composition {cv:+.0f}"
                     if np.isfinite(ev) else ""))
        if gas is not None and species_rows:
            print("\n  at t_end, worst cases: mass fraction predicted / TRUE, "
                  "and element drift of the prediction")
            print("    case  " + "".join(f"{sp:>15}" for sp in MAJ)
                  + "      dC      dH      dO")
            for c, *_ in worst:
                maj, dZ = species_rows.get(c, (None, None))
                if maj is None:
                    continue
                print(f"    {c:>4}  " + "".join(
                    f"{maj[sp][0]:>7.3f}/{maj[sp][1]:<7.3f}" for sp in MAJ)
                    + "".join(f"{100 * v:+7.1f}%" for v in dZ))
            allZ = np.array([v[1] for v in species_rows.values()
                             if v[1] is not None])
            allM = [v[0] for v in species_rows.values() if v[0] is not None]
            if allZ.size:
                print(f"  element drift over all cases (median |.|): "
                      + "  ".join(f"{e} {100 * np.median(np.abs(allZ[:, i])):.1f} %"
                                  for i, e in enumerate(EL))
                      + f"   (max {100 * np.abs(allZ).max():.1f} %)")
                for sp in ("O2", "CH4"):
                    if sp in MAJ:
                        d = np.array([m[sp][0] - m[sp][1] for m in allM])
                        print(f"  {sp} excess at t_end (predicted - true): median "
                              f"{np.median(d):+.4f}, {int((d > 0.01).sum())}/"
                              f"{len(d)} cases > +0.01 mass fraction")
        E_ = np.array([(p[8], p[9]) for p in pairs], dtype=np.float64)
        if np.isfinite(E_).any():
            okE = np.isfinite(E_).all(1)
            print(f"  end-temperature error split over {okE.sum()} cases (median |.|): "
                  f"energy violation {np.median(np.abs(E_[okE, 0])):.0f} K, "
                  f"composition error {np.median(np.abs(E_[okE, 1])):.0f} K")
            print("    energy violation = predicted T minus the T that the "
                  "predicted composition has at the initial enthalpy "
                  "(0 for an energy-conserving model)")
        D = np.array([(p[1], p[3], p[6], p[7]) for p in pairs], dtype=np.float64)
        okd = np.isfinite(D).all(1)
        if okd.sum() > 5:
            r_ig = np.corrcoef(np.abs(D[okd, 2]), D[okd, 1])[0, 1]
            print(f"  over all cases: |ignition-time error| median "
                  f"{np.median(np.abs(D[okd, 2])):.2f} dec, "
                  f"{int((np.abs(D[okd, 2]) > 0.3).sum())} cases off by > 2x;  "
                  f"correlation with dopri5 T RMSE {r_ig:.2f}")
            for lo, hi in ((0, 1000), (1000, 1200), (1200, 1500), (1500, 3000)):
                mk = okd & (D[:, 0] >= lo) & (D[:, 0] < hi)
                if mk.any():
                    print(f"    T0 {lo:4d}-{hi:4d} K: {int(mk.sum()):>2} cases, "
                          f"dopri5 T RMSE median {np.median(D[mk, 1]):6.1f} K, "
                          f"|ignition error| median {np.median(np.abs(D[mk, 2])):.2f}"
                          f" dec, |T(t_end) error| median "
                          f"{np.median(np.abs(D[mk, 3])):.0f} K")
        big = (P_[:, 0] - P_[:, 1]) > 50
        print(f"  cases where RK4 adds > 50 K of error: {int(big.sum())}")

    if bad and not a.no_adaptive:
        from torchdiffeq import odeint
        print("\n  adaptive dopri5 re-integration of the escaping cases "
              "(rtol 1e-5, atol 1e-7, max 20000 steps):")
        n_ok = n_bad = n_fail = 0
        for r in sorted(bad, key=lambda r: r["r_exit"])[:a.show]:
            c = r["c"]
            u = torch.tensor(ds.u[c].astype(np.float32), device=dev)
            cond = torch.tensor(ds.cond[c][None, :], dtype=torch.float32, device=dev)
            z0 = torch.tensor(ds.z[c][None, 0].astype(np.float32), device=dev)
            model.ode.set_cond(cond)
            with torch.no_grad():
                z1 = model.anchor(z0, cond, u[1:2].reshape(1, 1)) \
                    if model.init is not None else z0
                try:
                    zz = odeint(model.ode, model._pad(z1), u[1:], method="dopri5",
                                rtol=1e-5, atol=1e-7,
                                options={"max_num_steps": 20000})[:, 0]
                    zz = zz[:, :model.d_phys].cpu().numpy()
                    zm = float(np.abs(zz).max())
                    verdict = ("bounded -> RK4 STEP problem" if zm <= 1.1
                               else "diverges too -> FIELD problem")
                    n_ok += zm <= 1.1
                    n_bad += zm > 1.1
                except Exception as e:                           # noqa: BLE001
                    zm, verdict = np.nan, f"solver gave up ({type(e).__name__})" \
                                          f" -> field stiff/singular there"
                    n_fail += 1
            print(f"    case {c:>4}: RK4 max|z| {min(r['zmax'], 9e9):.3g}, "
                  f"dopri5 max|z| {zm:.3g}  {verdict}")
        print(f"  => {n_ok} step-size problems, {n_bad} field problems, "
              f"{n_fail} solver failures")


if __name__ == "__main__":
    main()
