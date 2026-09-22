#!/usr/bin/env python3
"""chemnode_deploy.py -- turn a training run into ONE self-contained model file.

    python chemnode_deploy.py --run_dir runs/fixed --out chemnode_model.pt
    python chemnode_deploy.py --model chemnode_model.pt --check      # no data

WHY THIS EXISTS
---------------
`final.pt` holds the weights and `rate_scale` and nothing else. Every other
number the model needs to be used -- the log-time constants, the state scaling,
the conditioning ranges, the ignition-delay correlation, which species are
modelled -- lives in `ChemDataset`, i.e. it is rebuilt from the training .npy
files every time. So "load the model" has, until now, silently meant "load the
model AND the training set".

That is wrong on its own terms. The thesis claim is that the surrogate needs
four numbers and a time grid. A surrogate that also needs the dataset it was
fitted on is not deployable, and cannot be handed to anyone.

This writes a single file containing the network and every constant required to
evaluate it. After that the training data is not needed again, by anything.

WHAT IS IN IT
-------------
  weights + architecture   enough to rebuild the network exactly
  y_floor, raw_lo/hi       the log-species state transform, and its inverse
  keep_mask, species       which of the mechanism's species are modelled
  delta, s_min, s_span     the log-time coordinate
  cond_lo/hi + names       conditioning scaled in the units the network learnt
  t_ign correlation        basis, coefficients, and whether steam is a regressor
  train ranges             min/max of T0, P, O2/CH4, H2O/CH4 over the TRAINING
                           split -- what a GUI should offer as slider limits and
                           where the envelope ends

Every one of these was fitted on the training split, so the file carries no
information about the validation or test cases beyond what the weights already
encode.

THE EXPORT IS VERIFIED, NOT ASSUMED
-----------------------------------
The correlation coefficients are refitted here from the dataset's own design
matrix, then checked against `ds.predict_tign` on every case in the dataset.
The state transform, the time coordinate and the conditioning vector are checked
the same way. If any of them disagrees by more than a tight tolerance the export
fails loudly rather than producing a file that quietly predicts something else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json
from dataclasses import fields

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chemnode_train as ct                                    # noqa: E402

POLY_BASES = ("arrhenius", "arrhenius_log", "quadratic", "cubic")
ARCH_KEYS = ("hidden", "n_blocks", "dropout", "out_gain", "rate_head",
             "rate_head_k", "n_time_fourier", "use_time_input", "box", "box_w",
             "n_augment", "use_initializer", "init_hidden", "init_gain",
             "y_floor", "rate_space", "phys_bound", "phys_cap_mult",
             "target_cap", "rk4_tol", "rk4_max_substeps", "state_clamp")


# ==========================================================================
# the maths, standalone -- no ChemDataset anywhere below this line
# ==========================================================================
def tign_design(T0, P, phi, steam, mode, use_steam):
    """The correlation's design matrix. A transcription of ChemDataset's, kept
    here so the deployed file does not depend on the dataset class."""
    T0 = np.atleast_1d(np.asarray(T0, dtype=np.float64))
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    v = [np.broadcast_to(1000.0 / T0, T0.shape),
         np.broadcast_to(np.log10(np.atleast_1d(P)), T0.shape),
         np.broadcast_to(np.log10(np.maximum(phi, 1e-12))
                         if mode == "arrhenius_log" else phi, T0.shape)]
    if use_steam:
        v.append(np.broadcast_to(np.atleast_1d(steam), T0.shape))
    v = [c.astype(np.float64) for c in v]
    cols = list(v)
    if mode in ("quadratic", "cubic"):
        cols += [v[i] * v[j] for i in range(len(v)) for j in range(i, len(v))]
    if mode == "cubic":
        cols += [v[i] * v[j] * v[k] for i in range(len(v))
                 for j in range(i, len(v)) for k in range(j, len(v))]
    cols.append(np.ones(v[0].size))
    return np.column_stack(cols)


class Deployed:
    """A trained ChemNODE that needs no training data.

        m = Deployed("chemnode_model.pt")
        r = m.predict(T0=1500, P=55, phi=0.62, steam=0.40, t_end=1e-2, n=400)
    """

    def __init__(self, path, device="cpu"):
        b = torch.load(path, map_location="cpu", weights_only=False)
        self.b = b
        self.device = torch.device(device)
        cfg = ct.Config()
        for k, v in b["arch"].items():
            setattr(cfg, k, v)
        self.cfg = cfg
        self.rate_scale = np.asarray(b["rate_scale"])
        self.model = ct.ChemNODE(int(b["d_state"]), int(b["d_cond"]), cfg,
                                 self.rate_scale).to(self.device)
        self.model.load_state_dict(b["state_dict"], strict=True)
        self.model.eval()

        for k in ("raw_lo", "raw_hi", "raw_range", "keep_mask",
                  "cond_lo", "cond_hi", "cond_rng"):
            setattr(self, k, np.asarray(b[k]))
        for k in ("delta", "s_min", "s_span"):
            setattr(self, k, float(b[k]))
        self.time_align = b["time_align"]
        self.cond_names = list(b["cond_names"])
        self.species_names = list(b["species_names"])
        self.species_names_all = list(b["species_names_all"])
        self.state_columns = list(b["state_columns"])
        self.n_points = int(b["n_points"])
        self.ranges = b["ranges"]
        self.tign = b["tign"]
        self.notes = b.get("notes", {})

    # ------------------------------------------------------------ transforms
    def predict_tign(self, T0, P, phi, steam=0.0):
        if self.time_align != "ignition":
            return None                       # global coordinate: no delay used
        t = self.tign
        X = tign_design(T0, P, phi, steam, t["mode"], bool(t["use_steam"]))
        return 10.0 ** (X @ np.asarray(t["coef"]))

    def u_of_t(self, t, t_ign=None):
        sh = 0.0 if (self.time_align != "ignition" or t_ign is None) \
            else float(np.log10(t_ign))
        return (np.log10(np.asarray(t, dtype=np.float64) + self.delta)
                - sh - self.s_min) / self.s_span

    def z_from_physical(self, T0, Y0):
        """Y0 over ALL mechanism species, in mechanism order."""
        Y0 = np.asarray(Y0, dtype=np.float64)
        if Y0.shape[-1] != len(self.species_names_all):
            raise ValueError(f"expected {len(self.species_names_all)} species, "
                             f"got {Y0.shape[-1]}")
        raw = np.concatenate([np.atleast_1d(T0).astype(np.float64)[:1],
                              np.log10(np.clip(Y0, 0.0, None)
                                       + self.cfg.y_floor)])[self.keep_mask]
        return (2.0 * (raw - self.raw_lo) / self.raw_range - 1.0
                ).astype(np.float32)[None, :]

    def z_to_physical(self, z):
        raw = (np.asarray(z) + 1.0) / 2.0 * self.raw_range + self.raw_lo
        Y = 10.0 ** raw[..., 1:] - self.cfg.y_floor
        return np.concatenate([raw[..., :1], np.clip(Y, 0.0, None)], axis=-1)

    def cond_vector(self, P, T0=None, phi=None, steam=None):
        avail = {"p": P, "T0": T0, "O2/CH4": phi, "H2O/CH4": steam}
        v = np.array([[float(avail[n]) for n in self.cond_names]])
        return (2.0 * (v - self.cond_lo) / self.cond_rng - 1.0).astype(np.float32)

    # -------------------------------------------------------------- the model
    def grid(self, t_end, n):
        """A log grid anchored at t = 0, starting at the coordinate's delta.

        The floor is delta and NOTHING ELSE. An earlier version used
        max(delta, t_end * 1e-12), which tied the START of the grid to its END:
        raising t_end from 12 s to 1000 s moved the first positive point from
        1.2e-11 s to 1e-9 s and grew the first step in u from 0.14 to 0.27 --
        a single RK4 step across a quarter of the whole coordinate, taken at
        t = 0 where every radical is still at the log floor. It destroyed the
        radical pool before induction began, and the model ignited 70x late
        while Cantera, which adapts internally, was unaffected. The lesson is
        that with a fixed point budget the EARLY resolution must not be a
        function of the horizon.

        delta is the right floor because it is the coordinate's own anchor:
        u(0) and u(delta) differ by log10(2)/s_span ~ 0.02, so the first step
        is small whatever t_end is.
        """
        return np.concatenate(([0.0], np.logspace(np.log10(self.delta),
                                                  np.log10(float(t_end)),
                                                  int(n) - 1)))

    def phase_grid(self, t_end, n, t_ign, frac=(0.30, 0.40, 0.30)):
        """A grid that spends its budget by PHASE, the way the generator did.

        Anchors are 0.5*t_ign and 3*t_ign -- the same boundaries
        chemnode_train.py uses to define induction / ignition / post. The delay
        is the PREDICTED one, so this grid is still buildable at a new operating
        point with no Cantera solve: the anchors come from the correlation, not
        from a reference trajectory.

        A uniform log grid spends points in proportion to DECADES, and the
        induction phase is most of the decades while the post-ignition
        relaxation is almost none of them. If the error after the peak were a
        resolution problem, moving the budget would fix it. That is the
        experiment this method exists to run.
        """
        t_end = float(t_end)
        lo = self.delta
        a1 = max(min(0.5 * float(t_ign), t_end * 0.5), lo * 10.0)
        a2 = max(min(3.0 * float(t_ign), t_end * 0.9), a1 * 1.001)
        n = int(n)
        n1 = max(2, int(frac[0] * n))
        n2 = max(2, int(frac[1] * n))
        n3 = max(2, n - 1 - n1 - n2)
        seg = (np.logspace(np.log10(lo), np.log10(a1), n1),
               np.logspace(np.log10(a1), np.log10(a2), n2 + 1)[1:],
               np.logspace(np.log10(a2), np.log10(t_end), n3 + 1)[1:])
        t = np.concatenate(([0.0], *seg))
        return t[np.concatenate(([True], np.diff(t) > 0))]

    @staticmethod
    def phase_masks(times, t_ign):
        """induction / ignition / post, on the generator's own boundaries."""
        t = np.asarray(times)
        if not t_ign or not np.isfinite(t_ign):
            return {}
        return {"induction": t < 0.5 * t_ign,
                "ignition": (t >= 0.5 * t_ign) & (t <= 3.0 * t_ign),
                "post": t > 3.0 * t_ign}

    def grid_quality(self, times, t_ign=None):
        """How coarse this grid is in the model's own coordinate.

        The model marches RK4 on exactly these points, so max|du| is the real
        accuracy knob -- and it is invisible on a time axis. Cantera adapts and
        is unaffected by either number, which is why a grid can look innocent
        and still break only the model.
        """
        u = self.u_of_t(times, t_ign)
        du = np.diff(u)
        return {"u_min": float(u.min()), "u_max": float(u.max()),
                "du_max": float(du.max()), "du_med": float(np.median(du)),
                "outside": bool(u.min() < 0.0 or u.max() > 1.0)}

    @torch.no_grad()
    def predict(self, T0, P, phi, steam, t_end=None, n=None, times=None,
                Y0=None, t_ign=None, substeps=2, grid_mode="log"):
        """Integrate at a new operating point. Y0 optional; see initial_Y."""
        if t_ign is None:
            tp = self.predict_tign(T0, P, phi, steam)
            t_ign = None if tp is None else float(np.atleast_1d(tp)[0])
        if times is None:
            te = t_end if t_end else 1e-2
            nn = n if n else self.n_points
            times = (self.phase_grid(te, nn, t_ign)
                     if (grid_mode == "phase" and t_ign) else self.grid(te, nn))
        times = np.asarray(times, dtype=np.float64)
        if Y0 is None:
            raise ValueError("Y0 is required: build it from the operating point "
                             "with chemnode_gui.initial_Y(T0, P, phi, steam)")
        u = self.u_of_t(times, t_ign).astype(np.float32)
        z0 = self.z_from_physical(T0, Y0)
        c = self.cond_vector(P, T0, phi, steam)
        dev = self.device
        t_ = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)   # noqa: E731
        pred = self.model.rollout(t_(z0), t_(u[None, :]), t_(c),
                                  substeps=substeps)[0].cpu().numpy()
        phys = self.z_to_physical(pred)
        return {"t": times, "u": u, "T": phys[:, 0], "Y": phys[:, 1:],
                "columns": self.state_columns, "t_ign_pred": t_ign,
                "extrapolated_u": bool(u.min() < 0.0 or u.max() > 1.0)}


# ==========================================================================
# CONFIG / CHECKPOINT RECONSTRUCTION
# ==========================================================================
# Inlined rather than imported: the export step must run from this one file
# plus chemnode_train.py, so a model can be packaged without dragging the
# diagnostic scripts along. deployed_eval.py keeps its own copy for its own
# use; the two are identical and neither imports the other.

# A field missing from config.json means that run PREDATES the field, so the
# right value is the behaviour before it existed -- not today's default, which
# is the opposite in every case that matters. Getting this wrong is silent:
# align_range_pad_dex and tign_source both change the time coordinate, so a
# wrong guess evaluates a correct model on a grid it never saw and reports the
# result as a modelling failure.
PRE_REVISION = {
    "tign_source": "measured",
    "align_range_pad_dex": 0.0,
    "use_initializer": False,
    "n_augment": 0,
    "species_scale": "per_channel",
    "rate_loss": "absolute",
    "rate_head": "linear",
    "phase_balance": False,
    "init_noise_decades": None,
    "rate_space": "logtime",
}


# --------------------------------------------------------------------------- #
def build_cfg(run_dir: Path, data_dir: str | None, legacy: bool, verbose=True):
    """Reconstruct the Config this checkpoint was trained with, as far as the
    run directory records it."""
    cfg = ct.Config()
    known = {f.name for f in fields(ct.Config)}

    cj = run_dir / "config.json"
    if cj.exists():
        raw = json.loads(cj.read_text())
        stale = sorted(set(raw) - known)
        absent = sorted(known - set(raw))
        for k, v in raw.items():
            if k in known:
                setattr(cfg, k, v)
        rolled = []
        for k in absent:
            if k in PRE_REVISION:
                setattr(cfg, k, PRE_REVISION[k])
                rolled.append(f"{k}={PRE_REVISION[k]!r}")
        if verbose:
            print(f"config: {cj}  ({len(raw)} keys)")
            if stale:
                print(f"        ignored, no longer in Config: {', '.join(stale)}")
            if rolled:
                print(f"        not recorded -> pre-revision value: "
                      f"{', '.join(rolled)}")
            rest = [k for k in absent if k not in PRE_REVISION]
            if rest:
                print(f"        not recorded, today's default (harmless here): "
                      f"{', '.join(rest)}")
    else:
        print(f"config: {cj} NOT FOUND -- falling back to current defaults"
              + (" + --legacy" if legacy else "")
              + "\n        this is a guess; check the printed regime below "
                "against the run's train.log")

    if legacy:
        cfg.time_align = "ignition"
        cfg.align_noise_dex = 0.0
        cfg.species_scale = "per_channel"
        cfg.rate_loss = "absolute"
        cfg.rate_head = "linear"
        cfg.phase_balance = False
        cfg.init_noise_decades = None
        cfg.tign_source = "measured"
        cfg.rate_loss_eps = 0.02
        cfg.box, cfg.box_w = 1.15, 0.08
        cfg.n_time_fourier = 8

    if data_dir:
        cfg.data_dir = data_dir
    cfg.out_dir = str(run_dir)
    # `stages` round-trips through JSON as lists; nothing here trains, but keep
    # the type honest in case something downstream indexes it.
    if isinstance(cfg.stages, list):
        cfg.stages = tuple(tuple(s) for s in cfg.stages)
    return cfg


def infer_arch(sd, ds, cfg):
    """Read the architecture back off the weights.

    The state dict is ground truth about the network in a way config.json is
    not: it survives renamed flags, changed defaults and hand-edited configs.
    Every quantity below is recoverable from a tensor shape, so recover it
    rather than trusting a field that may be missing.
    """
    notes = []

    def setv(k, v):
        if getattr(cfg, k, None) != v:
            notes.append(f"{k}: {getattr(cfg, k, None)!r} -> {v!r}")
            setattr(cfg, k, v)

    if "ode.net.out.weight" in sd:
        setv("n_augment", int(sd["ode.net.out.weight"].shape[0]) - ds.d_state)
    if "ode.net.inp.weight" in sd:
        setv("hidden", int(sd["ode.net.inp.weight"].shape[0]))
    # The Fourier bank is a registered buffer, so its presence and length say
    # both whether the field reads a clock and at what resolution.
    if "ode.net.time_feat.freq" in sd:
        setv("use_time_input", True)
        setv("n_time_fourier", int(sd["ode.net.time_feat.freq"].shape[0]))
    elif "ode.net.inp.weight" in sd:
        setv("use_time_input", False)
    setv("rate_space", "physical" if "ode.phys_Q" in sd else "logtime")
    setv("use_initializer", any(k.startswith("init.") for k in sd))
    if any(k.startswith("init.net.0.weight") for k in sd):
        setv("init_hidden", int(sd["init.net.0.weight"].shape[0]))
    n_blocks = len({k.split(".")[3] for k in sd
                    if k.startswith("ode.net.blocks.")})
    if n_blocks:
        setv("n_blocks", n_blocks)

    if notes:
        print("arch  : recovered from the weights -- " + "; ".join(notes))
    return cfg


def load_model(ckpt_path: Path, ds, cfg, device):
    """Build the network and load the weights strictly.

    The checkpoint's own `rate_scale` wins over the one the dataset just
    recomputed. They are normally identical, but if anything about the config
    reconstruction is off, the recomputed one would silently rescale the whole
    field and the error would look like a modelling result instead of a loading
    bug.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    rs = ck.get("rate_scale", None) if isinstance(ck, dict) else None
    if rs is None:
        rs = ds.rate_scale
        print("weights: checkpoint carries no rate_scale; using the recomputed "
              "one (fine when the config is right, silently wrong when it is not)")
    else:
        rs = np.asarray(rs)
        d = float(np.max(np.abs(np.asarray(rs) - np.asarray(ds.rate_scale))
                         / np.maximum(np.abs(rs), 1e-30)))
        if d > 1e-6:
            print(f"weights: checkpoint rate_scale differs from the recomputed "
                  f"one by up to {100*d:.2f} % -- using the checkpoint's. "
                  f"A large number here means the config or the dataset is not "
                  f"the one this model was trained on.")

    cfg = infer_arch(sd, ds, cfg)
    model = ct.ChemNODE(ds.d_state, ds.d_cond, cfg, rs).to(device)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        have, want = set(sd), set(model.state_dict())
        print("\n!! the checkpoint does not match the reconstructed "
              "architecture.\n"
              "   The config was reconstructed wrongly -- the numbers below "
              "would be meaningless, so nothing is printed.\n")
        if want - have:
            print(f"   missing from checkpoint : {sorted(want - have)[:8]}")
        if have - want:
            print(f"   present but unexpected  : {sorted(have - want)[:8]}")
        print("\n   Most common causes, in order:\n"
              "     use_initializer  (adds/removes every 'init.*' key)\n"
              "     n_augment        (changes the RateNet input width)\n"
              "     hidden / layers / n_time_fourier (changes layer shapes)\n"
              f"\n   torch message: {e}")
        raise SystemExit(2)
    model.eval()
    return model


# ==========================================================================
# EXPORT  (the only place the dataset is touched, once)
# ==========================================================================
def export(run_dir, out, data_dir=None, ckpt=None, tol=1e-8):
    run_dir = Path(run_dir)
    cfg = build_cfg(run_dir, data_dir, legacy=False, verbose=True)
    ds = ct.ChemDataset(cfg)
    ck = Path(ckpt) if ckpt else run_dir / "final.pt"
    model = load_model(ck, ds, cfg, torch.device("cpu"))
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    rate_scale = np.asarray(raw.get("rate_scale", ds.rate_scale))

    # THE ONE WAY TO GET A SILENTLY WRONG FILE: exporting against a dataset the
    # model was not trained on. Every constant below is refitted from whatever
    # data is loaded here, and the self-checks further down would all pass --
    # they verify that the export matches THIS dataset, not that this dataset is
    # the right one. rate_scale is the witness: it is computed from the data and
    # stored in the checkpoint at training time, so the two disagreeing means
    # the data has changed under the model.
    d = float(np.max(np.abs(rate_scale - ds.rate_scale)
                     / np.maximum(np.abs(rate_scale), 1e-30)))
    if d > 1e-3:
        raise SystemExit(
            f"\nrate_scale in {ck} differs from the one recomputed on\n"
            f"  {cfg.data_dir}\nby up to {100 * d:.1f} %.\n\n"
            f"This model was trained on different data. Exporting anyway would\n"
            f"produce a file whose normalisation constants are wrong, and every\n"
            f"check below would still pass. Point --data_dir at the dataset this\n"
            f"run used (config.json records it as the value above).")
    if d > 1e-9:
        print(f"note: rate_scale differs by {100 * d:.2g} % from the checkpoint's "
              f"-- small enough to be float noise, but worth a glance")

    tign = {"mode": None, "use_steam": False, "coef": None}
    if ds.time_align == "ignition":
        mode = ds.tign_fit_mode
        if mode not in POLY_BASES:
            raise SystemExit(
                f"basis '{mode}' cannot be exported as coefficients. Retrain or "
                f"re-evaluate with one of {POLY_BASES} (cubic is the one your "
                f"runs use), or the deployed file would need the training data "
                f"to rebuild the interpolant -- which is the thing this avoids.")
        tr = ds.train_idx
        X = tign_design(ds.T0[tr], ds.pressures[tr], ds.phi[tr], ds.steam[tr],
                        mode, ds.tign_use_steam)
        coef, *_ = np.linalg.lstsq(X, np.log10(ds.ignition_times[tr]), rcond=None)
        tign = {"mode": mode, "use_steam": bool(ds.tign_use_steam),
                "coef": coef, "terms": list(ds.tign_terms)}

    tr = ds.train_idx
    ranges = {"T0": (float(ds.T0[tr].min()), float(ds.T0[tr].max())),
              "P": (float(ds.pressures[tr].min()), float(ds.pressures[tr].max())),
              "phi": (float(ds.phi[tr].min()), float(ds.phi[tr].max())),
              "steam": (float(ds.steam[tr].min()), float(ds.steam[tr].max())),
              "t_data_max": float(ds.t.max())}

    b = {"state_dict": model.state_dict(),
         "arch": {k: getattr(cfg, k) for k in ARCH_KEYS if hasattr(cfg, k)},
         "rate_scale": rate_scale,
         "d_state": int(ds.d_state), "d_cond": int(ds.d_cond),
         "raw_lo": ds.raw_lo, "raw_hi": ds.raw_hi, "raw_range": ds.raw_range,
         "keep_mask": ds.keep_mask,
         "delta": ds.delta, "s_min": ds.s_min, "s_span": ds.s_span,
         "time_align": ds.time_align,
         "tign_source": getattr(ds, "tign_source", "n/a"),
         "cond_names": list(ds.cond_names), "cond_lo": ds.cond_lo,
         "cond_hi": ds.cond_hi, "cond_rng": ds.cond_rng,
         "species_names": list(ds.species_names),
         "species_names_all": list(ds.species_names_all),
         "state_columns": list(ds.state_columns),
         "n_points": int(ds.n_points),
         "ranges": ranges,
         "tign": tign,
         "notes": {"run_dir": str(run_dir), "ckpt": str(ck),
                   "data_dir": str(cfg.data_dir),
                   "n_train": int(len(tr))}}

    # ---- verify against the live dataset before writing -------------------
    print("\nverifying the export against ChemDataset:")
    torch.save(b, out)
    dep = Deployed(out)

    if ds.time_align == "ignition":
        a = np.asarray(dep.predict_tign(ds.T0, ds.pressures, ds.phi, ds.steam))
        c = np.asarray(ds.predict_tign(ds.T0, ds.pressures, ds.phi, ds.steam))
        e = float(np.max(np.abs(np.log10(a) - np.log10(c))))
        print(f"  t_ign correlation, all {len(a)} cases : {e:.2e} dex")
        assert e < 1e-6, "correlation mismatch"

    e = float(np.max(np.abs(dep.u_of_t(ds.t[0], None if ds.tign is None
                                       else ds.tign_used[0]) - ds.u[0])))
    print(f"  time coordinate, case 0             : {e:.2e}")
    assert e < 1e-9, "time coordinate mismatch"

    z_a = dep.z_from_physical(ds.states_phys[0, 0, 0], ds.states_phys[0, 0, 1:])
    e = float(np.max(np.abs(z_a[0] - ds.z[0, 0])))
    print(f"  state transform, case 0             : {e:.2e}")
    assert e < 1e-5, "state transform mismatch"

    c_a = dep.cond_vector(ds.pressures[0], ds.T0[0], ds.phi[0], ds.steam[0])
    e = float(np.max(np.abs(c_a[0] - ds.cond[0])))
    print(f"  conditioning vector, case 0         : {e:.2e}")
    assert e < 1e-5, "conditioning mismatch"

    # and the whole path: the deployed rollout must equal the dataset rollout
    p_a = dep.predict(ds.T0[0], ds.pressures[0], ds.phi[0], ds.steam[0],
                      times=ds.t[0], Y0=ds.states_phys[0, 0, 1:],
                      t_ign=None if ds.tign is None else float(ds.tign_used[0]))
    p_b = ct.rollout_case_aligned(model, ds, 0, torch.device("cpu"))
    e = float(np.max(np.abs(p_a["T"] - ds.from_z(p_b)[:, 0])))
    print(f"  full rollout, case 0                : {e:.2e} K")
    assert e < 1e-2, "rollout mismatch"

    mb = Path(out).stat().st_size / 1e6
    print(f"\nwrote {out}  ({mb:.1f} MB) -- the training data is no longer needed")
    print(f"  {len(dep.species_names)} species, conditions {dep.cond_names}")
    for k in ("T0", "P", "phi", "steam"):
        lo, hi = ranges[k]
        print(f"  {k:<6} train range {lo:.4g} .. {hi:.4g}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir")
    p.add_argument("--out", default="chemnode_model.pt")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--model", help="inspect an already-exported file")
    p.add_argument("--check", action="store_true")
    a = p.parse_args(argv)

    if a.model:
        d = Deployed(a.model)
        print(f"{a.model}: {len(d.species_names)} species, cond {d.cond_names}, "
              f"time_align={d.time_align}, tign basis={d.tign['mode']}")
        for k in ("T0", "P", "phi", "steam"):
            print(f"  {k:<6} {d.ranges[k][0]:.4g} .. {d.ranges[k][1]:.4g}")
        print(f"  from {d.notes.get('run_dir')} ({d.notes.get('n_train')} train cases)")
        return
    if not a.run_dir:
        raise SystemExit("--run_dir (to export) or --model (to inspect)")
    export(a.run_dir, a.out, a.data_dir, a.ckpt)


if __name__ == "__main__":
    main()
