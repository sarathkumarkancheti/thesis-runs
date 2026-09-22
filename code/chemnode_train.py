"""chemnode_train.py -- Neural-ODE for stiff chemical kinetics, no initialiser
head, REVISED to fix the induction-phase drift.

=============================================================================
WHAT CHANGED, AND WHY
=============================================================================
The symptom being fixed: in a continuous rollout the reactant channels drift by
half a decade to a decade through the induction phase, where the true
trajectory is flat to better than 1e-3 relative, and the predicted mass
fractions leave the physical range (CH4 above 1, O2 above 1). Ignition timing
is nevertheless almost exact. That combination is diagnostic -- it says the
loss never sees the induction phase, and the ignition-aligned coordinate keeps
ignition on time regardless, so the drift costs nothing during training.

Three changes, all in the REPRESENTATION rather than the architecture or the
training schedule, measured on a synthetic 5-species ignition system that
reproduces the failure (see make_testbed_data.py / run_testbed.py):

                                       spurious    T RMSE   typical   max
                                       drift                 factor   Y
  original configuration               1.150 dec    70.6 K    1.73x   24.6
  + box 1.005  (physical bound on Y)   0.905 dec    47.2 K    1.76x    3.0
  + species_scale = "physical"         0.578 dec    49.5 K    1.68x    1.5
  + rate_loss "symlog" & sinh head     0.044 dec    26.2 K    1.40x    1.2
                                       ^ the true drift is 0.050 dec

"spurious drift" is the induction-phase excursion of the channels whose true
excursion is below 0.1 decades, i.e. the ones that are supposed to sit still.
At 0.044 against a true 0.050 the reactants are as flat as the data.

  1  species_scale = "physical".  Every species channel is scaled by the same
     thing -- the physically possible range of log10(Y + y_floor) -- instead of
     by its own min-max. Under min-max, one z-unit was 6 decades in a channel
     that burns out to the floor and 0.25 in one that barely moves, so the
     squared loss weighted a one-decade error ~600x more heavily in H2O than in
     O2, inverted with respect to which channels are hard. It also makes
     |z| <= 1 mean 0 <= Y <= 1 exactly, which is what turns cfg.box from an
     arbitrary number into a physical bound.

  2  box = 1.005.  Follows from 1. The old 1.15 allowed a channel to run ~0.9
     decades past anything in the data, which is how a mass fraction of 24
     became representable.

  3  rate_loss = "symlog" WITH rate_head = "sinh".  rate_scale is an RMS set by
     the ignition spike, so the induction-phase rate target is 1e-5..1e-2 in
     normalised units while huber_delta is 0.3: the loss is flat-quadratic
     across that whole range and a field error of 1 % of rate_scale -- enough
     to drift a reactant by a decade -- is never corrected. symlog prices the
     error by its ratio; the sinh head gives the output exponential resolution
     near zero so the gradients stay conditioned. Either one alone is WORSE
     than neither (see the table in the rate_head comment).

Two things that were tried and are NOT on by default, because the measurement
said no: phase_balance (no benefit once the scaling is fixed) and the global
time coordinate (five times the T RMSE on the test bed). Both are implemented
and one flag away.

Also added: align_range_pad_dex, so a coordinate built from a PREDICTED
ignition delay cannot land outside the u range the model was trained on; and
deployed_metrics(), which reports accuracy with the delay predicted rather than
measured -- the only protocol that matches how the model will be used.

Run --legacy to reproduce the original configuration for the 'before' column.

=============================================================================
No initialiser head: the ODE covers the whole trajectory from t = 0.
=============================================================================

Identical to chemnode_train.py apart from one thing: there is no learned map
across the first interval. The ODE is responsible for the whole trajectory,
from t = 0 to the end, and every window in the shooting curriculum starts at
sample 0. The log-time coordinate, the log-mass-fraction state, the scaling,
the multiple-shooting curriculum and the anchored 550-point stage are unchanged.

    python chemnode_train_noinit.py --data_dir DATA --out_dir runs/noinit
    python chemnode_eval.py --out_dir runs/noinit --split test --solver both

(chemnode_eval.py does `from chemnode_train import ...`; change that one import
line to point here. Nothing else in the evaluation script cares.)

WHAT THIS COSTS, AND HOW TO TELL
--------------------------------
At t = 0 a CH4/O2/H2O/N2 mixture contains no radicals at all -- Y_OH, Y_H and
Y_O are identically zero -- so log10(Y + y_floor) sits exactly on the floor,
log10(y_floor). By the first stored sample those species have climbed several
decades. That first interval is a discontinuity in the *state representation*,
not a feature of the chemistry, and the baseline handled it by learning it as a
direct map rather than integrating across it.

Without that map the ODE has to cross it. Two things make this less hopeless
than it sounds:

* The regression target is already repaired. `_build_rate_targets` replaces the
  exact Cantera derivative with the data secant wherever a species sits within
  a decade of the floor (`at_floor`), because the exact value there is
  identically zero while the trajectory is about to move by decades. So
  pretraining sees a sensible slope across the first interval, not a zero.
* The field is bounded by out_gain * rate_scale with out_gain = 30, i.e. thirty
  times the largest rate anywhere in the data. Representing a steep first
  interval is therefore not the binding constraint.

What IS at risk is accuracy over that one interval, and it propagates: every
later state in a continuous rollout is downstream of it. Expect the damage to
show up as induction-phase error and ignition-delay error, which is exactly
what `chemnode_eval.py --solver both` breaks out by phase.

Before concluding anything, read the line `summary()` prints:

    first interval jump X decades -- the ODE must cross this unaided

If X is small, this file will do fine. If X is large, the cheapest fix is not
the initialiser -- it is raising `y_floor` (1e-12 -> 1e-10 or 1e-8), which
shrinks the jump by that many decades directly. You stop resolving species
below the floor, but `eval_metric_floor` already excludes those from the
reported metrics, so in practice you lose little. Run it both ways and report
the difference; that is a stronger result than either choice on its own.
"""
from __future__ import annotations

import argparse, copy, json, math, time, warnings
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# CONFIG
# ==========================================================================
#
# Central configuration for the data-driven Neural-ODE baseline.
#
# Everything that used to be scattered as module-level constants lives here so a
# run is fully described by one object (and is saved next to the checkpoint).
#

@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    data_dir: Path = Path("multipressure_hybrid_sampling_node_v2_with_derivatives/data")
    out_dir: Path = Path("runs/node_baseline")
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # ------------------------------------------------------- state representation
    # Mass fractions are learned as log10(Y + y_floor).  y_floor is the smallest
    # mass fraction we care about; anything below it is numerically irrelevant
    # for a combustion surrogate and would otherwise dominate every error metric.
    # ---- time coordinate ---------------------------------------------------
    # "global"   : u = (log10(t+delta) - s_min)/span, shared by all cases.
    # "ignition" : time is first divided by each case's own ignition delay, so
    #              tau = (t+delta)/t_ign and ignition lands at the SAME u for
    #              every case. In log space this is a per-case shift, so the
    #              transform stays smooth -- dt/du keeps exactly the same form
    #              and no kink is introduced.
    #
    # Why: with 3.1 decades of ignition-delay spread, a cold case spends nearly
    # its whole u range in induction where the true field is ~0, so a small
    # positive bias in the learned field accumulates over hundreds of steps and
    # ignites it early. Measured on the global coordinate: rank correlation
    # -0.83 between T0 and typical factor, +0.80 with ignition delay. Aligning
    # ignition makes every case occupy a comparable fraction of its integration
    # in each phase, so one rate_scale fits them all.
    #
    # KEPT AS "ignition", on evidence rather than on the argument that it hands
    # the model the answer. Measured on the synthetic test bed, with everything
    # else at the settings below:
    #
    #     coordinate            T RMSE    typical factor   spurious drift
    #     ignition-aligned       26.2 K        1.40x          0.044 dec
    #     global log-time       129.2 K        1.76x          0.519 dec
    #
    # and, more to the point, replacing the MEASURED delay with one predicted
    # from the operating conditions cost 26.2 -> 29.9 K and nothing at all in
    # species error. So alignment is not what breaks at inference: the
    # correlation is a function of (T0, P, phi, steam), which is exactly what a
    # new operating point supplies, and its error enters as a small rigid shift.
    #
    # What to check on the real dataset, because both of these are properties
    # of the data and not of the method:
    #   * chemnode_diagnose.py section 5 prints the correlation's val and WORST
    #     error. Below ~0.05 dex, alignment is close to free. Above it, set
    #     align_noise_dex and re-measure.
    #   * the same section prints s_span both ways. Alignment spreads the
    #     endpoints by roughly the spread of the delays, so it can WIDEN the
    #     shared range and cost each case resolution.
    #
    # Set "global" if you want a model with no auxiliary correlation anywhere in
    # the inference path; raise n_time_fourier to 12 if you do, because ignition
    # then sits in a ~0.02-wide window of u instead of a fixed one.
    time_align: str = "ignition"
    # Ignition delay is needed to BUILD the coordinate, so at inference on a new
    # condition it must be predicted rather than known. A log-linear Arrhenius
    # fit on the training cases does this to within a few percent; the fitted
    # coefficients are stored with the dataset and reported in summary().
    # Which delay DEFINES the coordinate.
    #   "measured"   log10(t_ign) from case_metadata.npz. Ignition then lands at
    #                exactly the same u in every case, which is the sharpest
    #                possible alignment -- but at inference the delay has to be
    #                predicted, so the deployed coordinate is shifted by the
    #                correlation's error and training never saw that shift.
    #   "predicted"  log10(t_ign) from the correlation, for TRAINING TOO. Train
    #                and inference then build the coordinate by the identical
    #                formula and the mismatch is exactly zero, not merely small.
    #                The measured delay is used only to FIT the correlation,
    #                which is a training target like the trajectories are.
    #
    # The cost of "predicted" is that ignition lands at u0 +/- eps instead of at
    # u0, so the alignment is blurred by the correlation's error. Judge that
    # against the ignition width: chemnode_diagnose.py section 7 prints the
    # 10-50 % rise as du, and section 5 prints eps/s_span. On this dataset that
    # is 0.0003 against 0.0065, a 4 % blur -- cheap next to removing the
    # mismatch outright.
    #
    # Note this is NOT the same as align_noise_dex. Jitter makes the offset
    # random, so the model can only average over it; "predicted" makes the
    # offset a deterministic function of the operating point, which the model
    # also receives as conditioning, so it can learn it instead.
    #
    # DEFAULT IS "predicted", because that is the only setting under which the
    # measured ignition delay never enters the model. Under "measured",
    # ds.u[c] is built from each case's own true delay INCLUDING test cases, so
    # every downstream script that uses ds.u -- chemnode_eval.py,
    # chemnode_eval_elements.py, the case plots -- silently reports an upper
    # bound rather than a deployable number, and deployed_metrics() is the only
    # honest line in the output. Under "predicted" the coordinate is the
    # deployed coordinate everywhere, so those scripts become correct with no
    # modification and there is one number to report instead of two.
    #
    # Use "measured" only to reproduce earlier runs. It is not the safer
    # choice: if the correlation is poor, "measured" hides that fact rather
    # than fixing it.
    tign_source: str = "predicted"
    tign_fit_report: bool = True
    # Basis for that correlation: "arrhenius" (1000/T0, log10 P, phi[, steam]),
    # "arrhenius_log" (same with log10 phi), or "quadratic" (those plus every
    # pairwise product). All three are fitted and scored on train/val/test every
    # run; this picks the one used. Choose on the VALIDATION column.
    #
    # This is the highest-leverage knob for DEPLOYED accuracy and it costs no
    # retraining: the network takes its ignition timing from t_ign, so its error
    # under predicted alignment is the correlation's error almost exactly.
    tign_fit_mode: str = "arrhenius"
    # ^ "arrhenius" | "arrhenius_log" | "quadratic" | "cubic" | "rbf".
    # Judge on the VALIDATION column that summary() prints, and judge it on the
    # WORST error as well as the mean: aggregate trajectory error under
    # predicted alignment is set by the tail, not the average.

    y_floor: float = 1e-12

    # ---- how the species channels are scaled -------------------------------
    # "physical"     every species channel uses the SAME scale, namely the
    #                physically possible range of a mass fraction:
    #                    z = 2*(log10(Y+y_floor) - log10(y_floor))/Dtot - 1,
    #                    Dtot = -log10(y_floor),
    #                so z = -1 is Y = 0, z = +1 is Y = 1, and one z-unit is
    #                Dtot/2 decades in EVERY channel.
    # "per_channel"  the original behaviour: min-max per channel over the
    #                training split, with scale_margin padding.
    #
    # Why this is the single most consequential line in the file. Under
    # "per_channel" a channel that burns out to the representation floor spans
    # ~12 decades while one that barely moves (H2O: 0.16 -> 0.5) spans ~0.5, so
    # one z-unit is worth 6 decades in the first and 0.25 in the second. The
    # squared loss then weights a one-decade error ~600x more heavily in H2O
    # than in O2 -- and O2 is the channel that is hard. Nobody chose that
    # weighting; it is a side effect of normalising each channel by its own
    # range. Under "physical" the loss is literally mean |dlog10 Y| in decades,
    # i.e. the quantity the evaluation reports, and |z| <= 1 is exactly the
    # statement 0 <= Y <= 1 -- so cfg.box below stops being an arbitrary
    # number and becomes the physical bound.
    species_scale: str = "physical"
    # Species whose log10 range over the training split is below this are treated
    # as constants (e.g. N2 / AR in a CH4/O2 mixture) and removed from the state.
    constant_species_tol: float = 1e-8
    drop_constant_species: bool = True
    # Fractional padding added to the min/max used for [-1, 1] scaling, so that
    # a slightly out-of-range prediction is still representable.
    scale_margin: float = 0.05
    # Per-channel scale of dz/du: 'rms' (default) or 'quantile'.
    rate_scale_mode: str = "rms"
    rate_quantile: float = 0.999
    # exact-derivative targets outside +/- target_cap * rate_scale are replaced
    # by the (resolvable) central difference of the sampled trajectory
    target_cap: float = 25.0

    # ------------------------------------------------------------------ splits
    train_frac: float = 0.70
    val_frac: float = 0.15

    # ------------------------------------------------------------------- model
    hidden: int = 512
    n_blocks: int = 4          # residual blocks
    dropout: float = 0.0
    use_time_input: bool = True
    # Octave-spaced: the finest feature has period 2 / 2^(n-1) in u, so n = 8
    # resolves nothing narrower than 0.0156 of the u range.
    #
    # RAISED FROM 8. Thermal runaway spans ~0.2 decades of log-time, so on a
    # 14.9-decade coordinate it occupies du ~ 0.013 -- NARROWER than the finest
    # feature n = 8 provides. The network then has no time feature capable of
    # representing the event it most needs to place, whichever coordinate is
    # used. n = 12 gives 0.00098, which resolves it with room to spare, at a
    # cost of 8 extra input dimensions.
    #
    # The number that matters is ignition_width_in_u, which
    # chemnode_diagnose.py section 7 prints for your dataset; pick n so that
    # 2 / 2^(n-1) is a few times smaller than it.
    n_time_fourier: int = 12   # random-free, log-spaced Fourier features of u
    # Output is out_gain * tanh(raw / out_gain) in *normalised rate* units, so
    # |dz/du| <= out_gain * rate_scale.  Bounded => the integration can never
    # blow up, but (unlike an exponentially decoded output) the bound is loose
    # enough that the network never fights it on real data.
    # ---- the first interval -----------------------------------------------
    # log10(Y) is singular at Y = 0. At t = 0 a CH4/O2/H2O/N2 mixture contains
    # no radicals at all, so 23 of the 27 retained channels sit exactly on the
    # representation floor and Cantera returns dY/dt identically zero for
    # precisely the species that are about to move fastest. By the first stored
    # sample they have climbed ~3 decades: measured on this dataset the state
    # moves |dz| = 0.78 across interval 0 and ~0.015 across every interval
    # after it, a factor of fifty.
    #
    # That is a discontinuity in the REPRESENTATION, not a feature of the
    # chemistry, and asking a bounded right-hand side to cross it is asking the
    # field to carry one pathological spike that has nothing to do with the
    # dynamics it is meant to learn. A direct map over that one interval,
    # z(u_1) = z_0 + d(z_0, c, u_1), handles it and the ODE takes over from the
    # first resolved point. ds.anchor becomes 1, so no window and no
    # pretraining sample starts at t = 0.
    #
    # Worth knowing for the autonomous variant: the map takes u_1, which is a
    # clock, but u_1 is the FIRST REQUESTED OUTPUT TIME and not t_ign -- the
    # caller supplies it as part of the time grid, so it is available at
    # inference and it is not the quantity the autonomy argument is about. It
    # also removes the single most degenerate state in the dataset (23 of 27
    # channels pinned, hence indistinguishable) from the ODE's domain, which
    # helps the autonomous field more than it helps the clock-driven one.
    #
    # Be honest about it in the write-up: with this on, the model is an
    # autonomous flow plus one explicit map across a representation
    # singularity, which is a stronger claim than pretending the flow can cross
    # it.
    use_initializer: bool = True
    init_hidden: int = 256
    # z1 = clamp(z0 + init_gain * tanh(d), -1, 1). Under species_scale
    # "physical" one z-unit is 6 decades, so 1.5 permits a 9-decade jump
    # against a measured 3.08 -- generous without being unbounded. The clamp to
    # +/-1 is exactly 0 <= Y <= 1 on that scale, so the map cannot emit an
    # unphysical mass fraction at all.
    init_gain: float = 1.5
    init_loss_weight: float = 1.0

    # ---- the autonomous variant -------------------------------------------
    # Setting use_time_input = False makes the field f(z, c) instead of
    # f(z, u, c). That is the physically correct structure: a closed,
    # constant-pressure, adiabatic reactor is an autonomous dynamical system, so
    # a right-hand side that reads a clock cannot be right, and a coordinate
    # built from t_ign stops being needed at all.
    #
    # It is not free. In log-time the TRUE right-hand side is
    #
    #     dz/du = ln(10) * s_span * (t + delta) * dz/dt
    #
    # which depends explicitly on t, so an autonomous field can only reproduce
    # it if the state z determines t -- i.e. if the trajectory never revisits a
    # state. The log floor breaks exactly that. While every radical still sits
    # at log10(y_floor) and the majors and T have not moved, z is CONSTANT while
    # t advances, so the true flow has to leave a state at which an autonomous
    # field is stationary. No autonomous field in z can do that, at any width or
    # depth: it is an obstruction in the phase space, not a fitting problem.
    # chemnode_diagnose.py section 7 measures how much of your data is affected.
    #
    # n_augment adds that many extra state channels, all starting at zero,
    # integrated by the same field and never compared against data. They give
    # the flow somewhere to keep the information the floor destroyed, which
    # restores the Markov property and removes the obstruction. This is the
    # augmented neural ODE of Dupont, Doucet & Teh (NeurIPS 2019), introduced
    # for precisely this class of unrepresentable-by-an-autonomous-flow problem,
    # so it is a citable fix rather than an invention.
    #
    # 0 disables it. 8 is a reasonable starting point; the cost is 8 input and
    # 8 output dimensions.
    n_augment: int = 0

    # ---- which rate the network learns --------------------------------------
    # "logtime"   the network emits dz/du / rate_scale (everything before this
    #             option existed).
    # "physical"  the network emits the PHYSICAL-time rate of the raw state
    #             (K/s for T, decades/s for log10 Y), symlog-compressed as
    #             sign(x)*log10(1+|x|) and divided by its training maximum per
    #             channel, through a tanh head bounded at phys_bound. The ODE
    #             still integrates in u, with the exact analytic factor
    #                 dz/du = ln10 * s_span * 10^(s_min + s_span u)
    #                         * (2 / raw_range) * dz_raw/dt .
    #             So the learned field g(z, c) is a genuine autonomous chemical
    #             source term: no clock has to be reconstructed from the state,
    #             because the clock is supplied analytically. rate_space_check.py
    #             on the 429-case data: 1-R^2 ratio physical/log-time 0.08-0.29
    #             for T in every phase, 0.29/0.95/0.65 for species.
    #             Requires time_align = "global" and n_augment = 0. The rate
    #             loss is compared directly in the compressed space, so
    #             rate_loss is forced to "absolute".
    rate_space: str = "logtime"
    phys_bound: float = 1.05     # just above the largest training target
    # The decoded dz/du is soft-limited at phys_cap_mult * target_cap *
    # rate_scale -- the band the log-time targets were clipped to, with room.
    # It is a solver-side limiter that keeps RK4 stable if the field produces a
    # large physical rate at a late time (where the analytic factor is large);
    # on the training targets it changes dz/du by < 2 %.
    phys_cap_mult: float = 4.0
    # Compression threshold of the physical rate: s = sign(x)log10(1 + |x|/eps),
    # in raw units (K/s for T, decades/s for species). Below eps the code is
    # linear, so eps is the finest rate the network can resolve cheaply. An
    # error of eps sustained over the whole horizon t_max drifts eps * t_max,
    # so eps is DERIVED as drift budget / t_max: rate_drift_tol_K / t_max for
    # T and rate_drift_tol_decades / t_max for species (5 K and 0.05 dec over
    # 1000 s -> 5e-3 K/s and 5e-5 dec/s). The notebook's eps = 1 is fine on a
    # millisecond horizon and resolves nothing that matters at t ~ 1000 s,
    # which is why the first physical runs diverged late. None = derive.
    phys_eps: float | None = None
    # Where the RATE loss is measured, for rate_space = "physical".
    # "logtime"  the decoded field is compared with the dz/du targets exactly
    #            as in auton_clean (same symlog, same derived eps). The network
    #            still REPRESENTS the autonomous physical rate g(z, c); only the
    #            weighting of its errors changes: an error in g at time t moves
    #            the trajectory by error * (t + delta) per unit u, so the loss
    #            weights it by (t + delta). Without that, an error of 1e-3 in q
    #            is harmless during ignition and ~2 decades per step at
    #            t ~ 1000 s -- which is how the first real-data pretrain
    #            (loss in q) diverged to +892 dex.
    # "q"        Huber directly on q. Time-blind; only safe on short horizons.
    phys_loss: str = "logtime"

    out_gain: float = 30.0
    # ---- how the rate head resolves SMALL rates ----------------------------
    # "linear"  out = out_gain * tanh(raw / out_gain).  Linear near zero, so
    #           the resolution near zero is the same as the resolution near the
    #           ignition spike: uniform in absolute rate.
    # "sinh"    out = out_gain * sinh(k * tanh(raw)) / sinh(k).  The same bound
    #           at +/- out_gain, analytic everywhere, but exponential in the
    #           pre-activation, so equal steps in raw give equal RATIOS in the
    #           output. The slope at zero is out_gain * k / sinh(k), so the
    #           zoom over a linear head is sinh(k)/(k*out_gain): NOTHING at
    #           k = 6 (33.6/30 ~ 1.1), 12x at k = 8, 37x at k = 10. Only k >= 9
    #           buys anything -- and the cost is that the slope near |a| = 1 is
    #           ~out_gain*k, so the field's Lipschitz constant in z grows by
    #           the same factor and the ODE gets stiffer exactly where it is
    #           already worst.
    #
    #           Only use this PAIRED with rate_loss = "symlog": the two
    #           together give a gradient w.r.t. the pre-activation that is
    #           roughly constant across five decades of rate, whereas either
    #           alone is badly conditioned in one direction or the other.
    #           "sinh" with k = 10 IS the default, on measurement. The pairing
    #           matters more than either half: on the synthetic test bed, with
    #           the physical species scale already in place,
    #
    #               rate_loss   head     T RMSE   typical   spurious drift
    #               absolute    linear    49.5 K   1.68x      0.578 dec
    #               symlog      linear   118.7 K   3.85x      2.962 dec
    #               symlog      sinh      26.2 K   1.40x      0.044 dec
    #
    #           The middle row is the badly conditioned half-fix: a symlog loss
    #           on a linear head gives the small rates a gradient ~1/eps and the
    #           large ones ~1/|rate|, so the ignition spike gets trampled. Do
    #           not ship one without the other.
    rate_head: str = "sinh"
    rate_head_k: float = 10.0
    cond_use_T0: bool = True
    cond_use_phi: bool = True
    # Steam is a fourth design dimension when the dataset carries it. Unlike
    # O2/CH4 -- which barely shifts ignition delay -- steam changes the product
    # composition directly through reforming and water-gas shift, so the network
    # needs it to place the syngas ratio correctly. Ignored automatically for a
    # dataset generated without steam, since the column is then constant.
    cond_use_steam: bool = True
    # log10(Y) is singular at Y = 0.  At t = 0 a CH4/O2 mixture contains no
    # radicals at all, so every radical jumps from the floor to a finite value
    # inside the *first* sampled interval -- several decades in one step, which
    # no bounded ODE right-hand side can reproduce and which is the single
    # biggest source of error in a naive log-space NODE.  Rather than pretend
    # the ODE can do it, a small initialiser head learns the flow map over that
    # one interval, z(u_1) = z_0 + d(z_0, c, u_1); the ODE runs from u_1 on.
    # No initialiser head in this variant: the ODE covers sample 0 onward.
    # y_floor above is the main lever on how hard that first interval is --
    # raising it shrinks the jump the field has to cross, decade for decade.

    # --------------------------------------------------------------- pretraining
    pretrain_epochs: int = 600
    pretrain_lr: float = 1e-3
    pretrain_bs: int = 8192
    pretrain_wd: float = 1e-6

    # ------------------------------------------------- multiple-shooting curriculum
    # (window_len, stride, epochs, lr)
    stages: tuple = (
        (8, 4, 40, 1.5e-4),
        (16, 8, 40, 1.0e-4),
        (32, 16, 40, 1e-4),
        (64, 32, 120, 6e-5),
        (128, 64, 120, 3e-5),
    )
    rk4_substeps: int = 2        # RK4 sub-steps per *data* interval
    # Local error tolerance of the RK4 stepper, in z units (one z unit is the
    # half-range of a channel: ~1000 K for T, 6 decades for a species), per
    # data interval. None = plain fixed-step RK4, as before. See rk4_rollout.
    rk4_tol: float | None = None
    rk4_max_substeps: int = 64
    # Hard bound on the STATE: after every RK4 sub-step the physical channels
    # are clamped to [-state_clamp, +state_clamp]. With species_scale
    # 'physical', |z| <= 1 is exactly y_floor <= Y <= 1 and T inside the data
    # range. The gate below only limits the continuous field; a discrete step
    # can jump past it (one sub-step may move z by ~1e4 under the physical
    # rate cap), and nothing brings the state back. None = off (old runs).
    state_clamp: float | None = None
    # Self-limiting field, replacing the old hard clamp inside the stepper.
    #
    # With species_scale = "physical" the species part of |z| <= 1 IS the
    # statement 0 <= Y <= 1, so this is no longer a free parameter: 1.02 leaves
    # a 2 % margin for the integrator to overshoot into and nothing more. At
    # 1.15 with per-channel scaling the gate allowed a channel to run ~0.9
    # decades past anything in the data, which is how a predicted CH4 mass
    # fraction of 1.3 and an O2 of 3 became representable in the first place.
    #
    # Note what this is not. _gate damps the outward field smoothly rather than
    # clamping the state, deliberately -- a clamp is invisible to the solver and
    # a restoring force makes the system stiff -- so a long RK4 step can still
    # overshoot the nominal bound. Measured on the test bed it turns a max
    # predicted Y of 24.6 into 1.18 against a nominal 1.07. That is a bounded
    # excursion instead of an unbounded one, not a guarantee; the guarantee
    # needs the element-conservation projection, which belongs in the
    # physics-constrained model and not in a data-driven baseline.
    # sized so that (box - 1) * decades_per_unit is a few per cent in Y rather
    # than a few tenths of a decade: with y_floor 1e-12 one z-unit is 6 decades,
    # so 1.005 caps Y at ~1.07 and the gate is 1 to machine precision anywhere
    # a real mass fraction lives. summary() prints the resulting bound -- check
    # it rather than trusting the number here, because it depends on y_floor.
    box: float = 1.005
    box_w: float = 0.003  # gate width; NOT a restoring rate (see ChemODE._gate)
    # Batch budget for the CURRICULUM stages. batch = budget // window_len,
    # so holding the budget fixed keeps memory roughly flat as windows grow.
    # 8192 is the value the working baseline used; raising it doubles the batch
    # at stages 2-4 without raising their learning rates, which under-trains
    # them.
    window_batch_budget: int = 8192
    # Separate budget for the ANCHOR stage, which is where memory actually
    # binds: the graph is anchor_len * substeps * 4 evaluations deep, so
    # activation memory scales as batch * anchor_len. A single shared budget
    # forces a compromise -- large enough for a sensible anchor batch means far
    # too large for the curriculum, and vice versa. At 550 points this gives
    # batch 16 (16 * 550 = 8800), close to the footprint the baseline ran at.
    # Raise only if the GPU has headroom; note it changes the optimisation, so
    # hold it fixed across runs being compared.
    anchor_batch_budget: int = 9000
    max_window_batch: int = 512
    traj_weight: float = 1.0
    end_weight: float = 1.0
    deriv_weight_start: float = 1.0   # derivative-matching regulariser at stage 1
    deriv_weight_end: float = 0.05    # ... at the last stage
    # Gaussian noise added to the window's initial state during training.  This
    # is what makes the *continuous* rollout robust: the model is forced to be
    # contractive towards the data manifold instead of only accurate on it.
    #
    # SPECIFIED IN DECADES, not in normalised units, because under
    # "per_channel" scaling the same normalised number meant 0.2 decades on O2
    # and 0.008 decades on H2O -- a 25x difference nobody asked for. Training
    # the model to tolerate 0.2 decades of uncertainty in O2 during induction
    # is training it to accept exactly the drift that shows up in the plots:
    # the noise budget has to be SMALLER than the accuracy you want, and 0.2
    # decades is a factor of 1.6.
    init_noise_decades: float = 0.03
    # Legacy path: used directly, in normalised units, when
    # init_noise_decades is None.
    init_noise: float = 0.03        # on species channels (normalised units)
    init_noise_T: float = 0.004     # on the temperature channel
    # Jitter on the ALIGNMENT, in decades of ignition delay. Same idea as the
    # initial-state noise above, applied to the other input the model gets.
    #
    # Why it exists. At inference t_ign is predicted, not known, so the u grid
    # is shifted by the correlation's error -- a constant shift, which leaves
    # every du untouched and only moves the absolute position in u. Training
    # never shows the model a shifted grid, so it is free to become
    # "clock-driven": ignition-alignment puts ignition at the same u in every
    # case, and the cheapest way to fit that is to read the clock. A
    # clock-driven field inherits the delay error one for one. A field that
    # leans on z and cond instead does not.
    #
    # Setting this to the measured `tign_fit_rms_dex` trains under exactly the
    # alignment error deployment will produce, which forces the field to stop
    # trusting u so precisely. 0.0 disables it and reproduces a run made before
    # this existed.
    #
    # Measure before you switch it on: chemnode_inference.py reports the gap
    # between true and predicted alignment, and if that gap is already small
    # the model is state-driven and needs no help.
    #
    # Left at 0.0, again on evidence: with a correlation good to ~0.01 dex there
    # is nothing to be robust to, and switching the jitter on cost 26.2 -> 30.1
    # K on the test bed. Set it to None ("use the correlation's own validation
    # RMS") when chemnode_diagnose.py section 5 reports an error above roughly
    # 0.05 dex, or when the WORST-case error is several times the RMS -- that is
    # the regime where the model's reliance on the clock starts to cost more
    # than the jitter does. Ignored entirely when time_align is "global".
    align_noise_dex: float | None = 0.0
    # Widen the shared s range by this many decades at each end when the
    # coordinate is ignition-aligned, so a grid built from a PREDICTED delay
    # still lands inside [0, 1]. None means 3x the correlation's validation
    # RMS. 0.0 reproduces the earlier runs.
    align_range_pad_dex: float | None = None
    t_weight: float = 5.0        # loss weight of the temperature channel
    huber_delta: float = 0.3
    grad_clip: float = 0.5

    # ---- rate-regression loss geometry ------------------------------------
    # "symlog"    the derivative loss is Huber( asinh(pred/eps), asinh(tgt/eps) ),
    #             i.e. an error measured in RATIOS above eps and linearly below
    #             it. "absolute" is the original Huber on the raw normalised
    #             rate.
    #
    # This is the other half of the induction problem. rate_scale is an RMS over
    # the whole trajectory and is therefore set by the ignition spike, so the
    # induction-phase target lands at 1e-5..1e-2 in normalised units while
    # huber_delta is 0.3. The loss is exactly quadratic across that entire
    # range, which means a field error of 1 % of rate_scale -- enough to drift a
    # reactant by a decade over the induction, see chemnode_diagnose.py section
    # 3 -- contributes ~1e-4 of the loss and is never fixed. Under "symlog" the
    # same error is a factor-of-two error in a small rate and is priced like
    # one.
    rate_loss: str = "symlog"
    # Where symlog turns over, in normalised rate units: below this a rate is
    # treated as indistinguishable from zero, so eps IS the accuracy the loss
    # demands of the field.
    #
    # None means DERIVE IT, per channel, from the drift budget below. That is
    # not a convenience -- a fixed value cannot be right, because the accuracy
    # needed to hold a channel still through induction is
    #
    #     eps_k = tol / ( rate_scale_k * du_induction * (range_k / 2) )
    #
    # and rate_scale depends on s_span, which depends on the time horizon and
    # on the first sample time of the dataset. Measured: on a 10-decade span
    # with rate_scale ~30 the answer is ~4e-4; on a 15-decade span with
    # rate_scale ~250-800 it is ~1e-4 for the reactants. Setting 0.02 on the
    # second dataset would declare the entire induction phase to be zero and
    # the loss would ignore exactly what it is meant to fix.
    rate_loss_eps: float | None = None
    # The drift budget the derivation targets: how many decades a species is
    # allowed to wander through the induction phase, and how many kelvin the
    # temperature channel is. 0.05 decades is a 12 % error in a mass fraction.
    rate_drift_tol_decades: float = 0.05
    rate_drift_tol_K: float = 5.0
    # ---- phase balance -----------------------------------------------------
    # Normalise the trajectory loss so induction, ignition and the post-ignition
    # tail each contribute one third, instead of in proportion to how much the
    # state happens to be moving. Without it the fit is driven almost entirely
    # by the ignition window: it holds ~15 % of the points and ~all of the |dz|.
    #
    # The phase boundaries come from each case's TRUE ignition time, which is in
    # case_metadata.npz. That is legitimate: it is a training-time loss weight,
    # not an input to the model, so it costs nothing at inference. Do not
    # confuse it with time_align, which does put t_ign inside the coordinate.
    phase_balance: bool = False

    # ------------------------------------------------- anchored long-horizon stage
    # Final stage: rollout from the true t = 0 state through anchor_len samples,
    # training the initialiser and the ODE jointly on the metric that matters.
    anchor_len: int = 550
    anchor_epochs: int = 120
    # The anchor stage previously optimised *only* trajectory error, which let
    # the field drift away from the true rates as long as the fixed-step map
    # still matched.  Keeping derivative matching alive pins the field itself.
    anchor_deriv_weight: float = 0.3
    # Step-doubling penalty: roll out at 2x sub-steps and require the same
    # answer.  A field that is a genuine ODE is insensitive to step size; a
    # field that is really a tuned discrete map is not.
    consistency_weight: float = 0.25
    consistency_every: int = 2
    # Select the anchor checkpoint on the metric that is actually reported.
    anchor_val_adaptive: bool = True
    # The adaptive validation solves each case separately (cases do not share a
    # time grid, so they cannot be batched).  On a long anchor stage that cost
    # is paid every `validate_every` epochs; subsampling keeps the selection
    # signal without paying for all of them.  0 = use every validation case.
    anchor_val_cases: int = 0
    # TF32 matmuls: ~2x on Ampere and later, well inside the tolerance the
    # step-doubling consistency check already enforces.
    allow_tf32: bool = True
    anchor_lr: float = 2e-5

    # ---------------------------------------------------------------- evaluation
    eval_method: str = "dopri5"
    eval_rtol: float = 1e-6
    eval_atol: float = 1e-8
    eval_metric_floor: float = 1e-10   # ignore Y below this in log-error metrics
    plot_species: tuple = ("CH4", "O2", "CO", "CO2", "H2", "H2O", "OH", "CH2O")

    # ------------------------------------------------------------------- misc
    validate_every: int = 2
    early_stop_patience: int = 15   # in validation checks
    # A validation loss this many times the stage's best (or NaN) restores the
    # best weights, clears the optimiser state and halves the LR. 0 disables.
    rollback_factor: float = 20.0
    lr_plateau_patience: int = 5
    lr_plateau_factor: float = 0.5
    min_lr: float = 1e-7
    num_workers: int = 0

    def resolved_device(self):
        import torch
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save(self, path: Path):
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        Path(path).write_text(json.dumps(d, indent=2, default=str))


# ==========================================================================
# DATA
# ==========================================================================
#
# Dataset handling for the chemical-kinetics Neural ODE.
#
# The single most important thing this module does is change the *independent
# variable* of the ODE from physical time t to a normalised logarithmic time
#
#         s  = log10(t + delta)
#         u  = (s - s_min) / (s_max - s_min)      in [0, 1]
#
# and rescale the exact Cantera derivatives accordingly
#
#         dz/du = ln(10) * (s_max - s_min) * (t + delta) * dz/dt .
#
# Why: an ignition trajectory contains a ~3-decade induction phase in which the
# radical pool grows like a power law in t, a sub-microsecond thermal runaway, and
# a millisecond-scale relaxation.  In physical time the normalised right-hand side
# spans ~1e7; in log-time it spans ~1e2.  That single change removes essentially
# all of the stiffness *of the learning problem*, which in turn removes the need
# for exponential (symlog) output decoding, derivative clipping, rejection
# penalties and 2000-step adaptive solvers.
#
# Everything downstream (windows, integration, evaluation) works in u.  Because u
# is a fixed, analytic, invertible function of t, predictions are converted back to
# physical time exactly.
#

LN10 = np.log(10.0)


# ---------------------------------------------------------------------------
def _get(meta, *names, default=None):
    for n in names:
        if n in meta:
            return meta[n]
    if default is None:
        raise KeyError(f"none of {names} found in metadata (has {list(meta)})")
    return default


class ChemDataset:
    """Loads the Cantera dataset and builds every array the trainer needs."""

    def __init__(self, cfg):
        self.cfg = cfg
        d = Path(cfg.data_dir)
        states = np.load(d / "sampled_states_physical.npy").astype(np.float64)
        derivs = np.load(d / "sampled_derivatives_physical.npy").astype(np.float64)
        times = np.load(d / "time_points_physical.npy").astype(np.float64)
        meta = np.load(d / "case_metadata.npz", allow_pickle=True)

        self.species_names_all = [str(s) for s in _get(meta, "species_names")]
        self.pressures = np.asarray(_get(meta, "pressures"), dtype=np.float64)
        self.T0 = np.asarray(_get(meta, "temperatures", "T0", default=states[:, 0, 0]),
                             dtype=np.float64)
        self.phi = np.asarray(_get(meta, "o2_ch4_ratios", "o2_ch4",
                                   default=np.zeros(len(states))), dtype=np.float64)
        self.steam = np.asarray(_get(meta, "h2o_ch4_ratios", "h2o_ch4",
                                     default=np.zeros(len(states))),
                                dtype=np.float64)
        self.ignition_times = np.asarray(
            _get(meta, "ignition_times", default=np.full(len(states), np.nan)),
            dtype=np.float64)

        self.n_cases, self.n_points, self.n_state_raw = states.shape
        assert self.n_state_raw == 1 + len(self.species_names_all)

        self.t = times
        self.states_phys = states          # [T, Y...]
        self.derivs_phys = derivs          # [dT/dt, dY/dt...]

        self._split()
        self._build_time_coordinate()
        self._build_state_transform()
        self._build_rate_targets()
        self._build_conditioning()
        self._build_phase_weights()
        self._build_rate_loss_eps()
        # Index of the first point the ODE is responsible for. 1 with an
        # initialiser head (the direct map covers interval 0), 0 without, in
        # which case the field owns the whole trajectory including the climb out
        # of the log floor.
        self.anchor = 1 if getattr(self.cfg, "use_initializer", False) else 0

    # ------------------------------------------------------------------ splits
    def _split(self):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        idx = np.arange(self.n_cases)
        rng.shuffle(idx)
        n_tr = int(round(cfg.train_frac * self.n_cases))
        n_va = int(round(cfg.val_frac * self.n_cases))
        self.train_idx = np.sort(idx[:n_tr])
        self.val_idx = np.sort(idx[n_tr:n_tr + n_va])
        self.test_idx = np.sort(idx[n_tr + n_va:])

    # --------------------------------------------------------- time coordinate
    def _build_time_coordinate(self):
        """Map physical time onto u in [0, 1].

        Constants come from the TRAINING split only, so the test set cannot
        influence the coordinate the model is trained in, and genuine
        extrapolation stays visible rather than being absorbed by a rescaling.
        The transform is then applied to all cases -- test trajectories still
        need coordinates.
        """
        cfg = self.cfg
        tr_idx = self.train_idx
        # delta = a typical first positive sample time: puts t=0 at a finite u
        # without wasting a large slice of the range on the empty (0, t_1) gap.
        self.delta = float(np.median([c[c > 0].min() for c in self.t[tr_idx]]))

        align = getattr(cfg, "time_align", "global")
        tign = np.asarray(self.ignition_times, dtype=np.float64)
        if align == "ignition" and not np.all(np.isfinite(tign) & (tign > 0)):
            align = "global"          # metadata lacks usable ignition times
        self.time_align = align

        if align == "ignition":
            # tau = (t + delta) / t_ign.  In log space this is a per-case shift,
            # so ignition sits at log10(tau) = 0 for every case and the map
            # remains smooth.
            #
            # The correlation is fitted FIRST, because with
            # tign_source="predicted" it defines the coordinate. It depends only
            # on T0/P/phi/steam, the measured delays of the TRAINING split and
            # the split indices -- never on s_all -- so the order is free.
            self.tign = tign
            self._fit_tign_correlation()
            self.tign_source = getattr(cfg, "tign_source", "measured")
            if self.tign_source == "predicted":
                tp = self.predict_tign(self.T0, self.pressures, self.phi,
                                       self.steam)
                tp = np.asarray(tp, dtype=np.float64).reshape(-1)
                if not np.all(np.isfinite(tp) & (tp > 0)):
                    raise RuntimeError(
                        "tign_source='predicted' but the correlation returned "
                        "a non-positive or non-finite delay for "
                        f"{int((~(np.isfinite(tp) & (tp > 0))).sum())} case(s). "
                        "Choose a different tign_fit_mode.")
                self.tign_used = tp
            else:
                self.tign_used = tign
            self.log_tign = np.log10(self.tign_used)[:, None]
            s_all = np.log10(self.t + self.delta) - self.log_tign
        else:
            self.tign = None
            self.log_tign = np.zeros((self.n_cases, 1))
            s_all = np.log10(self.t + self.delta)

        self.s_min = float(s_all[tr_idx].min())
        self.s_max = float(s_all[tr_idx].max())

        # ---- room for the alignment error ----------------------------------
        # At inference the coordinate is built from a PREDICTED delay, so every
        # u is shifted by the correlation's error. A case whose delay is
        # over-predicted lands below u = 0 and one whose delay is
        # under-predicted runs past u = 1, and out there the field has never
        # been trained and the Fourier features are extrapolating -- which is a
        # far more violent failure than the timing error itself, and is a
        # plausible reason a run that looks fine with the true delay falls apart
        # with a predicted one. predict_case() even reports it, as
        # `extrapolated_u`, but nothing prevents it.
        #
        # Widening the shared range by a few times the correlation's own error
        # costs a little resolution and buys the guarantee that the deployed
        # grid stays inside the domain the model was fitted on.
        if align == "ignition":
            pad = getattr(cfg, "align_range_pad_dex", None)
            if pad is None:
                # Nothing to pad for when the coordinate is already built from
                # the predicted delay: the deployed grid IS the training grid.
                pad = (0.0 if self.tign_source == "predicted" else
                       3.0 * float(getattr(self, "tign_fit_rms_dex_val", 0.0)
                                   or 0.0))
            pad = float(pad)
            if pad > 0:
                self.s_min -= pad
                self.s_max += pad
                self.align_range_pad = pad
        self.s_span = self.s_max - self.s_min
        self.u = (s_all - self.s_min) / self.s_span

        # dt/du. Dividing t by a per-case constant is a shift in log space, so
        # d(log10 tau)/dt = d(log10 t)/dt and this expression is unchanged --
        # which is exactly why the alignment introduces no discontinuity.
        self.dt_du = LN10 * self.s_span * (self.t + self.delta)
        assert np.all(np.diff(self.u, axis=1) > 0), \
            "time grid must be strictly increasing"

    # ---- ignition-delay correlation ------------------------------------
    # This fit sets a FLOOR on deployed accuracy. Because the coordinate is
    # ignition-aligned, the network learns the trajectory shape in aligned time
    # and takes the timing from t_ign; a delay wrong by eps decades shifts the
    # whole prediction by eps decades. Measured on this dataset the model's
    # ignition error under predicted alignment equals the correlation's error to
    # within 0.5%, i.e. the network inherits it exactly and corrects none of it.
    # Improving this fit therefore improves deployed accuracy with NO retraining
    # of the neural network.

    TIGN_BASES = ("arrhenius", "arrhenius_log", "quadratic", "cubic", "rbf")

    def _tign_vars(self, T0, pressure, phi, steam, logphi):
        """The four base regressors, broadcast to a common shape."""
        T0 = np.atleast_1d(np.asarray(T0, dtype=np.float64))
        phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
        cols = [1000.0 / T0,
                np.log10(np.atleast_1d(pressure)),
                np.log10(np.maximum(phi, 1e-12)) if logphi else phi]
        if self.tign_use_steam:
            cols.append(np.broadcast_to(np.atleast_1d(steam), T0.shape))
        return [np.broadcast_to(c, T0.shape).astype(np.float64) for c in cols]

    def _tign_var_names(self, logphi):
        n = ["1000/T0", "log10 P", "log10(O2/CH4)" if logphi else "O2/CH4"]
        if self.tign_use_steam:
            n.append("H2O/CH4")
        return n

    def _tign_terms(self, mode):
        """Regressor names for a basis, in column order."""
        if mode == "rbf":
            return self._tign_var_names(False) + ["(thin-plate RBF)"]
        v = self._tign_var_names(mode == "arrhenius_log")
        if mode in ("arrhenius", "arrhenius_log"):
            return v + ["const"]
        names = list(v)
        if mode in ("quadratic", "cubic"):
            names += [f"{v[i]}*{v[j]}"
                      for i in range(len(v)) for j in range(i, len(v))]
        if mode == "cubic":
            names += [f"{v[i]}*{v[j]}*{v[k]}"
                      for i in range(len(v)) for j in range(i, len(v))
                      for k in range(j, len(v))]
        return names + ["const"]

    def _tign_design(self, T0, pressure, phi, steam, mode=None):
        """Polynomial design matrix for the ignition-delay correlation.

        "arrhenius"      1000/T0, log10 P, phi [, steam], 1
                         The textbook form: rate ~ exp(-Ea/RT) makes log of a
                         delay linear in 1/T, with weak power-law dependence on
                         pressure and composition.
        "arrhenius_log"  the same with log10(phi), matching tau ~ P^n phi^m.
        "quadratic"      plus every pairwise product and square: curvature and
                         the temperature-pressure interaction a separable form
                         cannot express.
        "cubic"          plus every triple product. More flexible, and more
                         able to overfit -- watch the val column.

        ("rbf" does not use a design matrix; see _tign_fit_one.)
        """
        mode = mode or self.tign_fit_mode
        v = self._tign_vars(T0, pressure, phi, steam, mode == "arrhenius_log")
        cols = list(v)
        if mode in ("quadratic", "cubic"):
            cols += [v[i] * v[j]
                     for i in range(len(v)) for j in range(i, len(v))]
        if mode == "cubic":
            cols += [v[i] * v[j] * v[k]
                     for i in range(len(v)) for j in range(i, len(v))
                     for k in range(j, len(v))]
        cols.append(np.ones(v[0].size))
        return np.column_stack(cols)

    def _tign_fit_one(self, mode):
        """Fit one basis on TRAIN. Returns (predictor, {split: dex RMS}).

        The predictor maps operating conditions to log10 t_ign, so every basis
        -- polynomial or not -- is used through the same interface.
        """
        tr = self.train_idx
        y = np.log10(self.ignition_times[tr])

        if mode == "rbf":
            # Nonparametric alternative: a thin-plate-spline radial basis
            # interpolant over the four conditions, each scaled to [0, 1] by its
            # TRAINING range so no variable dominates the distance metric. The
            # design is a Latin hypercube, so coverage is even and interpolation
            # is well posed; the small smoothing term keeps it from chasing
            # round-off in the event-detected delays. It extrapolates poorly
            # outside the training hull, which is exactly what the val and test
            # columns are there to expose.
            from scipy.interpolate import RBFInterpolator

            def feats(idx):
                V = np.column_stack(self._tign_vars(
                    self.T0[idx], self.pressures[idx], self.phi[idx],
                    self.steam[idx], False))
                return (V - self._rbf_lo) / self._rbf_rng

            V = np.column_stack(self._tign_vars(
                self.T0[tr], self.pressures[tr], self.phi[tr], self.steam[tr],
                False))
            self._rbf_lo = V.min(0)
            span = np.ptp(V, axis=0)          # np.ptp, not V.ptp: removed in NumPy 2
            self._rbf_rng = np.where(span > 0, span, 1.0)
            interp = RBFInterpolator((V - self._rbf_lo) / self._rbf_rng, y,
                                     kernel="thin_plate_spline", smoothing=1e-8)
            lo, rng_ = self._rbf_lo.copy(), self._rbf_rng.copy()

            def predictor(T0, pressure, phi, steam):
                V = np.column_stack(self._tign_vars(T0, pressure, phi, steam,
                                                    False))
                return interp((V - lo) / rng_)
        else:
            X = self._tign_design(self.T0[tr], self.pressures[tr],
                                  self.phi[tr], self.steam[tr], mode)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)

            def predictor(T0, pressure, phi, steam, _c=coef, _m=mode):
                return self._tign_design(T0, pressure, phi, steam, _m) @ _c

        rms = {}
        for name, idx in (("train", tr), ("val", self.val_idx),
                          ("test", self.test_idx)):
            if len(idx) == 0:
                rms[name] = float("nan")
                continue
            r = predictor(self.T0[idx], self.pressures[idx], self.phi[idx],
                          self.steam[idx]) - np.log10(self.ignition_times[idx])
            rms[name] = float(np.sqrt(np.mean(r ** 2)))
        return predictor, rms

    def _fit_tign_correlation(self):
        """Fit the correlation used to build u at inference.

        Every basis in TIGN_BASES is fitted and scored so the choice is made on
        evidence; `cfg.tign_fit_mode` selects the one used. Coefficients come
        from the TRAINING split only. Choose on the VALIDATION column -- picking
        the best TEST number would be selecting on the test set.

        This fit sets a FLOOR on deployed accuracy. Because the coordinate is
        ignition-aligned, the network learns the trajectory shape in aligned
        time and takes its timing from t_ign; measured on this dataset the
        model's ignition error under predicted alignment equals the
        correlation's error to within 0.1%. Improving this fit improves deployed
        accuracy with NO retraining of the network.

        Note which statistic matters. Aggregate trajectory error tracks the
        WORST correlation errors, not the mean, because a large misalignment on
        a few cases dominates an average over cases. A basis that halves the
        mean but not the tail buys less than it appears to.

        Steam is a regressor here, not only a conditioning input to the network:
        it shifts the delay through H2O + M dissociation feeding the radical
        pool, so omitting it biases the coordinate for high-steam cases.
        """
        self.tign_use_steam = bool(getattr(self.cfg, "cond_use_steam", True)
                                   and np.ptp(self.steam) > 0)
        self.tign_fit_mode = getattr(self.cfg, "tign_fit_mode", "arrhenius")
        if self.tign_fit_mode not in self.TIGN_BASES:
            raise ValueError(f"tign_fit_mode must be one of {self.TIGN_BASES}")

        self.tign_fit_table, self._tign_models = {}, {}
        for mode in self.TIGN_BASES:
            try:
                pred, rms = self._tign_fit_one(mode)
            except Exception as e:          # degenerate basis, or scipy missing
                self.tign_fit_table[mode] = {"error": f"{type(e).__name__}: {e}"}
                continue
            self.tign_fit_table[mode] = rms
            self._tign_models[mode] = pred

        if self.tign_fit_mode not in self._tign_models:
            raise RuntimeError(f"basis '{self.tign_fit_mode}' failed to fit: "
                               f"{self.tign_fit_table[self.tign_fit_mode]}")
        sel = self.tign_fit_table[self.tign_fit_mode]
        self.tign_fit_rms_dex = sel["train"]
        self.tign_fit_rms_dex_val = sel["val"]
        self.tign_fit_rms_dex_test = sel["test"]
        self.tign_terms = self._tign_terms(self.tign_fit_mode)

    def tign_error_dex(self, idx):
        """Signed correlation error, in decades, for the given cases.

        Positive means the delay is over-predicted, so the model ignites late.
        Use it to find WHICH operating points the correlation struggles on --
        those are the cases that dominate the trajectory error under predicted
        alignment.
        """
        idx = np.asarray(idx)
        return (self._tign_models[self.tign_fit_mode](
                    self.T0[idx], self.pressures[idx], self.phi[idx],
                    self.steam[idx])
                - np.log10(self.ignition_times[idx]))

    def predict_tign(self, T0, pressure, phi, steam=0.0):
        """Ignition delay from operating conditions, for use at inference.

        `steam` is ignored for a dataset generated without steam variation
        (the column is then not in the fit), so existing three-argument calls
        keep working.
        """
        return 10.0 ** self._tign_models[self.tign_fit_mode](
            T0, pressure, phi, steam)

    def _shift(self, case, t_ign):
        """The per-case log-space shift that defines the aligned coordinate.

        `t_ign` overrides the stored value, which is what the inference path
        needs: at a new operating point the delay is predicted, not known.
        """
        if self.time_align == "global":
            return 0.0
        if t_ign is not None:
            return float(np.log10(t_ign))
        return float(self.log_tign[case, 0])
        # NB self.log_tign already holds whichever delay tign_source selected,
        # so a rollout with t_ign=None reproduces the training coordinate in
        # both modes, and deployed_metrics still overrides it explicitly.

    def u_of_t(self, t, case=None, t_ign=None):
        sh = self._shift(case, t_ign)
        return (np.log10(np.asarray(t) + self.delta) - sh - self.s_min) / self.s_span

    def t_of_u(self, u, case=None, t_ign=None):
        sh = self._shift(case, t_ign)
        return 10.0 ** (self.s_min + np.asarray(u) * self.s_span + sh) - self.delta

    # -------------------------------------------------------- state transform
    def _build_state_transform(self):
        cfg = self.cfg
        T = self.states_phys[..., :1]
        Y = self.states_phys[..., 1:]
        L = np.log10(np.clip(Y, 0.0, None) + cfg.y_floor)
        raw = np.concatenate([T, L], axis=-1)          # (C, K, 1+S)

        tr = raw[self.train_idx]
        lo = tr.min(axis=(0, 1))
        hi = tr.max(axis=(0, 1))
        rng_ = hi - lo

        # constant channels (species absent from the mixture) carry no signal
        keep = np.ones(raw.shape[-1], dtype=bool)
        if getattr(cfg, "drop_constant_species", True):
            keep[1:] = rng_[1:] > cfg.constant_species_tol
        self.keep_mask = keep
        self.const_values = lo.copy()                  # used to rebuild dropped ones
        self.species_names = [n for n, k in zip(self.species_names_all, keep[1:]) if k]
        self.dropped_species = [n for n, k in zip(self.species_names_all, keep[1:]) if not k]

        lo, hi, rng_ = lo[keep], hi[keep], rng_[keep]
        pad = cfg.scale_margin * np.maximum(rng_, 1e-12)
        self.raw_lo = lo - pad
        self.raw_hi = hi + pad

        if getattr(cfg, "species_scale", "per_channel") == "physical":
            # One scale for every species channel: the physically possible
            # range of log10(Y + y_floor), from Y = 0 to Y = 1. Temperature
            # keeps its own padded min-max.
            #
            # The consequence to keep in mind: a channel like H2O, whose data
            # covers 0.5 of the 12 decades, now occupies 4 % of [-1, 1]. That
            # is correct -- it IS a channel that barely moves -- and its
            # rate_scale shrinks to match, so the network's target stays O(1).
            # What changes is that its errors are no longer inflated 25x
            # relative to O2's.
            self.raw_lo[1:] = np.log10(cfg.y_floor)
            self.raw_hi[1:] = 0.0
        self.raw_range = self.raw_hi - self.raw_lo
        self.raw_range[self.raw_range <= 0] = 1.0

        self.raw = raw[..., keep]
        self.z = self.to_z(self.raw)                   # (C, K, D) in [-1, 1]
        self.d_state = self.z.shape[-1]
        self.state_columns = ["T"] + self.species_names

    def to_z(self, raw):
        return 2.0 * (raw - self.raw_lo) / self.raw_range - 1.0

    def from_z(self, z):
        return (np.asarray(z) + 1.0) / 2.0 * self.raw_range + self.raw_lo

    def z_to_physical(self, z):
        """(..., D) normalised state -> (T [K], Y [mass fractions]) on the kept
        channels only, in the order given by self.state_columns."""
        raw = self.from_z(z)
        T = raw[..., :1]
        Y = 10.0 ** raw[..., 1:] - self.cfg.y_floor
        return np.concatenate([T, np.clip(Y, 0.0, None)], axis=-1)

    # --------------------------------------------------------- rate targets
    def _build_rate_targets(self):
        """Rate targets dz/du.

        The *exact* Cantera derivative is used wherever it is meaningful.  It is
        not always meaningful in the log-species variable: whenever a species is
        still at the representation floor (Y ~ 0) while its production rate is
        already finite, d log10(Y+eps)/dt diverges like 1/eps -- values up to
        1e17 1/s appear at t = 0.  Those spikes are not resolvable on any grid
        and are not what the integrator has to reproduce; the *secant* of the
        sampled trajectory is.  So the target is the exact derivative where it
        lies inside the representable band, and a central difference of the data
        where it does not.  Everything is then clipped to the band the network
        can output, so the regression target is always attainable.
        """
        cfg = self.cfg
        Y = np.clip(self.states_phys[..., 1:], 0.0, None)
        dT = self.derivs_phys[..., :1]
        dY = self.derivs_phys[..., 1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            dL = dY / ((Y + cfg.y_floor) * LN10)       # d log10(Y+eps)/dt
        draw_dt = np.concatenate([dT, dL], axis=-1)[..., self.keep_mask]
        dz_dt = 2.0 * draw_dt / self.raw_range
        exact = np.nan_to_num(dz_dt * self.dt_du[..., None], nan=0.0,
                              posinf=0.0, neginf=0.0)

        # central difference in u (one-sided at the ends)
        cd = np.empty_like(self.z)
        du = np.diff(self.u, axis=1)
        dzc = np.diff(self.z, axis=1)
        cd[:, 1:-1] = ((self.z[:, 2:] - self.z[:, :-2]) /
                       (self.u[:, 2:] - self.u[:, :-2])[..., None])
        cd[:, 0] = dzc[:, 0] / du[:, 0][..., None]
        cd[:, -1] = dzc[:, -1] / du[:, -1][..., None]

        if cfg.rate_scale_mode == "rms":
            r = np.sqrt(np.mean(cd[self.train_idx] ** 2, axis=(0, 1)))
        else:
            r = np.quantile(np.abs(cd[self.train_idx]), cfg.rate_quantile, axis=(0, 1))
        self.rate_scale = np.maximum(r, 1e-2)
        cap = cfg.target_cap * self.rate_scale         # attainable band

        # The exact derivative is also meaningless (identically zero, while the
        # trajectory is about to move by several decades) wherever a species is
        # still exactly at zero: at t = 0 a CH4/O2 mixture has *no* radicals, so
        # every radical production rate is exactly 0 even though Y ~ t^n starts
        # immediately afterwards.  d log10 Y / du is then 0 at the sample point
        # and O(10) one instant later.  Use the data secant wherever the species
        # sits at (or within a decade of) the representation floor.
        at_floor = np.zeros_like(exact, dtype=bool)
        at_floor[..., 1:] = Y[..., self.keep_mask[1:]] <= 10.0 * cfg.y_floor
        use_cd = (np.abs(exact) > cap) | at_floor
        blended = np.where(use_cd, cd, exact)
        self.dz_du = np.clip(blended, -cap, cap)
        self.dz_du_exact = exact
        self.dz_du_n = self.dz_du / self.rate_scale    # network target, ~O(1)
        self.frac_replaced = float(np.mean(use_cd))
        if getattr(cfg, "rate_space", "logtime") == "physical":
            self._build_physical_targets()

        # how violent is the very first sampled interval?  (diagnostic: if this
        # is large, the log floor y_floor is set below what the time grid can
        # resolve and should be raised)
        jump = np.abs(self.z[:, 1] - self.z[:, 0]) * self.raw_range / 2.0
        self.first_jump_decades = float(np.max(jump[:, 1:]))
        self.first_jump_median = float(np.median(np.max(jump[:, 1:], axis=1)))

    def _build_physical_targets(self):
        """Replace the network target by the compressed physical-time rate.

        Built FROM the blended, clipped dz/du above, so the at-floor secants and
        the attainable band carry over unchanged: the integrator is asked to
        reproduce exactly the same trajectory; only what the network regresses
        changes. The log-time target is kept as dz_du_n_logtime.
        """
        cfg = self.cfg
        if self.time_align != "global":
            raise ValueError("rate_space = 'physical' needs time_align = "
                             "'global': the analytic factor 10^(s_min + s_span u)"
                             " is t + delta only when u carries no ignition shift")
        if int(getattr(cfg, "n_augment", 0)):
            raise ValueError("rate_space = 'physical' does not support n_augment "
                             "(the augmented channels have no physical rate)")
        draw_dt = (self.dz_du * (self.raw_range / 2.0)
                   / self.dt_du[..., None])                  # K/s, dex/s
        fixed = getattr(cfg, "phys_eps", None)
        t_max = float(self.t[self.train_idx].max())
        self.phys_t_max = t_max
        if fixed is not None:
            eps = np.full(self.d_state, float(fixed))
        else:
            eps = np.full(self.d_state,
                          float(getattr(cfg, "rate_drift_tol_decades", 0.05)))
            eps[0] = float(getattr(cfg, "rate_drift_tol_K", 5.0))
            eps = eps / t_max
        self.phys_eps = eps.astype(np.float32)
        s = np.sign(draw_dt) * np.log10(1.0 + np.abs(draw_dt) / eps)
        Q = np.abs(s[self.train_idx]).max(axis=(0, 1))
        self.phys_Q = np.maximum(Q, 1e-3).astype(np.float32)
        q = s / self.phys_Q
        self.phys_q = q
        if getattr(cfg, "phys_loss", "logtime") == "q":
            self.dz_du_n_logtime = self.dz_du_n
            self.dz_du_n = q
        # round trip: decoding the target must give back dz/du
        back = (eps * np.sign(q) * (10.0 ** (np.abs(q) * self.phys_Q) - 1.0)
                * (2.0 / self.raw_range) * self.dt_du[..., None])
        err = np.abs(back - self.dz_du) / (np.abs(self.dz_du) + self.rate_scale)
        self.phys_roundtrip = float(err.max())
        if self.phys_roundtrip > 1e-6:
            raise RuntimeError(f"physical target round trip error "
                               f"{self.phys_roundtrip:.2e}")

    # ----------------------------------------------------------- phase weights
    PHASE_NAMES = ("induction", "ignition", "post")

    def _build_phase_weights(self):
        """Per-point trajectory-loss weights, equalising the three phases.

        The ignition window holds ~15 % of the samples and essentially all of
        the |dz|, so an unweighted trajectory loss is a loss on the ignition
        window with a rounding error attached. Induction is where the error
        that matters is generated -- it is the phase in which the reactants are
        supposed to sit still for five decades of time and instead drift -- and
        it is also the phase the loss currently ignores.

        Each phase is given weight 1/3 spread over however many samples it
        contains, then the whole thing is renormalised to mean 1 so the loss
        stays on the same numerical scale as before and the existing learning
        rates remain sensible.

        The boundaries use the case's true ignition delay. That is a
        training-time quantity, like the data itself; it is not handed to the
        network and nothing at inference depends on it.
        """
        cfg = self.cfg
        if not getattr(cfg, "phase_balance", False):
            self.phase_w = np.ones((self.n_cases, self.n_points),
                                   dtype=np.float32)
            self.phase_frac = None
            return

        t = self.t
        tig = self._t_ign_for_phases()

        masks = (t < 0.5 * tig[:, None],
                 (t >= 0.5 * tig[:, None]) & (t <= 3.0 * tig[:, None]),
                 t > 3.0 * tig[:, None])
        w = np.zeros((self.n_cases, self.n_points), dtype=np.float64)
        for m in masks:
            n = m.sum(1, keepdims=True).astype(np.float64)
            w += np.where(m, 1.0 / np.maximum(n, 1.0), 0.0)
        w *= self.n_points / np.maximum(w.sum(1, keepdims=True), 1e-30)
        self.phase_w = w.astype(np.float32)
        self.phase_frac = [float(m.mean()) for m in masks]
        self.phase_wmean = [float(w[m].mean()) if m.any() else float("nan")
                            for m in masks]

    # -------------------------------------------------------- rate-loss scale
    def _t_ign_for_phases(self):
        """Per-case ignition time, from metadata, falling back to half-rise."""
        tig = np.asarray(self.ignition_times, dtype=np.float64).copy()
        bad = ~(np.isfinite(tig) & (tig > 0))
        if bad.any():
            T_t = self.raw[..., 0]
            thr = T_t[:, :1] + 0.5 * (T_t.max(1, keepdims=True) - T_t[:, :1])
            above = T_t >= thr
            first = np.where(above.any(1), above.argmax(1), self.n_points - 1)
            tig[bad] = np.take_along_axis(self.t, first[:, None], 1)[:, 0][bad]
        return tig

    def _build_rate_loss_eps(self):
        """Per-channel turnover point for the symlog rate loss.

        eps is not a regularisation constant to be tuned; it is the statement
        "rates below this are zero as far as the loss is concerned", so it has
        to be set to the accuracy the trajectory actually needs. Inverting the
        accumulation arithmetic,

            drift_decades = err * rate_scale_k * du_induction * (range_k / 2)

        gives the err that keeps channel k inside a chosen drift budget, and
        that err is the right eps. The result differs by more than an order of
        magnitude between datasets, because rate_scale is proportional to
        s_span and s_span depends on the time horizon and the first sample
        time -- which is exactly why a hard-coded value is a bug waiting to
        happen.

        The temperature channel gets its own budget in kelvin, since decades of
        a mass fraction do not mean anything there.
        """
        cfg = self.cfg
        fixed = getattr(cfg, "rate_loss_eps", None)
        dec = self.raw_range / 2.0
        if fixed is not None:
            self.rate_loss_eps = np.full(self.d_state, float(fixed),
                                         dtype=np.float32)
            self.rate_eps_auto = False
            self.du_induction = float("nan")
            return

        tig = self._t_ign_for_phases()
        ind = self.t < 0.5 * tig[:, None]
        du = [np.ptp(self.u[i][ind[i]]) if ind[i].sum() > 1 else 0.0
              for i in self.train_idx]
        self.du_induction = float(np.median(du))

        tol = np.full(self.d_state,
                      float(getattr(cfg, "rate_drift_tol_decades", 0.05)))
        tol[0] = float(getattr(cfg, "rate_drift_tol_K", 5.0))
        denom = np.maximum(self.rate_scale * self.du_induction * dec, 1e-30)
        # Clipped at both ends: below 1e-6 the loss would be chasing the noise
        # in the rate targets (14 % of which are data secants, not exact
        # derivatives); above 0.5 symlog is indistinguishable from the plain
        # absolute loss and there is no point paying for it.
        self.rate_loss_eps = np.clip(tol / denom, 1e-6, 0.5).astype(np.float32)
        self.rate_eps_auto = True

    # ------------------------------------------------------------ conditioning
    def _build_conditioning(self):
        cfg = self.cfg
        cols = [self.pressures]
        names = ["p"]
        if cfg.cond_use_T0:
            cols.append(self.T0); names.append("T0")
        if cfg.cond_use_phi and np.ptp(self.phi) > 0:
            cols.append(self.phi); names.append("O2/CH4")
        if getattr(cfg, "cond_use_steam", True) and np.ptp(self.steam) > 0:
            cols.append(self.steam); names.append("H2O/CH4")
        C = np.stack(cols, axis=-1)
        lo = C[self.train_idx].min(0)
        hi = C[self.train_idx].max(0)
        rng_ = np.where(hi > lo, hi - lo, 1.0)
        self.cond = (2.0 * (C - lo) / rng_ - 1.0).astype(np.float32)
        self.cond_names = names
        self.d_cond = self.cond.shape[-1]
        # Kept so an operating point that is NOT in the dataset can be scaled
        # the same way -- see cond_vector(), used by the inference path.
        self.cond_lo, self.cond_hi, self.cond_rng = lo, hi, rng_

    def cond_vector(self, pressure, T0=None, phi=None, steam=None):
        """Conditioning vector for an arbitrary operating point.

        Scaled with the TRAINING ranges, exactly as `self.cond` was, so a new
        condition is presented to the network in the same units it learned in.
        Only the entries this dataset actually conditions on are required; the
        rest are ignored. Returns (1, d_cond) float32.
        """
        avail = {"p": pressure, "T0": T0, "O2/CH4": phi, "H2O/CH4": steam}
        cols = []
        for n in self.cond_names:
            v = avail[n]
            if v is None:
                raise ValueError(f"this model conditions on {self.cond_names}; "
                                 f"'{n}' was not supplied")
            cols.append(np.atleast_1d(np.asarray(v, dtype=np.float64)))
        C = np.stack(cols, axis=-1)
        return (2.0 * (C - self.cond_lo) / self.cond_rng - 1.0).astype(np.float32)

    def z_from_physical(self, T, Y):
        """(T [K], Y over ALL mechanism species) -> normalised state z.

        Inverse of z_to_physical. Takes the full species vector in the
        mechanism's order and drops the constant channels, so an initial
        composition can be handed in exactly as Cantera would report it.
        """
        T = np.atleast_1d(np.asarray(T, dtype=np.float64))
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        if Y.shape[-1] != len(self.species_names_all):
            raise ValueError(f"expected {len(self.species_names_all)} species "
                             f"(mechanism order), got {Y.shape[-1]}")
        raw = np.concatenate(
            [T[:, None], np.log10(np.clip(Y, 0.0, None) + self.cfg.y_floor)],
            axis=-1)
        return self.to_z(raw[..., self.keep_mask]).astype(np.float32)

    # ---------------------------------------------------------------- windows
    def make_windows(self, case_idx, window_len, stride):
        """Index windows over the *original* sample grid -- no resampling, no
        interpolation.  Returns dict of arrays with a leading window axis."""
        if window_len > self.n_points - self.anchor:
            # Without this, np.arange returns an empty array and the starts[-1]
            # below raises a bare IndexError that says nothing about the cause.
            raise ValueError(
                f"window_len={window_len} exceeds the {self.n_points - self.anchor} "
                f"usable points per case (n_points={self.n_points}, "
                f"anchor={self.anchor}). Shorten cfg.stages.")
        starts = np.arange(self.anchor, self.n_points - window_len + 1, stride)
        if starts[-1] != self.n_points - window_len:
            starts = np.append(starts, self.n_points - window_len)
        offs = np.arange(window_len)
        gi = starts[:, None] + offs[None, :]                    # (W, L)

        cidx = np.repeat(np.asarray(case_idx), len(starts))
        gidx = np.tile(gi, (len(case_idx), 1))
        return {
            "case": cidx.astype(np.int64),
            "start": np.tile(starts, len(case_idx)).astype(np.int64),
            "u": self.u[cidx[:, None], gidx].astype(np.float32),        # (N, L)
            "z": self.z[cidx[:, None], gidx].astype(np.float32),        # (N, L, D)
            "dz": self.dz_du_n[cidx[:, None], gidx].astype(np.float32),
            "cond": self.cond[cidx].astype(np.float32),
            # per-point phase weight, (N, L); all ones when phase_balance is off
            "w": self.phase_w[cidx[:, None], gidx].astype(np.float32),
            "n": len(cidx),
        }

    def init_pairs(self, case_idx):
        """(z0, cond, u1, z1) for the first-interval map, one row per case."""
        c = np.asarray(case_idx)
        return (self.z[c, 0].astype(np.float32),
                self.cond[c].astype(np.float32),
                self.u[c, 1:2].astype(np.float32),
                self.z[c, 1].astype(np.float32))

    def flat_points(self, case_idx):
        """All (u, z, cond) -> dz/du_n samples of the given cases, for pretraining.

        Returns the phase weight as a fifth array. Pretraining is where the
        field is first shaped, so balancing the phases there matters as much as
        in the shooting stages -- a field that comes out of pretraining with a
        systematic bias through induction is one the later stages then have to
        undo.
        """
        c = np.asarray(case_idx)
        a = self.anchor
        k = self.n_points - a
        z = self.z[c, a:].reshape(-1, self.d_state).astype(np.float32)
        u = self.u[c, a:].reshape(-1, 1).astype(np.float32)
        dz = self.dz_du_n[c, a:].reshape(-1, self.d_state).astype(np.float32)
        cond = np.repeat(self.cond[c], k, axis=0).astype(np.float32)
        w = self.phase_w[c, a:].reshape(-1, 1).astype(np.float32)
        return u, z, cond, dz, w

    # ------------------------------------------------------------------ report
    def split_report(self, per_line=20):
        """Which cases landed in each split, by case index.

        The index is the row into sampled_states_physical.npy, i.e. the same
        identifier chemnode_eval.py reports as `worst_case_index`, so a bad
        case in the metrics can be looked up here and traced back to its
        operating point. Printed with the conditions spanned by each split, so
        it is visible at a glance whether the split is representative or
        whether one of them happens to miss a corner of the design space.
        """
        s = []
        for name, idx in (("train", self.train_idx),
                          ("val", self.val_idx),
                          ("test", self.test_idx)):
            idx = np.asarray(idx)
            s.append(f"  {name:<5} ({len(idx):3d} cases): "
                     f"T0 {self.T0[idx].min():.0f}-{self.T0[idx].max():.0f} K  "
                     f"P {self.pressures[idx].min():.1f}-{self.pressures[idx].max():.1f} bar  "
                     f"O2/CH4 {self.phi[idx].min():.3f}-{self.phi[idx].max():.3f}  "
                     f"H2O/CH4 {self.steam[idx].min():.3f}-{self.steam[idx].max():.3f}")
            if np.all(np.isfinite(self.ignition_times[idx])):
                s.append(f"        t_ign {self.ignition_times[idx].min():.3e} - "
                         f"{self.ignition_times[idx].max():.3e} s")
            for i in range(0, len(idx), per_line):
                chunk = " ".join(f"{c:4d}" for c in idx[i:i + per_line])
                s.append(f"        {chunk}")
        return "\n".join(s)

    def summary(self):
        s = []
        s.append(f"cases={self.n_cases}  points/case={self.n_points}  "
                 f"state dim={self.d_state} (dropped: {self.dropped_species or 'none'})")
        s.append(f"split  train={len(self.train_idx)} val={len(self.val_idx)} test={len(self.test_idx)}")
        if self.time_align == "ignition":
            src = getattr(self, "tign_source", "measured")
            s.append(f"t_ign defining the coordinate: {src.upper()}"
                     + ("  -- train and inference build u by the same formula, "
                        "so the mismatch is zero"
                        if src == "predicted" else
                        "  -- inference must predict it, so the deployed grid "
                        "is shifted by the correlation error"))
            if src == "predicted":
                err = np.abs(np.log10(self.tign_used)
                             - np.log10(self.ignition_times))
                s.append(f"  alignment blur: {err.mean():.4f} dex mean, "
                         f"{err.max():.4f} dex worst "
                         f"({err.mean()/self.s_span:.5f} of the u range)")
            s.append(f"time coordinate: IGNITION-ALIGNED "
                     f"(tau = (t+delta)/t_ign; ignition at the same u for every "
                     f"case)")
            s.append(f"  t_ign correlation: {self.tign_fit_rms_dex:.3f} dex RMS on "
                     f"train, {self.tign_fit_rms_dex_test:.3f} on test "
                     f"(needed to build u at inference)")
            s.append(f"  t_ign basis '{self.tign_fit_mode}': "
                     f"{' + '.join(self.tign_terms)}")
            s.append("  t_ign basis comparison (dex RMS; choose on val):")
            for mode, r in self.tign_fit_table.items():
                mark = "  <- in use" if mode == self.tign_fit_mode else ""
                if "error" in r:
                    s.append(f"     {mode:<14} failed: {r['error']}{mark}")
                else:
                    s.append(f"     {mode:<14} train {r['train']:.4f}  "
                             f"val {r['val']:.4f}  test {r['test']:.4f}{mark}")
            s.append("  NOTE: under predicted alignment the model's ignition "
                     "error is this number,")
            s.append("        not the t_ign_dex reported with true alignment.")
        else:
            s.append(f"time coordinate: global log-time")
        s.append(f"log-time: delta={self.delta:.3e}s  s in [{self.s_min:.2f},"
                 f"{self.s_max:.2f}] ({self.s_span:.2f} decades)")
        s.append(f"T range {self.raw_lo[0]:.1f}..{self.raw_hi[0]:.1f} K   "
                 f"log10Y range {self.raw_lo[1:].min():.2f}..{self.raw_hi[1:].max():.2f}")
        dzdt = np.abs(2.0 * np.concatenate([
            self.derivs_phys[..., :1],
            self.derivs_phys[..., 1:] / ((np.clip(self.states_phys[..., 1:], 0, None)
                                          + self.cfg.y_floor) * LN10)], -1)[..., self.keep_mask]
            / self.raw_range)
        s.append(f"|dz/dt| max = {dzdt.max():.3e} 1/s  ->  "
                 f"|dz/du| max = {np.abs(self.dz_du).max():.3e}  "
                 f"(rate_scale max {self.rate_scale.max():.3e}, "
                 f"{100*self.frac_replaced:.2f}% of targets replaced by central diff)")
        if getattr(self.cfg, "rate_space", "logtime") == "physical":
            s.append(f"RATE SPACE: PHYSICAL  network learns sign*log10(1+|dz_raw/dt|)"
                     f" / Q, Q = {self.phys_Q.min():.2f}..{self.phys_Q.max():.2f}"
                     f" decades (T: {self.phys_Q[0]:.2f}); eps T "
                     f"{self.phys_eps[0]:.1e} K/s, species "
                     f"{self.phys_eps[1]:.1e} dec/s"
                     f"{' (derived: budget / t_max ' + format(self.phys_t_max, '.3g') + ' s)' if getattr(self.cfg, 'phys_eps', None) is None else ''}; rate loss in "
                     f"{getattr(self.cfg, 'phys_loss', 'logtime')} space;"
                     f" round trip {self.phys_roundtrip:.1e}")
        s.append(f"first sampled interval jumps up to {self.first_jump_decades:.2f} "
                 f"decades (median case {self.first_jump_median:.2f}); the ODE "
                 f"must cross this unaided -- raise y_floor if it is large")

        # ---- the units the loss actually works in --------------------------
        dec = self.raw_range / 2.0
        mode = getattr(self.cfg, "species_scale", "per_channel")
        s.append(f"species scaling: '{mode}' -- one z-unit is "
                 + (f"{dec[1]:.2f} decades in every species channel"
                    if mode == "physical" else
                    f"{dec[1:].min():.2f}..{dec[1:].max():.2f} decades "
                    f"depending on the channel"))
        if mode != "physical":
            wgt = (dec[1:].max() / dec[1:].min()) ** 2
            s.append(f"  -> a one-decade error costs {wgt:.0f}x more in "
                     f"'{self.species_names[int(np.argmin(dec[1:]))]}' than in "
                     f"'{self.species_names[int(np.argmax(dec[1:]))]}'. "
                     f"Set species_scale='physical' to remove this.")
        else:
            s.append(f"  -> |z| <= 1 on the species channels IS 0 <= Y <= 1; "
                     f"box={self.cfg.box} targets Y <= "
                     f"{10.0 ** ((self.cfg.box - 1.0) * dec[1]):.3f} "
                     f"(soft gate, so a long step can still overshoot it)")
        dexn = getattr(self.cfg, "init_noise_decades", None)
        s.append(f"initial-state noise: "
                 + (f"{dexn:g} decades on every species channel"
                    if dexn is not None else
                    f"{self.cfg.init_noise:g} normalised = "
                    f"{self.cfg.init_noise*dec[1:].min():.3f}.."
                    f"{self.cfg.init_noise*dec[1:].max():.3f} decades"))
        if getattr(self.cfg, "rate_loss", "absolute") == "symlog":
            e = np.asarray(self.rate_loss_eps, dtype=np.float64)
            s.append(f"rate loss: 'symlog', eps "
                     + (f"DERIVED from a "
                        f"{self.cfg.rate_drift_tol_decades:g}-decade / "
                        f"{self.cfg.rate_drift_tol_K:g} K drift budget over "
                        f"du_induction={self.du_induction:.3f}"
                        if self.rate_eps_auto else f"fixed at {e[0]:g}"))
            if self.rate_eps_auto:
                o = np.argsort(e[1:])
                lo = ", ".join(f"{self.species_names[i]} {e[1+i]:.1e}"
                               for i in o[:4])
                s.append(f"  eps: T {e[0]:.1e}   species "
                         f"{e[1:].min():.1e}..{e[1:].max():.1e}   "
                         f"tightest: {lo}")
                s.append(f"  (a channel needing eps below 1e-6 is clipped "
                         f"there; {int((e < 1.01e-6).sum())} of "
                         f"{e.size} are)")
        elif (getattr(self.cfg, "rate_space", "logtime") == "physical"
              and getattr(self.cfg, "phys_loss", "logtime") == "q"):
            s.append("rate loss: Huber on the compressed physical rate q "
                     "(already log-scaled, so no second symlog)")
        else:
            s.append("rate loss: 'absolute' -- blind to the induction phase, "
                     "see chemnode_diagnose.py section 2")
        if getattr(self.cfg, "rate_space", "logtime") == "physical":
            s.append(f"rate head: {self.cfg.phys_bound:g}*tanh -- bounded by "
                     f"the training range of q (rate_head setting not used)")
        else:
          s.append(f"rate head: '{getattr(self.cfg, 'rate_head', 'linear')}'"
                 + (f" k={self.cfg.rate_head_k:g}, resolution near zero "
                    f"{math.sinh(self.cfg.rate_head_k)/(self.cfg.rate_head_k*self.cfg.out_gain):.0f}x"
                    f" finer than the linear head of the same bound"
                    if getattr(self.cfg, 'rate_head', '') == 'sinh' else ""))
        if self.phase_frac is not None:
            s.append("phase balance: ON -- induction/ignition/post hold "
                     + "/".join(f"{100*f:.0f}%" for f in self.phase_frac)
                     + " of the points and now get 1/3 of the loss each "
                     + "(mean weights "
                     + "/".join(f"{v:.2f}" for v in self.phase_wmean) + ")")
        else:
            s.append("phase balance: OFF -- the trajectory loss is dominated "
                     "by the ignition window")
        if self.anchor:
            j = self.first_jump_decades
            s.append(f"first interval: handled by a learned MAP (anchor={self.anchor}); "
                     f"the ODE starts at sample 1,")
            s.append(f"       so the {j:.2f}-decade jump is not the field's "
                     f"problem and no window starts at t=0")
        else:
            s.append(f"first interval: the ODE must cross "
                     f"{self.first_jump_decades:.2f} decades unaided "
                     f"(use_initializer=True hands it to a map)")
        m = int(getattr(self.cfg, "n_augment", 0))
        if getattr(self.cfg, "use_time_input", True):
            s.append(f"field: f(z, u, c) -- reads the clock. Ignition timing "
                     f"can be taken from u rather than from the state.")
        else:
            s.append(f"field: f(z, c) -- AUTONOMOUS. No clock, so nothing in "
                     f"the inference path needs t_ign;")
            s.append(f"       timing has to emerge from the state, which "
                     f"requires z to determine t (section 7).")
        s.append(f"augmented channels: {m}"
                 + ("  (none -- the flow must be representable in z alone)"
                    if m == 0 else
                    f"  -> integrating {self.d_state + m} channels, comparing "
                    f"{self.d_state} against data"))
        s.append(f"conditioning: {self.cond_names}")
        for name, idx in (("val", self.val_idx), ("test", self.test_idx)):
            um = float(self.u[idx].max())
            if um > 1.0:
                s.append(f"NOTE: {name} u reaches {um:.4f} > 1 -- "
                         f"{int((self.u[idx].max(axis=1) > 1).sum())} case(s) run "
                         f"longer than any training case, so the model "
                         f"extrapolates in time there")
        return "\n".join(s)


# ==========================================================================
# MODEL
# ==========================================================================
#
# Neural-ODE model.
#
# The network predicts the *normalised* right-hand side in log-time,
#
#         n_theta(z, u, c)  ~=  (dz/du) / rate_scale ,
#
# and the ODE function multiplies by ``rate_scale``.  Two structural properties
# matter:
#
# 1. The output passes through ``g * tanh(x / g)``.  This is linear for |x| < g
#    (so the network is not fighting the nonlinearity on real data, where the
#    target is O(1) by construction) but saturates at +/- g.  The right-hand side
#    is therefore *bounded*, which makes the integration unconditionally stable --
#    without the exponential ``symlog`` decoding that made the previous model's
#    bounded latent output decode to a 1e15 physical derivative.
#
# 2. There is no rejection / clipping machinery anywhere.  Nothing can blow up, so
#    nothing has to be thrown away, so every window contributes a real gradient.
#

# --------------------------------------------------------------------------- #
class FourierTime(nn.Module):
    """Fixed (non-learned) Fourier features of the log-time coordinate u.

    Ignition is a sharp feature in u (it occupies ~2% of the range).  A plain
    scalar input makes an MLP smooth it out; a small bank of octave-spaced
    sinusoids lets the network resolve it without extra depth.
    """

    def __init__(self, n_freq: int):
        super().__init__()
        f = 2.0 ** torch.arange(n_freq, dtype=torch.float32)  # 1, 2, 4, ...
        self.register_buffer("freq", math.pi * f)
        self.out_dim = 1 + 2 * n_freq

    def forward(self, u):
        a = u * self.freq
        return torch.cat([u, torch.sin(a), torch.cos(a)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return x + h


class RateNet(nn.Module):
    def __init__(self, d_state, d_cond, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_time = cfg.use_time_input
        self.time_feat = FourierTime(cfg.n_time_fourier) if self.use_time else None
        in_dim = d_state + d_cond + (self.time_feat.out_dim if self.use_time else 0)
        self.inp = nn.Linear(in_dim, cfg.hidden)
        self.blocks = nn.ModuleList([ResBlock(cfg.hidden, cfg.dropout)
                                     for _ in range(cfg.n_blocks)])
        self.norm = nn.LayerNorm(cfg.hidden)
        self.out = nn.Linear(cfg.hidden, d_state)
        self.act = nn.SiLU()
        self.gain = cfg.out_gain
        self.head = getattr(cfg, "rate_head", "linear")
        self.head_k = float(getattr(cfg, "rate_head_k", 6.0))
        # How much finer the resolution near zero is than a linear head of the
        # same bound: slope at zero is out_gain*k/sinh(k) against 1 for linear,
        # so the ratio is sinh(k)/(k*out_gain). It is 1.1 at k=6 -- i.e. nothing
        # -- and 37 at k=10, which is why k is not a free-and-easy knob.
        # Reported by summary() so the choice is visible rather than implied.
        self.head_zoom = math.sinh(self.head_k) / (self.head_k * cfg.out_gain)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out.weight)   # start from a zero vector field
        nn.init.zeros_(self.out.bias)

    def forward(self, z, u, cond):
        """z (B, D), u (B, 1), cond (B, C) -> normalised rate (B, D)."""
        feats = [z, cond]
        if self.use_time:
            feats.append(self.time_feat(u))
        h = self.act(self.inp(torch.cat(feats, dim=-1)))
        for b in self.blocks:
            h = b(h)
        raw = self.out(self.norm(h))
        if getattr(self.cfg, "rate_space", "logtime") == "physical":
            # Bounded by the training range of the compressed physical rate,
            # like the notebook's decoder: the field can never emit a rate far
            # outside anything it has seen.
            b = float(getattr(self.cfg, "phys_bound", 1.05))
            return b * torch.tanh(raw / b)
        if self.head == "sinh":
            # Same bound as the linear head (+/- out_gain at |tanh| = 1) and
            # analytic everywhere, but exponential in the pre-activation, so
            # equal steps in raw give equal ratios in the output. That is what
            # lets one field be accurate to 1e-5 through induction and to O(1)
            # at the ignition spike without the small values being rounding
            # error on the large ones.
            a = torch.tanh(raw)
            return self.gain * torch.sinh(self.head_k * a) / math.sinh(self.head_k)
        return self.gain * torch.tanh(raw / self.gain)


# --------------------------------------------------------------------------- #
class InitNet(nn.Module):
    """Learned flow map across the first sampled interval, z(u_1) = z_0 + d.

    Residual rather than absolute, so the map starts as the identity (the last
    layer is zero-initialised) and has only to learn the departure. Bounded by
    init_gain * tanh and then clamped, so it cannot emit a state outside the
    representable range no matter what the optimiser does to it.
    """

    def __init__(self, d_state, d_cond, cfg):
        super().__init__()
        h = cfg.init_hidden
        self.net = nn.Sequential(
            nn.Linear(d_state + d_cond + 1, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, d_state),
        )
        self.gain = cfg.init_gain
        # The clamp is at the ODE's box, NOT at 1.0, and that detail is load
        # bearing. Under species_scale = "physical", Y = 0 maps to exactly
        # z = -1, and at t = 0 that is where 23 of 27 channels sit -- precisely
        # the channels that carry the first-interval jump. A clamp at 1.0 puts
        # them exactly on the boundary, torch.clamp passes no gradient there,
        # and the map freezes at the identity: measured, the loss was identical
        # to five significant figures over 600 steps. Because anchor = 1 also
        # stops the ODE from ever training on interval 0, a frozen map means
        # the jump is never made by anything, which is worse than having no
        # initialiser at all. The original code escaped this only because
        # per-channel min-max with scale_margin put the floor at z = -0.909.
        self.box = float(getattr(cfg, "box", 1.005))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z0, cond, u1):
        d = self.net(torch.cat([z0, cond, u1], dim=-1))
        return (z0 + self.gain * torch.tanh(d)).clamp(-self.box, self.box)


# --------------------------------------------------------------------------- #
class ChemODE(nn.Module):
    """Wraps RateNet into dz/du, holding the per-channel rate scale."""

    def __init__(self, net: RateNet, rate_scale, box=1.15, box_w=0.08,
                 n_augment=0):
        super().__init__()
        self.net = net
        self.n_augment = int(n_augment)
        rs = np.asarray(rate_scale, dtype=np.float32)
        if self.n_augment:
            # The augmented channels have no data and therefore no measured
            # rate scale. Unit scale keeps them on the same footing as the
            # normalised physical channels, and the box gate below bounds them
            # exactly as it bounds the rest, so they cannot run away.
            rs = np.concatenate([rs, np.ones(self.n_augment, dtype=np.float32)])
        self.register_buffer("rate_scale",
                             torch.as_tensor(rs, dtype=torch.float32))
        self._cond = None
        # A *smooth restoring term in the field itself*, replacing the hard
        # clamp that used to live inside the RK4 stepper.  A clamp is invisible
        # to torchdiffeq, so the old model could learn a violently divergent
        # field and rely on the integrator to saw the tops off; dopri5 then
        # integrated the real thing and ignited three decades early.  Expressed
        # as part of dz/du, every integrator sees the same dynamics.
        self.box = box
        self.box_w = box_w
        # error-controlled RK4 (see rk4_rollout); None = plain fixed-step RK4
        self.rk4_tol = getattr(net.cfg, "rk4_tol", None)
        self.state_clamp = getattr(net.cfg, "state_clamp", None)
        self.rk4_max_substeps = int(getattr(net.cfg, "rk4_max_substeps", 64))
        self.rk4_stats = None
        self.physical = (getattr(net.cfg, "rate_space", "logtime") == "physical")
        if self.physical:
            # Filled by set_physical() from the dataset, and saved in the state
            # dict, so a loaded checkpoint carries its own decoder.
            d = rs.shape[0]
            self.register_buffer("phys_Q", torch.ones(d))
            self.register_buffer("phys_zfac", torch.ones(d))  # 2 / raw_range
            self.register_buffer("phys_time", torch.zeros(2))  # s_min, s_span
            self.register_buffer("phys_eps", torch.ones(d))   # K/s, dec/s
            self.phys_cap_mult = float(getattr(net.cfg, "phys_cap_mult", 4.0))
            self.target_cap = float(getattr(net.cfg, "target_cap", 25.0))

    def set_physical(self, Q, raw_range, s_min, s_span, eps):
        dev = self.rate_scale.device
        self.phys_eps.copy_(torch.as_tensor(np.asarray(eps), dtype=torch.float32,
                                            device=dev))
        self.phys_Q.copy_(torch.as_tensor(np.asarray(Q), dtype=torch.float32,
                                          device=dev))
        self.phys_zfac.copy_(torch.as_tensor(2.0 / np.asarray(raw_range),
                                             dtype=torch.float32, device=dev))
        self.phys_time.copy_(torch.tensor([float(s_min), float(s_span)],
                                          device=dev))

    def decode_physical(self, q, u):
        """Compressed physical rate q (B, D) at log-time u (B, 1) -> dz/du."""
        s = q * self.phys_Q
        # sign(s) * (10^|s| - 1), written as s * ln10 * expm1(x)/x with
        # x = ln10 |s|, so the gradient at s = 0 is ln10 rather than 0.
        # (sign() and abs() both have zero gradient at 0, and the output
        # layer starts at exactly zero: the naive form never trains.)
        x = LN10 * s.abs()
        ratio = torch.where(x > 1e-4, torch.expm1(x) / x.clamp_min(1e-4),
                            1.0 + 0.5 * x)
        draw_dt = self.phys_eps * s * LN10 * ratio
        s_min, s_span = self.phys_time[0], self.phys_time[1]
        # ln10 * s_span * (t + delta), computed in log space then exponentiated
        fac = LN10 * s_span * torch.exp(LN10 * (s_min + s_span * u))
        f = fac * self.phys_zfac * draw_dt
        cap = self.phys_cap_mult * self.target_cap * self.rate_scale
        return cap * torch.tanh(f / cap)

    def set_cond(self, cond):
        self._cond = cond

    def rate(self, z, u, cond=None):
        cond = self._cond if cond is None else cond
        return self.net(z, u, cond)

    def _gate(self, f, z):
        """Smoothly switch off the *outward* part of the field outside the box.

        An additive restoring force cannot work here.  To hold a field whose
        magnitude reaches ~20x rate_scale you need a restoring rate of the same
        order, and that rate is an eigenvalue: with rate_scale up to 3.2e2 and
        du ~ 0.02 it puts |lambda*h| far outside the RK4 stability region, so
        the rollout diverges to NaN.  Making the state stiff is exactly what
        this whole formulation exists to avoid.

        Damping the field instead adds no eigenvalue at all -- outside the box
        the dynamics simply stop rather than being yanked back.  Inside the box
        the gate is 1 to machine precision, so trajectories on the data
        manifold are untouched, and the gate is a plain function of z, so every
        integrator sees identical dynamics.
        """
        if self.box is None:
            return f
        gate = torch.sigmoid((self.box - z.abs()) / self.box_w)
        outward = f * torch.sign(z) > 0
        return torch.where(outward, f * gate, f)

    def forward(self, u, z):
        """torchdiffeq signature.  u scalar tensor, z (B, D)."""
        uu = u.reshape(1, 1).expand(z.shape[0], 1) if u.dim() == 0 else u
        if self.physical:
            f = self.decode_physical(self.net(z, uu, self._cond), uu)
        else:
            f = self.net(z, uu, self._cond) * self.rate_scale
        return self._gate(f, z)


# --------------------------------------------------------------------------- #
def _rk4_steps(ode, z, ua, H, n):
    """n equal RK4 sub-steps across an interval of length H (B, 1)."""
    h = H / n
    for j in range(n):
        u0 = ua + j * h
        k1 = ode(u0, z)
        k2 = ode(u0 + 0.5 * h, z + 0.5 * h * k1)
        k3 = ode(u0 + 0.5 * h, z + 0.5 * h * k2)
        k4 = ode(u0 + h, z + h * k3)
        z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        z = _clamp_state(ode, z)
    return z


def _clamp_state(ode, z):
    """Clamp the physical channels to the box (augmented ones are free)."""
    c = getattr(ode, "state_clamp", None)
    if c is None:
        return z
    m = int(getattr(ode, "n_augment", 0))
    if not m:
        return z.clamp(-c, c)
    return torch.cat([z[..., :-m].clamp(-c, c), z[..., -m:]], dim=-1)


def rk4_rollout(ode: ChemODE, z0, u_grid, substeps: int = 1, z_clip=None):
    """Batched explicit RK4 on *per-sample, non-uniform* grids.

    ``u_grid`` is (B, L): every window in the batch may have its own, different
    step sizes.  This is what lets us train on the raw Cantera sample points with
    zero resampling while still integrating the whole batch in lockstep.

    ERROR CONTROL (when ode.rk4_tol is set). A fixed number of sub-steps is
    right for most of a trajectory and wrong at a sharp ignition, where the
    fixed step overshoots (rollout_diagnose.py: RK4 left the physical range at
    ignition while dopri5 on the same field stayed inside, in 7/7 cases). Each
    data interval is therefore integrated with `substeps` sub-steps as before,
    and ALSO -- without gradients -- with half as many. By Richardson, the error
    of the fine result is about |fine - coarse| / (2^4 - 1). Only the samples
    whose estimate exceeds rk4_tol are redone, with sub-steps doubled until the
    estimate between two successive levels is below rk4_tol (or
    rk4_max_substeps is reached), and that result replaces theirs, with
    gradients. Smooth intervals -- induction included, however large the change
    in log Y, because it is smooth in log time -- cost 1.5x the plain stepper;
    only intervals that are genuinely under-resolved get refined. With
    rk4_tol = None the behaviour is exactly the old fixed-step RK4.

    Returns (B, L, D).
    """
    B, L = u_grid.shape
    tol = getattr(ode, "rk4_tol", None)
    n_max = int(getattr(ode, "rk4_max_substeps", 64))
    stats = getattr(ode, "rk4_stats", None)
    z = z0
    out = [z]
    for k in range(L - 1):
        ua = u_grid[:, k:k + 1]
        ub = u_grid[:, k + 1:k + 2]
        H = ub - ua
        zn = _rk4_steps(ode, z, ua, H, substeps)
        if tol is not None and substeps >= 1:
            with torch.no_grad():
                zc = _rk4_steps(ode, z.detach(), ua, H, max(substeps // 2, 1)) \
                    if substeps >= 2 else None
                if zc is None:           # substeps == 1: compare with 2
                    zf = _rk4_steps(ode, z.detach(), ua, H, 2)
                    est = (zf - zn.detach()).abs().amax(-1) / 15.0
                else:
                    est = (zn.detach() - zc).abs().amax(-1) / 15.0
                bad = torch.nonzero(~(est <= tol), as_tuple=False).flatten()
            if bad.numel():
                cond_all = ode._cond
                ode.set_cond(cond_all[bad] if cond_all is not None
                             and cond_all.shape[0] == B else cond_all)
                zb, uab, Hb = z[bad], ua[bad], H[bad]
                n, prev = substeps, zn.detach()[bad]
                with torch.no_grad():
                    while n < n_max:
                        n *= 2
                        cur = _rk4_steps(ode, zb.detach(), uab, Hb, n)
                        e = (cur - prev).abs().amax(-1) / 15.0
                        if not torch.isfinite(e).all():
                            prev = cur
                            continue
                        if (e <= tol).all():
                            break
                        prev = cur
                zr = _rk4_steps(ode, zb, uab, Hb, n)          # with gradients
                ode.set_cond(cond_all)
                zn = zn.clone()
                zn[bad] = zr
                if stats is not None:
                    stats["refined"] = stats.get("refined", 0) + int(bad.numel())
                    stats["max_n"] = max(stats.get("max_n", 0), n)
            if stats is not None:
                stats["intervals"] = stats.get("intervals", 0) + B
        z = zn
        if z_clip is not None:              # diagnostic only -- OFF by default
            z = z.clamp(-z_clip, z_clip)
        out.append(z)
    return torch.stack(out, dim=1)


@torch.no_grad()
def adaptive_rollout(ode: ChemODE, z0, u_grid, method="dopri5", rtol=1e-6, atol=1e-8):
    """Single adaptive-solver integration over one case's full u grid.

    This is the honest test: a *fixed-step* rollout can hide an ill-conditioned
    vector field, an adaptive solver cannot.
    """
    from torchdiffeq import odeint
    z = odeint(ode, z0, u_grid, method=method, rtol=rtol, atol=atol)
    return z.transpose(0, 1)


# --------------------------------------------------------------------------- #
class ChemNODE(nn.Module):
    """Container for the ODE. One object to save/load.

    The initialiser head is gone in this variant, so this is a thin wrapper --
    it is kept rather than using ChemODE directly so that the checkpoint format,
    the `model.net` / `model.rollout` interface and chemnode_eval.py all stay
    exactly as they were.
    """

    def __init__(self, d_state, d_cond, cfg, rate_scale):
        super().__init__()
        self.cfg = cfg
        m = int(getattr(cfg, "n_augment", 0))
        self.d_phys = d_state
        self.ode = ChemODE(RateNet(d_state + m, d_cond, cfg), rate_scale,
                           box=cfg.box, box_w=cfg.box_w, n_augment=m)
        self.init = (InitNet(d_state, d_cond, cfg)
                     if getattr(cfg, "use_initializer", False) else None)

    def _pad(self, z):
        """Append the augmented channels at their initial value, zero."""
        m = self.ode.n_augment
        if not m:
            return z
        return torch.cat([z, z.new_zeros(*z.shape[:-1], m)], dim=-1)

    def rate_phys(self, z, u, cond):
        """The field on the PHYSICAL channels only, with the augmented channels
        held at zero.

        Used by every term that regresses the field against a measured rate.
        There is no target for the augmented channels -- that is the point of
        them -- so the derivative-matching terms can only constrain the physical
        block, and zero is the one place in the augmented subspace where every
        trajectory demonstrably passes: all of them start there. It is a warm
        start for the field, not a claim about the field elsewhere.
        """
        out = self.ode.net(self._pad(z), u, cond)[..., :self.d_phys]
        if (self.ode.physical
                and getattr(self.cfg, "phys_loss", "logtime") == "logtime"):
            # compare in the same (normalised dz/du) space as the targets
            return self.ode.decode_physical(out, u) / self.ode.rate_scale
        return out

    @property
    def net(self):
        return self.ode.net

    def anchor(self, z0, cond, u1):
        """State at the first point the ODE is responsible for."""
        if self.init is None:
            return z0
        return self.init(z0, cond, u1)

    def rollout(self, z0, u_grid, cond, substeps=2, adaptive=False,
                method="dopri5", rtol=1e-6, atol=1e-8):
        """z0 is the state at u_grid[:, 0]. Returns (B, L, D) on u_grid.

        The ODE integrates the whole grid, first interval included.
        """
        self.ode.set_cond(cond)
        if self.init is None:
            zs = self._pad(z0)
            if adaptive:
                out = adaptive_rollout(self.ode, zs, u_grid[0], method, rtol,
                                       atol)
            else:
                out = rk4_rollout(self.ode, zs, u_grid, substeps=substeps)
            return out[..., :self.d_phys]
        # The map produces the state at u_grid[:, 1]; the ODE covers u_grid[1:].
        # Index 0 is returned as given -- it is the true initial condition and
        # nothing should be free to move it.
        z1 = _clamp_state(self.ode, self._pad(self.init(z0, cond, u_grid[:, 1:2])))
        if adaptive:
            rest = adaptive_rollout(self.ode, z1, u_grid[0, 1:], method, rtol,
                                    atol)
        else:
            rest = rk4_rollout(self.ode, z1, u_grid[:, 1:], substeps=substeps)
        return torch.cat([z0.unsqueeze(1), rest[..., :self.d_phys]], dim=1)


# ==========================================================================
# TRAINING
# ==========================================================================
#
# Training driver.
#
# Stages
# ------
# 0. ``pretrain``  -- regression of dz/du (and of the first-interval map).
# 1..N ``stageK``  -- multiple shooting on windows of growing length, taken
#                     straight off the Cantera sample grid (no resampling).
# N+1 ``anchor``   -- joint fine-tune of the initialiser and the ODE on a long
#                     rollout that starts from the true t = 0 state, i.e. on
#                     exactly the quantity the model is judged by.
#
# Run:
#     python -m chem_node.train --data_dir <.../data> --out_dir runs/baseline
# Re-running resumes: every stage skips itself if its ``done.pt`` exists.
#

# --------------------------------------------------------------------------- #
def channel_weights(ds, cfg, device):
    w = np.ones(ds.d_state, dtype=np.float32)
    w[0] = cfg.t_weight
    return torch.tensor(w / w.mean(), device=device)


def rate_eps_tensor(ds, cfg, device):
    """ds.rate_loss_eps as a broadcastable (D,) tensor."""
    return torch.tensor(np.asarray(ds.rate_loss_eps, dtype=np.float32),
                        device=device)


def weighted_huber(pred, true, w, delta, wpt=None):
    """Channel-weighted Huber, optionally weighted per time point as well.

    `wpt` broadcasts against the leading dimensions of `pred` (so (B, L, 1) for
    a trajectory, (N, 1) for flat points). ds.phase_w is normalised to mean 1,
    so passing it changes where the loss looks without changing its scale, and
    the learning rates tuned without it stay usable.
    """
    e = torch.nn.functional.huber_loss(pred, true, delta=delta, reduction="none")
    e = e * w
    if wpt is not None:
        e = e * wpt
    return e.mean()


_SKIPPED = [0]          # optimiser steps skipped since the last log line


def safe_step(model, opt, loss, cfg):
    """backward + clip + step, skipping any batch whose loss or gradient is
    not finite.

    clip_grad_norm_ on an inf gradient multiplies it by 0 and writes NaN into
    every weight, and one such step kills the run (stage 3 of auton_phys_p3).
    The exponential decoder of the physical rate makes rare overflowing
    batches possible; skipping them costs one batch, not the run. This is the
    per-window rejection of the old multi-window notebook, at batch level.
    """
    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True)
        _SKIPPED[0] += 1
        return "loss"
    loss.backward()
    gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    if not torch.isfinite(gn):
        opt.zero_grad(set_to_none=True)
        _SKIPPED[0] += 1
        return "grad"
    opt.step()
    return True


def rate_label(cfg):
    """Log label for the rate-matching term.

    Always measured at the TRUE (data) states, never along the predicted
    trajectory. The bracket says which space the comparison is made in:
      rate[dz/du]    log-time model: network rate vs dz/du target
      rate[dz/dt*t]  physical model: network dz/dt, error weighted by
                     (t + delta) -- compared as dz/du, which is the same thing
      rate[q]        physical model: compressed dz/dt, unweighted
    """
    if getattr(cfg, "rate_space", "logtime") != "physical":
        return "rate[dz/du]"          # network learns the log-time rate
    if getattr(cfg, "phys_loss", "logtime") == "q":
        return "rate[q]"              # compressed dz/dt, unweighted
    return "rate[dz/dt*t]"            # dz/dt error weighted by (t + delta)


def symlog(x, eps):
    """Signed log magnitude: linear for |x| << eps, logarithmic above it.

    asinh(x/eps) rather than sign(x)*log1p(|x|/eps) because it is analytic at
    zero, so the gradient of the loss is well behaved for targets that are
    exactly zero -- which a great many rate targets are.
    """
    return torch.asinh(x / eps)


def rate_huber(pred, true, w, cfg, wpt=None, eps=None):
    """The loss used wherever a RATE is regressed (pretraining, the derivative
    term of the shooting loss, the anchor stage's derivative term).

    Under cfg.rate_loss == "symlog" the comparison happens in log-magnitude,
    which is the only way a single loss can care about a target of 1e-5 and a
    target of 10 at the same time. Under "absolute" this is exactly the
    original Huber and the behaviour is unchanged.
    """
    if getattr(cfg, "rate_loss", "absolute") == "symlog":
        if eps is None:                    # caller did not pass ds's eps
            eps = getattr(cfg, "rate_loss_eps", None) or 0.02
        return weighted_huber(symlog(pred, eps), symlog(true, eps), w,
                              cfg.huber_delta, wpt)
    return weighted_huber(pred, true, w, cfg.huber_delta, wpt)


def noise_vector(ds, cfg, device):
    """Initial-state noise, per channel.

    Specified in decades when cfg.init_noise_decades is set, so the budget is
    the same physical statement in every channel instead of being scaled by
    each channel's arbitrary min-max range.
    """
    dex = getattr(cfg, "init_noise_decades", None)
    if dex is not None:
        # decades -> normalised units: one z-unit is raw_range/2 decades
        v = (2.0 * float(dex) / ds.raw_range).astype(np.float32)
        v[0] = cfg.init_noise_T          # temperature stays in its own units
    else:
        v = np.full(ds.d_state, cfg.init_noise, dtype=np.float32)
        v[0] = cfg.init_noise_T
    return torch.tensor(v, device=device)


def T(x, device):
    return torch.as_tensor(np.ascontiguousarray(x), device=device)


def align_jitter_u(ds, cfg, log=None):
    """cfg.align_noise_dex, expressed in u units.

    A delay wrong by eps decades shifts s by eps and therefore u by
    eps / s_span, since u = (s - s_min) / s_span. Returns 0.0 when the feature
    is off or the coordinate is not ignition-aligned (with a global coordinate
    there is no per-case delay to get wrong).
    """
    if ds.time_align != "ignition":
        return 0.0                       # no per-case delay, nothing to get wrong
    dex = getattr(cfg, "align_noise_dex", None)
    if dex is None:
        # "auto": train under exactly the misalignment deployment will produce.
        dex = float(getattr(ds, "tign_fit_rms_dex_val", 0.0) or 0.0)
        if log is not None:
            log(f"alignment jitter: auto -> {dex:.4f} dex "
                f"(the t_ign correlation's validation RMS)")
    dex = float(dex)
    if dex <= 0:
        return 0.0
    uj = dex / ds.s_span
    if log is not None:
        log(f"alignment jitter: {dex:.4f} dex = {uj:.5f} in u "
            f"(correlation error is {ds.tign_fit_rms_dex_test:.4f} dex on test)")
    return uj


# --------------------------------------------------------------------------- #
def pretrain(model, ds, cfg, device, log):
    """Stage 0.  In log-time the dz/du target is O(1) and smooth, so plain
    regression already yields a usable integrator; the shooting stages only have
    to remove the accumulation of small errors."""
    ck = Path(cfg.out_dir) / "pretrain.pt"
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=device)["model"])
        log("pretrain: loaded existing checkpoint")
        return

    u_tr, z_tr, c_tr, d_tr, w_tr = (T(x, device)
                                    for x in ds.flat_points(ds.train_idx))
    u_va, z_va, c_va, d_va, w_va = (T(x, device)
                                    for x in ds.flat_points(ds.val_idx))
    i_tr = i_va = None
    if model.init is not None:
        i_tr = [T(x, device) for x in ds.init_pairs(ds.train_idx)]
        i_va = [T(x, device) for x in ds.init_pairs(ds.val_idx)]
    w = channel_weights(ds, cfg, device)
    reps = rate_eps_tensor(ds, cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain_lr,
                            weight_decay=cfg.pretrain_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg.pretrain_epochs, eta_min=cfg.pretrain_lr * 0.01)
    n = z_tr.shape[0]
    best, best_state = float("inf"), None

    for ep in range(1, cfg.pretrain_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, cfg.pretrain_bs):
            j = perm[i:i + cfg.pretrain_bs]
            opt.zero_grad(set_to_none=True)
            loss = rate_huber(model.rate_phys(z_tr[j], u_tr[j], c_tr[j]), d_tr[j],
                              w, cfg, w_tr[j], reps)
            if model.init is not None:
                # One term over all cases, not over the sampled batch: there is
                # one first interval per case, so this is a few hundred rows and
                # costs nothing next to the flat-point regression.
                loss = loss + cfg.init_loss_weight * weighted_huber(
                    model.anchor(i_tr[0], i_tr[1], i_tr[2]), i_tr[3], w,
                    cfg.huber_delta)
            if safe_step(model, opt, loss, cfg) is True:
                tot += loss.item() * len(j)
        sched.step()
        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                vl = rate_huber(model.rate_phys(z_va, u_va, c_va), d_va, w, cfg,
                                w_va, reps).item()
                vi = (weighted_huber(model.anchor(i_va[0], i_va[1], i_va[2]),
                                     i_va[3], w, cfg.huber_delta).item()
                      if model.init is not None else 0.0)
            score = vl + cfg.init_loss_weight * vi
            star = ""
            if score < best:
                best, best_state = score, copy.deepcopy(model.state_dict())
                star = "  *"
            log(f"pretrain {ep:4d}/{cfg.pretrain_epochs}  train {tot/n:.4e}  "
                f"val {rate_label(cfg)} {vl:.4e}"
                + (f"  val init {vi:.4e}" if model.init is not None else "")
                + star)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "val": best}, ck)


# --------------------------------------------------------------------------- #
def shooting_loss(model, u, z, dz, cond, cfg, w, deriv_weight, noise,
                  u_jitter=0.0, wpt=None, reps=None, per_window=False):
    """Multiple-shooting loss on one batch of windows.

    Two perturbations, both of which exist so the model survives conditions it
    will meet at rollout but would never meet on clean training data:

    `noise`     perturbs the window's starting STATE, so the field has to be
                contractive toward the data manifold rather than merely
                accurate on it.
    `u_jitter`  perturbs the window's position in ALIGNED TIME by one scalar
                per window -- the same shift a mispredicted ignition delay
                produces at inference. Broadcasting one value across the window
                keeps every du identical, which is exactly the structure of the
                real error. The TARGET is unchanged: the model is asked to
                produce the true trajectory even when told the wrong phase.
    """
    model.ode.set_cond(cond)
    if u_jitter:
        u = u + u_jitter * torch.randn(u.shape[0], 1, device=u.device,
                                       dtype=u.dtype)
    z0 = z[:, 0]
    if noise is not None:
        z0 = (z0 + noise * torch.randn_like(z0)).clamp(-1.05, 1.05)
    # The ODE covers the WHOLE window, with no initialiser applied.
    #
    # Windows start at ds.anchor, which is 1 when the first-interval map is in
    # use, so z[:, 0] here is a mid-trajectory sample and u[:, 0] -> u[:, 1] is
    # an ordinary interval. model.rollout would apply the first-interval map to
    # it -- a map fitted for the t = 0 -> t_1 transition, handed an interior
    # pair it was never trained on and evaluated frozen, since the curriculum
    # stages optimise model.ode.parameters() only. The ODE would then be fit to
    # continue from a wrong state. Only the anchor stage, which genuinely starts
    # at t = 0, goes through model.rollout.
    #
    # The augmented channels still have to be padded and stripped here, which is
    # why this is not simply rk4_rollout on z0.
    zs = model._pad(z0)
    pred = rk4_rollout(model.ode, zs, u,
                       substeps=cfg.rk4_substeps)[..., :model.d_phys]
    wp = None if wpt is None else wpt[..., None]          # (B, L, 1)
    traj = weighted_huber(pred, z, w, cfg.huber_delta, wp)
    end = weighted_huber(pred[:, -1], z[:, -1], w, cfg.huber_delta)
    if deriv_weight > 0:
        B, L, D = z.shape
        flat_c = cond[:, None, :].expand(B, L, cond.shape[-1]).reshape(B * L, -1)
        drv = rate_huber(
            model.rate_phys(z.reshape(B * L, D), u.reshape(B * L, 1), flat_c),
            dz.reshape(B * L, D), w, cfg,
            None if wpt is None else wpt.reshape(B * L, 1), reps)
    else:
        drv = torch.zeros((), device=z.device)
    total = cfg.traj_weight * traj + cfg.end_weight * end + deriv_weight * drv
    if per_window:
        # mean |pred - true| per window, for naming the culprit of a skipped
        # batch; NaN/inf exactly where that window's rollout blew up
        pw = (pred.detach() - z).abs().mean(dim=(1, 2))
        return total, traj.detach(), end.detach(), drv.detach(), pw
    return total, traj.detach(), end.detach(), drv.detach()


# ==========================================================================
# INFERENCE
# ==========================================================================
#
# The path a deployed model takes, which training and evaluation never
# exercise.
#
# Every case seen during training, validation and testing comes out of the
# dataset, so its ignition delay is known -- it is in case_metadata.npz -- and
# `_build_time_coordinate` uses that true value to build the coordinate. At a
# genuinely new operating point there is no Cantera run and no measured delay,
# so it has to be predicted, and the coordinate built from the prediction.
#
# That distinction is not cosmetic. Alignment places ignition at the same u for
# every case, so the network never learns WHEN ignition happens -- it learns the
# trajectory shape in aligned time, and the timing comes entirely from t_ign.
# Predicting the delay with an error of eps decades therefore shifts the whole
# predicted trajectory by eps decades in physical time. The deployed model's
# ignition-delay accuracy is the CORRELATION's accuracy, not the t_ign_dex
# measured with true alignment. Report both; see chemnode_inference.py.
#

@torch.no_grad()
def predict_case(model, ds, device, pressure, T0, phi=None, steam=None,
                 Y0=None, times=None, t_ign=None, t_end=None, substeps=2,
                 adaptive=False, method="dopri5", rtol=1e-6, atol=1e-8):
    """Run the model at an operating point that is not in the dataset.

    pressure, T0, phi, steam : the operating point. Only the entries this model
        conditions on are needed (see ds.cond_names).
    Y0 : initial mass fractions over ALL mechanism species, in mechanism order
        (ds.species_names_all). If None, the composition is taken from the
        training case nearest this operating point, which is a convenience for
        smoke tests rather than something to rely on.
    times : physical times to report on, in seconds, strictly increasing and
        starting at 0. If None, a log grid is built spanning the same range
        relative to ignition that the training cases cover.
    t_ign : override the predicted delay. Pass the true value to measure what
        the prediction costs.

    Returns a dict with the predicted delay, the times, T [K] and Y [mass
    fractions] on the kept channels (ds.state_columns).
    """
    if t_ign is None:
        t_ign = float(np.atleast_1d(
            ds.predict_tign(T0, pressure, phi, 0.0 if steam is None else steam))[0])
    if not np.isfinite(t_ign) or t_ign <= 0:
        raise ValueError(f"predicted ignition delay is not usable: {t_ign}")

    if times is None:
        # Cover the same span in ALIGNED time that training did, so the model is
        # never asked to extrapolate in u.
        #
        # The grid starts at u_of_t(0), NOT at u = 0. t_of_u subtracts delta, so
        # for u below u_of_t(0) it returns a negative time -- t = 0 is already
        # an interior point of the [0, 1] range in u, sitting at
        # (log10(delta) - log10(t_ign) - s_min) / s_span. Starting at 0 and
        # clamping would collapse every point below that into a run of zeros,
        # which is not a strictly increasing grid.
        u_lo = float(ds.u_of_t(0.0, t_ign=t_ign))
        u_hi = 1.0
        if t_end is not None:
            # u = 1 is the longest time RELATIVE to ignition that any training
            # case reached. For a short predicted delay that can be a very long
            # ABSOLUTE time -- far past where anything is still happening -- so
            # allow the horizon to be capped in seconds.
            u_hi = min(1.0, float(ds.u_of_t(t_end, t_ign=t_ign)))
            if u_hi <= u_lo:
                raise ValueError(f"t_end={t_end:g}s is at or before t=0 in this "
                                 f"coordinate; nothing to integrate")
        u = np.linspace(u_lo, u_hi, ds.n_points)
        times = ds.t_of_u(u, t_ign=t_ign)
        times[0] = 0.0                       # kill the round-off at the anchor
    times = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        bad = int(np.argmin(np.diff(times)))
        raise ValueError(
            f"times must be strictly increasing; first violation at index {bad} "
            f"({times[bad]:.6e} -> {times[bad + 1]:.6e}). If you passed `times` "
            f"yourself, check for duplicates; a grid that starts below "
            f"u_of_t(0) produces negative times.")

    if Y0 is None:
        # nearest training case in normalised conditioning space
        c = ds.cond_vector(pressure, T0, phi, steam)
        j = int(np.argmin(np.linalg.norm(ds.cond[ds.train_idx] - c, axis=1)))
        Y0 = ds.states_phys[ds.train_idx[j], 0, 1:]

    u = ds.u_of_t(times, t_ign=t_ign).astype(np.float32)
    z0 = ds.z_from_physical(T0, Y0)
    cond = ds.cond_vector(pressure, T0, phi, steam)

    model.eval()
    pred = model.rollout(T(z0, device), T(u[None, :], device), T(cond, device),
                         substeps=substeps, adaptive=adaptive,
                         method=method, rtol=rtol, atol=atol)[0].cpu().numpy()
    phys = ds.z_to_physical(pred)
    return {"t_ign_pred": t_ign, "t": times, "u": u,
            "T": phys[:, 0], "Y": phys[:, 1:],
            "columns": ds.state_columns,
            "extrapolated_u": bool(u.min() < 0.0 or u.max() > 1.0)}


@torch.no_grad()
def rollout_case_aligned(model, ds, c, device, t_ign=None, substeps=2,
                         adaptive=False, method="dopri5", rtol=1e-6, atol=1e-8):
    """Roll out dataset case `c`, optionally re-aligning with a different delay.

    With `t_ign=None` this is the ordinary evaluation rollout: the coordinate
    uses the case's TRUE delay. Passing a predicted delay rebuilds the grid the
    way deployment would.

    Changing t_ign is a constant shift in log space, so the step sizes du are
    untouched and only the absolute position in u moves -- which is exactly the
    mechanism by which a delay error becomes a timing error.
    """
    u = ds.u[c].astype(np.float32)
    if t_ign is not None:
        u = u - np.float32((np.log10(t_ign) - ds.log_tign[c, 0]) / ds.s_span)
    z0 = ds.z[c, 0].astype(np.float32)[None, :]
    cond = ds.cond[c][None, :]
    model.eval()
    return model.rollout(T(z0, device), T(u[None, :], device), T(cond, device),
                         substeps=substeps, adaptive=adaptive, method=method,
                         rtol=rtol, atol=atol)[0].cpu().numpy()


@torch.no_grad()
def full_rollout_metrics(model, ds, case_idx, device, cfg, substeps=2,
                         adaptive=False, extra=None):
    """Continuous rollout from the true state at t = 0 -- the headline metric.

    Returns (T RMSE, mean |dlog10 Y| over true Y > eval_metric_floor, max
    log10 Y) -- unchanged, so numbers stay comparable with earlier runs.

    If `extra` is a dict it is filled with two species metrics that do not
    depend on any floor:
      logY_1e6   mean |dlog10 Y| only where the TRUE Y > 1e-6 -- the species
                 that carry mass; radicals in induction are excluded
      absY_pct   mean |Y_pred - Y_true| in mass-%, over all species and times
      absY_major_pct  the same over species whose true max Y > 1e-3
    """
    model.eval()
    Tr, Lm, Sm = [], [], []
    L6, Aa, Am = [], [], []
    for c in np.atleast_1d(case_idx):
        u = T(ds.u[c][None, :].astype(np.float32), device)
        z0 = T(ds.z[c][None, 0].astype(np.float32), device)
        cond = T(ds.cond[c][None, :], device)
        try:
            pred = model.rollout(z0, u, cond, substeps=substeps, adaptive=adaptive,
                                 method=cfg.eval_method, rtol=cfg.eval_rtol,
                                 atol=cfg.eval_atol)[0].cpu().numpy()
        except Exception:
            continue
        rp, rt = ds.from_z(pred), ds.from_z(ds.z[c])
        Tr.append(float(np.sqrt(np.mean((rp[:, 0] - rt[:, 0]) ** 2))))
        m = rt[:, 1:] > np.log10(cfg.eval_metric_floor)
        if m.any():
            Lm.append(float(np.mean(np.abs(rp[:, 1:][m] - rt[:, 1:][m]))))
        # Cheapest possible physical-plausibility monitor: the largest mass
        # fraction the rollout produced. Anything above 1 is impossible, so a
        # number above 1 in the training log says the run is unphysical long
        # before the evaluation plots do. This is a DIAGNOSTIC only -- nothing
        # is constrained -- so it stays a fair data-driven baseline.
        #
        # Reported as log10(Y), in decades, and NOT as Y. A diverged rollout
        # reaches z of order 50, which is log10(Y) of order 300, and 10 ** that
        # overflows float64 -- so computing Y first turns a useful number into
        # inf plus a RuntimeWarning. In decades it stays finite and readable no
        # matter how badly the field has blown up, and it says how far past the
        # physical bound the excursion went.
        Sm.append(float(rp[:, 1:].max()))
        if extra is not None:
            yf = cfg.y_floor
            Yp = np.clip(10.0 ** np.minimum(rp[:, 1:], 1.0) - yf, 0.0, None)
            Yt = np.clip(10.0 ** rt[:, 1:] - yf, 0.0, None)
            m6 = Yt > 1e-6
            if m6.any():
                L6.append(float(np.mean(np.abs(rp[:, 1:][m6] - rt[:, 1:][m6]))))
            Aa.append(100.0 * float(np.mean(np.abs(Yp - Yt))))
            maj = Yt.max(0) > 1e-3
            if maj.any():
                Am.append(100.0 * float(np.mean(np.abs(Yp[:, maj] - Yt[:, maj]))))
    if extra is not None:
        f = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
        extra.update(logY_1e6=f(L6), absY_pct=f(Aa), absY_major_pct=f(Am))
    return (float(np.mean(Tr)) if Tr else float("nan"),
            float(np.mean(Lm)) if Lm else float("nan"),
            float(np.max(Sm)) if Sm else float("nan"))


@torch.no_grad()
def deployed_metrics(model, ds, cfg, device, case_idx, substeps=2):
    """Accuracy with the ignition delay PREDICTED, not measured.

    This is the number that belongs in the thesis, because it is the only one
    that corresponds to the stated use of the model: initial conditions and a
    time grid, nothing else. Every other metric in this pipeline builds the
    coordinate from the case's measured delay, which at a new operating point
    does not exist.

    With time_align == "global" there is no delay in the coordinate, so this
    returns exactly what full_rollout_metrics does -- and that identity is the
    main argument for that coordinate. With "ignition" the gap between the two
    is the price of the alignment, and it is a number worth reporting
    explicitly rather than leaving implicit.
    """
    model.eval()
    aligned = ds.time_align == "ignition"
    Tr, Lm, Ym, Ig = [], [], [], []
    for c in np.atleast_1d(case_idx):
        tp = None
        if aligned:
            tp = float(np.atleast_1d(ds.predict_tign(
                ds.T0[c], ds.pressures[c], ds.phi[c], ds.steam[c]))[0])
            if not np.isfinite(tp) or tp <= 0:
                continue
        try:
            pred = rollout_case_aligned(model, ds, c, device, t_ign=tp,
                                        substeps=substeps)
        except Exception:                                    # noqa: BLE001
            continue
        rp, rt = ds.from_z(pred), ds.from_z(ds.z[c])
        Tr.append(float(np.sqrt(np.mean((rp[:, 0] - rt[:, 0]) ** 2))))
        m = rt[:, 1:] > np.log10(cfg.eval_metric_floor)
        if m.any():
            Lm.append(float(np.mean(np.abs(rp[:, 1:][m] - rt[:, 1:][m]))))
        Ym.append(float(rp[:, 1:].max()))          # decades, never overflows
        # ignition time from the half-rise crossing, on both traces
        def half_rise(T_):
            thr = T_[0] + 0.5 * (T_.max() - T_[0])
            k = np.flatnonzero(T_ >= thr)
            return ds.t[c][k[0]] if k.size else np.nan
        a, b = half_rise(rp[:, 0]), half_rise(rt[:, 0])
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            Ig.append(abs(np.log10(a) - np.log10(b)))
    f = lambda v: float(np.mean(v)) if v else float("nan")    # noqa: E731
    return {"T_rmse": f(Tr), "logY_mae": f(Lm), "t_ign_dex": f(Ig),
            "logY_max": float(np.max(Ym)) if Ym else float("nan"),
            "n": len(Tr)}


# --------------------------------------------------------------------------- #
def _run_epochs(tag, model, cfg, log, params, epochs, lr, batch_fn, val_fn, sdir,
                ep_counter=None):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=cfg.lr_plateau_factor, patience=cfg.lr_plateau_patience,
        min_lr=cfg.min_lr, threshold=1e-3)
    best, best_state, bad = float("inf"), None, 0
    start = 1
    lp = sdir / "latest.pt"
    if lp.exists():   # cluster-timeout resume, per stage
        st = torch.load(lp, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        if "opt" in st:
            opt.load_state_dict(st["opt"])
            ok = all(torch.isfinite(v).all() for stt in opt.state.values()
                     for v in stt.values() if torch.is_tensor(v))
            if not ok:
                # saved after a NaN step: the weights in latest.pt are the
                # finite best_state, but Adam's moments are poisoned
                opt.state.clear()
                log(f"[{tag}] optimiser state was non-finite -- reset")
        best = st.get("val", float("inf"))
        best_state = copy.deepcopy(model.state_dict())
        start = st.get("ep", 0) + 1
        # Restore the LR schedule and the early-stop counter too.  Without
        # these a resumed stage restarts at the base LR and has to re-plateau
        # its way back down, which shows up as a loss bump after every restart.
        if "sched" in st:
            sched.load_state_dict(st["sched"])
        bad = st.get("bad", 0)
        if ep_counter is not None:
            ep_counter[0] = st.get("ep", 0)
        log(f"[{tag}] resuming at epoch {start} (best val {best:.4e}, "
            f"lr {opt.param_groups[0]['lr']:.2e})")
    for ep in range(start, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        acc = batch_fn(opt)
        if ep % cfg.validate_every == 0 or ep == epochs:
            model.eval()
            vl = val_fn()
            rb = float(getattr(cfg, "rollback_factor", 20.0))
            if best_state is not None and (not np.isfinite(vl)
                                           or (rb > 0 and vl > rb * best)):
                # The field jumped to a much worse (or NaN) state. Go back to
                # the best weights, clear Adam's moments (they carry the jump)
                # and halve the LR, instead of spending the patience budget
                # climbing back -- stage 1 of auton_phys_p3 lost ~20 epochs
                # to one such jump, stage 3 lost everything to a NaN.
                model.load_state_dict(best_state)
                opt.state.clear()
                for g in opt.param_groups:
                    g["lr"] = max(g["lr"] * 0.5, cfg.min_lr)
                log(f"[{tag}] ep {ep:4d}/{epochs}  val {vl:.4e} vs best "
                    f"{best:.4e} -- ROLLED BACK to best, lr -> "
                    f"{opt.param_groups[0]['lr']:.2e}"
                    + (f"  ({_SKIPPED[0]} non-finite batches skipped)"
                       if _SKIPPED[0] else ""))
                _SKIPPED[0] = 0
                bad += 1
                if bad >= cfg.early_stop_patience:
                    log(f"[{tag}] early stop at epoch {ep}")
                    break
                continue
            sched.step(vl)
            improved = vl < best * (1 - 1e-3)
            star = ""
            if vl < best:
                best = vl
                best_state = copy.deepcopy(model.state_dict())
                star = "  *"
            bad = 0 if improved else bad + 1
            log(f"[{tag}] ep {ep:4d}/{epochs}  " +
                "  ".join(f"{k} {v:.3e}" for k, v in acc.items()) +
                f"  val {vl:.4e}  lr {opt.param_groups[0]['lr']:.2e}  "
                f"{time.perf_counter()-t0:.1f}s{star}"
                + (f"  [{_SKIPPED[0]} non-finite batches skipped]"
                   if _SKIPPED[0] else ""))
            _SKIPPED[0] = 0
            torch.save({"model": best_state if best_state is not None
                        else model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "bad": bad,
                        "ep": ep, "val": best}, sdir / "latest.pt")
            if bad >= cfg.early_stop_patience:
                log(f"[{tag}] early stop at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def train_stage(model, ds, cfg, device, log, stage_i, window_len, stride, epochs, lr):
    tag = f"stage{stage_i}_L{window_len}"
    sdir = Path(cfg.out_dir) / tag
    sdir.mkdir(parents=True, exist_ok=True)
    if (sdir / "done.pt").exists():
        st = torch.load(sdir / "done.pt", map_location=device)
        model.load_state_dict(st["model"])
        log(f"[{tag}] already complete -- loaded (val {st['val']:.4e})")
        return

    tr = ds.make_windows(ds.train_idx, window_len, stride)
    va = ds.make_windows(ds.val_idx, window_len, stride)
    bs = int(np.clip(cfg.window_batch_budget // window_len, 16, cfg.max_window_batch))
    log(f"\n=== {tag}: window_len={window_len} stride={stride} epochs={epochs} "
        f"lr={lr:.1e} batch={bs} | train windows {tr['n']} val {va['n']} ===")

    w = channel_weights(ds, cfg, device)
    reps = rate_eps_tensor(ds, cfg, device)
    noise = noise_vector(ds, cfg, device)
    frac = stage_i / max(len(cfg.stages) - 1, 1)
    dw = cfg.deriv_weight_start * (cfg.deriv_weight_end / cfg.deriv_weight_start) ** frac
    # Alignment jitter, converted from decades of ignition delay into u units.
    uj = align_jitter_u(ds, cfg, log if stage_i == 0 else None)
    va_t = {k: T(va[k], device) for k in ("u", "z", "dz", "cond", "w")}
    skip_csv = sdir / "skipped_windows.csv"

    def report_skip(reason, pw, j):
        """Name the windows behind a skipped batch.

        reason "loss": the forward pass itself went non-finite, and pw is
        non-finite exactly for the windows whose rollout blew up -- those ARE
        the culprits. reason "grad": the loss was finite but the gradient
        overflowed; no window is non-finite, so the largest-error windows are
        listed as SUSPECTS. Everything is also appended to skipped_windows.csv
        in the stage directory.
        """
        pw = pw.cpu().numpy()
        bad = np.where(~np.isfinite(pw))[0]
        kind = "culprit"
        if bad.size == 0:
            bad = np.argsort(-pw)[:3]
            kind = "suspect"
        new = not skip_csv.exists()
        with open(skip_csv, "a") as fh:
            if new:
                fh.write("reason,kind,case,start,t_start_s,t_over_tign,"
                         "mean_abs_err\n")
            for k in bad:
                c, st = int(tr["case"][j[k]]), int(tr["start"][j[k]])
                t0 = float(ds.t[c, st])
                ti = float(ds.ignition_times[c])
                fh.write(f"{reason},{kind},{c},{st},{t0:.4e},"
                         f"{t0 / ti if ti > 0 else float('nan'):.4g},"
                         f"{pw[k]:.4e}\n")
        items = []
        for k in bad[:5]:
            c, st = int(tr["case"][j[k]]), int(tr["start"][j[k]])
            t0 = float(ds.t[c, st])
            ti = float(ds.ignition_times[c])
            items.append(f"case {c} start {st} (t={t0:.2e}s, "
                         f"{t0 / ti:.3g}x t_ign)")
        log(f"[{tag}] SKIPPED batch ({'non-finite loss' if reason == 'loss' else 'gradient overflow'}; "
            f"{len(bad)} {kind}{'s' if len(bad) != 1 else ''}"
            f"{' of ' + str(int((~np.isfinite(pw)).sum())) if kind == 'culprit' and len(bad) > 5 else ''}): "
            + "; ".join(items))

    def batch_fn(opt):
        perm = np.random.permutation(tr["n"])
        acc = np.zeros(4)
        for i in range(0, tr["n"], bs):
            j = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss, a, b, c, pw = shooting_loss(
                model, T(tr["u"][j], device), T(tr["z"][j], device),
                T(tr["dz"][j], device), T(tr["cond"][j], device),
                cfg, w, dw, noise, u_jitter=uj, wpt=T(tr["w"][j], device),
                reps=reps, per_window=True)
            r = safe_step(model, opt, loss, cfg)
            if r is True:
                acc += np.array([loss.item(), a.item(), b.item(),
                                 c.item()]) * len(j)
            else:
                report_skip(r, pw, j)
        acc /= tr["n"]
        return {"tot": acc[0], "traj": acc[1], "end": acc[2],
                rate_label(cfg): acc[3]}

    @torch.no_grad()
    def val_fn():
        s = 0.0
        for i in range(0, va["n"], bs):
            sl = slice(i, i + bs)
            l, *_ = shooting_loss(model, va_t["u"][sl], va_t["z"][sl],
                                  va_t["dz"][sl], va_t["cond"][sl], cfg, w,
                                  0.0, None, wpt=va_t["w"][sl])
            s += l.item() * va_t["u"][sl].shape[0]
        return s / va["n"]

    best = _run_epochs(tag, model, cfg, log, list(model.ode.parameters()),
                       epochs, lr, batch_fn, val_fn, sdir)
    torch.save({"model": model.state_dict(), "val": best}, sdir / "done.pt")


def train_anchor(model, ds, cfg, device, log):
    """Final stage: optimise the initialiser *and* the ODE on a long rollout
    that starts at t = 0 -- the objective the model is actually evaluated on."""
    tag = "anchor"
    sdir = Path(cfg.out_dir) / tag
    sdir.mkdir(parents=True, exist_ok=True)
    if (sdir / "done.pt").exists():
        st = torch.load(sdir / "done.pt", map_location=device)
        model.load_state_dict(st["model"])
        log(f"[{tag}] already complete -- loaded (val {st['val']:.4e})")
        return
    L = min(cfg.anchor_len, ds.n_points)
    log(f"\n=== {tag}: rollout length {L} from t=0, epochs={cfg.anchor_epochs}, "
        f"lr={cfg.anchor_lr:.1e} ===")
    w = channel_weights(ds, cfg, device)
    reps = rate_eps_tensor(ds, cfg, device)
    uj = align_jitter_u(ds, cfg, log)
    ztr = T(ds.z[ds.train_idx, :L].astype(np.float32), device)
    utr = T(ds.u[ds.train_idx, :L].astype(np.float32), device)
    ctr = T(ds.cond[ds.train_idx], device)
    zva = T(ds.z[ds.val_idx, :L].astype(np.float32), device)
    uva = T(ds.u[ds.val_idx, :L].astype(np.float32), device)
    cva = T(ds.cond[ds.val_idx], device)
    bs = max(4, getattr(cfg, 'anchor_batch_budget',
                        cfg.window_batch_budget) // L)

    dtr = T(ds.dz_du_n[ds.train_idx, :L].astype(np.float32), device)
    dva = T(ds.dz_du_n[ds.val_idx, :L].astype(np.float32), device)
    ptr = T(ds.phase_w[ds.train_idx, :L].astype(np.float32), device)
    pva = T(ds.phase_w[ds.val_idx, :L].astype(np.float32), device)

    def one(zb, ub, cb, db=None, consistency=False, u_jitter=0.0, pb=None):
        # Same alignment jitter as the curriculum stages: one shift per
        # trajectory, so every du is preserved and only the phase moves.
        if u_jitter:
            ub = ub + u_jitter * torch.randn(ub.shape[0], 1, device=ub.device,
                                             dtype=ub.dtype)
        pred = model.rollout(zb[:, 0], ub, cb, substeps=cfg.rk4_substeps)
        wp = None if pb is None else pb[..., None]
        loss = weighted_huber(pred, zb, w, cfg.huber_delta, wp)
        # Derivative matching at the TRUE states.  Without this the anchor
        # stage only constrains the discrete map, and the underlying field is
        # free to drift -- which is exactly how a model with 9e-4 fixed-step
        # rollout loss ignites three decades early under an adaptive solver.
        if db is not None and cfg.anchor_deriv_weight > 0:
            B, Ls, D = zb.shape
            model.ode.set_cond(cb)
            r = model.rate_phys(zb.reshape(B * Ls, D),
                                ub.reshape(B * Ls, 1),
                                cb.repeat_interleave(Ls, 0))
            loss = loss + cfg.anchor_deriv_weight * rate_huber(
                r, db.reshape(B * Ls, D), w, cfg,
                None if pb is None else pb.reshape(B * Ls, 1), reps)
        # Step doubling: a real ODE gives the same trajectory at half the step.
        if consistency and cfg.consistency_weight > 0:
            fine = model.rollout(zb[:, 0], ub, cb, substeps=2 * cfg.rk4_substeps)
            loss = loss + cfg.consistency_weight * weighted_huber(
                pred, fine.detach(), w, cfg.huber_delta, wp)
        return loss

    ep_counter = [0]

    def batch_fn(opt):
        ep_counter[0] += 1
        perm = torch.randperm(ztr.shape[0], device=device)
        tot = 0.0
        for i in range(0, ztr.shape[0], bs):
            j = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = one(ztr[j], utr[j], ctr[j], dtr[j],
                       consistency=(ep_counter[0] % cfg.consistency_every == 0),
                       u_jitter=uj, pb=ptr[j])
            if safe_step(model, opt, loss, cfg) is True:
                tot += loss.item() * len(j)
        return {"traj": tot / ztr.shape[0]}

    @torch.no_grad()
    def val_fn():
        if not cfg.anchor_val_adaptive:
            return float(np.mean([one(zva[i:i + bs], uva[i:i + bs],
                                      cva[i:i + bs],
                                      pb=pva[i:i + bs]).item()
                                  for i in range(0, zva.shape[0], bs)]))
        n_val = zva.shape[0]
        sel = range(n_val)
        if cfg.anchor_val_cases and cfg.anchor_val_cases < n_val:
            sel = np.linspace(0, n_val - 1, cfg.anchor_val_cases).astype(int)
        sel = list(sel)
        tot = 0.0
        for i in sel:                        # adaptive solver is per-case
            try:
                pred = model.rollout(zva[i:i + 1, 0], uva[i:i + 1], cva[i:i + 1],
                                     adaptive=True, method=cfg.eval_method,
                                     rtol=cfg.eval_rtol, atol=cfg.eval_atol)
            except Exception:                # solver gave up == a bad field
                tot += 1.0
                continue
            tot += weighted_huber(pred, zva[i:i + 1], w, cfg.huber_delta,
                                  pva[i:i + 1, :, None]).item()
        return tot / len(sel)

    best = _run_epochs(tag, model, cfg, log, list(model.parameters()),
                       cfg.anchor_epochs, cfg.anchor_lr, batch_fn, val_fn, sdir,
                       ep_counter=ep_counter)
    torch.save({"model": model.state_dict(), "val": best}, sdir / "done.pt")


# --------------------------------------------------------------------------- #
def build(cfg):
    """Seeding is done here, from cfg.seed, so a run is reproducible without
    anything being passed on the command line.

    torch.manual_seed covers weight initialisation and torch.randperm (the
    anchor stage's shuffling); np.random.seed covers np.random.permutation (the
    curriculum stages' shuffling) and the noise draws. The data split uses its
    own np.random.default_rng(cfg.seed), so it is reproducible independently.

    Runs are not bit-identical across machines: TF32 and cuDNN kernel selection
    are hardware-dependent. Set allow_tf32 = False and
    torch.use_deterministic_algorithms(True) if exact reproduction is needed,
    at a substantial cost in speed.
    """
    device = cfg.resolved_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    ds = ChemDataset(cfg)
    model = ChemNODE(ds.d_state, ds.d_cond, cfg, ds.rate_scale).to(device)
    if getattr(cfg, "rate_space", "logtime") == "physical":
        model.ode.set_physical(ds.phys_Q, ds.raw_range, ds.s_min, ds.s_span,
                               ds.phys_eps)
    return ds, model, device


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--from_config", type=str, default=None,
                   help="start from ALL settings of an earlier run's "
                        "config.json (exact replay of that recipe); flags given "
                        "on the command line still override, e.g. --data_dir "
                        "and --out_dir. Settings the old file lacks keep "
                        "today's defaults and are printed.")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--epochs_scale", type=float, default=1.0)
    p.add_argument("--pretrain_epochs", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--n_blocks", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no_time_input", action="store_true")
    p.add_argument("--anchor_batch_budget", type=int, default=None,
                   help="anchor batch = budget // anchor_len. Memory scales as "
                        "batch * anchor_len; 9000 gives batch 16 at 550 points. "
                        "Separate from the curriculum budget because the two "
                        "stages bind on different resources.")
    p.add_argument("--window_batch_budget", type=int, default=None,
                   help="anchor batch = budget // anchor_len. The default (8192) "
                        "gives 17 on a 475-point trajectory, which badly "
                        "under-uses a 16GB GPU. Try 475*48 = 22800.")
    p.add_argument("--anchor_epochs", type=int, default=None)
    p.add_argument("--anchor_val_cases", type=int, default=None)
    p.add_argument("--consistency_every", type=int, default=None)
    p.add_argument("--tign_source", type=str, default=None,
                   choices=("measured", "predicted"),
                   help="which delay DEFINES the coordinate. 'predicted' builds "
                        "it from the correlation during training too, so the "
                        "train/inference mismatch is exactly zero. 'measured' "
                        "(default) uses the true delay and leaves the "
                        "correlation's error as a deployment-time shift.")
    p.add_argument("--tign_fit_mode", type=str, default=None,
                   choices=("arrhenius", "arrhenius_log", "quadratic",
                            "cubic", "rbf"),
                   help="basis for the ignition-delay correlation. All three "
                        "are scored in summary(); this picks the one used. "
                        "Sets the floor on deployed accuracy.")
    p.add_argument("--init_noise", type=float, default=None,
                   help="Gaussian noise on each training window's initial SPECIES "
                        "state, in normalised units. Makes the field contractive, "
                        "but also blurs the state -- and with --no_time_input the "
                        "state is the only phase signal the field has. The default "
                        "0.03 is ~0.2 decades of log10(Y), which is far more phase "
                        "uncertainty than the ignition-delay correlation carries. "
                        "Reduce it for an autonomous model.")
    p.add_argument("--init_noise_T", type=float, default=None,
                   help="the same, for the temperature channel. Temperature is "
                        "flat during induction so it carries little phase "
                        "information; this matters much less.")
    p.add_argument("--align_noise_dex", type=float, default=None,
                   help="train with the u grid jittered by this many decades of "
                        "ignition delay, so the field stops relying on a clock "
                        "it will not have at inference. The principled value is "
                        "the correlation's own error, which summary() reports as "
                        "tign_fit_rms_dex. 0 (default) disables it.")
    # ---- the four switches this revision adds ----------------------------
    p.add_argument("--time_align", type=str, default=None,
                   choices=("global", "ignition"),
                   help="'global' (default) needs nothing at inference but the "
                        "initial conditions and a time grid. 'ignition' "
                        "reproduces the earlier runs and needs a predicted "
                        "t_ign at inference.")
    p.add_argument("--species_scale", type=str, default=None,
                   choices=("physical", "per_channel"),
                   help="'physical' gives every species channel the same "
                        "decades-per-z-unit, so the loss is mean |dlog10 Y| "
                        "and |z|<=1 means 0<=Y<=1. 'per_channel' is the "
                        "original min-max.")
    p.add_argument("--rate_loss", type=str, default=None,
                   choices=("symlog", "absolute"),
                   help="'symlog' prices a rate error by its RATIO, so the "
                        "1e-5-scale induction targets are not rounding error "
                        "on the ignition spike.")
    p.add_argument("--rate_head", type=str, default=None,
                   choices=("sinh", "linear"))
    p.add_argument("--n_augment", type=int, default=None,
                   help="extra state channels, starting at zero, integrated by "
                        "the same field and never compared against data "
                        "(augmented neural ODE). They restore the Markov "
                        "property the log floor destroys, which an AUTONOMOUS "
                        "field (--no_time_input) needs. Only turn this on if "
                        "chemnode_diagnose.py section 7 reports stationary "
                        "samples: with no obstruction to fix it is pure extra "
                        "freedom and measurably hurts.")
    p.add_argument("--rate_head_k", type=float, default=None)
    p.add_argument("--rate_loss_eps", type=float, default=None,
                   help="fix the symlog turnover instead of deriving it from "
                        "the drift budget. Leave unset: the derivation depends "
                        "on rate_scale, which depends on your time horizon.")
    p.add_argument("--rate_drift_tol_decades", type=float, default=None,
                   help="how far a species may drift through induction, in "
                        "decades. Sets eps. 0.05 is a 12 %% error in Y.")
    p.add_argument("--init_noise_decades", type=float, default=None,
                   help="initial-state noise in DECADES of log10(Y), applied "
                        "identically to every species channel. Must be smaller "
                        "than the accuracy you want out of the induction "
                        "phase.")
    p.add_argument("--no_phase_balance", action="store_true",
                   help="let the ignition window dominate the trajectory loss, "
                        "as before")
    p.add_argument("--box", type=float, default=None)
    p.add_argument("--y_floor", type=float, default=None,
                   help="raising this shrinks the first-interval jump decade "
                        "for decade and shortens every species' dynamic range; "
                        "1e-10 and 1e-8 are the values worth trying.")
    p.add_argument("--legacy_init", action="store_true",
                   help="like --legacy but keeps the first-interval map on. Use "
                        "this if the baseline you are comparing against was "
                        "chemnode_train_init.py, so the 'before' column differs "
                        "by the representation changes ONLY.")
    p.add_argument("--legacy", action="store_true",
                   help="reproduce the original configuration exactly: "
                        "ignition alignment with no jitter, per-channel "
                        "min-max scaling, absolute rate loss, linear head, no "
                        "phase balance, box 1.15, 8 Fourier features. Use this "
                        "for the 'before' column of the ablation table.")
    p.add_argument("--no_initializer", action="store_true",
                   help="make the ODE cross the first sampled interval itself "
                        "instead of handing it to a direct map. On this dataset "
                        "that interval is ~3 decades and 23 of 27 channels "
                        "start pinned at the floor, so expect it to cost "
                        "induction and ignition-delay accuracy.")
    p.add_argument("--rate_space", type=str, default=None,
                   choices=["logtime", "physical"],
                   help="'physical': learn the compressed physical-time rate "
                        "dz/dt (autonomous source term), integrate in log-time "
                        "with the analytic factor. Needs --time_align global.")
    p.add_argument("--phys_loss", type=str, default=None,
                   choices=["logtime", "q"],
                   help="with --rate_space physical: where the rate loss is "
                        "measured (default logtime, time-weighted; 'q' is "
                        "time-blind and diverged on the 1000 s horizon)")
    p.add_argument("--phys_eps", type=float, default=None,
                   help="with --rate_space physical: compression threshold in "
                        "K/s and dec/s. Default: derived from the drift budget "
                        "and the time horizon. 1.0 = the notebook's choice.")
    p.add_argument("--lr_scale", type=float, default=1.0,
                   help="multiply every stage's LR and the anchor LR (not the "
                        "pretrain LR). 0.3 for --rate_space physical, whose "
                        "exponential decoder does not tolerate the LR jump at "
                        "the start of each stage.")
    p.add_argument("--rk4_tol", type=float, default=None,
                   help="error-controlled RK4: redo only the intervals whose "
                        "local error estimate exceeds this (z units; 1e-3 ~ "
                        "1 K in T, 0.006 dec in a species). Default: off")
    p.add_argument("--rk4_max_substeps", type=int, default=None,
                   help="cap on sub-steps per interval under --rk4_tol "
                        "(default 64). Cost grows with it: measured per stage-2 "
                        "epoch 1.7x at 8, 2.5x at 16, 7x at 32, 18x at 256")
    p.add_argument("--state_clamp", type=float, default=None,
                   help="clamp the state to |z| <= this after every RK4 "
                        "sub-step (1.0 = the physical range: y_floor <= Y <= 1, "
                        "T inside the data range). Default: off")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the rate network and the ODE function. "
                        "Same weights, same maths (verified: max difference "
                        "0 to 6e-8 against eager); fuses the many small GPU "
                        "kernels of an RK4 step. Linux/Kaggle only; falls back "
                        "to eager with a warning if compilation is unavailable. "
                        "The first epoch is slower while it compiles.")
    p.add_argument("--cond_p_only", action="store_true",
                   help="condition the field on pressure only. T0, O2/CH4 and "
                        "H2O/CH4 are initial conditions of a trajectory, not "
                        "inputs of the chemical source term: T and Y already "
                        "carry them. Required for a field usable in CFD.")
    p.add_argument("--no_tf32", action="store_true",
                   help="disable TF32 matmuls. TF32 is ~2x on Ampere+ but uses a "
                        "10-bit mantissa; use this to reproduce a full-precision "
                        "run bit-for-bit against an earlier one.")
    a = p.parse_args(argv)

    cfg = Config()
    if a.from_config:
        old = json.loads(Path(a.from_config).read_text())
        known = {f.name for f in fields(Config)}
        for k, v in old.items():
            if k not in known:
                continue
            if k in ("data_dir", "out_dir"):
                v = Path(v)
            elif k == "stages":
                v = tuple(tuple(x) for x in v)
            setattr(cfg, k, v)
        miss = sorted(known - set(old))
        # Runs older than the rollback safeguard never rolled back; replay
        # them without it so the recipe is the one that produced them.
        if "rollback_factor" in miss:
            cfg.rollback_factor = 0.0
        print(f"settings replayed from {a.from_config} ({len(old)} keys)")
        if miss:
            print("  not in that file, today's default used: "
                  + ", ".join(f"{k}={getattr(cfg, k)!r}" for k in miss))
    if a.data_dir: cfg.data_dir = Path(a.data_dir)
    if a.out_dir: cfg.out_dir = Path(a.out_dir)
    if a.hidden: cfg.hidden = a.hidden
    if a.n_blocks: cfg.n_blocks = a.n_blocks
    if a.device: cfg.device = a.device
    if a.no_time_input: cfg.use_time_input = False
    for k in ("window_batch_budget", "anchor_batch_budget",
              "anchor_epochs", "anchor_val_cases",
              "consistency_every", "align_noise_dex", "tign_fit_mode",
              "init_noise", "init_noise_T",
              "time_align", "species_scale", "rate_loss", "rate_loss_eps",
              "rate_head", "rate_head_k", "init_noise_decades", "box",
              "y_floor", "rate_drift_tol_decades", "n_augment",
              "tign_source", "rate_space", "phys_loss", "phys_eps", "rk4_tol",
              "rk4_max_substeps", "state_clamp"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.no_phase_balance: cfg.phase_balance = False
    if a.no_initializer: cfg.use_initializer = False
    if a.legacy or a.legacy_init:
        # Everything this revision changed, put back. Kept as one flag so the
        # 'before' row of the ablation is one command, not eight.
        cfg.time_align = "ignition"
        cfg.align_noise_dex = 0.0
        cfg.species_scale = "per_channel"
        cfg.rate_loss = "absolute"
        cfg.rate_head = "linear"
        cfg.phase_balance = False
        cfg.init_noise_decades = None
        cfg.use_initializer = bool(a.legacy_init)
        cfg.tign_source = "measured"
        cfg.rate_loss_eps = 0.02
        cfg.box, cfg.box_w = 1.15, 0.08
        cfg.n_time_fourier = 8
    if (getattr(cfg, "rate_space", "logtime") == "physical"
            and getattr(cfg, "phys_loss", "logtime") == "q"):
        # the target is already log-compressed; a second symlog would distort it
        cfg.rate_loss = "absolute"
    if a.cond_p_only:
        cfg.cond_use_T0 = cfg.cond_use_phi = cfg.cond_use_steam = False
    if a.no_tf32: cfg.allow_tf32 = False
    if a.epochs_scale != 1.0:
        f = a.epochs_scale
        cfg.stages = tuple((wl, st, max(1, int(ep * f)), lr)
                           for wl, st, ep, lr in cfg.stages)
        cfg.pretrain_epochs = max(1, int(cfg.pretrain_epochs * f))
        cfg.anchor_epochs = max(1, int(cfg.anchor_epochs * f))
    if a.lr_scale != 1.0:
        f = a.lr_scale
        cfg.stages = tuple((wl, st, ep, lr * f) for wl, st, ep, lr in cfg.stages)
        cfg.anchor_lr = cfg.anchor_lr * f
    if a.pretrain_epochs is not None:
        cfg.pretrain_epochs = a.pretrain_epochs

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    logf = open(Path(cfg.out_dir) / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(str(msg) + "\n"); logf.flush()

    if getattr(cfg, "allow_tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    ds, model, device = build(cfg)
    if a.compile:
        # Compile the bound methods, not the modules: the state dict keys stay
        # exactly as they are, so checkpoints remain interchangeable with
        # uncompiled runs, the GUI and chemnode_deploy.py. Compilation is lazy,
        # so it is exercised once here and undone if it fails.
        orig = (model.ode.net.forward, model.ode.forward)
        try:
            model.ode.net.forward = torch.compile(orig[0])
            model.ode.forward = torch.compile(orig[1])
            d = model.ode.rate_scale.shape[0]
            zt = torch.zeros(4, d, device=device)
            ct_ = torch.zeros(4, ds.d_cond, device=device)
            model.ode.set_cond(ct_)
            model.ode(torch.tensor(0.5, device=device), zt).sum().backward()
            model.zero_grad(set_to_none=True)
            print("torch.compile: ON (first epochs include compilation)")
        except Exception as e:                                 # noqa: BLE001
            model.ode.net.forward, model.ode.forward = orig
            model.zero_grad(set_to_none=True)
            print(f"torch.compile unavailable ({type(e).__name__}: "
                  f"{str(e).splitlines()[0][:120]}); continuing uncompiled")
    cfg.save(Path(cfg.out_dir) / "config.json")
    log(f"device: {device}")
    log(ds.summary())
    log("split membership (case indices, as reported by chemnode_eval.py):")
    log(ds.split_report())
    log(f"parameters: {sum(q.numel() for q in model.parameters())/1e6:.2f} M")

    pretrain(model, ds, cfg, device, log)
    # ---- continuous-rollout line after each stage, cached across restarts --
    # A stage loaded from its checkpoint has not changed, so its rollout number
    # has not either: re-running a 550-point rollout over every val case for it
    # on every restart is pure cost. The number is taken from, in order:
    #   1. rollout_cache.json, if the entry matches the checkpoint's mtime;
    #   2. train.log, for runs made before the cache existed;
    #   3. a fresh computation -- always, if the stage was trained in THIS run.
    out = Path(cfg.out_dir)
    cache_p = out / "rollout_cache.json"
    t_start = time.time()
    try:
        cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    except Exception:                                          # noqa: BLE001
        cache = {}

    def from_log():
        """{tag: rollout message} from an existing train.log (last wins)."""
        found, cur = {}, None
        lp = out / "train.log"
        if not lp.exists():
            return found
        for line in lp.read_text(errors="ignore").splitlines():
            if line.startswith("=== ") and ":" in line:
                cur = line[4:].split(":")[0].strip()
            elif line.startswith("[") and "already complete" in line:
                cur = line[1:line.index("]")]
            elif "after pretraining -- val continuous rollout" in line:
                found["pretrain"] = line.split(" -- ", 1)[1]
            elif line.lstrip().startswith("-> val continuous rollout") and cur:
                found[cur] = line.strip()[3:]
        for k, v in found.items():                  # drop earlier suffixes
            found[k] = v.replace("   (cached)", "").replace(
                "   (from train.log)", "")
        return found
    logged = from_log()

    def rollout_line(prefix, tag, ckpt):
        ckpt = Path(ckpt)
        mt = ckpt.stat().st_mtime if ckpt.exists() else None
        fresh = mt is None or mt >= t_start - 1.0     # trained in this run
        if not fresh:
            e = cache.get(tag)
            if e and abs(e.get("mtime", -1) - mt) < 1e-3:
                log(f"{prefix} {e['msg']}   (cached)")
                return
            if tag in logged:
                log(f"{prefix} {logged[tag]}   (from train.log)")
                return
        ex = {}
        tm, lm, ym = full_rollout_metrics(model, ds, ds.val_idx, device, cfg,
                                          extra=ex)
        msg = (f"val continuous rollout: T RMSE {tm:.2f} K, "
               f"mean |dlog10 Y| {lm:.3f}, max log10 Y {ym:+.2f} dex | "
               f"|dlog10 Y|(Y>1e-6) {ex['logY_1e6']:.3f}, "
               f"|dY| majors {ex['absY_major_pct']:.3f} mass-%")
        if ym > 20.0:
            msg += "   <-- DIVERGED (the field left the box; see the log note)"
        elif ym > 0.0:
            msg += f"   <-- UNPHYSICAL (Y up to {10.0 ** min(ym, 30.0):.3g})"
        log(f"{prefix} {msg}")
        if mt is not None or ckpt.exists():
            cache[tag] = {"mtime": ckpt.stat().st_mtime, "msg": msg}
            cache_p.write_text(json.dumps(cache, indent=1))

    rollout_line("after pretraining --", "pretrain", out / "pretrain.pt")

    for i, (wl, st, ep, lr) in enumerate(cfg.stages):
        train_stage(model, ds, cfg, device, log, i, wl, st, ep, lr)
        tag = f"stage{i}_L{wl}"
        rollout_line("  ->", tag, out / tag / "done.pt")

    if cfg.anchor_epochs > 0:
        train_anchor(model, ds, cfg, device, log)
        rollout_line("  ->", "anchor", out / "anchor" / "done.pt")

    torch.save({"model": model.state_dict(), "rate_scale": ds.rate_scale},
               Path(cfg.out_dir) / "final.pt")
    log(f"saved {Path(cfg.out_dir)/'final.pt'}")

    # ---- the two numbers to report side by side ---------------------------
    # Cached like the stage lines: if the last stage was loaded rather than
    # trained in this run, the model is unchanged and so are these numbers.
    last = (out / "anchor" / "done.pt" if cfg.anchor_epochs > 0 else
            out / f"stage{len(cfg.stages) - 1}_L{cfg.stages[-1][0]}" / "done.pt")
    mt = last.stat().st_mtime if last.exists() else None
    e = cache.get("final")
    if (mt is not None and mt < t_start - 1.0 and e
            and abs(e.get("mtime", -1) - mt) < 1e-3):
        for line in e["lines"]:
            log(line)
        log("(final numbers cached -- the model has not changed since they "
            "were computed)")
        return
    lines = []
    for name, idx in (("val", ds.val_idx), ("test", ds.test_idx)):
        ex = {}
        tm, lm, ym = full_rollout_metrics(model, ds, idx, device, cfg, extra=ex)
        dp = deployed_metrics(model, ds, cfg, device, idx)
        lines.append(f"\n{name}: with the MEASURED delay   T RMSE {tm:7.2f} K   "
                     f"mean |dlog10 Y| {lm:.3f}   max log10 Y {ym:+.2f} dex")
        lines.append(f"{name}: with the PREDICTED delay  T RMSE "
                     f"{dp['T_rmse']:7.2f} K   "
                     f"mean |dlog10 Y| {dp['logY_mae']:.3f}   "
                     f"max log10 Y {dp['logY_max']:+.2f} dex"
                     f"   t_ign err {dp['t_ign_dex']:.3f} dex")
        lines.append(f"{name}: floor-free species error  |dlog10 Y| where "
                     f"Y > 1e-6: {ex['logY_1e6']:.3f} dec   |dY| majors (max "
                     f"Y > 1e-3): {ex['absY_major_pct']:.3f} mass-%   |dY| "
                     f"all: {ex['absY_pct']:.4f} mass-%")
        log(lines[-3])
        log(lines[-2])
        log(lines[-1])
    if ds.time_align == "ignition":
        lines.append("the second line is the deployed number: at a new "
                     "operating point the delay has to\nbe predicted. Report "
                     "it, not the first line.")
    else:
        lines.append("the coordinate needs no delay, so the two lines are the "
                     "same measurement.")
    log(lines[-1])
    if mt is not None:
        cache["final"] = {"mtime": mt, "lines": lines}
        cache_p.write_text(json.dumps(cache, indent=1))


if __name__ == "__main__":
    main()
