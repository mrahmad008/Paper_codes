# =============================================================================
# IRS-Assisted Backscatter Wireless Communication Simulation
#
# This script reproduces part of the numerical results reported in Section 5
# of the manuscript. The system consists of a transmitter, a conventional
# backscatter tag (Tag 1), an IRS-assisted backscatter tag (Tag 2), and a
# receiver.
#
# The simulations follow the 3GPP Indoor Hotspot (InH) propagation setting
# considered in the manuscript, with a carrier frequency of 2.4 GHz.
# Unless otherwise stated, the main system parameters follow Table 2.
# Parameters that are varied for a particular experiment are defined
# separately in the corresponding section of this script.
#
# Random channel realizations are generated independently for Monte Carlo
# evaluation. The generated figures correspond to those reported in the
# revised manuscript.
# ============================================================================
# Noncoherent detector performance - Fig. 10
#
# This script evaluates the direct LRT-based detector and the indirect
# plug-in energy-based detector for M = 128 IRS elements.
#
# The direct detector evaluates the marginalized likelihood of the received
# energy. The indirect detector first estimates the channel energy and then
# evaluates the corresponding plug-in density ratio.
#
# The simulation reports the ROC, BER, probability of false alarm, and
# probability of miss detection. The influence of the number of observations
# Ns is also evaluated.
# -----------------------------------------------------------------------------

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import gammaln, logsumexp
from scipy.stats import kstest

# ============================================================
# DETECTOR PERFORMANCE — M = 128 (UPDATED MARKER SCHEME)
# ============================================================

# ============================================================
# 1. USER-EDITABLE SETTINGS
# ============================================================

M = 128
ETA = 0.8
BETA = 0.0
SIGMA_S2 = 1.0

FC_GHZ = 2.4

D_SR = 20.0
D_ST = 10.0
D_TR = 10.0
D_SI = 8.0
D_IR = 12.0

PT_DBM = 10.0
NOISE_POWER_DBM = -95.0

NS_VALUES = np.arange(5, 41, 5, dtype=int)
MAX_NS = int(NS_VALUES.max())
ROC_NS = 20

SBF_ITERATIONS = 100
SBF_EPS_LOW = -np.pi / 20.0
SBF_EPS_HIGH = np.pi / 20.0

N_TRAIN = 50_000
N_TEST = 500_000

TRAIN_BATCH_SIZE = 1_000
TEST_BATCH_SIZE = 1_000

TRAIN_SEED = 2042
TEST_U1_SEED = 7042
TEST_U0_SEED = 8042
OBS_SEED = 9042

LOG_U_MIN = -20.0
LOG_U_MAX = 8.0
LOG_U_POINTS = 2400

T_GRID_POINTS = 2400
T_GRID_MAX = 9.0
LUT_CHUNK = 200

CONF_Z = 1.959963984540054  # 95%

# ============================================================
# 2. FIGURE STYLE (UPDATED STYLING)
# ============================================================

FIGSIZE = (11.0, 8.2)

LABEL_FONT = 11
TICK_FONT = 9
LEGEND_FONT = 9
PANEL_FONT = 11

GRID_STYLE = "--"
GRID_WIDTH = 0.55
GRID_ALPHA = 0.35

LINE_WIDTH = 1.4
ROC_LINE_WIDTH = 1.4
MARKER_SIZE = 5.0

# Direct: Blue Square (Solid Line)
DIRECT_COLOR = "blue"
DIRECT_MARKER = "s"
DIRECT_LINESTYLE = "-"

# Indirect: Red Triangle (Dashed Line)
INDIRECT_COLOR = "red"
INDIRECT_MARKER = "^"
INDIRECT_LINESTYLE = "-"

# Benchmark: Black Dotted
RANDOM_COLOR = "black"

PANEL_Y = -0.24

# ============================================================
# 3. OUTPUT
# ============================================================

if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()

OUT = BASE_DIR / "detector_performance_M128_2x2_updated"
OUT.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUT / "detector_performance_M128_2x2_updated.png"
PDF_PATH = OUT / "detector_performance_M128_2x2_updated.pdf"
NS_CSV_PATH = OUT / "detector_metrics_vs_Ns_M128.csv"
ROC_CSV_PATH = OUT / "detector_ROC_M128.csv"
META_PATH = OUT / "detector_performance_M128_2x2_metadata.json"

# ============================================================
# 4. 3GPP InH PATH LOSS
# ============================================================

def pl_los(distance_m, frequency_GHz):
    return 32.4 + 17.3 * np.log10(distance_m) + 20.0 * np.log10(frequency_GHz)

def pl_nlos(distance_m, frequency_GHz):
    los = pl_los(distance_m, frequency_GHz)
    candidate = 17.30 + 38.3 * np.log10(distance_m) + 24.9 * np.log10(frequency_GHz)
    return max(los, candidate)

PL_SR = pl_nlos(D_SR, FC_GHZ)
PL_ST = pl_nlos(D_ST, FC_GHZ)
PL_TR = pl_nlos(D_TR, FC_GHZ)
PL_SI = pl_los(D_SI, FC_GHZ)
PL_IR = pl_los(D_IR, FC_GHZ)

VAR_HSR = 10.0 ** (-PL_SR / 10.0)
VAR_HST = 10.0 ** (-PL_ST / 10.0)
VAR_HTR = 10.0 ** (-PL_TR / 10.0)
VAR_HSI = 10.0 ** (-PL_SI / 10.0)
VAR_HIR = 10.0 ** (-PL_IR / 10.0)

U_REF = VAR_HSR
U0_MEAN = 1.0

RHO_DB = PT_DBM - NOISE_POWER_DBM
SIGMA_W2 = (10.0 ** (-RHO_DB / 10.0)) / U_REF

# ============================================================
# 5. COMPLEX GAUSSIAN GENERATOR
# ============================================================

def cn(rng, shape, variance):
    return np.sqrt(variance / 2.0) * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )

# ============================================================
# 6. POST-SBF U1 GENERATOR
# ============================================================

def generate_u1_batch(rng, n):
    h_sr = cn(rng, n, VAR_HSR)
    h_st = cn(rng, n, VAR_HST)
    h_tr = cn(rng, n, VAR_HTR)
    h_si = cn(rng, (n, M), VAR_HSI)
    h_ir = cn(rng, (n, M), VAR_HIR)

    direct_non_irs = h_sr + h_st * np.exp(1j * BETA) * h_tr
    cascaded = h_ir * h_si

    phi = rng.uniform(-np.pi, np.pi, size=(n, M))
    phase_state = np.exp(1j * phi)

    G = direct_non_irs + ETA * np.sum(cascaded * phase_state, axis=1)
    best_power = np.abs(G) ** 2

    for _ in range(SBF_ITERATIONS):
        epsilon = rng.uniform(SBF_EPS_LOW, SBF_EPS_HIGH, size=(n, M))
        trial_state = phase_state * np.exp(1j * epsilon)
        G_trial = direct_non_irs + ETA * np.sum(cascaded * trial_state, axis=1)
        trial_power = np.abs(G_trial) ** 2
        improve = trial_power > best_power
        best_power[improve] = trial_power[improve]
        phase_state[improve] = trial_state[improve]

    return best_power / U_REF

def generate_u1(total_samples, seed, batch_size):
    rng = np.random.default_rng(seed)
    parts = []
    completed = 0
    while completed < total_samples:
        n = min(batch_size, total_samples - completed)
        parts.append(generate_u1_batch(rng, n))
        completed += n
    return np.concatenate(parts)

# ============================================================
# 7. GAMMA FIT
# ============================================================

def fit_gamma_moments(samples):
    mu = float(np.mean(samples))
    var = float(np.var(samples, ddof=1))
    alpha_g = mu**2 / var
    theta_g = var / mu
    return mu, var, alpha_g, theta_g

print(f"Generating {N_TRAIN:,} training realizations...")

u1_train = generate_u1(N_TRAIN, TRAIN_SEED, TRAIN_BATCH_SIZE)
MEAN_U1, VAR_U1, ALPHA_G, THETA_G = fit_gamma_moments(u1_train)

KS_D, KS_P = kstest(u1_train, "gamma", args=(ALPHA_G, 0.0, THETA_G))

# ============================================================
# 8. DIRECT-DETECTOR PRIOR GRID
# ============================================================

Q_GRID = np.linspace(LOG_U_MIN, LOG_U_MAX, LOG_U_POINTS)
DQ = Q_GRID[1] - Q_GRID[0]
U_GRID = np.exp(Q_GRID)

LOGW = np.full(LOG_U_POINTS, np.log(DQ))
LOGW[0] += np.log(0.5)
LOGW[-1] += np.log(0.5)

LOGPRIOR0 = -U_GRID + Q_GRID + LOGW
LOGPRIOR1 = (
    (ALPHA_G - 1.0) * Q_GRID
    - U_GRID / THETA_G
    - gammaln(ALPHA_G)
    - ALPHA_G * np.log(THETA_G)
    + Q_GRID
    + LOGW
)

T_GRID = np.linspace(0.0, T_GRID_MAX, T_GRID_POINTS)
Z_GRID = np.expm1(T_GRID)

CONDITIONAL_VARIANCE = U_GRID * SIGMA_S2 + SIGMA_W2

DIRECT_LUT = {}

print("Precomputing Direct-detector lookup tables...")

for Ns in NS_VALUES:
    Ns_int = int(Ns)
    common = -Ns_int * np.log(np.pi * CONDITIONAL_VARIANCE)
    logp0_parts = []
    logp1_parts = []

    for start in range(0, T_GRID_POINTS, LUT_CHUNK):
        stop = min(start + LUT_CHUNK, T_GRID_POINTS)
        zz = Z_GRID[start:stop, None]
        exponent = -zz / CONDITIONAL_VARIANCE[None, :]

        logp0_parts.append(
            logsumexp((LOGPRIOR0 + common)[None, :] + exponent, axis=1)
        )
        logp1_parts.append(
            logsumexp((LOGPRIOR1 + common)[None, :] + exponent, axis=1)
        )

    DIRECT_LUT[Ns_int] = np.concatenate(logp1_parts) - np.concatenate(logp0_parts)

def direct_score_from_lut(z, Ns):
    t = np.log1p(np.maximum(z, 0.0))
    if np.max(t) > T_GRID_MAX:
        raise RuntimeError(
            f"Observed log-energy {np.max(t):.4f} exceeds T_GRID_MAX={T_GRID_MAX}."
        )
    return np.interp(t, T_GRID, DIRECT_LUT[int(Ns)])

# ============================================================
# 9. INDIRECT DETECTOR
# ============================================================

def indirect_score(z, Ns):
    u_hat = np.maximum((z / Ns - SIGMA_W2) / SIGMA_S2, 0.0)
    score = np.empty_like(u_hat)
    positive = u_hat > 0.0
    constant = -gammaln(ALPHA_G) - ALPHA_G * np.log(THETA_G)

    score[positive] = (
        (ALPHA_G - 1.0) * np.log(u_hat[positive])
        + u_hat[positive] * (1.0 - 1.0 / THETA_G)
        + constant
    )

    if ALPHA_G > 1.0:
        score[~positive] = -1e12
    elif np.isclose(ALPHA_G, 1.0):
        score[~positive] = -np.log(THETA_G)
    else:
        score[~positive] = 1e12

    return score

# ============================================================
# 10. CONFIDENCE INTERVALS
# ============================================================

def wilson_interval(successes, n, z=CONF_Z):
    p = successes / n
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denominator
    half_width = z / denominator * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2))
    return max(0.0, center - half_width), min(1.0, center + half_width)

def ber_interval(fp_count, fn_count, n, z=CONF_Z):
    pfa = fp_count / n
    pmd = fn_count / n
    ber = 0.5 * (pfa + pmd)
    variance = (pfa * (1.0 - pfa) / n + pmd * (1.0 - pmd) / n) / 4.0
    half_width = z * np.sqrt(variance)
    return max(0.0, ber - half_width), min(1.0, ber + half_width)

# ============================================================
# 11. TEST SIMULATION
# ============================================================

metric_counts = {
    int(Ns): {
        "fp_direct": 0,
        "fn_direct": 0,
        "fp_indirect": 0,
        "fn_indirect": 0
    }
    for Ns in NS_VALUES
}

roc_direct_h0_parts = []
roc_direct_h1_parts = []
roc_indirect_h0_parts = []
roc_indirect_h1_parts = []

rng_u1 = np.random.default_rng(TEST_U1_SEED)
rng_u0 = np.random.default_rng(TEST_U0_SEED)
rng_obs = np.random.default_rng(OBS_SEED)

completed = 0
print(f"Running {N_TEST:,} test realizations per hypothesis...")

while completed < N_TEST:
    n = min(TEST_BATCH_SIZE, N_TEST - completed)

    u1 = generate_u1_batch(rng_u1, n)
    u0 = rng_u0.exponential(scale=U0_MEAN, size=n)

    exp_h0 = rng_obs.exponential(scale=1.0, size=(n, MAX_NS))
    exp_h1 = rng_obs.exponential(scale=1.0, size=(n, MAX_NS))

    sample_energy_h0 = (u0[:, None] + SIGMA_W2) * exp_h0
    sample_energy_h1 = (u1[:, None] + SIGMA_W2) * exp_h1

    cumulative_z0 = np.cumsum(sample_energy_h0, axis=1)
    cumulative_z1 = np.cumsum(sample_energy_h1, axis=1)

    for Ns in NS_VALUES:
        Ns_int = int(Ns)
        z0 = cumulative_z0[:, Ns_int - 1]
        z1 = cumulative_z1[:, Ns_int - 1]

        d0 = direct_score_from_lut(z0, Ns_int)
        d1 = direct_score_from_lut(z1, Ns_int)
        i0 = indirect_score(z0, Ns_int)
        i1 = indirect_score(z1, Ns_int)

        metric_counts[Ns_int]["fp_direct"] += int(np.count_nonzero(d0 > 0.0))
        metric_counts[Ns_int]["fn_direct"] += int(np.count_nonzero(d1 <= 0.0))
        metric_counts[Ns_int]["fp_indirect"] += int(np.count_nonzero(i0 > 0.0))
        metric_counts[Ns_int]["fn_indirect"] += int(np.count_nonzero(i1 <= 0.0))

        if Ns_int == ROC_NS:
            roc_direct_h0_parts.append(d0.copy())
            roc_direct_h1_parts.append(d1.copy())
            roc_indirect_h0_parts.append(i0.copy())
            roc_indirect_h1_parts.append(i1.copy())

    completed += n
    if completed % 50_000 == 0 or completed == N_TEST:
        print(f"Completed {completed:,}/{N_TEST:,}")

# ============================================================
# 12. BUILD RESULTS TABLE
# ============================================================

rows = []

for Ns in NS_VALUES:
    Ns_int = int(Ns)
    c = metric_counts[Ns_int]

    fp_d = c["fp_direct"]
    fn_d = c["fn_direct"]
    fp_i = c["fp_indirect"]
    fn_i = c["fn_indirect"]

    pfa_d = fp_d / N_TEST
    pmd_d = fn_d / N_TEST
    ber_d = 0.5 * (pfa_d + pmd_d)

    pfa_i = fp_i / N_TEST
    pmd_i = fn_i / N_TEST
    ber_i = 0.5 * (pfa_i + pmd_i)

    pfa_d_lo, pfa_d_hi = wilson_interval(fp_d, N_TEST)
    pmd_d_lo, pmd_d_hi = wilson_interval(fn_d, N_TEST)
    ber_d_lo, ber_d_hi = ber_interval(fp_d, fn_d, N_TEST)

    pfa_i_lo, pfa_i_hi = wilson_interval(fp_i, N_TEST)
    pmd_i_lo, pmd_i_hi = wilson_interval(fn_i, N_TEST)
    ber_i_lo, ber_i_hi = ber_interval(fp_i, fn_i, N_TEST)

    rows.append({
        "N_s": Ns_int,
        "P_FA_Direct": pfa_d,
        "P_FA_Direct_CI95_low": pfa_d_lo,
        "P_FA_Direct_CI95_high": pfa_d_hi,
        "P_MD_Direct": pmd_d,
        "P_MD_Direct_CI95_low": pmd_d_lo,
        "P_MD_Direct_CI95_high": pmd_d_hi,
        "BER_Direct": ber_d,
        "BER_Direct_CI95_low": ber_d_lo,
        "BER_Direct_CI95_high": ber_d_hi,
        "P_FA_Indirect": pfa_i,
        "P_FA_Indirect_CI95_low": pfa_i_lo,
        "P_FA_Indirect_CI95_high": pfa_i_hi,
        "P_MD_Indirect": pmd_i,
        "P_MD_Indirect_CI95_low": pmd_i_lo,
        "P_MD_Indirect_CI95_high": pmd_i_hi,
        "BER_Indirect": ber_i,
        "BER_Indirect_CI95_low": ber_i_lo,
        "BER_Indirect_CI95_high": ber_i_hi
    })

ns_df = pd.DataFrame(rows)
ns_df.to_csv(NS_CSV_PATH, index=False)

# ============================================================
# 13. ROC / AUC
# ============================================================

direct_h0 = np.concatenate(roc_direct_h0_parts)
direct_h1 = np.concatenate(roc_direct_h1_parts)
indirect_h0 = np.concatenate(roc_indirect_h0_parts)
indirect_h1 = np.concatenate(roc_indirect_h1_parts)

def empirical_roc(h0_scores, h1_scores):
    scores = np.concatenate([h0_scores, h1_scores])
    labels = np.concatenate([
        np.zeros(h0_scores.size, dtype=np.int8),
        np.ones(h1_scores.size, dtype=np.int8)
    ])

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)

    change = np.where(np.diff(sorted_scores) != 0)[0]
    idx = np.r_[change, sorted_scores.size - 1]

    p_d = tp[idx] / h1_scores.size
    p_fa = fp[idx] / h0_scores.size

    p_d = np.r_[0.0, p_d]
    p_fa = np.r_[0.0, p_fa]

    auc = float(np.trapezoid(p_d, p_fa))
    return p_fa, p_d, auc

roc_pfa_direct, roc_pd_direct, auc_direct = empirical_roc(direct_h0, direct_h1)
roc_pfa_indirect, roc_pd_indirect, auc_indirect = empirical_roc(indirect_h0, indirect_h1)

roc_df = pd.DataFrame({
    "P_FA_Direct": roc_pfa_direct,
    "P_D_Direct": roc_pd_direct,
    "P_FA_Indirect": np.nan,
    "P_D_Indirect": np.nan
})
roc_df.loc[:len(roc_pfa_indirect)-1, "P_FA_Indirect"] = roc_pfa_indirect
roc_df.loc[:len(roc_pd_indirect)-1, "P_D_Indirect"] = roc_pd_indirect
roc_df.to_csv(ROC_CSV_PATH, index=False)

# ============================================================
# 14. PLOTTING HELPERS
# ============================================================

def asymmetric_yerr(values, lows, highs):
    values = np.asarray(values)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    return np.vstack([values - lows, highs - values])

def style_axis(ax):
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)
    ax.tick_params(axis="both", labelsize=TICK_FONT)

# ============================================================
# 15. CREATE FINAL 2x2 FIGURE
#     Direct   = blue square
#     Indirect = red triangle
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
axes = axes.ravel()

# Clean legend handles: exactly one marker per detector.
direct_handle = Line2D(
    [0], [0],
    color=DIRECT_COLOR,
    marker=DIRECT_MARKER,
    linestyle="-",
    linewidth=LINE_WIDTH,
    markersize=MARKER_SIZE,
    label="Direct"
)

indirect_handle = Line2D(
    [0], [0],
    color=INDIRECT_COLOR,
    marker=INDIRECT_MARKER,
    linestyle="-",
    linewidth=LINE_WIDTH,
    markersize=MARKER_SIZE,
    label="Indirect"
)

random_handle = Line2D(
    [0], [0],
    color=RANDOM_COLOR,
    linestyle=":",
    linewidth=1.1,
    label="Random"
)

# (a) ROC
ax = axes[0]

# The ROC curves may be very close or overlap.
# Use different marker positions so square and triangle are not drawn
# on top of each other while keeping the actual curves unchanged.
roc_step_direct = max(2, len(roc_pfa_direct) // 12)
roc_step_indirect = max(2, len(roc_pfa_indirect) // 12)

ax.plot(
    roc_pfa_direct, roc_pd_direct,
    color=DIRECT_COLOR,
    marker=DIRECT_MARKER,
    markevery=(0, roc_step_direct),
    linestyle="-",
    linewidth=ROC_LINE_WIDTH,
    markersize=MARKER_SIZE
)

ax.plot(
    roc_pfa_indirect, roc_pd_indirect,
    color=INDIRECT_COLOR,
    marker=INDIRECT_MARKER,
    markevery=(max(1, roc_step_indirect // 2), roc_step_indirect),
    linestyle="-",
    linewidth=ROC_LINE_WIDTH,
    markersize=MARKER_SIZE
)

ax.plot(
    [0.0, 1.0], [0.0, 1.0],
    color=RANDOM_COLOR,
    linestyle=":",
    linewidth=1.1
)

ax.set_xlabel(r"Probability of False Alarm (PFA)", fontsize=LABEL_FONT)
ax.set_ylabel(r"Probability of Detection (PD)", fontsize=LABEL_FONT)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
style_axis(ax)

roc_direct_handle = Line2D(
    [0], [0],
    color=DIRECT_COLOR,
    marker=DIRECT_MARKER,
    linestyle="-",
    linewidth=ROC_LINE_WIDTH,
    markersize=MARKER_SIZE,
    label=rf"Direct"
)

roc_indirect_handle = Line2D(
    [0], [0],
    color=INDIRECT_COLOR,
    marker=INDIRECT_MARKER,
    linestyle="-",
    linewidth=ROC_LINE_WIDTH,
    markersize=MARKER_SIZE,
    label=rf"Indirect"
)

ax.legend(
    handles=[roc_direct_handle, roc_indirect_handle, random_handle],
    loc="lower right",
    frameon=True,
    edgecolor="black",
    fontsize=LEGEND_FONT
)

# Helper for panels (b)-(d):
# draw CI bars separately, then draw one clean line+marker curve.
def plot_metric_with_ci(
    ax,
    x,
    y_direct,
    lo_direct,
    hi_direct,
    y_indirect,
    lo_indirect,
    hi_indirect,
):
    # CI only — no marker and no legend entry.
    ax.errorbar(
        x, y_direct,
        yerr=asymmetric_yerr(y_direct, lo_direct, hi_direct),
        fmt="none",
        ecolor=DIRECT_COLOR,
        elinewidth=0.9,
        capsize=2.5,
        alpha=0.75,
        label="_nolegend_",
        zorder=1
    )

    ax.errorbar(
        x, y_indirect,
        yerr=asymmetric_yerr(y_indirect, lo_indirect, hi_indirect),
        fmt="none",
        ecolor=INDIRECT_COLOR,
        elinewidth=0.9,
        capsize=2.5,
        alpha=0.75,
        label="_nolegend_",
        zorder=1
    )

    # Main curves — exactly one marker style for each detector.
    ax.plot(
        x, y_direct,
        color=DIRECT_COLOR,
        marker=DIRECT_MARKER,
        linestyle="-",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        zorder=3
    )

    ax.plot(
        x, y_indirect,
        color=INDIRECT_COLOR,
        marker=INDIRECT_MARKER,
        linestyle="-",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        zorder=4
    )


def plot_metric_curves_only(
    ax,
    x,
    y_direct,
    y_indirect,
):
    # Panels (c) and (d): curves only, no CI shading and no error
    # bars — the earlier translucent fill_between band read as a
    # "projection"/shadow behind the lines, so it's removed here.
    x_arr = np.asarray(x, dtype=float)
    yd = np.asarray(y_direct, dtype=float)
    yi = np.asarray(y_indirect, dtype=float)

    # Direct = blue square
    # Indirect = red triangle
    # Marker positions alternate so the two marker shapes never overlap.
    ax.plot(
        x_arr, yd,
        color=DIRECT_COLOR,
        marker=DIRECT_MARKER,
        markevery=(0, 2),
        linestyle="-",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        zorder=3
    )

    ax.plot(
        x_arr, yi,
        color=INDIRECT_COLOR,
        marker=INDIRECT_MARKER,
        markevery=(1, 2),
        linestyle="-",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        zorder=4
    )


# (b) BER vs N_s
ax = axes[1]
plot_metric_with_ci(
    ax,
    ns_df["N_s"],
    ns_df["BER_Direct"],
    ns_df["BER_Direct_CI95_low"],
    ns_df["BER_Direct_CI95_high"],
    ns_df["BER_Indirect"],
    ns_df["BER_Indirect_CI95_low"],
    ns_df["BER_Indirect_CI95_high"]
)
ax.axhline(0.5, color=RANDOM_COLOR, linestyle=":", linewidth=1.1)
ax.set_xlabel(r"Number of Observations $N_s$", fontsize=LABEL_FONT)
ax.set_ylabel("Bit Error Rate (BER)", fontsize=LABEL_FONT)
ax.set_xlim(NS_VALUES.min(), NS_VALUES.max())
ax.set_ylim(0.38, 0.505)
style_axis(ax)
ax.legend(
    handles=[direct_handle, indirect_handle, random_handle],
    loc="upper right",
    frameon=True,
    edgecolor="black",
    fontsize=LEGEND_FONT
)

# (c) P_FA vs N_s
ax = axes[2]
plot_metric_curves_only(
    ax,
    ns_df["N_s"],
    ns_df["P_FA_Direct"],
    ns_df["P_FA_Indirect"]
)
ax.set_xlabel(r"Number of Observations $N_s$", fontsize=LABEL_FONT)
ax.set_ylabel(r"Probability of False Alarm (PFA)", fontsize=LABEL_FONT)
ax.set_xlim(NS_VALUES.min(), NS_VALUES.max())
ax.set_ylim(0.40, 0.47)
style_axis(ax)
ax.legend(
    handles=[direct_handle, indirect_handle],
    loc="best",
    frameon=True,
    edgecolor="black",
    fontsize=LEGEND_FONT
)

# (d) P_MD vs N_s
ax = axes[3]
plot_metric_curves_only(
    ax,
    ns_df["N_s"],
    ns_df["P_MD_Direct"],
    ns_df["P_MD_Indirect"]
)
ax.set_xlabel(r"Number of Observations $N_s$", fontsize=LABEL_FONT)
ax.set_ylabel(r"Probability of Miss Detection (PMD)", fontsize=LABEL_FONT)
ax.set_xlim(NS_VALUES.min(), NS_VALUES.max())
ax.set_ylim(0.32, 0.40)
style_axis(ax)
ax.legend(
    handles=[direct_handle, indirect_handle],
    loc="best",
    frameon=True,
    edgecolor="black",
    fontsize=LEGEND_FONT
)

fig.subplots_adjust(
    left=0.08, right=0.985, top=0.97, bottom=0.12,
    wspace=0.22, hspace=0.30
)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()

LABEL_GAP_FIGFRAC = 0.012

for ax, label in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
    tight_bbox = ax.get_tightbbox(renderer)
    tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
    x_center = (tight_bbox_fig.x0 + tight_bbox_fig.x1) / 2.0
    y_bottom = tight_bbox_fig.y0
    fig.text(
        x_center,
        y_bottom - LABEL_GAP_FIGFRAC,
        label,
        ha="center", va="top",
        fontsize=PANEL_FONT
    )

fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
plt.show()

metadata = {
    "M": M,
    "eta": ETA,
    "beta": BETA,
    "Pt_dBm": PT_DBM,
    "noise_power_dBm": NOISE_POWER_DBM,
    "rho_dB": RHO_DB,
    "N_s_values": NS_VALUES.tolist(),
    "ROC_N_s": ROC_NS,
    "phase_error": "none",
    "SBF_iterations": SBF_ITERATIONS,
    "SBF_perturbation": "Uniform[-pi/20,+pi/20]",
    "physical_geometry_m": {
        "d_sr": D_SR,
        "d_st": D_ST,
        "d_tr": D_TR,
        "d_si": D_SI,
        "d_ir": D_IR
    },
    "training_realizations": N_TRAIN,
    "test_realizations_per_hypothesis": N_TEST,
    "gamma_fit": {
        "mean_U1_normalized": float(MEAN_U1),
        "variance_U1_normalized": float(VAR_U1),
        "alpha_g": float(ALPHA_G),
        "theta_g_normalized": float(THETA_G),
        "KS_D": float(KS_D),
        "KS_p_value": float(KS_P)
    },
    "ROC": {
        "AUC_Direct": float(auc_direct),
        "AUC_Indirect": float(auc_indirect)
    }
}

META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("\nSaved:")
print(PNG_PATH.resolve())
print(PDF_PATH.resolve())
print(NS_CSV_PATH.resolve())
print(ROC_CSV_PATH.resolve())
print(META_PATH.resolve())
