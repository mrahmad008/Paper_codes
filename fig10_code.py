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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import gammaln, logsumexp
from scipy.stats import kstest

# ============================================================
# MATPLOTLIB CONFIGURATION (IEEE / PUBLICATION BOLD & VECTOR PDF)
# ============================================================
mpl.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

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

PT_DBM = 20.0
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

# ============================================================
# 2. FIGURE STYLE & LAYOUT
# ============================================================

FIGSIZE = (7.2, 5.2)

LABEL_FONT = 10
TICK_FONT = 9
LEGEND_FONT = 8
PANEL_FONT = 11

GRID_STYLE = "--"
GRID_WIDTH = 0.6
GRID_ALPHA = 0.35

LINE_WIDTH = 1.8
ROC_LINE_WIDTH = 1.8
MARKER_SIZE = 4.5

DIRECT_COLOR = "blue"
DIRECT_MARKER = "s"

INDIRECT_COLOR = "red"
INDIRECT_MARKER = "^"

RANDOM_COLOR = "black"

# ============================================================
# 3. OUTPUT PATHS
# ============================================================

if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()

OUT = BASE_DIR / "detector_results_middle_labels"
OUT.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUT / "detector_performance_M128_2x2_clean.png"
PDF_PATH = OUT / "detector_performance_M128_2x2_clean.pdf"

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

u1_train = generate_u1(N_TRAIN, TRAIN_SEED, TRAIN_BATCH_SIZE)
MEAN_U1, VAR_U1, ALPHA_G, THETA_G = fit_gamma_moments(u1_train)

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
        - u_hat[positive] / THETA_G
        + u_hat[positive]
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
# 10. SIMULATION EVALUATION
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

    rows.append({
        "N_s": Ns_int,
        "P_FA_Direct": pfa_d,
        "P_MD_Direct": pmd_d,
        "BER_Direct": ber_d,
        "P_FA_Indirect": pfa_i,
        "P_MD_Indirect": pmd_i,
        "BER_Indirect": ber_i,
    })

ns_df = pd.DataFrame(rows)

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

    return p_fa, p_d

roc_pfa_direct, roc_pd_direct = empirical_roc(direct_h0, direct_h1)
roc_pfa_indirect, roc_pd_indirect = empirical_roc(indirect_h0, indirect_h1)

# ============================================================
# 11. GENERATE FINAL PLOT WITH CENTERED SUB-PANEL TAGS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.3))
fig.align_ylabels()
axes = axes.ravel()

def style_axis_clean(ax, panel_label):
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # Place panel label in the middle below x-axis (ha="center")
    ax.text(
        -0.1,
        -0.1,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold"
    )

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

# Panel (a)
ax = axes[0]
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

ax.set_xlabel(r"Probability of False Alarm ($P_{\mathrm{FA}}$)", fontweight="bold", labelpad=3)
ax.set_ylabel(r"Probability of Detection ($P_{\mathrm{D}}$)", fontweight="bold")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
style_axis_clean(ax, "(a)")
ax.legend(
    handles=[direct_handle, indirect_handle, random_handle],
    loc="lower right",
    frameon=True,
    edgecolor="black",
    fontsize= 10
)

# Panel (b)
ax = axes[1]
ax.plot(ns_df["N_s"], ns_df["BER_Direct"], color=DIRECT_COLOR, marker=DIRECT_MARKER, linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
ax.plot(ns_df["N_s"], ns_df["BER_Indirect"], color=INDIRECT_COLOR, marker=INDIRECT_MARKER, linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
ax.axhline(0.5, color=RANDOM_COLOR, linestyle=":", linewidth=1.1)
ax.set_xlabel(r"Number of Observations $N_s$", fontweight="bold", labelpad=3)
ax.set_ylabel("Bit Error Rate (BER)", fontweight="bold")
ax.set_xlim(NS_VALUES.min(), NS_VALUES.max())
style_axis_clean(ax, "(b)")
ax.legend(
    handles=[direct_handle, indirect_handle, random_handle],
    loc="best",
    frameon=True,
    edgecolor="black",
    fontsize=10
)

# Panel (c)
ax = axes[2]
ax.plot(ns_df["N_s"], ns_df["P_FA_Direct"], color=DIRECT_COLOR, marker=DIRECT_MARKER, linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
ax.plot(ns_df["N_s"], ns_df["P_FA_Indirect"], color=INDIRECT_COLOR, marker=INDIRECT_MARKER, linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
ax.set_xlabel(r"Number of Observations $N_s$", fontweight="bold", labelpad=3)
ax.set_ylabel(r"Probability of False Alarm ($P_{\mathrm{FA}}$)", fontweight="bold")
ax.set_xlim(NS_VALUES.min(), NS_VALUES.max())
style_axis_clean(ax, "(c)")
ax.legend(
    handles=[direct_handle, indirect_handle],
    loc="lower right",
    frameon=True,
    edgecolor="black",
    fontsize=10
)

# Panel (d)
ax = axes[3]
ax.plot(ns_df["N_s"], ns_df["P_MD_Direct"], color=DIRECT_COLOR, marker=DIRECT_MARKER, linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
ax.plot(ns_df["N_s"], ns_df["P_MD_Indirect"], color=INDIRECT_COLOR, marker=INDIRECT_MARKER, linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
ax.set_xlabel(r"Number of Observations $N_s$", fontweight="bold", labelpad=3)
ax.set_ylabel(r"Probability of Miss Detection ($P_{\mathrm{MD}}$)", fontweight="bold")
ax.set_xlim(NS_VALUES.min(), NS_VALUES.max())
style_axis_clean(ax, "(d)")
ax.legend(
    handles=[direct_handle, indirect_handle],
    loc="upper right",
    frameon=True,
    edgecolor="black",
    fontsize=10
)

# Subplot margins and layout tuning to completely prevent label overlaps
fig.subplots_adjust(
    left=0.09, right=0.98, top=0.98, bottom=0.10,
    wspace=0.3, hspace=0.42
)

fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_PATH, format="pdf", bbox_inches="tight", facecolor="white")

plt.show()
