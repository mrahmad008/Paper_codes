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
# Detector sensitivity to parameter mismatch - Fig. 11
#
# This script evaluates the effect of imperfect detector parameters on the
# direct and indirect noncoherent detectors for M = 128 and Ns = 20.
#
# Two types of mismatch are considered:
#   1. Received-SNR estimation error, Delta_SNR
#   2. Channel-statistics mismatch, Delta_ch
#
# Both mismatch parameters are varied from -3 dB to +3 dB. The received
# observations are generated using the nominal model, while the assumed
# detector parameters are perturbed during detection.
#
# The zero-mismatch point represents perfect knowledge of the corresponding
# detector parameters.
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
# MATPLOTLIB CONFIGURATION (IEEE PUBLICATION BOLD & VECTOR PDF)
# ============================================================
mpl.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# 1. SETTINGS
# ============================================================
M = 128
ETA = 0.8
BETA = 0.0
SIGMA_S2 = 1.0
FC_GHZ = 2.4

D_SR, D_ST, D_TR = 20.0, 10.0, 10.0
D_SI, D_IR = 8.0, 12.0

PT_DBM = 10.0
NOISE_POWER_DBM = -95.0
N_S = 20

SNR_MISMATCH_DB = np.arange(-3.0, 3.01, 0.5)
CHANNEL_MISMATCH_DB = np.arange(-3.0, 3.01, 0.5)

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
CONF_Z = 1.959963984540054

# Layout & Colors
FIGSIZE = (7.2, 3.2)
DIRECT_COLOR = "blue"
INDIRECT_COLOR = "red"
REFERENCE_COLOR = "black"
DIRECT_MARKER = "s"
INDIRECT_MARKER = "^"
LINE_WIDTH = 1.8
MARKER_SIZE = 4.5
GRID_STYLE = "--"
GRID_WIDTH = 0.6
GRID_ALPHA = 0.35
PANEL_FONT = 11
LEGEND_FONT = 8

# ============================================================
# 2. OUTPUT PATHS
# ============================================================
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()

OUT = BASE_DIR / "detector_robustness_results"
OUT.mkdir(parents=True, exist_ok=True)
PNG_PATH = OUT / "BER_SNR_channel_mismatch_M128_clean.png"
PDF_PATH = OUT / "BER_SNR_channel_mismatch_M128_clean.pdf"
SNR_CSV = OUT / "BER_vs_SNR_mismatch_M128.csv"
CH_CSV = OUT / "BER_vs_channel_mismatch_M128.csv"
META_PATH = OUT / "detector_robustness_M128_metadata.json"

# ============================================================
# 3. PATH LOSS & CHANNELS
# ============================================================
def pl_los(d, fc):
    return 32.4 + 17.3 * np.log10(d) + 20.0 * np.log10(fc)

def pl_nlos(d, fc):
    return max(pl_los(d, fc), 17.30 + 38.3 * np.log10(d) + 24.9 * np.log10(fc))

PL_SR = pl_nlos(D_SR, FC_GHZ)
PL_ST = pl_nlos(D_ST, FC_GHZ)
PL_TR = pl_nlos(D_TR, FC_GHZ)
PL_SI = pl_los(D_SI, FC_GHZ)
PL_IR = pl_los(D_IR, FC_GHZ)

VAR_HSR = 10 ** (-PL_SR / 10)
VAR_HST = 10 ** (-PL_ST / 10)
VAR_HTR = 10 ** (-PL_TR / 10)
VAR_HSI = 10 ** (-PL_SI / 10)
VAR_HIR = 10 ** (-PL_IR / 10)

U_REF = VAR_HSR
TRUE_U0_MEAN = 1.0
TRUE_RHO_DB = PT_DBM - NOISE_POWER_DBM
TRUE_SIGMA_W2 = (10 ** (-TRUE_RHO_DB / 10)) / U_REF

def cn(rng, shape, var):
    return np.sqrt(var / 2.0) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

def generate_u1_batch(rng, n):
    h_sr = cn(rng, n, VAR_HSR)
    h_st = cn(rng, n, VAR_HST)
    h_tr = cn(rng, n, VAR_HTR)
    h_si = cn(rng, (n, M), VAR_HSI)
    h_ir = cn(rng, (n, M), VAR_HIR)

    direct = h_sr + h_st * np.exp(1j * BETA) * h_tr
    cascaded = h_ir * h_si

    phi = rng.uniform(-np.pi, np.pi, size=(n, M))
    phase_state = np.exp(1j * phi)

    G = direct + ETA * np.sum(cascaded * phase_state, axis=1)
    best = np.abs(G) ** 2

    for _ in range(SBF_ITERATIONS):
        eps = rng.uniform(SBF_EPS_LOW, SBF_EPS_HIGH, size=(n, M))
        trial_state = phase_state * np.exp(1j * eps)
        G_trial = direct + ETA * np.sum(cascaded * trial_state, axis=1)
        trial = np.abs(G_trial) ** 2
        improve = trial > best
        best[improve] = trial[improve]
        phase_state[improve] = trial_state[improve]

    return best / U_REF

def generate_u1(total, seed, batch_size):
    rng = np.random.default_rng(seed)
    parts = []
    done = 0
    while done < total:
        n = min(batch_size, total - done)
        parts.append(generate_u1_batch(rng, n))
        done += n
    return np.concatenate(parts)

# ============================================================
# 4. FIT GAMMA MODEL
# ============================================================
u1_train = generate_u1(N_TRAIN, TRAIN_SEED, TRAIN_BATCH_SIZE)
mu = float(np.mean(u1_train))
var = float(np.var(u1_train, ddof=1))
ALPHA_G = mu ** 2 / var
TRUE_THETA_G = var / mu
KS_D, KS_P = kstest(u1_train, "gamma", args=(ALPHA_G, 0.0, TRUE_THETA_G))

# ============================================================
# 5. NUMERICAL GRID
# ============================================================
Q_GRID = np.linspace(LOG_U_MIN, LOG_U_MAX, LOG_U_POINTS)
DQ = Q_GRID[1] - Q_GRID[0]
U_GRID = np.exp(Q_GRID)
LOGW = np.full(LOG_U_POINTS, np.log(DQ))
LOGW[0] += np.log(0.5)
LOGW[-1] += np.log(0.5)

T_GRID = np.linspace(0.0, T_GRID_MAX, T_GRID_POINTS)
Z_GRID = np.expm1(T_GRID)

def build_direct_lut(sigma_w2_assumed, u0_mean_assumed, theta_g_assumed):
    logprior0 = -np.log(u0_mean_assumed) - U_GRID / u0_mean_assumed + Q_GRID + LOGW
    logprior1 = (
        (ALPHA_G - 1.0) * Q_GRID
        - U_GRID / theta_g_assumed
        - gammaln(ALPHA_G)
        - ALPHA_G * np.log(theta_g_assumed)
        + Q_GRID
        + LOGW
    )

    cond_var = U_GRID * SIGMA_S2 + sigma_w2_assumed
    common = -N_S * np.log(np.pi * cond_var)

    p0, p1 = [], []
    for start in range(0, T_GRID_POINTS, LUT_CHUNK):
        stop = min(start + LUT_CHUNK, T_GRID_POINTS)
        zz = Z_GRID[start:stop, None]
        exponent = -zz / cond_var[None, :]
        p0.append(logsumexp((logprior0 + common)[None, :] + exponent, axis=1))
        p1.append(logsumexp((logprior1 + common)[None, :] + exponent, axis=1))

    return np.concatenate(p1) - np.concatenate(p0)

def score_from_lut(z, lut):
    t = np.log1p(np.maximum(z, 0.0))
    return np.interp(t, T_GRID, lut)

def indirect_score(z, sigma_w2_assumed, u0_mean_assumed, theta_g_assumed):
    u_hat = np.maximum((z / N_S - sigma_w2_assumed) / SIGMA_S2, 0.0)
    score = np.empty_like(u_hat)
    positive = u_hat > 0.0

    constant = (
        np.log(u0_mean_assumed)
        - gammaln(ALPHA_G)
        - ALPHA_G * np.log(theta_g_assumed)
    )

    score[positive] = (
        (ALPHA_G - 1.0) * np.log(u_hat[positive])
        + u_hat[positive] * (1.0 / u0_mean_assumed - 1.0 / theta_g_assumed)
        + constant
    )

    if ALPHA_G > 1.0:
        score[~positive] = -1e12
    elif np.isclose(ALPHA_G, 1.0):
        score[~positive] = np.log(u0_mean_assumed / theta_g_assumed)
    else:
        score[~positive] = 1e12

    return score

# ============================================================
# 6. EVALUATE MISMATCH PERFORMANCES
# ============================================================
u1_test = generate_u1(N_TEST, TEST_U1_SEED, TEST_BATCH_SIZE)
rng0 = np.random.default_rng(TEST_U0_SEED)
u0_test = rng0.exponential(scale=TRUE_U0_MEAN, size=N_TEST)

rng_obs = np.random.default_rng(OBS_SEED)
g0 = rng_obs.gamma(shape=N_S, scale=1.0, size=N_TEST)
g1 = rng_obs.gamma(shape=N_S, scale=1.0, size=N_TEST)

z0 = (u0_test + TRUE_SIGMA_W2) * g0
z1 = (u1_test + TRUE_SIGMA_W2) * g1

def metrics(h0, h1):
    fp = int(np.count_nonzero(h0 > 0.0))
    fn = int(np.count_nonzero(h1 <= 0.0))
    pfa = fp / N_TEST
    pmd = fn / N_TEST
    return 0.5 * (pfa + pmd)

snr_rows = []
for delta_db in SNR_MISMATCH_DB:
    assumed_rho_db = TRUE_RHO_DB + delta_db
    sigma_assumed = (10 ** (-assumed_rho_db / 10)) / U_REF

    lut = build_direct_lut(sigma_assumed, TRUE_U0_MEAN, TRUE_THETA_G)
    d0 = score_from_lut(z0, lut)
    d1 = score_from_lut(z1, lut)
    i0 = indirect_score(z0, sigma_assumed, TRUE_U0_MEAN, TRUE_THETA_G)
    i1 = indirect_score(z1, sigma_assumed, TRUE_U0_MEAN, TRUE_THETA_G)

    snr_rows.append({
        "SNR_Mismatch_dB": float(delta_db),
        "BER_Direct": metrics(d0, d1),
        "BER_Indirect": metrics(i0, i1),
    })

snr_df = pd.DataFrame(snr_rows)

ch_rows = []
for delta_db in CHANNEL_MISMATCH_DB:
    scale = 10 ** (delta_db / 10.0)
    u0_assumed = TRUE_U0_MEAN * scale
    theta_assumed = TRUE_THETA_G * scale

    lut = build_direct_lut(TRUE_SIGMA_W2, u0_assumed, theta_assumed)
    d0 = score_from_lut(z0, lut)
    d1 = score_from_lut(z1, lut)
    i0 = indirect_score(z0, TRUE_SIGMA_W2, u0_assumed, theta_assumed)
    i1 = indirect_score(z1, TRUE_SIGMA_W2, u0_assumed, theta_assumed)

    ch_rows.append({
        "Channel_Mismatch_dB": float(delta_db),
        "BER_Direct": metrics(d0, d1),
        "BER_Indirect": metrics(i0, i1),
    })

ch_df = pd.DataFrame(ch_rows)

# ============================================================
# 7. GENERATE FINAL 1x2 PLOT WITH CENTERED SUB-PANEL TAGS
# ============================================================
def style_axis_clean(ax, panel_label):
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # Place panel label in the middle below x-axis (ha="center")
    ax.text(
        -0.12,
        -0.1,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_FONT,
        fontweight="bold"
    )

LEGEND_HANDLES = [
    Line2D(
        [0], [0],
        color=DIRECT_COLOR,
        marker=DIRECT_MARKER,
        linestyle="-",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="Direct"
    ),
    Line2D(
        [0], [0],
        color=INDIRECT_COLOR,
        marker=INDIRECT_MARKER,
        linestyle="-",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="Indirect"
    ),
    Line2D(
        [0], [0],
        color=REFERENCE_COLOR,
        linestyle=":",
        linewidth=1.1,
        label="Perfect knowledge"
    )
]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.8))

# (a) SNR mismatch
ax = axes[0]
ax.plot(snr_df["SNR_Mismatch_dB"], snr_df["BER_Direct"], color=DIRECT_COLOR, marker=DIRECT_MARKER, linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
ax.plot(snr_df["SNR_Mismatch_dB"], snr_df["BER_Indirect"], color=INDIRECT_COLOR, marker=INDIRECT_MARKER, linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
ax.axvline(0.0, color=REFERENCE_COLOR, linestyle=":", linewidth=1.1)
ax.set_xlabel(r"SNR-estimation error, $\Delta_{\mathrm{SNR}}$ (dB)", fontweight="bold", labelpad=3)
ax.set_ylabel("Bit Error Rate (BER)", fontweight="bold")
ax.set_xlim(-3, 3)
ax.set_ylim(0.38, 0.40)
style_axis_clean(ax, "(a)")
ax.legend(handles=LEGEND_HANDLES, loc="upper left", frameon=True, edgecolor="black", fontsize=8)

# (b) Channel mismatch
ax = axes[1]
ax.plot(ch_df["Channel_Mismatch_dB"], ch_df["BER_Direct"], color=DIRECT_COLOR, marker=DIRECT_MARKER, linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
ax.plot(ch_df["Channel_Mismatch_dB"], ch_df["BER_Indirect"], color=INDIRECT_COLOR, marker=INDIRECT_MARKER, linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
ax.axvline(0.0, color=REFERENCE_COLOR, linestyle=":", linewidth=1.1)
ax.set_xlabel(r"Channel-statistics mismatch, $\Delta_{\mathrm{ch}}$ (dB)", fontweight="bold", labelpad=3)
ax.set_ylabel("Bit Error Rate (BER)", fontweight="bold")
ax.set_xlim(-3, 3)
style_axis_clean(ax, "(b)")
ax.legend(handles=LEGEND_HANDLES, loc="upper left", frameon=True, edgecolor="black", fontsize=8)

fig.subplots_adjust(
    left=0.09, right=0.98, top=0.98, bottom=0.10,
    wspace=0.25
)

fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_PATH, format="pdf", bbox_inches="tight", facecolor="white")

plt.show()
