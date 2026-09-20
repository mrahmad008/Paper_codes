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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import gammaln, logsumexp
from scipy.stats import kstest

# ============================================================
# DETECTOR ROBUSTNESS — M = 128, NOMINAL GEOMETRY
# (a) BER vs SNR-estimation error
# (b) BER vs channel-statistics mismatch
# ============================================================

# -----------------------------
# 1. SETTINGS
# -----------------------------
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

# Figure style
FIGSIZE = (10.8, 4.3)
DIRECT_COLOR = "blue"
INDIRECT_COLOR = "red"
REFERENCE_COLOR = "black"
DIRECT_MARKER = "s"
INDIRECT_MARKER = "^"
LINE_WIDTH = 1.4
MARKER_SIZE = 4.5
CAP_SIZE = 2.5
LABEL_FONT = 10
TICK_FONT = 8
LEGEND_FONT = 8
PANEL_FONT = 9

# Legend handles: kept identical to the plotted marker/line scheme
LEGEND_HANDLES = [
    Line2D([0], [0], color=DIRECT_COLOR, marker=DIRECT_MARKER, linestyle="-",
           linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label="Direct"),
    Line2D([0], [0], color=INDIRECT_COLOR, marker=INDIRECT_MARKER, linestyle="-",
           linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label="Indirect"),
    Line2D([0], [0], color=REFERENCE_COLOR, linestyle=":",
           linewidth=1.1, label="Perfect knowledge")
]

# -----------------------------
# 2. OUTPUT
# -----------------------------
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()

OUT = BASE_DIR / "detector_robustness_M128"
OUT.mkdir(parents=True, exist_ok=True)
PNG_PATH = OUT / "BER_SNR_channel_mismatch_M128.png"
PDF_PATH = OUT / "BER_SNR_channel_mismatch_M128.pdf"
SNR_CSV = OUT / "BER_vs_SNR_mismatch_M128.csv"
CH_CSV = OUT / "BER_vs_channel_mismatch_M128.csv"
META_PATH = OUT / "detector_robustness_M128_metadata.json"

# -----------------------------
# 3. PATH LOSS + CHANNELS
# -----------------------------
def pl_los(d, fc):
    return 32.4 + 17.3*np.log10(d) + 20.0*np.log10(fc)

def pl_nlos(d, fc):
    return max(pl_los(d, fc), 17.30 + 38.3*np.log10(d) + 24.9*np.log10(fc))

PL_SR = pl_nlos(D_SR, FC_GHZ)
PL_ST = pl_nlos(D_ST, FC_GHZ)
PL_TR = pl_nlos(D_TR, FC_GHZ)
PL_SI = pl_los(D_SI, FC_GHZ)
PL_IR = pl_los(D_IR, FC_GHZ)

VAR_HSR = 10**(-PL_SR/10)
VAR_HST = 10**(-PL_ST/10)
VAR_HTR = 10**(-PL_TR/10)
VAR_HSI = 10**(-PL_SI/10)
VAR_HIR = 10**(-PL_IR/10)

U_REF = VAR_HSR
TRUE_U0_MEAN = 1.0
TRUE_RHO_DB = PT_DBM - NOISE_POWER_DBM
TRUE_SIGMA_W2 = (10**(-TRUE_RHO_DB/10))/U_REF

def cn(rng, shape, var):
    return np.sqrt(var/2.0)*(rng.standard_normal(shape) + 1j*rng.standard_normal(shape))

def generate_u1_batch(rng, n):
    h_sr = cn(rng, n, VAR_HSR)
    h_st = cn(rng, n, VAR_HST)
    h_tr = cn(rng, n, VAR_HTR)
    h_si = cn(rng, (n, M), VAR_HSI)
    h_ir = cn(rng, (n, M), VAR_HIR)

    direct = h_sr + h_st*np.exp(1j*BETA)*h_tr
    cascaded = h_ir*h_si

    phi = rng.uniform(-np.pi, np.pi, size=(n, M))
    phase_state = np.exp(1j*phi)

    G = direct + ETA*np.sum(cascaded*phase_state, axis=1)
    best = np.abs(G)**2

    for _ in range(SBF_ITERATIONS):
        eps = rng.uniform(SBF_EPS_LOW, SBF_EPS_HIGH, size=(n, M))
        trial_state = phase_state*np.exp(1j*eps)
        G_trial = direct + ETA*np.sum(cascaded*trial_state, axis=1)
        trial = np.abs(G_trial)**2
        improve = trial > best
        best[improve] = trial[improve]
        phase_state[improve] = trial_state[improve]

    return best/U_REF

def generate_u1(total, seed, batch_size):
    rng = np.random.default_rng(seed)
    parts = []
    done = 0
    while done < total:
        n = min(batch_size, total-done)
        parts.append(generate_u1_batch(rng, n))
        done += n
    return np.concatenate(parts)

# -----------------------------
# 4. FIT GAMMA MODEL
# -----------------------------
print(f"Generating {N_TRAIN:,} training realizations...")
u1_train = generate_u1(N_TRAIN, TRAIN_SEED, TRAIN_BATCH_SIZE)
mu = float(np.mean(u1_train))
var = float(np.var(u1_train, ddof=1))
ALPHA_G = mu**2/var
TRUE_THETA_G = var/mu
KS_D, KS_P = kstest(u1_train, "gamma", args=(ALPHA_G, 0.0, TRUE_THETA_G))

print(f"alpha_g={ALPHA_G:.8f}, theta_g={TRUE_THETA_G:.8f}, KS D={KS_D:.8f}")

# -----------------------------
# 5. NUMERICAL GRID
# -----------------------------
Q_GRID = np.linspace(LOG_U_MIN, LOG_U_MAX, LOG_U_POINTS)
DQ = Q_GRID[1]-Q_GRID[0]
U_GRID = np.exp(Q_GRID)
LOGW = np.full(LOG_U_POINTS, np.log(DQ))
LOGW[0] += np.log(0.5)
LOGW[-1] += np.log(0.5)

T_GRID = np.linspace(0.0, T_GRID_MAX, T_GRID_POINTS)
Z_GRID = np.expm1(T_GRID)

def build_direct_lut(sigma_w2_assumed, u0_mean_assumed, theta_g_assumed):
    logprior0 = -np.log(u0_mean_assumed) - U_GRID/u0_mean_assumed + Q_GRID + LOGW
    logprior1 = ((ALPHA_G-1.0)*Q_GRID - U_GRID/theta_g_assumed
                 - gammaln(ALPHA_G) - ALPHA_G*np.log(theta_g_assumed)
                 + Q_GRID + LOGW)

    cond_var = U_GRID*SIGMA_S2 + sigma_w2_assumed
    common = -N_S*np.log(np.pi*cond_var)

    p0, p1 = [], []
    for start in range(0, T_GRID_POINTS, LUT_CHUNK):
        stop = min(start+LUT_CHUNK, T_GRID_POINTS)
        zz = Z_GRID[start:stop, None]
        exponent = -zz/cond_var[None, :]
        p0.append(logsumexp((logprior0+common)[None, :] + exponent, axis=1))
        p1.append(logsumexp((logprior1+common)[None, :] + exponent, axis=1))

    return np.concatenate(p1) - np.concatenate(p0)

def score_from_lut(z, lut):
    t = np.log1p(np.maximum(z, 0.0))
    if np.max(t) > T_GRID_MAX:
        raise RuntimeError("Increase T_GRID_MAX")
    return np.interp(t, T_GRID, lut)

def indirect_score(z, sigma_w2_assumed, u0_mean_assumed, theta_g_assumed):
    u_hat = np.maximum((z/N_S - sigma_w2_assumed)/SIGMA_S2, 0.0)
    score = np.empty_like(u_hat)
    positive = u_hat > 0.0

    constant = (np.log(u0_mean_assumed) - gammaln(ALPHA_G)
                - ALPHA_G*np.log(theta_g_assumed))

    score[positive] = ((ALPHA_G-1.0)*np.log(u_hat[positive])
                       + u_hat[positive]*(1.0/u0_mean_assumed - 1.0/theta_g_assumed)
                       + constant)

    if ALPHA_G > 1.0:
        score[~positive] = -1e12
    elif np.isclose(ALPHA_G, 1.0):
        score[~positive] = np.log(u0_mean_assumed/theta_g_assumed)
    else:
        score[~positive] = 1e12

    return score

# -----------------------------
# 6. TRUE NOMINAL TEST BANK
# -----------------------------
print(f"Generating {N_TEST:,} independent test realizations per hypothesis...")
u1_test = generate_u1(N_TEST, TEST_U1_SEED, TEST_BATCH_SIZE)
rng0 = np.random.default_rng(TEST_U0_SEED)
u0_test = rng0.exponential(scale=TRUE_U0_MEAN, size=N_TEST)

rng_obs = np.random.default_rng(OBS_SEED)
g0 = rng_obs.gamma(shape=N_S, scale=1.0, size=N_TEST)
g1 = rng_obs.gamma(shape=N_S, scale=1.0, size=N_TEST)

z0 = (u0_test + TRUE_SIGMA_W2)*g0
z1 = (u1_test + TRUE_SIGMA_W2)*g1

def metrics(h0, h1):
    fp = int(np.count_nonzero(h0 > 0.0))
    fn = int(np.count_nonzero(h1 <= 0.0))
    pfa = fp/N_TEST
    pmd = fn/N_TEST
    ber = 0.5*(pfa+pmd)
    var_ber = (pfa*(1-pfa)/N_TEST + pmd*(1-pmd)/N_TEST)/4.0
    half = CONF_Z*np.sqrt(var_ber)
    return ber, max(0.0, ber-half), min(1.0, ber+half), pfa, pmd

# -----------------------------
# 7. SNR-ESTIMATION MISMATCH
# -----------------------------
snr_rows = []
print("Evaluating SNR-estimation mismatch...")
for delta_db in SNR_MISMATCH_DB:
    assumed_rho_db = TRUE_RHO_DB + delta_db
    sigma_assumed = (10**(-assumed_rho_db/10))/U_REF

    lut = build_direct_lut(sigma_assumed, TRUE_U0_MEAN, TRUE_THETA_G)
    d0 = score_from_lut(z0, lut)
    d1 = score_from_lut(z1, lut)
    i0 = indirect_score(z0, sigma_assumed, TRUE_U0_MEAN, TRUE_THETA_G)
    i1 = indirect_score(z1, sigma_assumed, TRUE_U0_MEAN, TRUE_THETA_G)

    bd, bdl, bdh, pfad, pmdd = metrics(d0, d1)
    bi, bil, bih, pfai, pmdi = metrics(i0, i1)

    snr_rows.append({
        "SNR_Mismatch_dB": float(delta_db),
        "BER_Direct": bd, "BER_Direct_CI95_low": bdl, "BER_Direct_CI95_high": bdh,
        "BER_Indirect": bi, "BER_Indirect_CI95_low": bil, "BER_Indirect_CI95_high": bih,
        "P_FA_Direct": pfad, "P_MD_Direct": pmdd,
        "P_FA_Indirect": pfai, "P_MD_Indirect": pmdi
    })

snr_df = pd.DataFrame(snr_rows)
snr_df.to_csv(SNR_CSV, index=False)

# -----------------------------
# 8. CHANNEL-STATISTICS MISMATCH
# -----------------------------
ch_rows = []
print("Evaluating channel-statistics mismatch...")
for delta_db in CHANNEL_MISMATCH_DB:
    scale = 10**(delta_db/10.0)
    u0_assumed = TRUE_U0_MEAN*scale
    theta_assumed = TRUE_THETA_G*scale

    lut = build_direct_lut(TRUE_SIGMA_W2, u0_assumed, theta_assumed)
    d0 = score_from_lut(z0, lut)
    d1 = score_from_lut(z1, lut)
    i0 = indirect_score(z0, TRUE_SIGMA_W2, u0_assumed, theta_assumed)
    i1 = indirect_score(z1, TRUE_SIGMA_W2, u0_assumed, theta_assumed)

    bd, bdl, bdh, pfad, pmdd = metrics(d0, d1)
    bi, bil, bih, pfai, pmdi = metrics(i0, i1)

    ch_rows.append({
        "Channel_Mismatch_dB": float(delta_db),
        "BER_Direct": bd, "BER_Direct_CI95_low": bdl, "BER_Direct_CI95_high": bdh,
        "BER_Indirect": bi, "BER_Indirect_CI95_low": bil, "BER_Indirect_CI95_high": bih,
        "P_FA_Direct": pfad, "P_MD_Direct": pmdd,
        "P_FA_Indirect": pfai, "P_MD_Indirect": pmdi
    })

ch_df = pd.DataFrame(ch_rows)
ch_df.to_csv(CH_CSV, index=False)

# -----------------------------
# 9. FINAL 1x2 FIGURE
# -----------------------------
def style(ax):
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_FONT)

# Clean legend handles: one marker only for each detector.
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

def plot_ber_curves_only(ax, x, direct, indirect):
    """
    Plot only one visible marker type per detector, no CI shading
    and no error bars — the earlier translucent fill_between band
    read as a "projection"/shadow behind the lines and is removed.
    """
    x = np.asarray(x, dtype=float)
    direct = np.asarray(direct, dtype=float)
    indirect = np.asarray(indirect, dtype=float)

    # Direct: blue solid line + square marker only.
    ax.plot(
        x, direct,
        color=DIRECT_COLOR,
        marker=DIRECT_MARKER,
        linestyle="-",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        markerfacecolor=DIRECT_COLOR,
        markeredgecolor=DIRECT_COLOR,
        zorder=3
    )

    # Indirect: red solid line + triangle marker only.
    ax.plot(
        x, indirect,
        color=INDIRECT_COLOR,
        marker=INDIRECT_MARKER,
        linestyle="-",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        markerfacecolor=INDIRECT_COLOR,
        markeredgecolor=INDIRECT_COLOR,
        zorder=4
    )


fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

# (a) SNR mismatch
ax = axes[0]

plot_ber_curves_only(
    ax,
    snr_df["SNR_Mismatch_dB"],
    snr_df["BER_Direct"],
    snr_df["BER_Indirect"]
)

ax.axvline(
    0.0,
    color=REFERENCE_COLOR,
    linestyle=":",
    linewidth=1.1
)

ax.set_xlabel(
    r"SNR-estimation error, $\Delta_{\mathrm{SNR}}$ (dB)",
    fontsize=LABEL_FONT
)
ax.set_ylabel("Bit Error Rate (BER)", fontsize=LABEL_FONT)
ax.set_xlim(-3, 3)
ax.set_ylim(0.38, 0.40)
style(ax)

ax.legend(
    handles=LEGEND_HANDLES,
    loc="best",
    frameon=True,
    edgecolor="black",
    fontsize=LEGEND_FONT,
    handlelength=2.4,
    numpoints=1
)

# (b) Channel mismatch
ax = axes[1]

plot_ber_curves_only(
    ax,
    ch_df["Channel_Mismatch_dB"],
    ch_df["BER_Direct"],
    ch_df["BER_Indirect"]
)

ax.axvline(
    0.0,
    color=REFERENCE_COLOR,
    linestyle=":",
    linewidth=1.1
)

ax.set_xlabel(
    r"Channel-statistics mismatch, $\Delta_{\mathrm{ch}}$ (dB)",
    fontsize=LABEL_FONT
)
ax.set_ylabel("Bit Error Rate (BER)", fontsize=LABEL_FONT)
ax.set_xlim(-3, 3)
style(ax)

ax.legend(
    handles=LEGEND_HANDLES,
    loc="best",
    frameon=True,
    edgecolor="black",
    fontsize=LEGEND_FONT,
    handlelength=2.4,
    numpoints=1
)

# Tighten the gap between the two panels first, THEN compute
# label positions against the final layout (order matters — see
# below), so labels can't drift onto the neighbouring panel.
fig.subplots_adjust(
    left=0.08, right=0.99, bottom=0.24, top=0.97, wspace=0.16
)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()

LABEL_GAP_FIGFRAC = 0.012

for ax, label in zip(axes, ["(a)", "(b)"]):
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
    "M": M, "eta": ETA, "Pt_dBm": PT_DBM, "noise_power_dBm": NOISE_POWER_DBM,
    "N_s": N_S,
    "geometry_m": {"d_sr":D_SR,"d_st":D_ST,"d_tr":D_TR,"d_si":D_SI,"d_ir":D_IR},
    "gamma_fit": {"alpha_g":float(ALPHA_G),"theta_g":float(TRUE_THETA_G),"KS_D":float(KS_D),"KS_p_value":float(KS_P)},
    "SNR_mismatch_definition": "Delta_SNR = assumed SNR - true SNR; true observations are unchanged.",
    "channel_mismatch_definition": "The assumed H0 exponential mean and H1 Gamma scale are both multiplied by 10^(Delta_ch/10), while alpha_g remains fixed; true observations are unchanged.",
    "decision_threshold": 0.0,
    "training_realizations": N_TRAIN,
    "test_realizations_per_hypothesis": N_TEST,
    "phase_error": "none",
    "smoothing": "none"
}
META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("\nSNR mismatch:")
print(snr_df[["SNR_Mismatch_dB","BER_Direct","BER_Indirect"]].to_string(index=False))
print("\nChannel-statistics mismatch:")
print(ch_df[["Channel_Mismatch_dB","BER_Direct","BER_Indirect"]].to_string(index=False))
print("\nSaved:")
print(PNG_PATH.resolve())
print(PDF_PATH.resolve())
print(SNR_CSV.resolve())
print(CH_CSV.resolve())
print(META_PATH.resolve())
