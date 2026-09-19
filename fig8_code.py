from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erfc

# ============================================================
# FIGURE 8 — REVISED STATE-OF-THE-ART BENCHMARK
# ============================================================
# Replaces PARAFAC with a backscatter-specific RIS phase benchmark
# adapted from:
#   D. L. Galappaththige, F. Rezaei, C. Tellambura, and S. P. Herath,
#   "Optimizing Passive Tag Performance With Reconfigurable Intelligent
#   Surfaces in Bistatic Backscatter Networks," IEEE Transactions on
#   Vehicular Technology, vol. 73, no. 9, pp. 12917-12933, 2024.
#   DOI: 10.1109/TVT.2024.3384447
#
# Source-paper principle used here:
#   1) obtain the continuous CSI-based phase that constructively aligns
#      each cascaded RIS path with the non-RIS reference path;
#   2) quantize that phase to a finite-resolution RIS alphabet.
#
# IMPORTANT ADAPTATION:
# The TVT-2024 paper optimizes the emitter->tag direct/RIS paths in a
# bistatic-backscatter topology. Here the same constructive-alignment
# principle is specialized to the composite coefficient of THIS paper:
#
#   C = h_sr + h_st*x*exp(j*beta)*h_tr
#   c_m = h_ir,m * h_si,m
#
# so that the continuous target phase is
#
#   phi_m* = angle(C) - angle(c_m).
#
# The target phase is then quantized to D bits. D=2 is selected because
# the source paper explicitly studies 1-, 2-, and 4-bit phase resolution,
# while 2-bit gives a practical non-trivial benchmark distinct from the
# continuous Perfect-CSI oracle.
#
# Curves:
#   1) TVT'24-adapted 2-bit quantized RIS phase alignment (M=4)
#   2) MPS (M=4, N=8)
#   3) SBF (M=4)
#   4) Perfect CSI continuous phase (M=4)
#   5) SBF (M=128)
#   6) Perfect CSI continuous phase (M=128)
#
# Physical model:
#   fc = 2.4 GHz
#   noise power = -95 dBm
#   source->receiver = 20 m, NLOS
#   source->conventional tag = 10 m, NLOS
#   conventional tag->receiver = 10 m, NLOS
#   source->IRS = 8 m, LOS path-loss expression
#   IRS->receiver = 12 m, LOS path-loss expression
#   eta = 0.8
#   all small-scale links Rayleigh
#
# Static channel, no phase-error, no feedback-delay, no drift.
# ============================================================

# ============================================================
# 1. PARAMETERS / REPRODUCIBILITY
# ============================================================

PIPELINE_VERSION = "figure7_TVT2024_quantized_v1.0"
MASTER_SEED = 42
rng = np.random.default_rng(MASTER_SEED)

M4 = 4
M128 = 128

# Set to 5000 to match the manuscript statement for Figure 7.
NUM_RUNS = 5000

x = 1.0
beta_phase = 0.0
eta = 0.8

# SBF — same algorithmic settings as the previous Figure 7
SBF_ITERATIONS = 100
epsilon_low = -np.pi / 20.0
epsilon_high = np.pi / 20.0

# TVT-2024 adapted quantized phase benchmark
QUANT_BITS = 2
QUANT_LEVELS = 2 ** QUANT_BITS
QUANT_STEP = 2.0 * np.pi / QUANT_LEVELS

carrier_frequency_GHz = 2.4
noise_power_dBm = -95.0

Pt_dBm = np.arange(-20.0, 31.0, 2.0)
rho_dB = Pt_dBm - noise_power_dBm
rho_lin = 10.0 ** (rho_dB / 10.0)

# Geometry
d_sr = 20.0
d_st = 10.0
d_tr = 10.0
d_si = 8.0
d_ir = 12.0

# ============================================================
# 2. 3GPP InH PATH LOSS
# ============================================================

def indoor_los_path_loss_dB(distance_m, frequency_GHz):
    return (
        32.4
        + 17.3 * np.log10(distance_m)
        + 20.0 * np.log10(frequency_GHz)
    )


def indoor_nlos_path_loss_dB(distance_m, frequency_GHz):
    los = indoor_los_path_loss_dB(distance_m, frequency_GHz)
    nlos_candidate = (
        17.30
        + 38.3 * np.log10(distance_m)
        + 24.9 * np.log10(frequency_GHz)
    )
    return max(los, nlos_candidate)


PL_sr = indoor_nlos_path_loss_dB(d_sr, carrier_frequency_GHz)
PL_st = indoor_nlos_path_loss_dB(d_st, carrier_frequency_GHz)
PL_tr = indoor_nlos_path_loss_dB(d_tr, carrier_frequency_GHz)
PL_si = indoor_los_path_loss_dB(d_si, carrier_frequency_GHz)
PL_ir = indoor_los_path_loss_dB(d_ir, carrier_frequency_GHz)

var_sr = 10.0 ** (-PL_sr / 10.0)
var_st = 10.0 ** (-PL_st / 10.0)
var_tr = 10.0 ** (-PL_tr / 10.0)
var_si = 10.0 ** (-PL_si / 10.0)
var_ir = 10.0 ** (-PL_ir / 10.0)

# ============================================================
# 3. EXACT MANUSCRIPT MPS CODEBOOK — M=4, N=8
# ============================================================

Phi_3 = np.array([
    [0.0,      0.0,       0.0,       0.0,      0.0,      0.0,       0.0,       0.0],
    [np.pi,   -np.pi/2,  -np.pi/2,  -np.pi/2,  0.0,      np.pi/2,  np.pi/2,   np.pi/2],
    [0.0,      np.pi/2,   np.pi,     -np.pi/2,  0.0,      np.pi/2,  np.pi,    -np.pi/2],
    [np.pi,    0.0,       np.pi/2,    np.pi,     0.0,      np.pi,   -np.pi/2,   0.0],
], dtype=float)

codebook_8 = Phi_3.T

# ============================================================
# 4. STATIC RAYLEIGH CHANNELS — COMMON TO ALL METHODS
# ============================================================

def complex_gaussian(variance, shape):
    return np.sqrt(variance / 2.0) * (
        rng.standard_normal(shape)
        + 1j * rng.standard_normal(shape)
    )


h_sr = complex_gaussian(var_sr, NUM_RUNS)
h_st = complex_gaussian(var_st, NUM_RUNS)
h_tr = complex_gaussian(var_tr, NUM_RUNS)

# Generate M=128 once; M=4 uses the first four elements.
# Generate the M=4 channels first so that they are exactly
# identical to the M=4 channel realizations used in Fig. 7.
h_si_4 = complex_gaussian(
    var_si,
    (NUM_RUNS, M4)
)

h_ir_4 = complex_gaussian(
    var_ir,
    (NUM_RUNS, M4)
)

# Generate the additional elements required for M=128.
h_si_extra = complex_gaussian(
    var_si,
    (NUM_RUNS, M128 - M4)
)

h_ir_extra = complex_gaussian(
    var_ir,
    (NUM_RUNS, M128 - M4)
)

h_si_128 = np.concatenate(
    (h_si_4, h_si_extra),
    axis=1
)

h_ir_128 = np.concatenate(
    (h_ir_4, h_ir_extra),
    axis=1
)

# Composite non-IRS contribution from the manuscript model.
direct = (
    h_sr
    + h_st * x * np.exp(1j * beta_phase) * h_tr
)

cascaded_4 = h_ir_4 * h_si_4
cascaded_128 = h_ir_128 * h_si_128

# ============================================================
# 5. TVT-2024-ADAPTED 2-BIT QUANTIZED RIS PHASE ALIGNMENT
# ============================================================
# Continuous constructive-alignment phase:
#       phi_m* = angle(direct) - angle(cascaded_m)
# Then nearest-neighbour quantization to L=2^D uniformly spaced
# phases over a 2*pi interval.
# ============================================================

def wrap_to_pi(phi):
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def quantize_phase_nearest(phi, bits):
    levels = 2 ** bits
    step = 2.0 * np.pi / levels
    phi_wrapped = wrap_to_pi(phi)
    phi_quantized = step * np.round(phi_wrapped / step)
    return wrap_to_pi(phi_quantized)


phi_continuous_4 = (
    np.angle(direct)[:, None]
    - np.angle(cascaded_4)
)

phi_tvt24_4 = quantize_phase_nearest(
    phi_continuous_4,
    QUANT_BITS,
)

G_tvt24_4 = (
    direct
    + eta
    * np.sum(
        cascaded_4 * np.exp(1j * phi_tvt24_4),
        axis=1,
    )
    * x
)

gain_tvt24_4 = np.abs(G_tvt24_4) ** 2

# ============================================================
# 6. MPS — M=4, N=8
# ============================================================

exp_codebook_8 = np.exp(1j * codebook_8)

G_mps_all = (
    direct[:, None]
    + eta * (cascaded_4 @ exp_codebook_8.T) * x
)

gain_mps = np.max(
    np.abs(G_mps_all) ** 2,
    axis=1,
)

# ============================================================
# 7. PERFECT CSI CONTINUOUS-PHASE ORACLE
# ============================================================

def perfect_csi_gain(direct_channel, cascaded_channel):
    phi_opt = (
        np.angle(direct_channel)[:, None]
        - np.angle(cascaded_channel)
    )

    G_opt = (
        direct_channel
        + eta
        * np.sum(
            cascaded_channel * np.exp(1j * phi_opt),
            axis=1,
        )
        * x
    )

    return np.abs(G_opt) ** 2


gain_csi_4 = perfect_csi_gain(direct, cascaded_4)
gain_csi_128 = perfect_csi_gain(direct, cascaded_128)

# ============================================================
# 8. SBF — SAME STATIC IDEAL-FEEDBACK ALGORITHM
# ============================================================

sbf_rng = np.random.default_rng(MASTER_SEED + 100)

phi_128 = sbf_rng.uniform(
    -np.pi,
    np.pi,
    size=(NUM_RUNS, M128),
)
phi_4 = phi_128[:, :M4].copy()

G_sbf_4 = (
    direct
    + eta
    * np.sum(
        cascaded_4 * np.exp(1j * phi_4),
        axis=1,
    )
    * x
)
gain_sbf_4 = np.abs(G_sbf_4) ** 2

G_sbf_128 = (
    direct
    + eta
    * np.sum(
        cascaded_128 * np.exp(1j * phi_128),
        axis=1,
    )
    * x
)
gain_sbf_128 = np.abs(G_sbf_128) ** 2

for _ in range(SBF_ITERATIONS):
    epsilon_128 = sbf_rng.uniform(
        epsilon_low,
        epsilon_high,
        size=(NUM_RUNS, M128),
    )
    epsilon_4 = epsilon_128[:, :M4]

    # M=4
    phi_trial_4 = phi_4 + epsilon_4
    G_trial_4 = (
        direct
        + eta
        * np.sum(
            cascaded_4 * np.exp(1j * phi_trial_4),
            axis=1,
        )
        * x
    )
    gain_trial_4 = np.abs(G_trial_4) ** 2

    improve_4 = gain_trial_4 > gain_sbf_4
    gain_sbf_4[improve_4] = gain_trial_4[improve_4]
    phi_4[improve_4] = phi_trial_4[improve_4]

    # M=128
    phi_trial_128 = phi_128 + epsilon_128
    G_trial_128 = (
        direct
        + eta
        * np.sum(
            cascaded_128 * np.exp(1j * phi_trial_128),
            axis=1,
        )
        * x
    )
    gain_trial_128 = np.abs(G_trial_128) ** 2

    improve_128 = gain_trial_128 > gain_sbf_128
    gain_sbf_128[improve_128] = gain_trial_128[improve_128]
    phi_128[improve_128] = phi_trial_128[improve_128]

# ============================================================
# 9. BER + 95% MONTE-CARLO CONFIDENCE INTERVALS
# ============================================================
# BPSK conditional BER:
#       P_b(gamma) = 0.5 * erfc(sqrt(gamma))
# We average this exact conditional BER over independent fading
# realizations. The CI therefore describes Monte-Carlo uncertainty
# of this channel-average estimate; it is not a binomial bit-count CI.
# ============================================================

def conditional_bpsk_ber(rho, channel_power_gain):
    return 0.5 * erfc(
        np.sqrt(rho * channel_power_gain)
    )


def mean_and_ci95(values):
    values = np.asarray(values, dtype=float)
    mean_value = float(np.mean(values))
    if values.size <= 1:
        return mean_value, mean_value, mean_value

    se = float(
        np.std(values, ddof=1)
        / np.sqrt(values.size)
    )
    half = 1.959963984540054 * se

    return (
        mean_value,
        max(0.0, mean_value - half),
        min(1.0, mean_value + half),
    )


method_gains = {
    "TVT24_Quantized_M4_D2": gain_tvt24_4,
    "MPS_M4_N8": gain_mps,
    "SBF_M4": gain_sbf_4,
    "Perfect_CSI_M4": gain_csi_4,
    "SBF_M128": gain_sbf_128,
    "Perfect_CSI_M128": gain_csi_128,
}

summary_rows = []

for p_idx, (pt, rho_db_value, rho) in enumerate(
    zip(Pt_dBm, rho_dB, rho_lin)
):
    for method_name, gain in method_gains.items():
        ber_runs = conditional_bpsk_ber(rho, gain)
        mean_ber, ci_low, ci_high = mean_and_ci95(ber_runs)

        summary_rows.append({
            "Pt_dBm": float(pt),
            "rho_dB": float(rho_db_value),
            "method": method_name,
            "mean_BER": mean_ber,
            "CI95_low": ci_low,
            "CI95_high": ci_high,
        })

summary_df = pd.DataFrame(summary_rows)


def get_curve(method_name):
    return (
        summary_df[summary_df["method"] == method_name]
        .sort_values("Pt_dBm")
        .reset_index(drop=True)
    )


curve_tvt24 = get_curve("TVT24_Quantized_M4_D2")
curve_mps = get_curve("MPS_M4_N8")
curve_sbf4 = get_curve("SBF_M4")
curve_csi4 = get_curve("Perfect_CSI_M4")
curve_sbf128 = get_curve("SBF_M128")
curve_csi128 = get_curve("Perfect_CSI_M128")

# ============================================================
# 10. VERIFICATION CHECKS
# ============================================================

# Continuous CSI must be at least as good as quantized benchmark.
check_csi_vs_quantized = np.all(
    gain_csi_4 + 1e-30 >= gain_tvt24_4
)

# Exact manuscript MPS candidate count.
check_mps_candidates = codebook_8.shape == (8, 4)

# M=128 CSI oracle should dominate M=128 SBF in BER.
check_csi128_vs_sbf128 = np.all(
    curve_csi128["mean_BER"].to_numpy()
    <= curve_sbf128["mean_BER"].to_numpy() + 1e-14
)

# Quantized phases must belong to the D-bit alphabet modulo 2*pi.
quantized_indices = np.mod(
    np.round(phi_tvt24_4 / QUANT_STEP),
    QUANT_LEVELS,
)
check_quantized_alphabet = np.allclose(
    phi_tvt24_4,
    wrap_to_pi(QUANT_STEP * np.round(phi_tvt24_4 / QUANT_STEP)),
    atol=1e-12,
)

# ============================================================
# USER-EDITABLE PLOT SETTINGS
# ============================================================

X_LABEL = r"Transmit Power (dBm)"
Y_LABEL = "Bit Error Rate (BER)"

LINE_WIDTH = 2.1
MARKER_SIZE = 6
MARK_EVERY = 2

# Same color/marker combinations as the previous finalized Figure 7.
# The new TVT'24 benchmark takes the previous PARAFAC style:
# black diamond, solid line.
TVT24_COLOR = "black"
TVT24_MARKER = "D"
TVT24_LINESTYLE = "-"

MPS_COLOR = "red"
MPS_MARKER = "^"
MPS_LINESTYLE = "-"

SBF4_COLOR = "blue"
SBF4_MARKER = "s"
SBF4_LINESTYLE = "-"

CSI4_COLOR = "green"
CSI4_MARKER = "o"
CSI4_LINESTYLE = "-"

SBF128_COLOR = "blue"
SBF128_MARKER = "s"
SBF128_LINESTYLE = "--"

CSI128_COLOR = "green"
CSI128_MARKER = "o"
CSI128_LINESTYLE = "--"

X_LIMITS = (-21, 31)
Y_LIMITS = (1e-12, 1.0)

# Set True only if you want CI bands on the final manuscript plot.
SHOW_CI_BANDS = False

# ============================================================
# 11. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

plot_specs = [
    (
        curve_tvt24,
        r"QPSA ($M=4$)",
        TVT24_COLOR,
        TVT24_MARKER,
        TVT24_LINESTYLE,
    ),
    (
        curve_mps,
        r"MPS ($M=4,N=8$)",
        MPS_COLOR,
        MPS_MARKER,
        MPS_LINESTYLE,
    ),
    (
        curve_sbf4,
        r"SBF ($M=4$)",
        SBF4_COLOR,
        SBF4_MARKER,
        SBF4_LINESTYLE,
    ),
    (
        curve_csi4,
        r"Perfect CSI ($M=4$)",
        CSI4_COLOR,
        CSI4_MARKER,
        CSI4_LINESTYLE,
    ),
    (
        curve_sbf128,
        r"SBF ($M=128$)",
        SBF128_COLOR,
        SBF128_MARKER,
        SBF128_LINESTYLE,
    ),
    (
        curve_csi128,
        r"Perfect CSI ($M=128$)",
        CSI128_COLOR,
        CSI128_MARKER,
        CSI128_LINESTYLE,
    ),
]

for curve, label, color, marker, linestyle in plot_specs:
    line, = ax.semilogy(
        curve["Pt_dBm"],
        curve["mean_BER"],
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        markevery=MARK_EVERY,
        label=label,
    )

    if SHOW_CI_BANDS:
        line_color = line.get_color()
        ax.fill_between(
            curve["Pt_dBm"].to_numpy(),
            np.maximum(curve["CI95_low"].to_numpy(), Y_LIMITS[0]),
            np.maximum(curve["CI95_high"].to_numpy(), Y_LIMITS[0]),
            alpha=0.10,
            color=line_color,
        )

ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)
ax.set_xlim(X_LIMITS)
ax.set_ylim(Y_LIMITS)
ax.minorticks_on()

ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.50)
ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.25)

ax.legend(
    loc="upper right",
    frameon=True,
    fancybox=False,
    edgecolor="black",
    framealpha=1.0,
    fontsize=8.5,
    handlelength=3.2,      # <-- added: long enough handle to show dashes clearly
    handletextpad=0.8,     # <-- added: keeps spacing tidy with the longer handle
)

fig.tight_layout(pad=0.8)

# ============================================================
# 12. SAVE OUTPUTS — PORTABLE PATH
# ============================================================

if "__file__" in globals():
    base_dir = Path(__file__).resolve().parent
else:
    base_dir = Path.cwd()

output_dir = base_dir / "figure7_TVT2024_quantized_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / "figure7_TVT2024_quantized_benchmark_same_styles.png"
csv_path = output_dir / "figure7_TVT2024_quantized_results.csv"
metadata_path = output_dir / "figure7_TVT2024_quantized_metadata.json"

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

summary_df.to_csv(csv_path, index=False)

metadata = {
    "pipeline_version": PIPELINE_VERSION,
    "master_seed": MASTER_SEED,
    "num_channel_realizations": NUM_RUNS,
    "carrier_frequency_GHz": carrier_frequency_GHz,
    "noise_power_dBm": noise_power_dBm,
    "eta": eta,
    "quantized_benchmark": {
        "reference": (
            "Galappaththige et al., IEEE Transactions on Vehicular "
            "Technology, vol. 73, no. 9, 2024, DOI 10.1109/TVT.2024.3384447"
        ),
        "adaptation": (
            "Constructive CSI phase alignment specialized to the present "
            "composite backscatter coefficient, followed by nearest-neighbour "
            "uniform D-bit phase quantization."
        ),
        "quantization_bits": QUANT_BITS,
        "quantization_levels": QUANT_LEVELS,
        "quantization_step_rad": float(QUANT_STEP),
    },
    "SBF": {
        "iterations": SBF_ITERATIONS,
        "perturbation": "Uniform[-pi/20, pi/20]",
    },
    "path_loss_dB": {
        "source_receiver": PL_sr,
        "source_tag": PL_st,
        "tag_receiver": PL_tr,
        "source_IRS": PL_si,
        "IRS_receiver": PL_ir,
    },
    "checks": {
        "Perfect_CSI_gain_not_below_quantized_gain": bool(check_csi_vs_quantized),
        "MPS_codebook_shape_is_8x4": bool(check_mps_candidates),
        "Perfect_CSI_M128_BER_not_worse_than_SBF_M128": bool(check_csi128_vs_sbf128),
        "quantized_phases_on_D_bit_alphabet": bool(check_quantized_alphabet),
    },
}

metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

plt.show()
