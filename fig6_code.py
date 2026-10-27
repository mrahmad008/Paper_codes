# ============================================================
# FAIR COMBINED FIGURE 6
# Static vs time-varying SBF target-SNR attainment
#
# Fairness rule:
# For every Monte Carlo realization and every M, the static
# and time-varying cases use EXACTLY the same:
#   - initial channel realization
#   - initial IRS phase vector
#   - SBF perturbation sequence epsilon_m(k)
#
# The time-varying case differs only through channel phase drift:
#   delta_m(k) ~ U[-pi/25, pi/25]
#
# The dynamic SBF discount factor remains lambda = 0.98.
#
# Static curves: solid
# Time-varying curves: dashed
# Same color/marker for the same M.
#
# Saves PNG only.
# ============================================================

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================
# MATPLOTLIB IEEE SINGLE-COLUMN CONFIGURATION (EXACT MATCH FOR FIGURE 7)
# ============================================================

mpl.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,    # Perfectly proportioned legend font size
    "mathtext.default": "regular",
    "pdf.fonttype": 42,       # Embed TrueType fonts in PDF
    "ps.fonttype": 42,
})


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

MASTER_SEED = 42
master_rng = np.random.default_rng(MASTER_SEED)

K = 100
NUM_REALIZATIONS = 200

x = 1.0
beta = 0.0
eta = 0.8

epsilon_low = -np.pi / 20.0
epsilon_high = np.pi / 20.0

lambda_discount = 0.98
delta_bound = np.pi / 25.0

carrier_frequency_GHz = 2.4

ambient_source_power_dBm = 20.0
ambient_noise_power_dBm = -95.0

rho_dB = ambient_source_power_dBm - ambient_noise_power_dBm
rho = 10.0 ** (rho_dB / 10.0)

source_to_receiver_distance_m = 20.0
source_to_conventional_tag_distance_m = 10.0
conventional_tag_to_receiver_distance_m = 10.0
source_to_irs_distance_m = 8.0
irs_to_receiver_distance_m = 12.0

M_values = [4, 32, 128, 512]
MAX_M = max(M_values)

target_snr_dB_values = np.arange(5.0, 41.0, 1.0)


# ============================================================
# 3GPP InH PATH LOSS
# ============================================================

def indoor_los_path_loss_dB(distance_m, frequency_GHz):
    return 32.4 + 17.3 * np.log10(distance_m) + 20.0 * np.log10(frequency_GHz)


def indoor_nlos_path_loss_dB(distance_m, frequency_GHz):
    los = indoor_los_path_loss_dB(distance_m, frequency_GHz)
    nlos_candidate = 17.30 + 38.3 * np.log10(distance_m) + 24.9 * np.log10(frequency_GHz)
    return max(los, nlos_candidate)


source_to_receiver_PL_dB = indoor_nlos_path_loss_dB(source_to_receiver_distance_m, carrier_frequency_GHz)
source_to_conventional_tag_PL_dB = indoor_nlos_path_loss_dB(source_to_conventional_tag_distance_m, carrier_frequency_GHz)
conventional_tag_to_receiver_PL_dB = indoor_nlos_path_loss_dB(conventional_tag_to_receiver_distance_m, carrier_frequency_GHz)
source_to_irs_PL_dB = indoor_los_path_loss_dB(source_to_irs_distance_m, carrier_frequency_GHz)
irs_to_receiver_PL_dB = indoor_los_path_loss_dB(irs_to_receiver_distance_m, carrier_frequency_GHz)

source_to_receiver_variance = 10.0 ** (-source_to_receiver_PL_dB / 10.0)
source_to_conventional_tag_variance = 10.0 ** (-source_to_conventional_tag_PL_dB / 10.0)
conventional_tag_to_receiver_variance = 10.0 ** (-conventional_tag_to_receiver_PL_dB / 10.0)
source_to_irs_variance = 10.0 ** (-source_to_irs_PL_dB / 10.0)
irs_to_receiver_variance = 10.0 ** (-irs_to_receiver_PL_dB / 10.0)


# ============================================================
# COMPLEX GAUSSIAN HELPERS
# ============================================================

def cn_scalar(rng, variance):
    return np.sqrt(variance / 2.0) * (rng.standard_normal() + 1j * rng.standard_normal())


def cn_vector(rng, variance, size):
    return np.sqrt(variance / 2.0) * (rng.standard_normal(size) + 1j * rng.standard_normal(size))


def generate_common_trial(seed):
    rng = np.random.default_rng(seed)
    h_sr = cn_scalar(rng, source_to_receiver_variance)
    h_st = cn_scalar(rng, source_to_conventional_tag_variance)
    h_tr = cn_scalar(rng, conventional_tag_to_receiver_variance)
    h_si_full = cn_vector(rng, source_to_irs_variance, MAX_M)
    h_ir_full = cn_vector(rng, irs_to_receiver_variance, MAX_M)
    phi0_full = rng.uniform(-np.pi, np.pi, MAX_M)
    epsilon_full = rng.uniform(epsilon_low, epsilon_high, size=(K - 1, MAX_M))
    drift_unit_full = rng.uniform(-1.0, 1.0, size=(K - 1, MAX_M))

    C = h_sr + h_st * x * np.exp(1j * beta) * h_tr
    return {
        "C": C,
        "h_si_full": h_si_full,
        "h_ir_full": h_ir_full,
        "phi0_full": phi0_full,
        "epsilon_full": epsilon_full,
        "drift_unit_full": drift_unit_full,
    }


def static_sbf_history_from_common(trial, M):
    C, h_si, h_ir = trial["C"], trial["h_si_full"][:M], trial["h_ir_full"][:M]
    phi_best, epsilon_seq = trial["phi0_full"][:M].copy(), trial["epsilon_full"][:, :M]

    gamma_history_linear = np.zeros(K)
    irs_sum = np.sum(h_ir * np.exp(1j * phi_best) * h_si)
    G = C + eta * irs_sum * x
    gamma_best = rho * np.abs(G) ** 2
    gamma_history_linear[0] = gamma_best

    for k in range(1, K):
        phi_trial = phi_best + epsilon_seq[k - 1]
        irs_sum_trial = np.sum(h_ir * np.exp(1j * phi_trial) * h_si)
        G_trial = C + eta * irs_sum_trial * x
        gamma_trial = rho * np.abs(G_trial) ** 2

        if gamma_trial > gamma_best:
            phi_best, gamma_best = phi_trial, gamma_trial

        gamma_history_linear[k] = gamma_best

    return 10.0 * np.log10(np.maximum(gamma_history_linear, 1e-30))


def dynamic_sbf_history_from_common(trial, M):
    C, h_si, h_ir = trial["C"], trial["h_si_full"][:M], trial["h_ir_full"][:M]
    phi_best, epsilon_seq, drift_unit_seq = trial["phi0_full"][:M].copy(), trial["epsilon_full"][:, :M], trial["drift_unit_full"][:, :M]

    cascaded_channel = h_ir * h_si
    cascaded_magnitude, channel_phase = np.abs(cascaded_channel), np.angle(cascaded_channel)

    gamma_history_linear = np.zeros(K)
    current_cascaded = cascaded_magnitude * np.exp(1j * channel_phase)
    irs_sum = np.sum(current_cascaded * np.exp(1j * phi_best))

    G = C + eta * irs_sum * x
    gamma_best = rho * np.abs(G) ** 2
    gamma_history_linear[0] = gamma_best

    for k in range(1, K):
        phi_trial = phi_best + epsilon_seq[k - 1]
        delta = delta_bound * drift_unit_seq[k - 1]

        channel_phase += delta
        current_cascaded = cascaded_magnitude * np.exp(1j * channel_phase)

        irs_sum_trial = np.sum(current_cascaded * np.exp(1j * phi_trial))
        G_trial = C + eta * irs_sum_trial * x
        gamma_trial = rho * np.abs(G_trial) ** 2

        if gamma_trial > gamma_best:
            phi_best, gamma_best = phi_trial, gamma_trial
        else:
            gamma_best = lambda_discount * gamma_best

        gamma_history_linear[k] = gamma_best

    return 10.0 * np.log10(np.maximum(gamma_history_linear, 1e-30))


def first_iteration_to_reach_target(gamma_history_dB, target_dB):
    reached = np.where(gamma_history_dB >= target_dB)[0]
    return K if reached.size == 0 else int(reached[0])


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

static_counts = {M: np.zeros((NUM_REALIZATIONS, len(target_snr_dB_values))) for M in M_values}
dynamic_counts = {M: np.zeros((NUM_REALIZATIONS, len(target_snr_dB_values))) for M in M_values}
trial_seeds = master_rng.integers(0, 2**31 - 1, size=NUM_REALIZATIONS, dtype=np.uint32)

for realization_index, seed in enumerate(trial_seeds):
    trial = generate_common_trial(int(seed))
    for M in M_values:
        static_history_dB = static_sbf_history_from_common(trial, M)
        dynamic_history_dB = dynamic_sbf_history_from_common(trial, M)
        for target_index, target_dB in enumerate(target_snr_dB_values):
            static_counts[M][realization_index, target_index] = first_iteration_to_reach_target(static_history_dB, target_dB)
            dynamic_counts[M][realization_index, target_index] = first_iteration_to_reach_target(dynamic_history_dB, target_dB)

static_curves = {M: static_counts[M].mean(axis=0) for M in M_values}
dynamic_curves = {M: dynamic_counts[M].mean(axis=0) for M in M_values}


# ============================================================
# FIGURE GENERATION (3.4 in x 3.1 in IEEE SIZE)
# ============================================================

fig, ax = plt.subplots(figsize=(3.5, 2.5))

CURVE_COLORS = {4: "red", 32: "black", 128: "blue", 512: "green"}
CURVE_MARKERS = {4: "^", 32: "D", 128: "o", 512: "s"}

LINE_WIDTH = 1.2
MARKER_SIZE = 3.5
MARK_EVERY = 3

# Full untruncated legend text entries in a single column
for M in M_values:
    ax.plot(
        target_snr_dB_values, static_curves[M],
        color=CURVE_COLORS[M], marker=CURVE_MARKERS[M],
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE, markevery=MARK_EVERY,
        linestyle="-", label=rf"$M={M}$ static channel"
    )

for M in M_values:
    ax.plot(
        target_snr_dB_values, dynamic_curves[M],
        color=CURVE_COLORS[M], marker=CURVE_MARKERS[M],
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE, markevery=MARK_EVERY,
        linestyle=":", label=rf"$M={M}$ time-varying channel"
    )

ax.set_xlabel("Target Received SNR $\gamma$ (dB)")
ax.set_ylabel("Average Number of Iterations")

ax.set_xticks(np.arange(5, 41, 5))
ax.set_yticks(np.arange(0, 101, 20))
ax.set_ylim(0, K)

ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
ax.minorticks_on()

# Single-column legend with complete untruncated text
ax.legend(
    loc="upper left",
    ncol=1,                   # One column layout
    fontsize=8,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    framealpha=1.0,
    handlelength=2.0,
    handletextpad=0.4,
    borderpad=0.35,
    labelspacing=0.25,
)

fig.tight_layout(pad=0.4)


# ============================================================
# SAVE VECTOR PDF
# ============================================================

output_dir = Path.cwd() / "figure6_untruncated_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

pdf_path = output_dir / "Figure6_IEEE_untruncated_single_column.pdf"
fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")

print(f"IEEE Publication Vector PDF saved to: {pdf_path}")
plt.show()
