from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
# 3GPP InH path loss
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

source_to_receiver_PL_dB = indoor_nlos_path_loss_dB(
    source_to_receiver_distance_m,
    carrier_frequency_GHz
)

source_to_conventional_tag_PL_dB = indoor_nlos_path_loss_dB(
    source_to_conventional_tag_distance_m,
    carrier_frequency_GHz
)

conventional_tag_to_receiver_PL_dB = indoor_nlos_path_loss_dB(
    conventional_tag_to_receiver_distance_m,
    carrier_frequency_GHz
)

source_to_irs_PL_dB = indoor_los_path_loss_dB(
    source_to_irs_distance_m,
    carrier_frequency_GHz
)

irs_to_receiver_PL_dB = indoor_los_path_loss_dB(
    irs_to_receiver_distance_m,
    carrier_frequency_GHz
)

source_to_receiver_variance = 10.0 ** (
    -source_to_receiver_PL_dB / 10.0
)

source_to_conventional_tag_variance = 10.0 ** (
    -source_to_conventional_tag_PL_dB / 10.0
)

conventional_tag_to_receiver_variance = 10.0 ** (
    -conventional_tag_to_receiver_PL_dB / 10.0
)

source_to_irs_variance = 10.0 ** (
    -source_to_irs_PL_dB / 10.0
)

irs_to_receiver_variance = 10.0 ** (
    -irs_to_receiver_PL_dB / 10.0
)

# ============================================================
# Complex Gaussian helpers
# ============================================================

def cn_scalar(rng, variance):
    return np.sqrt(variance / 2.0) * (
        rng.standard_normal()
        + 1j * rng.standard_normal()
    )

def cn_vector(rng, variance, size):
    return np.sqrt(variance / 2.0) * (
        rng.standard_normal(size)
        + 1j * rng.standard_normal(size)
    )

# ============================================================
# One common trial shared by all M and both channel cases
# ============================================================

def generate_common_trial(seed):
    rng = np.random.default_rng(seed)

    h_sr = cn_scalar(rng, source_to_receiver_variance)
    h_st = cn_scalar(rng, source_to_conventional_tag_variance)
    h_tr = cn_scalar(rng, conventional_tag_to_receiver_variance)

    h_si_full = cn_vector(
        rng,
        source_to_irs_variance,
        MAX_M
    )

    h_ir_full = cn_vector(
        rng,
        irs_to_receiver_variance,
        MAX_M
    )

    phi0_full = rng.uniform(
        -np.pi,
        np.pi,
        MAX_M
    )

    epsilon_full = rng.uniform(
        epsilon_low,
        epsilon_high,
        size=(K - 1, MAX_M)
    )

    drift_unit_full = rng.uniform(
        -1.0,
        1.0,
        size=(K - 1, MAX_M)
    )

    C = (
        h_sr
        + h_st
        * x
        * np.exp(1j * beta)
        * h_tr
    )

    return {
        "C": C,
        "h_si_full": h_si_full,
        "h_ir_full": h_ir_full,
        "phi0_full": phi0_full,
        "epsilon_full": epsilon_full,
        "drift_unit_full": drift_unit_full,
    }

# ============================================================
# Static SBF
# ============================================================

def static_sbf_history_from_common(trial, M):
    C = trial["C"]

    h_si = trial["h_si_full"][:M]
    h_ir = trial["h_ir_full"][:M]

    phi_best = trial["phi0_full"][:M].copy()
    epsilon_seq = trial["epsilon_full"][:, :M]

    gamma_history_linear = np.zeros(K)

    irs_sum = np.sum(
        h_ir
        * np.exp(1j * phi_best)
        * h_si
    )

    G = C + eta * irs_sum * x

    gamma_best = rho * np.abs(G) ** 2
    gamma_history_linear[0] = gamma_best

    for k in range(1, K):

        phi_trial = (
            phi_best
            + epsilon_seq[k - 1]
        )

        irs_sum_trial = np.sum(
            h_ir
            * np.exp(1j * phi_trial)
            * h_si
        )

        G_trial = C + eta * irs_sum_trial * x

        gamma_trial = rho * np.abs(G_trial) ** 2

        if gamma_trial > gamma_best:
            phi_best = phi_trial
            gamma_best = gamma_trial

        gamma_history_linear[k] = gamma_best

    return 10.0 * np.log10(
        np.maximum(
            gamma_history_linear,
            1e-30
        )
    )

# ============================================================
# Time-varying SBF
# Same channel / phi0 / epsilon as static; drift only
# ============================================================

def dynamic_sbf_history_from_common(trial, M):
    C = trial["C"]

    h_si = trial["h_si_full"][:M]
    h_ir = trial["h_ir_full"][:M]

    phi_best = trial["phi0_full"][:M].copy()

    epsilon_seq = trial[
        "epsilon_full"
    ][:, :M]

    drift_unit_seq = trial[
        "drift_unit_full"
    ][:, :M]

    cascaded_channel = (
        h_ir
        * h_si
    )

    cascaded_magnitude = np.abs(
        cascaded_channel
    )

    channel_phase = np.angle(
        cascaded_channel
    )

    gamma_history_linear = np.zeros(K)

    current_cascaded = (
        cascaded_magnitude
        * np.exp(1j * channel_phase)
    )

    irs_sum = np.sum(
        current_cascaded
        * np.exp(1j * phi_best)
    )

    G = C + eta * irs_sum * x

    gamma_best = rho * np.abs(G) ** 2
    gamma_history_linear[0] = gamma_best

    for k in range(1, K):

        # Same perturbation as static case
        phi_trial = (
            phi_best
            + epsilon_seq[k - 1]
        )

        # Channel drift only
        delta = (
            delta_bound
            * drift_unit_seq[k - 1]
        )

        channel_phase = (
            channel_phase
            + delta
        )

        current_cascaded = (
            cascaded_magnitude
            * np.exp(1j * channel_phase)
        )

        irs_sum_trial = np.sum(
            current_cascaded
            * np.exp(1j * phi_trial)
        )

        G_trial = (
            C
            + eta
            * irs_sum_trial
            * x
        )

        gamma_trial = (
            rho
            * np.abs(G_trial) ** 2
        )

        if gamma_trial > gamma_best:
            phi_best = phi_trial
            gamma_best = gamma_trial
        else:
            gamma_best = (
                lambda_discount
                * gamma_best
            )

        gamma_history_linear[k] = gamma_best

    return 10.0 * np.log10(
        np.maximum(
            gamma_history_linear,
            1e-30
        )
    )

# ============================================================
# First-passage iteration
# ============================================================

def first_iteration_to_reach_target(
    gamma_history_dB,
    target_dB
):
    reached = np.where(
        gamma_history_dB >= target_dB
    )[0]

    if reached.size == 0:
        return K

    return int(reached[0])

# ============================================================
# Paired Monte Carlo averaging
# ============================================================

static_counts = {
    M: np.zeros(
        (
            NUM_REALIZATIONS,
            len(target_snr_dB_values)
        )
    )
    for M in M_values
}

dynamic_counts = {
    M: np.zeros(
        (
            NUM_REALIZATIONS,
            len(target_snr_dB_values)
        )
    )
    for M in M_values
}

trial_seeds = master_rng.integers(
    0,
    2**32 - 1,
    size=NUM_REALIZATIONS,
    dtype=np.uint32
)

for realization_index, seed in enumerate(
    trial_seeds
):

    trial = generate_common_trial(
        int(seed)
    )

    for M in M_values:

        static_history_dB = (
            static_sbf_history_from_common(
                trial,
                M
            )
        )

        dynamic_history_dB = (
            dynamic_sbf_history_from_common(
                trial,
                M
            )
        )

        for target_index, target_dB in enumerate(
            target_snr_dB_values
        ):

            static_counts[M][
                realization_index,
                target_index
            ] = first_iteration_to_reach_target(
                static_history_dB,
                target_dB
            )

            dynamic_counts[M][
                realization_index,
                target_index
            ] = first_iteration_to_reach_target(
                dynamic_history_dB,
                target_dB
            )

static_curves = {
    M: static_counts[M].mean(axis=0)
    for M in M_values
}

dynamic_curves = {
    M: dynamic_counts[M].mean(axis=0)
    for M in M_values
}

# ============================================================
# USER-EDITABLE PLOT SETTINGS
# ============================================================

X_LABEL = r"Target Received SNR $\gamma$ (dB)"
Y_LABEL = "Average Number of Iterations"

GRID_ON = True
GRID_STYLE = "--"
GRID_ALPHA = 0.35
GRID_WIDTH = 0.6

X_TICK_STEP = 5
Y_TICK_STEP = 10

LINE_WIDTH = 2.0
MARKER_SIZE = 5
MARK_EVERY = 3

CURVE_COLORS = {
    4: "red",
    32: "black",
    128: "blue",
    512: "green",
}

CURVE_MARKERS = {
    4: "^",
    32: "D",
    128: "o",
    512: "s",
}

STATIC_LINESTYLE = "-"
DYNAMIC_LINESTYLE = ":"

OUTPUT_FILE_NAME = (
    "figure5_static_vs_time_varying_paired.png"
)

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

for M in M_values:

    ax.plot(
        target_snr_dB_values,
        static_curves[M],
        color=CURVE_COLORS[M],
        marker=CURVE_MARKERS[M],
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        markevery=MARK_EVERY,
        linestyle=STATIC_LINESTYLE,
        label=rf"$M={M}$ static channel (SBF-based algorithm)"
    )

for M in M_values:

    ax.plot(
        target_snr_dB_values,
        dynamic_curves[M],
        color=CURVE_COLORS[M],
        marker=CURVE_MARKERS[M],
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        markevery=MARK_EVERY,
        linestyle=DYNAMIC_LINESTYLE,
        label=rf"$M={M}$ time-varying channel (SBF-based algorithm)"
    )

ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)

ax.grid(
    GRID_ON,
    which="major",
    linestyle=GRID_STYLE,
    linewidth=GRID_WIDTH,
    alpha=GRID_ALPHA
)

ax.set_xticks(
    np.arange(
        target_snr_dB_values.min(),
        target_snr_dB_values.max() + 1,
        X_TICK_STEP
    )
)

ax.set_yticks(
    np.arange(
        0,
        K + 1,
        Y_TICK_STEP
    )
)

ax.set_ylim(
    0,
    K
)

ax.legend(
    loc="best",
    frameon=True,
    edgecolor="black",
    framealpha=1.0,
    fontsize=9,
    ncol=2
)

fig.tight_layout()

output_dir = (
    Path.cwd()
    / "figure5_combined_paired_fair"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

save_path = (
    output_dir
    / OUTPUT_FILE_NAME
)

fig.savefig(
    save_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("PNG saved to:")
print(save_path)
