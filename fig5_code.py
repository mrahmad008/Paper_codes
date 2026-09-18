# -----------------------------------------------------------------------------
# SBF tracking under time-varying channels - Fig. 5
#
# This script evaluates the SBF algorithm when the channel changes between
# successive iterations. The simulations examine the effects of channel drift,
# IRS phase-setting error, feedback delay, and reflection coefficient.
#
# Unless a parameter is being varied, the simulation uses M = 128,
# epsilon_max = pi/20, delta_max = pi/25, and lambda = 0.98.
#
# The plotted results are averaged over 200 independent channel realizations.
# The channel-drift model is also related to an equivalent mobility level using
# the Doppler relation described in the manuscript.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 4 — DYNAMIC SBF ROBUSTNESS STUDY
# Latest 2.4 GHz indoor physically grounded baseline
#
# PNG-only version
#
# Four separate studies now combined into ONE 2x2 figure:
#   Fig. 4(a): channel drift
#   Fig. 4(b): time-varying phase error
#   Fig. 4(c): feedback delay
#   Fig. 4(d): reflection magnitude / hardware loss
# ============================================================

MASTER_SEED = 42
master_rng = np.random.default_rng(MASTER_SEED)

# -----------------------------
# Manuscript / Algorithm-2 settings
# -----------------------------
M = 128
K = 100

x = 1.0
beta = 0.0

lambda_discount = 0.98

epsilon_low = -np.pi / 20.0
epsilon_high = np.pi / 20.0

NUM_REALIZATIONS = 200

# -----------------------------
# Latest indoor physical parameters
# -----------------------------
carrier_frequency_GHz = 2.4
carrier_frequency_Hz = carrier_frequency_GHz * 1e9

ambient_source_power_dBm = 20.0
ambient_noise_power_dBm = -95.0

# Manuscript definition:
# rho is average transmit SNR.
rho_dB = ambient_source_power_dBm - ambient_noise_power_dBm
rho = 10.0 ** (rho_dB / 10.0)

# Distances
source_to_receiver_distance_m = 20.0
source_to_conventional_tag_distance_m = 10.0
conventional_tag_to_receiver_distance_m = 10.0
source_to_irs_distance_m = 8.0
irs_to_receiver_distance_m = 12.0

# -----------------------------
# 3GPP TR 38.901 InH path loss
# -----------------------------
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

source_to_receiver_variance = 10.0 ** (-source_to_receiver_PL_dB / 10.0)
source_to_conventional_tag_variance = 10.0 ** (-source_to_conventional_tag_PL_dB / 10.0)
conventional_tag_to_receiver_variance = 10.0 ** (-conventional_tag_to_receiver_PL_dB / 10.0)
source_to_irs_variance = 10.0 ** (-source_to_irs_PL_dB / 10.0)
irs_to_receiver_variance = 10.0 ** (-irs_to_receiver_PL_dB / 10.0)

# -----------------------------
# Mobility for feedback-delay aging
# -----------------------------
speed_of_light = 3.0e8
velocity_m_per_s = 2.0

wavelength_m = speed_of_light / carrier_frequency_Hz
doppler_frequency_Hz = velocity_m_per_s / wavelength_m

# ------------------------------------------------------------
# Equivalent-speed interpretation for channel-drift legend
# ------------------------------------------------------------
T_update_seconds = 1.0e-3

def equivalent_speed_from_delta(delta_max_rad, wavelength_m, T_update_seconds):
    return (
        delta_max_rad * wavelength_m
        / (2.0 * np.pi * T_update_seconds)
    )

# -----------------------------
# Sweep definitions
# -----------------------------
channel_drift_bounds = [
    np.pi / 50.0,
    np.pi / 25.0,
    3.0 * np.pi / 50.0,
    2.0 * np.pi / 25.0,
]

channel_drift_speed_mps = [
    equivalent_speed_from_delta(
        bound,
        wavelength_m,
        T_update_seconds
    )
    for bound in channel_drift_bounds
]

phase_error_bounds = [
    np.deg2rad(0.0),
    np.deg2rad(5.0),
    np.deg2rad(10.0),
    np.deg2rad(20.0),
]

feedback_delays_s = [
    0.0,
    1.0e-3,
    3.0e-3,
    5.0e-3,
]

reflection_magnitude_values = [
    1.0,
    0.9,
    0.8,
    0.7,
]

# -----------------------------
# CSCG generators
# -----------------------------
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

# -----------------------------
# One Monte Carlo trial
# -----------------------------
def generate_trial_inputs(seed):
    rng = np.random.default_rng(seed)

    source_to_receiver_channel = cn_scalar(
        rng,
        source_to_receiver_variance
    )

    source_to_conventional_tag_channel = cn_scalar(
        rng,
        source_to_conventional_tag_variance
    )

    conventional_tag_to_receiver_channel = cn_scalar(
        rng,
        conventional_tag_to_receiver_variance
    )

    source_to_irs_channel = cn_vector(
        rng,
        source_to_irs_variance,
        M
    )

    irs_to_receiver_channel = cn_vector(
        rng,
        irs_to_receiver_variance,
        M
    )

    C = (
        source_to_receiver_channel
        + source_to_conventional_tag_channel
        * x
        * np.exp(1j * beta)
        * conventional_tag_to_receiver_channel
    )

    initial_cascaded_channel = (
        irs_to_receiver_channel
        * source_to_irs_channel
    )

    cascaded_magnitude = np.abs(
        initial_cascaded_channel
    )

    channel_phase_initial = np.angle(
        initial_cascaded_channel
    )

    initial_irs_phases = rng.uniform(
        -np.pi,
        np.pi,
        M
    )

    epsilon_sequence = rng.uniform(
        epsilon_low,
        epsilon_high,
        size=(K - 1, M)
    )

    channel_drift_unit_sequence = rng.uniform(
        -1.0,
        1.0,
        size=(K - 1, M)
    )

    feedback_delay_unit_sequence = rng.uniform(
        -1.0,
        1.0,
        size=(K - 1, M)
    )

    phase_error_unit_sequence = rng.uniform(
        -1.0,
        1.0,
        size=(K, M)
    )

    return {
        "C": C,
        "cascaded_magnitude": cascaded_magnitude,
        "channel_phase_initial": channel_phase_initial,
        "initial_irs_phases": initial_irs_phases,
        "epsilon_sequence": epsilon_sequence,
        "channel_drift_unit_sequence": channel_drift_unit_sequence,
        "feedback_delay_unit_sequence": feedback_delay_unit_sequence,
        "phase_error_unit_sequence": phase_error_unit_sequence,
    }

# -----------------------------
# Dynamic SBF for one case
# -----------------------------
def simulate_dynamic_case(
    inputs,
    channel_drift_bound,
    phase_error_bound=0.0,
    feedback_delay_s=0.0,
    reflection_magnitude=0.8,
):
    C = inputs["C"]

    cascaded_magnitude = inputs["cascaded_magnitude"]
    channel_phase = inputs["channel_phase_initial"].copy()
    phi_best = inputs["initial_irs_phases"].copy()

    epsilon_sequence = inputs["epsilon_sequence"]
    channel_drift_unit_sequence = inputs["channel_drift_unit_sequence"]
    feedback_delay_unit_sequence = inputs["feedback_delay_unit_sequence"]
    phase_error_unit_sequence = inputs["phase_error_unit_sequence"]

    feedback_delay_phase_bound = (
        2.0
        * np.pi
        * doppler_frequency_Hz
        * feedback_delay_s
    )

    e0 = phase_error_bound * phase_error_unit_sequence[0]

    current_cascaded_channel = (
        cascaded_magnitude
        * np.exp(1j * channel_phase)
    )

    G_initial = (
        C
        + reflection_magnitude
        * np.sum(
            current_cascaded_channel
            * np.exp(1j * (phi_best + e0))
        )
        * x
    )

    gamma_best = rho * np.abs(G_initial) ** 2

    gamma_history = np.zeros(K)
    gamma_history[0] = gamma_best

    for k in range(1, K):
        epsilon = epsilon_sequence[k - 1]
        phi_trial = phi_best + epsilon

        delta = (
            channel_drift_bound
            * channel_drift_unit_sequence[k - 1]
        )
        channel_phase = channel_phase + delta

        delay_phase = (
            feedback_delay_phase_bound
            * feedback_delay_unit_sequence[k - 1]
        )
        channel_phase = channel_phase + delay_phase

        current_cascaded_channel = (
            cascaded_magnitude
            * np.exp(1j * channel_phase)
        )

        phase_error = (
            phase_error_bound
            * phase_error_unit_sequence[k]
        )

        phi_applied = phi_trial + phase_error

        G_trial = (
            C
            + reflection_magnitude
            * np.sum(
                current_cascaded_channel
                * np.exp(1j * phi_applied)
            )
            * x
        )

        gamma_trial = rho * np.abs(G_trial) ** 2

        if gamma_trial > gamma_best:
            phi_best = phi_trial
            gamma_best = gamma_trial
        else:
            gamma_best = lambda_discount * gamma_best

        gamma_history[k] = gamma_best

    return gamma_history

# -----------------------------
# Common Monte Carlo seeds
# -----------------------------
trial_seeds = master_rng.integers(
    0,
    2**32 - 1,
    size=NUM_REALIZATIONS,
    dtype=np.uint32
)

def average_family(case_values, family_name):
    averaged_curves = [
        np.zeros(K)
        for _ in case_values
    ]

    for seed in trial_seeds:
        inputs = generate_trial_inputs(int(seed))

        for index, case in enumerate(case_values):
            if family_name == "channel_drift":
                curve = simulate_dynamic_case(
                    inputs,
                    channel_drift_bound=case,
                    phase_error_bound=0.0,
                    feedback_delay_s=0.0,
                    reflection_magnitude=0.8
                )

            elif family_name == "phase_error":
                curve = simulate_dynamic_case(
                    inputs,
                    channel_drift_bound=np.pi / 25.0,
                    phase_error_bound=case,
                    feedback_delay_s=0.0,
                    reflection_magnitude=0.8
                )

            elif family_name == "feedback_delay":
                curve = simulate_dynamic_case(
                    inputs,
                    channel_drift_bound=np.pi / 25.0,
                    phase_error_bound=0.0,
                    feedback_delay_s=case,
                    reflection_magnitude=0.8
                )

            elif family_name == "reflection_magnitude":
                curve = simulate_dynamic_case(
                    inputs,
                    channel_drift_bound=np.pi / 25.0,
                    phase_error_bound=0.0,
                    feedback_delay_s=0.0,
                    reflection_magnitude=case
                )

            else:
                raise ValueError("Unknown family name.")

            averaged_curves[index] += curve

    return [
        curve / NUM_REALIZATIONS
        for curve in averaged_curves
    ]

channel_drift_avg = average_family(
    channel_drift_bounds,
    "channel_drift"
)

phase_error_avg = average_family(
    phase_error_bounds,
    "phase_error"
)

feedback_delay_avg = average_family(
    feedback_delays_s,
    "feedback_delay"
)

reflection_magnitude_avg = average_family(
    reflection_magnitude_values,
    "reflection_magnitude"
)

# ============================================================
# USER-EDITABLE PLOT SETTINGS
# ============================================================

# -----------------------------
# Axis labels
# -----------------------------
X_LABEL = r"Number of iterations ($k$)"
Y_LABEL = r"Received SNR ($\gamma$)"

# -----------------------------
# Grid settings
# -----------------------------
GRID_ON = True
GRID_STYLE = "--"
GRID_ALPHA = 0.35
GRID_WIDTH = 0.6
X_TICK_STEP = 10

# -----------------------------
# Line / marker settings
# -----------------------------
LINE_WIDTH = 2.0
MARKER_SIZE = 5
MARK_EVERY = 8

# Marker mapping:
# green -> circle
# blue  -> square
# black -> diamond
# red   -> triangle
CURVE_MARKERS = [
    "o",   # circle
    "s",   # square
    "D",   # diamond
    "^",   # triangle
]

# -----------------------------
# Fig. 4(a): Channel-drift colors and legends
# -----------------------------
CHANNEL_DRIFT_COLORS = [
    "green",
    "blue",
    "black",
    "red",
]

CHANNEL_DRIFT_LEGENDS = [
    rf"$\delta_m(k)=\pi/50$,  v={channel_drift_speed_mps[0]:.2f} m/s",
    rf"$\delta_m(k)=\pi/25$,  v={channel_drift_speed_mps[1]:.2f} m/s",
    rf"$\delta_m(k)= 3\pi/50$, v={channel_drift_speed_mps[2]:.2f} m/s",
    rf"$\delta_m(k)= 2\pi/25$, v={channel_drift_speed_mps[3]:.2f} m/s",
]

# -----------------------------
# Fig. 4(b): Phase-error colors and legends
# -----------------------------
PHASE_ERROR_COLORS = [
    "green",
    "blue",
    "black",
    "red",
]

PHASE_ERROR_LEGENDS = [
    r"$e_m(k)=0$",
    r"$e_m(k)=\pi/36$",
    r"$e_m(k)=\pi/18$",
    r"$e_m(k)=\pi/9$",
]

# -----------------------------
# Fig. 4(c): Feedback-delay colors and legends
# -----------------------------
FEEDBACK_DELAY_COLORS = [
    "green",
    "blue",
    "black",
    "red",
]

FEEDBACK_DELAY_LEGENDS = [
    r"$\tau_{\mathrm{fb}}=0$ ms",
    r"$\tau_{\mathrm{fb}}=1$ ms",
    r"$\tau_{\mathrm{fb}}=3$ ms",
    r"$\tau_{\mathrm{fb}}=5$ ms",
]

# -----------------------------
# Fig. 4(d): Reflection-magnitude colors and legends
# -----------------------------
REFLECTION_COLORS = [
    "green",
    "blue",
    "black",
    "red",
]

REFLECTION_LEGENDS = [
    r"$\eta=1.0$",
    r"$\eta=0.9$",
    r"$\eta=0.8$",
    r"$\eta=0.7$",
]

# -----------------------------
# Output folder (PNG only)
# -----------------------------
output_dir = Path.cwd() / "figure4_png_results"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Plot function for subplot
# ============================================================
def plot_family_on_axis(ax, curves, legends, colors, markers, panel_label):
    k_axis = np.arange(K)

    for curve, legend, color, marker in zip(
        curves,
        legends,
        colors,
        markers
    ):
        ax.plot(
            k_axis,
            curve,
            color=color,
            linewidth=LINE_WIDTH,
            marker=marker,
            markersize=MARKER_SIZE,
            markevery=MARK_EVERY,
            label=legend,
        )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    ax.grid(
        GRID_ON,
        which="major",
        linestyle=GRID_STYLE,
        linewidth=GRID_WIDTH,
        alpha=GRID_ALPHA,
    )

    ax.set_xticks(np.arange(0, K + 1, X_TICK_STEP))

    ax.legend(
        loc="best",
        frameon=True,
        edgecolor="black",
        framealpha=1.0,
        fontsize=8,
    )

    ax.text(
        0.5,
        -0.22,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11
    )

# ============================================================
# Generate ONE 2x2 combined figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

plot_family_on_axis(
    axes[0],
    channel_drift_avg,
    CHANNEL_DRIFT_LEGENDS,
    CHANNEL_DRIFT_COLORS,
    CURVE_MARKERS,
    "(a)"
)

plot_family_on_axis(
    axes[1],
    phase_error_avg,
    PHASE_ERROR_LEGENDS,
    PHASE_ERROR_COLORS,
    CURVE_MARKERS,
    "(b)"
)

plot_family_on_axis(
    axes[2],
    feedback_delay_avg,
    FEEDBACK_DELAY_LEGENDS,
    FEEDBACK_DELAY_COLORS,
    CURVE_MARKERS,
    "(c)"
)

plot_family_on_axis(
    axes[3],
    reflection_magnitude_avg,
    REFLECTION_LEGENDS,
    REFLECTION_COLORS,
    CURVE_MARKERS,
    "(d)"
)

fig.tight_layout()

save_path = output_dir / "figure4_combined_2x2.png"
fig.savefig(
    save_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

print("PNG figure saved in:")
print(save_path)
