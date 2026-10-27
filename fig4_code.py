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
# Static-channel SBF convergence - Fig. 4
#
# This script evaluates the convergence behavior of the Single-Bit Feedback
# (SBF) algorithm under static channel conditions.
#
# The IRS contains M = 128 elements. At each iteration, a random phase
# perturbation is applied and the receiver returns one feedback bit indicating
# whether the measured SNR has improved. A successful perturbation is retained;
# otherwise, the previous phase configuration is kept.
#
# The perturbation is drawn from
# U[-pi/20, pi/20].
#
# Two independent simulation instances are shown to illustrate the evolution
# of the received SNR from different initial phase configurations.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================
# MATPLOTLIB CONFIGURATION (IEEE SINGLE-COLUMN BOLD)
# ============================================================

mpl.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9.5,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,     # Embed TrueType fonts in PDF
    "ps.fonttype": 42,
})


# ============================================================
# FIGURE 4 — STATIC SBF CONVERGENCE
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

M = 128
K = 100

x = 1.0
beta = 0.0
eta = 0.8

epsilon_low = -np.pi / 20.0
epsilon_high = np.pi / 20.0

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
    source_to_receiver_distance_m, carrier_frequency_GHz
)

source_to_conventional_tag_PL_dB = indoor_nlos_path_loss_dB(
    source_to_conventional_tag_distance_m, carrier_frequency_GHz
)

conventional_tag_to_receiver_PL_dB = indoor_nlos_path_loss_dB(
    conventional_tag_to_receiver_distance_m, carrier_frequency_GHz
)

source_to_irs_PL_dB = indoor_los_path_loss_dB(
    source_to_irs_distance_m, carrier_frequency_GHz
)

irs_to_receiver_PL_dB = indoor_los_path_loss_dB(
    irs_to_receiver_distance_m, carrier_frequency_GHz
)

source_to_receiver_variance = 10.0 ** (-source_to_receiver_PL_dB / 10.0)
source_to_conventional_tag_variance = 10.0 ** (-source_to_conventional_tag_PL_dB / 10.0)
conventional_tag_to_receiver_variance = 10.0 ** (-conventional_tag_to_receiver_PL_dB / 10.0)
source_to_irs_variance = 10.0 ** (-source_to_irs_PL_dB / 10.0)
irs_to_receiver_variance = 10.0 ** (-irs_to_receiver_PL_dB / 10.0)


def cn_scalar(variance):
    return np.sqrt(variance / 2.0) * (
        rng.standard_normal()
        + 1j * rng.standard_normal()
    )


def cn_vector(variance, size):
    return np.sqrt(variance / 2.0) * (
        rng.standard_normal(size)
        + 1j * rng.standard_normal(size)
    )


source_to_receiver_channel = cn_scalar(source_to_receiver_variance)
source_to_conventional_tag_channel = cn_scalar(source_to_conventional_tag_variance)
conventional_tag_to_receiver_channel = cn_scalar(conventional_tag_to_receiver_variance)

source_to_irs_channel = cn_vector(source_to_irs_variance, M)
irs_to_receiver_channel = cn_vector(irs_to_receiver_variance, M)

C = (
    source_to_receiver_channel
    + source_to_conventional_tag_channel
    * x
    * np.exp(1j * beta)
    * conventional_tag_to_receiver_channel
)


def simulate_static_sbf(initial_phases):
    phi_best = initial_phases.copy()

    irs_sum = np.sum(
        irs_to_receiver_channel
        * np.exp(1j * phi_best)
        * source_to_irs_channel
    )

    G = C + eta * irs_sum * x
    gamma_best = rho * np.abs(G) ** 2

    gamma_history = np.zeros(K)
    gamma_history[0] = gamma_best

    for k in range(1, K):
        epsilon = rng.uniform(epsilon_low, epsilon_high, M)
        phi_trial = phi_best + epsilon

        irs_sum_trial = np.sum(
            irs_to_receiver_channel
            * np.exp(1j * phi_trial)
            * source_to_irs_channel
        )

        G_trial = C + eta * irs_sum_trial * x
        gamma_trial = rho * np.abs(G_trial) ** 2

        if gamma_trial > gamma_best:
            phi_best = phi_trial
            gamma_best = gamma_trial

        gamma_history[k] = gamma_best

    return gamma_history


phi_initial_1 = rng.uniform(-np.pi, np.pi, M)
phi_initial_2 = rng.uniform(-np.pi, np.pi, M)

gamma_1 = simulate_static_sbf(phi_initial_1)
gamma_2 = simulate_static_sbf(phi_initial_2)


# ============================================================
# IEEE SINGLE-COLUMN FIGURE GENERATION (3.4 in x 3.1 in)
# ============================================================

fig, ax = plt.subplots(figsize=(3.5, 2.5))

ax.plot(
    np.arange(K),
    gamma_1,
    color="blue",
    linewidth=1.4,
    label="Instance 1"
)

ax.plot(
    np.arange(K),
    gamma_2,
    color="red",
    linewidth=1.4,
    label="Instance 2"
)

# Bold Axis Labels
ax.set_xlabel("Number of Iterations ($k$)", fontweight='bold')
ax.set_ylabel("Received SNR ($\gamma$)", fontweight='bold')

ax.set_xlim(0, K)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

# Bold Tick Labels (Axes Values)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight("bold")

# Bold Legend Box and Labels
legend = ax.legend(
    loc="upper left",
    frameon=True,
    fancybox=False,
    edgecolor="black",
    framealpha=1.0,
    handlelength=2.2,
    handletextpad=0.4,
    borderpad=0.4,
)

legend.get_frame().set_linewidth(1.2)
for text in legend.get_texts():
    text.set_fontweight("bold")

fig.tight_layout(pad=0.4)


# ============================================================
# SAVE RESULTS
# ============================================================

output_dir = Path.cwd() / "figure4_bold_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

pdf_path = output_dir / "Figure4_static_SBF_IEEE_bold.pdf"
png_path = output_dir / "Figure4_static_SBF_IEEE_bold.png"

fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")

print(f"Bold IEEE Vector PDF saved to: {pdf_path}")
plt.show()
