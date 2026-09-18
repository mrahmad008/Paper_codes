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
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 3 — STATIC SBF CONVERGENCE
# Latest 2.4 GHz indoor physically grounded baseline
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

# -----------------------------
# Manuscript / algorithm settings
# -----------------------------
M = 128
K = 100

x = 1.0
beta = 0.0
eta = 0.8

epsilon_low = -np.pi / 20.0
epsilon_high = np.pi / 20.0

# -----------------------------
# Latest indoor physical parameters
# -----------------------------
carrier_frequency_GHz = 2.4

ambient_source_power_dBm = 20.0
ambient_noise_power_dBm = -95.0

# Manuscript definition:
# rho is the average transmit SNR.
# Keep rho explicit in the simulation rather than deriving it
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

# NLOS for source->receiver and conventional-tag links
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

# LOS path-loss cases for IRS-associated links
source_to_irs_PL_dB = indoor_los_path_loss_dB(
    source_to_irs_distance_m,
    carrier_frequency_GHz
)

irs_to_receiver_PL_dB = indoor_los_path_loss_dB(
    irs_to_receiver_distance_m,
    carrier_frequency_GHz
)

# Existing manuscript Rayleigh variances
source_to_receiver_variance = 10.0 ** (-source_to_receiver_PL_dB / 10.0)
source_to_conventional_tag_variance = 10.0 ** (-source_to_conventional_tag_PL_dB / 10.0)
conventional_tag_to_receiver_variance = 10.0 ** (-conventional_tag_to_receiver_PL_dB / 10.0)
source_to_irs_variance = 10.0 ** (-source_to_irs_PL_dB / 10.0)
irs_to_receiver_variance = 10.0 ** (-irs_to_receiver_PL_dB / 10.0)

# -----------------------------
# CSCG / Rayleigh generators
# -----------------------------
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

# -----------------------------
# ONE common static channel realization
# -----------------------------
source_to_receiver_channel = cn_scalar(
    source_to_receiver_variance
)

source_to_conventional_tag_channel = cn_scalar(
    source_to_conventional_tag_variance
)

conventional_tag_to_receiver_channel = cn_scalar(
    conventional_tag_to_receiver_variance
)

source_to_irs_channel = cn_vector(
    source_to_irs_variance,
    M
)

irs_to_receiver_channel = cn_vector(
    irs_to_receiver_variance,
    M
)

# Manuscript non-IRS component
C = (
    source_to_receiver_channel
    + source_to_conventional_tag_channel
    * x
    * np.exp(1j * beta)
    * conventional_tag_to_receiver_channel
)

# -----------------------------
# Static SBF
# -----------------------------
def simulate_static_sbf(initial_phases):
    phi_best = initial_phases.copy()

    irs_sum = np.sum(
        irs_to_receiver_channel
        * np.exp(1j * phi_best)
        * source_to_irs_channel
    )

    G = (
        C
        + eta * irs_sum * x
    )

    gamma_best = rho * np.abs(G) ** 2

    gamma_history = np.zeros(K)
    gamma_history[0] = gamma_best

    for k in range(1, K):
        epsilon = rng.uniform(
            epsilon_low,
            epsilon_high,
            M
        )

        phi_trial = phi_best + epsilon

        irs_sum_trial = np.sum(
            irs_to_receiver_channel
            * np.exp(1j * phi_trial)
            * source_to_irs_channel
        )

        G_trial = (
            C
            + eta * irs_sum_trial * x
        )

        gamma_trial = rho * np.abs(G_trial) ** 2

        if gamma_trial > gamma_best:
            phi_best = phi_trial
            gamma_best = gamma_trial

        gamma_history[k] = gamma_best

    return gamma_history

# Algorithm-1-consistent initialization
phi_initial_1 = rng.uniform(-np.pi, np.pi, M)
phi_initial_2 = rng.uniform(-np.pi, np.pi, M)

gamma_1 = simulate_static_sbf(phi_initial_1)
gamma_2 = simulate_static_sbf(phi_initial_2)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Original color convention requested by user
ax.plot(
    np.arange(K),
    gamma_1,
    color="blue",
    linewidth=2.0,
    label="Instance 1"
)

ax.plot(
    np.arange(K),
    gamma_2,
    color="red",
    linewidth=2.0,
    label="Instance 2"
)

ax.set_xlabel(r"Number of iterations ($k$)")
ax.set_ylabel(r"Received SNR ($\gamma$)")
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

legend = ax.legend(
    frameon=True,
    fancybox=False,
    edgecolor="black"
)
legend.get_frame().set_facecolor("white")

fig.tight_layout()

# -----------------------------
# Portable output folder
# -----------------------------
output_dir = Path.cwd() / "figure3_latest_2p4GHz_indoor"
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / "figure3_static_SBF_2p4GHz_indoor.png"

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)


print("=" * 70)
print("FIGURE 3 — LATEST 2.4 GHz INDOOR BASELINE")
print("=" * 70)
print(f"rho = {rho_dB:.2f} dB")
print(f"Source -> receiver path loss = {source_to_receiver_PL_dB:.3f} dB")
print(f"Source -> conventional tag path loss = {source_to_conventional_tag_PL_dB:.3f} dB")
print(f"Conventional tag -> receiver path loss = {conventional_tag_to_receiver_PL_dB:.3f} dB")
print(f"Source -> IRS path loss = {source_to_irs_PL_dB:.3f} dB")
print(f"IRS -> receiver path loss = {irs_to_receiver_PL_dB:.3f} dB")
print()
print(f"Instance 1: {gamma_1[0]:.6f} -> {gamma_1[-1]:.6f}")
print(f"Instance 2: {gamma_2[0]:.6f} -> {gamma_2[-1]:.6f}")
print()
print(f"PNG: {png_path}")
print("=" * 70)

plt.show()
