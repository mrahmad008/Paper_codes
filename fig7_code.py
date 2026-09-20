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

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erfc


# ============================================================
# Simulation parameters
# ============================================================

MASTER_SEED = 42
rng = np.random.default_rng(MASTER_SEED)

M = 4
x = 1.0
beta_phase = 0.0
eta = 0.8
NUM_TRIALS = 5000

carrier_frequency_GHz = 2.4
noise_power_dBm = -95.0

Pt_dBm_values = np.arange(-20.0, 31.0, 2.0)

# Distances (m)
d_sr = 20.0
d_st = 10.0
d_tr = 10.0
d_si = 8.0
d_ir = 12.0


# ============================================================
# 3GPP Indoor Hotspot path-loss model
# ============================================================

def indoor_los_pl(d, fc):
    return (
        32.4
        + 17.3 * np.log10(d)
        + 20.0 * np.log10(fc)
    )


def indoor_nlos_pl(d, fc):
    return max(
        indoor_los_pl(d, fc),
        17.30
        + 38.3 * np.log10(d)
        + 24.9 * np.log10(fc)
    )


PL_sr = indoor_nlos_pl(d_sr, carrier_frequency_GHz)
PL_st = indoor_nlos_pl(d_st, carrier_frequency_GHz)
PL_tr = indoor_nlos_pl(d_tr, carrier_frequency_GHz)

PL_si = indoor_los_pl(d_si, carrier_frequency_GHz)
PL_ir = indoor_los_pl(d_ir, carrier_frequency_GHz)

beta_sr = 10.0 ** (-PL_sr / 10.0)
beta_st = 10.0 ** (-PL_st / 10.0)
beta_tr = 10.0 ** (-PL_tr / 10.0)
beta_si = 10.0 ** (-PL_si / 10.0)
beta_ir = 10.0 ** (-PL_ir / 10.0)


# ============================================================
# MPS phase-shift matrices
# ============================================================

Phi1 = np.array([
    [0, 0, 0, 0],
    [np.pi, -np.pi / 2, 0, np.pi / 2],
    [0, np.pi, 0, np.pi],
    [np.pi, np.pi / 2, 0, -np.pi / 2]
], dtype=float)


Phi2 = np.array([
    [0, 0, 0, 0, 0, 0],
    [np.pi, -np.pi / 2, 0, 0, np.pi / 2, np.pi],
    [0, np.pi, -np.pi / 2, 0, np.pi, -np.pi / 2],
    [np.pi, 0, np.pi, 0, np.pi, 0]
], dtype=float)


Phi3 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [
        np.pi,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        0,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2
    ],
    [
        0,
        np.pi / 2,
        np.pi,
        -np.pi / 2,
        0,
        np.pi / 2,
        np.pi,
        -np.pi / 2
    ],
    [
        np.pi,
        0,
        np.pi / 2,
        np.pi,
        0,
        np.pi,
        -np.pi / 2,
        0
    ]
], dtype=float)


mps_codebooks = {
    4: Phi1,
    6: Phi2,
    8: Phi3
}

N_values = [4, 6, 8]


# ============================================================
# Complex Gaussian channel generators
# ============================================================

def cn_scalar(rng, variance, size):
    return np.sqrt(variance / 2.0) * (
        rng.standard_normal(size)
        + 1j * rng.standard_normal(size)
    )


def cn_vector(rng, variance, shape):
    return np.sqrt(variance / 2.0) * (
        rng.standard_normal(shape)
        + 1j * rng.standard_normal(shape)
    )


# ============================================================
# BPSK BER
# ============================================================

def bpsk_ber(gamma):
    return 0.5 * erfc(
        np.sqrt(
            np.maximum(gamma, 0.0)
        )
    )


# ============================================================
# Generate common channel realizations
# ============================================================

h_sr = cn_scalar(
    rng,
    beta_sr,
    NUM_TRIALS
)

h_st = cn_scalar(
    rng,
    beta_st,
    NUM_TRIALS
)

h_tr = cn_scalar(
    rng,
    beta_tr,
    NUM_TRIALS
)

h_si = cn_vector(
    rng,
    beta_si,
    (NUM_TRIALS, M)
)

h_ir = cn_vector(
    rng,
    beta_ir,
    (NUM_TRIALS, M)
)


# Direct + conventional backscatter component
C = (
    h_sr
    + h_st
    * x
    * np.exp(1j * beta_phase)
    * h_tr
)


# ============================================================
# MPS simulation
# ============================================================

rows = []


for N in N_values:

    Phi = mps_codebooks[N]

    ideal_candidate_sums = np.zeros(
        (NUM_TRIALS, N),
        dtype=complex
    )

    # --------------------------------------------------------
    # Evaluate every MPS phase-shift candidate
    # --------------------------------------------------------

    for l in range(N):

        phi_l = Phi[:, l]

        ideal_candidate_sums[:, l] = np.sum(
            h_ir
            * np.exp(1j * phi_l)[None, :]
            * h_si,
            axis=1
        )

    G_all_ideal = (
        C[:, None]
        + eta
        * ideal_candidate_sums
        * x
    )

    # Select the MPS configuration giving maximum SNR
    best_index_ideal = np.argmax(
        np.abs(G_all_ideal) ** 2,
        axis=1
    )

    selected_codewords = (
        Phi[:, best_index_ideal].T
    )

    # --------------------------------------------------------
    # Apply the selected MPS phase shifts directly
    # No IRS phase-setting error is included
    # --------------------------------------------------------

    irs_sum = np.sum(
        h_ir
        * np.exp(1j * selected_codewords)
        * h_si,
        axis=1
    )

    G_selected = (
        C
        + eta
        * irs_sum
        * x
    )

    gain_selected = np.abs(G_selected) ** 2

    # --------------------------------------------------------
    # BER versus transmit power
    # --------------------------------------------------------

    for Pt_dBm in Pt_dBm_values:

        rho_dB = (
            Pt_dBm
            - noise_power_dBm
        )

        rho = 10.0 ** (
            rho_dB / 10.0
        )

        gamma = (
            rho
            * gain_selected
        )

        ber = bpsk_ber(gamma)

        rows.append({
            "N": N,
            "Pt_dBm": float(Pt_dBm),
            "noise_power_dBm": float(noise_power_dBm),
            "rho_dB": float(rho_dB),
            "eta": float(eta),
            "mean_BER": float(
                np.mean(ber)
            ),
            "num_trials": NUM_TRIALS
        })


# ============================================================
# Convert results to DataFrame
# ============================================================

results_df = pd.DataFrame(rows)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.8)
)


style_map = {
    4: {
        "color": "red",
        "marker": "^"
    },
    6: {
        "color": "blue",
        "marker": "s"
    },
    8: {
        "color": "green",
        "marker": "o"
    },
}


for N in N_values:

    d = (
        results_df[
            results_df["N"] == N
        ]
        .sort_values("Pt_dBm")
    )

    s = style_map[N]

    ax.semilogy(
        d["Pt_dBm"],
        d["mean_BER"],
        color=s["color"],
        marker=s["marker"],
        linestyle="-",
        linewidth=2.1,
        markersize=6,
        label=rf"$N={N}$"
    )


ax.set_xlabel(
    "Transmit Power (dBm)"
)

ax.set_ylabel(
    "Bit Error Rate (BER)"
)

ax.set_xlim(
    Pt_dBm_values.min() - 1,
    Pt_dBm_values.max() + 1
)

ax.set_ylim(
    1e-12,
    1
)

ax.minorticks_on()

ax.grid(
    True,
    which="major",
    linestyle="--",
    linewidth=0.7,
    alpha=0.5
)

ax.grid(
    True,
    which="minor",
    linestyle=":",
    linewidth=0.5,
    alpha=0.25
)

ax.legend(
    loc="best",
    frameon=True,
    edgecolor="black",
    framealpha=1.0,
    fontsize=9
)

fig.tight_layout()


# ============================================================
# Save results
# ============================================================

output_dir = (
    Path.cwd()
    / "figure7_outputs"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

png_path = (
    output_dir
    / "Figure7_BER_vs_transmit_power.png"
)

csv_path = (
    output_dir
    / "Figure7_BER_vs_transmit_power.csv"
)


fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

results_df.to_csv(
    csv_path,
    index=False
)


print("Figure saved to:")
print(png_path.resolve())

print("\nCSV saved to:")
print(csv_path.resolve())


plt.show()
