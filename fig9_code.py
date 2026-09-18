# -----------------------------------------------------------------------------
# Effect of feedback bit errors on SBF - Fig. 9
#
# This simulation evaluates the sensitivity of the SBF algorithm to errors
# in the one-bit feedback channel.
#
# The IRS contains M = 128 elements and the SBF phase perturbation follows
# U[-pi/20, pi/20]. Feedback bit-flip probabilities of
# p_e = {0, 0.05, 0.10} are considered.
#
# The curves show the average received SNR over 200 independent channel
# realizations.
# -----------------------------------------------------------------------------

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# SBF FEEDBACK BIT-FLIP ROBUSTNESS
# Updated to the physical simulation parameters used in the paper
# ============================================================

# ============================================================
# 1. REPRODUCIBILITY / ALGORITHM PARAMETERS
# ============================================================
MASTER_SEED = 42

M = 128
K = 100

x = 1.0
beta = 0.0
eta = 0.8

PERT_LOW = -np.pi / 20.0
PERT_HIGH = np.pi / 20.0

NUM_REALIZATIONS = 200

# Feedback bit-flip probabilities
P_E_VALUES = [0.00, 0.05, 0.10]

# ============================================================
# 2. PHYSICAL SYSTEM PARAMETERS
# ============================================================
FC_GHZ = 2.4

PT_DBM = 20.0
NOISE_DBM = -95.0

# With E{|s|^2}=1, rho = P_t / P_noise
RHO_DB = PT_DBM - NOISE_DBM
rho = 10.0 ** (RHO_DB / 10.0)

# Distances (m)
D_SR = 20.0
D_ST = 10.0
D_TR = 10.0
D_SI = 8.0
D_IR = 12.0


# ============================================================
# 3. 3GPP INDOOR HOTSPOT (InH) PATH-LOSS MODEL
# ============================================================
def path_loss_los_db(distance_m, fc_ghz):
    return (
        32.4
        + 17.3 * np.log10(distance_m)
        + 20.0 * np.log10(fc_ghz)
    )


def path_loss_nlos_db(distance_m, fc_ghz):
    pl_los = path_loss_los_db(distance_m, fc_ghz)

    pl_nlos_candidate = (
        17.30
        + 38.3 * np.log10(distance_m)
        + 24.9 * np.log10(fc_ghz)
    )

    return max(pl_los, pl_nlos_candidate)


# NLOS links
PL_SR_DB = path_loss_nlos_db(D_SR, FC_GHZ)
PL_ST_DB = path_loss_nlos_db(D_ST, FC_GHZ)
PL_TR_DB = path_loss_nlos_db(D_TR, FC_GHZ)

# LOS path-loss expressions
PL_SI_DB = path_loss_los_db(D_SI, FC_GHZ)
PL_IR_DB = path_loss_los_db(D_IR, FC_GHZ)

# Large-scale channel variances beta_ab = 10^(-PL_ab/10)
SIGMA_HSR2 = 10.0 ** (-PL_SR_DB / 10.0)
SIGMA_HST2 = 10.0 ** (-PL_ST_DB / 10.0)
SIGMA_HTR2 = 10.0 ** (-PL_TR_DB / 10.0)
SIGMA_HSI2 = 10.0 ** (-PL_SI_DB / 10.0)
SIGMA_HIR2 = 10.0 ** (-PL_IR_DB / 10.0)


# ============================================================
# 4. COMPLEX GAUSSIAN CHANNEL GENERATOR
# ============================================================
def complex_gaussian(rng, variance, size=None):
    return np.sqrt(variance / 2.0) * (
        rng.standard_normal(size)
        + 1j * rng.standard_normal(size)
    )


# ============================================================
# 5. GENERATE ONE COMMON MONTE CARLO TRIAL
# ============================================================
def generate_trial_inputs(seed):
    """
    Generate the channel, initial IRS phases, SBF perturbations,
    and common feedback-error random numbers for one realization.

    The same trial inputs are reused for all p_e values so that the
    feedback-error curves are compared fairly.
    """
    rng = np.random.default_rng(seed)

    h_sr = complex_gaussian(rng, SIGMA_HSR2)
    h_st = complex_gaussian(rng, SIGMA_HST2)
    h_tr = complex_gaussian(rng, SIGMA_HTR2)

    h_si = complex_gaussian(rng, SIGMA_HSI2, M)
    h_ir = complex_gaussian(rng, SIGMA_HIR2, M)

    direct = (
        h_sr
        + h_st * x * np.exp(1j * beta) * h_tr
    )

    initial_phases = rng.uniform(
        -np.pi,
         np.pi,
        M
    )

    epsilon_sequence = rng.uniform(
        PERT_LOW,
        PERT_HIGH,
        size=(K - 1, M)
    )

    # One common U[0,1] sequence is used for all p_e values.
    # A feedback bit flips when flip_uniform[k] < p_e.
    flip_uniform_sequence = rng.random(K - 1)

    return {
        "direct": direct,
        "h_ir": h_ir,
        "h_si": h_si,
        "initial_phases": initial_phases,
        "epsilon_sequence": epsilon_sequence,
        "flip_uniform_sequence": flip_uniform_sequence,
    }


# ============================================================
# 6. SBF WITH FEEDBACK BIT-FLIP ERRORS
# ============================================================
def simulate_sbf_with_feedback_errors(inputs, p_e):
    """
    Static-channel SBF simulation with feedback bit errors.

    The receiver forms the true improvement bit:
        b(k) = 1, if gamma_trial > gamma_current
             = 0, otherwise.

    The feedback channel flips the bit with probability p_e.
    The transmitter updates the IRS phase configuration according
    to the received bit.

    No channel drift, feedback delay, or IRS phase error is included
    here so that the effect of feedback bit errors is isolated.
    """
    direct = inputs["direct"]
    h_ir = inputs["h_ir"]
    h_si = inputs["h_si"]

    phi_current = inputs["initial_phases"].copy()

    epsilon_sequence = inputs["epsilon_sequence"]
    flip_uniform_sequence = inputs["flip_uniform_sequence"]

    # Initial received coefficient
    G_current = (
        direct
        + eta
        * np.sum(
            h_ir
            * np.exp(1j * phi_current)
            * h_si
        )
        * x
    )

    gamma_current = rho * np.abs(G_current) ** 2

    gamma_history = np.zeros(K, dtype=float)
    gamma_history[0] = gamma_current

    # --------------------------------------------------------
    # Iterative SBF procedure
    # --------------------------------------------------------
    for k in range(1, K):

        epsilon = epsilon_sequence[k - 1]

        phi_trial = phi_current + epsilon

        G_trial = (
            direct
            + eta
            * np.sum(
                h_ir
                * np.exp(1j * phi_trial)
                * h_si
            )
            * x
        )

        gamma_trial = rho * np.abs(G_trial) ** 2

        # True one-bit receiver decision
        true_feedback_bit = int(
            gamma_trial > gamma_current
        )

        # Binary symmetric feedback channel
        bit_flip = int(
            flip_uniform_sequence[k - 1] < p_e
        )

        received_feedback_bit = (
            true_feedback_bit ^ bit_flip
        )

        # Transmitter acts on the received feedback bit
        if received_feedback_bit == 1:
            phi_current = phi_trial
            gamma_current = gamma_trial

        gamma_history[k] = gamma_current

    return gamma_history


# ============================================================
# 7. MONTE CARLO AVERAGING
# ============================================================
master_rng = np.random.default_rng(MASTER_SEED)

trial_seeds = master_rng.integers(
    0,
    2**32 - 1,
    size=NUM_REALIZATIONS,
    dtype=np.uint32
)

trajectories = {
    p_e: np.zeros(K, dtype=float)
    for p_e in P_E_VALUES
}

print("============================================================")
print("SBF FEEDBACK BIT-FLIP ROBUSTNESS")
print("============================================================")
print(f"M                     : {M}")
print(f"K                     : {K}")
print(f"Realizations          : {NUM_REALIZATIONS}")
print(f"fc                    : {FC_GHZ:.1f} GHz")
print(f"Transmit power        : {PT_DBM:.1f} dBm")
print(f"Noise power           : {NOISE_DBM:.1f} dBm")
print(f"rho                   : {RHO_DB:.1f} dB")
print(f"eta                   : {eta:.1f}")
print(
    "Perturbation          : "
    f"U[-pi/20, pi/20]"
)
print()

print("Nominal path losses:")
print(f"PL_sr                  : {PL_SR_DB:.6f} dB")
print(f"PL_st                  : {PL_ST_DB:.6f} dB")
print(f"PL_tr                  : {PL_TR_DB:.6f} dB")
print(f"PL_si                  : {PL_SI_DB:.6f} dB")
print(f"PL_ir                  : {PL_IR_DB:.6f} dB")
print("============================================================")
print()

for realization_index, seed in enumerate(trial_seeds):

    inputs = generate_trial_inputs(int(seed))

    for p_e in P_E_VALUES:
        trajectories[p_e] += simulate_sbf_with_feedback_errors(
            inputs,
            p_e
        )

    if (
        (realization_index + 1) % 25 == 0
        or realization_index == NUM_REALIZATIONS - 1
    ):
        print(
            f"Completed "
            f"{realization_index + 1}/{NUM_REALIZATIONS} "
            f"realizations"
        )

# Average over realizations
for p_e in P_E_VALUES:
    trajectories[p_e] /= NUM_REALIZATIONS


# ============================================================
# 8. NUMERICAL SUMMARY
# ============================================================
print()
print("============================================================")
print("FINAL AVERAGE RECEIVED SNR")
print("============================================================")

reference_final = trajectories[0.0][-1]

for p_e in P_E_VALUES:
    final_gamma = trajectories[p_e][-1]
    final_gamma_db = 10.0 * np.log10(final_gamma)

    retention_percent = (
        100.0 * final_gamma / reference_final
    )

    print(
        f"p_e = {p_e:0.2f} : "
        f"gamma = {final_gamma:.6f}, "
        f"gamma_dB = {final_gamma_db:.4f} dB, "
        f"retention = {retention_percent:.2f}%"
    )

print("============================================================")


# ============================================================
# 9. SAVE NUMERICAL RESULTS
# ============================================================
OUTPUT_DIR = Path.cwd() / "feedback_bitflip_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_file = OUTPUT_DIR / "sbf_feedback_bitflip_M128.csv"

with csv_file.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["iteration"]
        + [
            f"avg_gamma_pe_{p_e:.2f}"
            for p_e in P_E_VALUES
        ]
    )

    for k in range(K):
        writer.writerow(
            [k]
            + [
                trajectories[p_e][k]
                for p_e in P_E_VALUES
            ]
        )


# ============================================================
# 10. PUBLICATION-QUALITY PLOT
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

fig, ax = plt.subplots(
    figsize=(7.0, 4.5)
)

colors = [
    "green",
    "blue",
    "red",
]

line_styles = [
    "-",
    "-",
    "-",
]

markers = [
    "o",
    "s",
    "^",
]

for index, p_e in enumerate(P_E_VALUES):

    avg_gamma = trajectories[p_e]

    k_values = np.arange(K)

    ax.plot(
        k_values,
        avg_gamma,
        color=colors[index],
        linestyle=line_styles[index],
        linewidth=1.8,
        marker=markers[index],
        markersize=4.0,
        markevery=10,
        markeredgewidth=0.8,
        label=rf"$p_e = {p_e:.2f}$",
    )

ax.set_xlabel(
    r"Number of Iterations ($k$)"
)

ax.set_ylabel(
    r"Received SNR ($\gamma$)"
)

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    length=4,
    width=0.8,
)

ax.minorticks_on()

ax.tick_params(
    axis="both",
    which="minor",
    direction="in",
    length=2.5,
    width=0.6,
)

ax.grid(
    True,
    which="major",
    linestyle=":",
    linewidth=0.6,
    alpha=0.5,
)

legend = ax.legend(
    loc="best",
    frameon=True,
    fancybox=False,
    edgecolor="black",
    framealpha=1.0,
    borderpad=0.5,
    handlelength=2.5,
)

legend.get_frame().set_facecolor("white")
legend.get_frame().set_alpha(1.0)

fig.tight_layout(
    pad=0.8
)

png_file = (
    OUTPUT_DIR
    / "sbf_feedback_bitflip_convergence_M128.png"
)

fig.savefig(
    png_file,
    dpi=600,
    format="png",
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor="white",
    edgecolor="white",
)

# ============================================================
# 11. VERIFY PNG PROPERTIES
# ============================================================
img = Image.open(png_file)

print()
print("============================================================")
print("OUTPUT FILES")
print("============================================================")
print(f"PNG  : {png_file}")
print(f"CSV  : {csv_file}")
print(f"Image size : {img.size}")
print(f"Color mode : {img.mode}")
print(f"DPI        : {img.info.get('dpi', 'Not stored')}")
print("============================================================")

plt.show()
