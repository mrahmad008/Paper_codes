import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Parameters
# ============================================================

M = 128
K = 100

rho = 1.0
eta = 1.0
theta = 0.0
x = 1.0
var = 1.0

# SBF perturbation range
eps_low = -np.pi / 7
eps_high = np.pi / 7

# Channel drift range
delta_low = -np.pi / 25
delta_high = np.pi / 25

# Forgetting/decay factor
lam = 0.98


# ============================================================
# Time-varying SBF simulation
# ============================================================

def simulate_sbf_timevarying(seed=None):

    if seed is not None:
        np.random.seed(seed)

    # ========================================================
    # Static channel coefficients
    # ========================================================

    h_sr = np.sqrt(var / 2) * (
        np.random.randn()
        + 1j * np.random.randn()
    )

    h_st = np.sqrt(var / 2) * (
        np.random.randn()
        + 1j * np.random.randn()
    )

    h_tr = np.sqrt(var / 2) * (
        np.random.randn()
        + 1j * np.random.randn()
    )

    h_si = np.sqrt(var / 2) * (
        np.random.randn(M)
        + 1j * np.random.randn(M)
    )

    h_ir = np.sqrt(var / 2) * (
        np.random.randn(M)
        + 1j * np.random.randn(M)
    )


    # ========================================================
    # Direct path and IRS channel
    # ========================================================

    direct = (
        h_sr
        + h_st
        * x
        * np.exp(1j * theta)
        * h_tr
    )

    g = h_ir * h_si


    # ========================================================
    # Initial phase configuration
    # ========================================================

    # Same initial phase concept for SBF and baseline
    phi_init = np.random.uniform(
        0,
        2 * np.pi,
        M
    )

    phi_best = phi_init.copy()

    # Initial channel phase drift
    phi_channel = np.zeros(M)


    # ========================================================
    # Initial SNR
    # ========================================================

    sum_irs = np.sum(
        g
        * np.exp(
            1j * (
                phi_best
                + phi_channel
            )
        )
    )

    gamma = (
        rho
        * np.abs(
            direct
            + eta * sum_irs
        ) ** 2
    )

    gamma_best = gamma


    # ========================================================
    # Allocate history arrays
    # ========================================================

    gamma_history = np.zeros(K)

    gamma_history[0] = gamma_best


    # ========================================================
    # Baseline:
    # Initial phases remain fixed while the channel drifts
    # ========================================================

    baseline_history = np.zeros(K)

    baseline_history[0] = gamma


    # ========================================================
    # Iterative simulation
    # ========================================================

    for k in range(1, K):

        # ----------------------------------------------------
        # Random SBF phase perturbation
        # ----------------------------------------------------

        eps = np.random.uniform(
            eps_low,
            eps_high,
            M
        )


        # ----------------------------------------------------
        # Channel phase drift
        # ----------------------------------------------------

        delta = np.random.uniform(
            delta_low,
            delta_high,
            M
        )

        phi_channel_new = (
            phi_channel
            + delta
        )


        # ====================================================
        # SBF update
        # ====================================================

        phi_candidate = (
            phi_best
            + eps
        )

        sum_irs_new = np.sum(
            g
            * np.exp(
                1j * (
                    phi_candidate
                    + phi_channel_new
                )
            )
        )

        gamma_new = (
            rho
            * np.abs(
                direct
                + eta * sum_irs_new
            ) ** 2
        )


        # ----------------------------------------------------
        # Accept or decay
        # ----------------------------------------------------

        if gamma_new > gamma_best:

            phi_best = phi_candidate

            gamma_best = gamma_new

        else:

            gamma_best = (
                lam
                * gamma_best
            )


        # ====================================================
        # Baseline: fixed initial phases
        # ====================================================

        sum_irs_baseline = np.sum(
            g
            * np.exp(
                1j * (
                    phi_init
                    + phi_channel_new
                )
            )
        )

        baseline = (
            rho
            * np.abs(
                direct
                + eta * sum_irs_baseline
            ) ** 2
        )


        # ====================================================
        # Store results
        # ====================================================

        gamma_history[k] = gamma_best

        baseline_history[k] = baseline


        # ====================================================
        # Update channel for next iteration
        # ====================================================

        phi_channel = phi_channel_new


    return gamma_history, baseline_history


# ============================================================
# Run two independent instances
# ============================================================

instance1, baseline1 = (
    simulate_sbf_timevarying(None)
)

instance2, baseline2 = (
    simulate_sbf_timevarying(None)
)


# ============================================================
# Publication-quality plotting parameters
# ============================================================

plt.rcParams.update({

    # Font
    'font.family': 'Arial',

    # General font size
    'font.size': 10,

    # Axis labels
    'axes.labelsize': 11,

    # Tick labels
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,

    # Legend
    'legend.fontsize': 9,

    # Axis line width
    'axes.linewidth': 0.8,

    # Tick widths
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,

    # Tick lengths
    'xtick.major.size': 4,
    'ytick.major.size': 4
})


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


# ============================================================
# SBF Instance 1
# Original: BLUE solid
# ============================================================

ax.plot(
    instance1,
    label='Instance 1 (SBF)',
    linewidth=2.0,
    color='blue',
    linestyle='-'
)


# ============================================================
# SBF Instance 2
# Original: RED solid
# ============================================================

ax.plot(
    instance2,
    label='Instance 2 (SBF)',
    linewidth=2.0,
    color='red',
    linestyle='-'
)


# ============================================================
# Baseline 1
# Original: BLUE dashed
# ============================================================

ax.plot(
    baseline1,
    label='Channel condition (baseline)',
    linewidth=1.5,
    color='blue',
    linestyle='--',
    alpha=0.7
)


# ============================================================
# Baseline 2
# Original: RED dashed
# ============================================================

ax.plot(
    baseline2,
    label='Channel condition (baseline)',
    linewidth=1.5,
    color='red',
    linestyle='--',
    alpha=0.7
)


# ============================================================
# X-axis label
# ============================================================

ax.set_xlabel(
    'Number of iterations ($k$)',
    fontsize=11
)


# ============================================================
# Y-axis label
# ============================================================

ax.set_ylabel(
    r'Received SNR ($\gamma$)',
    fontsize=11
)


# ============================================================
# Tick formatting
# ============================================================

ax.tick_params(
    axis='both',
    which='major',
    direction='in',
    length=4,
    width=0.8
)


# Minor ticks
ax.minorticks_on()

ax.tick_params(
    axis='both',
    which='minor',
    direction='in',
    length=2.5,
    width=0.6
)


# ============================================================
# Grid
# ============================================================

ax.grid(
    True,
    which='major',
    linestyle='--',
    linewidth=0.6,
    alpha=0.6
)


# ============================================================
# Legend
# ============================================================

legend = ax.legend(
    loc='best',
    frameon=True,
    fancybox=False,
    edgecolor='black',
    framealpha=1.0,
    borderpad=0.5,
    handlelength=2.5
)

legend.get_frame().set_facecolor(
    'white'
)

legend.get_frame().set_alpha(
    1.0
)


# ============================================================
# Layout
# ============================================================

fig.tight_layout(
    pad=0.8
)


# ============================================================
# Save high-resolution PNG
# ============================================================

output_file = (
    'figure4_with_channel.png'
)

fig.savefig(
    output_file,

    # High resolution
    dpi=600,

    # PNG format
    format='png',

    # Prevent clipping
    bbox_inches='tight',

    # Small padding
    pad_inches=0.05,

    # White background
    facecolor='white',

    # White edge
    edgecolor='white'
)


# ============================================================
# Verify PNG properties
# ============================================================

img = Image.open(
    output_file
)

print()
print("============================================")
print("Figure successfully saved")
print("============================================")
print(
    f"File       : {output_file}"
)
print(
    f"Format     : {img.format}"
)
print(
    f"Image size : {img.size}"
)
print(
    f"Color mode : {img.mode}"
)
print(
    f"Resolution : "
    f"{img.info.get('dpi', 'Not stored')}"
)
print("============================================")


# ============================================================
# Display figure
# ============================================================

plt.show()
