import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Simulation parameters
# ============================================================

M_values = [64, 128, 256]          # number of IRS elements
NUM_TRIALS = 200                   # channel realisations per M
K_MAX = 100                        # max iterations (cap)
RHO = 1.0                          # transmit SNR scale
ETA = 1.0                          # IRS reflection coefficient
THETA = 0.0                        # tag phase
X = 1.0                            # backscattered symbol
VAR = 1.0                          # channel variance (0 dB)

# Perturbation distribution: uniform [-pi/20, pi/20]
EPS_LOW = -np.pi / 20
EPS_HIGH = np.pi / 20


# ============================================================
# Helper functions
# ============================================================

def compute_max_snr(direct, h_ir, h_si, eta, rho):
    """
    Theoretical maximum SNR with perfect IRS alignment.
    """

    ir_mags = np.abs(h_ir * h_si)

    max_sum = np.sum(ir_mags)

    max_total = (
        np.abs(direct)
        + eta * max_sum
    )

    return rho * (max_total ** 2)


# ============================================================
# Run one SBF instance
# ============================================================

def run_sbf_static(
    M,
    target_lin,
    direct,
    h_ir,
    h_si
):
    """
    Runs one SBF instance.

    Returns the iteration index (0-based)
    when SNR >= target_lin.

    Returns K_MAX if the target is not reached.
    """

    # Random initial IRS phases
    phi_best = np.random.uniform(
        0,
        2 * np.pi,
        M
    )

    # Initial IRS contribution
    sum_irs = np.sum(
        h_ir
        * np.exp(1j * phi_best)
        * h_si
    )

    # Initial received SNR
    gamma = (
        RHO
        * np.abs(
            direct + ETA * sum_irs
        ) ** 2
    )

    gamma_best = gamma


    # Check whether target is already achieved
    if gamma_best >= target_lin:
        return 0


    # ========================================================
    # SBF iterations
    # ========================================================

    for k in range(1, K_MAX + 1):

        # Random phase perturbation
        eps = np.random.uniform(
            EPS_LOW,
            EPS_HIGH,
            M
        )

        # New IRS phase configuration
        phi_new = phi_best + eps

        # New IRS contribution
        sum_irs_new = np.sum(
            h_ir
            * np.exp(1j * phi_new)
            * h_si
        )

        # New received SNR
        gamma_new = (
            RHO
            * np.abs(
                direct
                + ETA * sum_irs_new
            ) ** 2
        )


        # Accept only an improved solution
        if gamma_new > gamma_best:

            phi_best = phi_new

            gamma_best = gamma_new


            # Check whether target has been reached
            if gamma_best >= target_lin:
                return k


    return K_MAX


# ============================================================
# Simulation for one M
# ============================================================

def simulate_for_M(
    M,
    target_dB_list
):
    """
    Returns average iterations for each
    target SNR value.
    """

    avg_iters = []

    # Convert dB targets to linear scale
    target_lin_list = [
        10 ** (dB / 10)
        for dB in target_dB_list
    ]


    # ========================================================
    # Loop over target SNR values
    # ========================================================

    for target_lin in target_lin_list:

        iters_all = []


        # ====================================================
        # Independent channel realisations
        # ====================================================

        for _ in range(NUM_TRIALS):

            # Generate random channels

            h_sr = np.sqrt(VAR / 2) * (
                np.random.randn()
                + 1j * np.random.randn()
            )

            h_st = np.sqrt(VAR / 2) * (
                np.random.randn()
                + 1j * np.random.randn()
            )

            h_tr = np.sqrt(VAR / 2) * (
                np.random.randn()
                + 1j * np.random.randn()
            )

            h_si = np.sqrt(VAR / 2) * (
                np.random.randn(M)
                + 1j * np.random.randn(M)
            )

            h_ir = np.sqrt(VAR / 2) * (
                np.random.randn(M)
                + 1j * np.random.randn(M)
            )


            # Direct path + tag contribution

            direct = (
                h_sr
                + h_st
                * X
                * np.exp(1j * THETA)
                * h_tr
            )


            # =================================================
            # Maximum theoretically achievable SNR
            # =================================================

            max_snr = compute_max_snr(
                direct,
                h_ir,
                h_si,
                ETA,
                RHO
            )


            # Target cannot be achieved
            if max_snr < target_lin:
                continue


            # =================================================
            # Run SBF
            # =================================================

            iters = run_sbf_static(
                M,
                target_lin,
                direct,
                h_ir,
                h_si
            )


            # Only count successful trials
            if iters < K_MAX:
                iters_all.append(iters)


        # ====================================================
        # Average iterations
        # ====================================================

        if iters_all:

            avg_iters.append(
                np.mean(iters_all)
            )

        else:

            avg_iters.append(
                np.nan
            )


    return avg_iters


# ============================================================
# Main execution
# ============================================================

# Target SNR values from 0 to 25 dB
target_dB = np.arange(
    0,
    26,
    1
)


# Store results
all_avg_iters = {}


# ============================================================
# Run simulations
# ============================================================

for M in M_values:

    print(
        f"Simulating M = {M} ..."
    )

    all_avg_iters[M] = simulate_for_M(
        M,
        target_dB
    )


print("\nSimulation completed.")


# ============================================================
# Print average iterations at 20 dB
# ============================================================

idx_20dB = np.where(
    target_dB == 20
)[0][0]


print(
    "\n=== Average iterations at 20 dB SNR ==="
)


for M in M_values:

    val = all_avg_iters[M][idx_20dB]

    if np.isnan(val):

        print(
            f"M = {M:3d} : Not reached"
        )

    else:

        print(
            f"M = {M:3d} : "
            f"{val:.2f} iterations"
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
# Original color and marker scheme
#
# M = 64   -> red + triangle
# M = 128  -> blue + square
# M = 256  -> green + circle
# ============================================================

style_map = {

    64: {
        'color': 'red',
        'marker': '^'
    },

    128: {
        'color': 'blue',
        'marker': 's'
    },

    256: {
        'color': 'green',
        'marker': 'o'
    }
}


# ============================================================
# Plot each M
# ============================================================

for M in M_values:

    # Remove NaN values
    valid = ~np.isnan(
        all_avg_iters[M]
    )

    dB_vals = target_dB[valid]

    iter_vals = np.array(
        all_avg_iters[M]
    )[valid]

    style = style_map[M]


    ax.plot(
        dB_vals,
        iter_vals,

        # Original color
        color=style['color'],

        # Original marker
        marker=style['marker'],

        # Original line width
        linewidth=2.0,

        # Original marker size
        markersize=6,

        # Marker every second point
        markevery=2,

        # Legend
        label=f'M = {M}'
    )


# ============================================================
# X-axis label
# ============================================================

ax.set_xlabel(
    r'Target Received SNR $\gamma$ (dB)',
    fontsize=11
)


# ============================================================
# Y-axis label
# ============================================================

ax.set_ylabel(
    'Average Number of Iterations',
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


# White opaque legend background
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
    'figure5_M64_128_256.png'
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
