import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Parameters
# ============================================================

M = 128                 # number of IRS elements
K = 100                 # number of iterations
rho = 1.0               # average transmit SNR
eta = 1.0               # reflection coefficient magnitude
theta = 0.0
x = 1.0

# Channel variance (0 dB)
var = 1.0


# ============================================================
# Generate static channel coefficients
# ============================================================

h_sr = np.sqrt(var / 2) * (
    np.random.randn() + 1j * np.random.randn()
)

h_st = np.sqrt(var / 2) * (
    np.random.randn() + 1j * np.random.randn()
)

h_tr = np.sqrt(var / 2) * (
    np.random.randn() + 1j * np.random.randn()
)

h_si = np.sqrt(var / 2) * (
    np.random.randn(M) + 1j * np.random.randn(M)
)

h_ir = np.sqrt(var / 2) * (
    np.random.randn(M) + 1j * np.random.randn(M)
)


# ============================================================
# Direct path + tag-1 contribution
# ============================================================

direct = (
    h_sr
    + h_st * x * np.exp(1j * theta) * h_tr
)


# ============================================================
# Perturbation range
# ============================================================

pert_low = -np.pi / 20
pert_high = np.pi / 20


# ============================================================
# SBF simulation
# ============================================================

def simulate_sbf(initial_phases):

    phi_best = initial_phases.copy()

    # Initial IRS contribution
    sum_irs = np.sum(
        h_ir
        * np.exp(1j * phi_best)
        * h_si
    )

    # Initial received signal
    G = (
        direct
        + eta * sum_irs * x
    )

    # Initial received SNR
    gamma_best = rho * np.abs(G) ** 2

    gamma_history = np.zeros(K)

    gamma_history[0] = gamma_best


    # ========================================================
    # Iterative SBF optimization
    # ========================================================

    for k in range(1, K):

        # Generate random perturbation
        eps = np.random.uniform(
            pert_low,
            pert_high,
            M
        )

        # New phase vector
        phi_new = phi_best + eps

        # New IRS contribution
        sum_irs_new = np.sum(
            h_ir
            * np.exp(1j * phi_new)
            * h_si
        )

        # New received signal
        G_new = (
            direct
            + eta * sum_irs_new * x
        )

        # New received SNR
        gamma_new = rho * np.abs(G_new) ** 2


        # Accept new phase configuration
        # only when SNR improves
        if gamma_new > gamma_best:

            phi_best = phi_new
            gamma_best = gamma_new


        # Store best SNR
        gamma_history[k] = gamma_best


    return gamma_history


# ============================================================
# Two independent instances
# Different random initial phase configurations
# ============================================================

initial_phases_1 = np.random.uniform(
    0,
    2 * np.pi,
    M
)

initial_phases_2 = np.random.uniform(
    0,
    2 * np.pi,
    M
)


# ============================================================
# Run SBF for both instances
# ============================================================

gamma_hist_1 = simulate_sbf(
    initial_phases_1
)

gamma_hist_2 = simulate_sbf(
    initial_phases_2
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

    # Axes line width
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
# Plot Instance 1
# Original color: BLUE
# ============================================================

ax.plot(
    gamma_hist_1,
    label='Instance 1',
    linewidth=2.0,
    color='blue'
)


# ============================================================
# Plot Instance 2
# Original color: RED
# ============================================================

ax.plot(
    gamma_hist_2,
    label='Instance 2',
    linewidth=2.0,
    color='red'
)


# ============================================================
# Axis labels
# ============================================================

ax.set_xlabel(
    'Number of iterations ($k$)',
    fontsize=11
)

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

output_file = 'figure3_M128_K100.png'

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
print(f"File       : {output_file}")
print(f"Format     : {img.format}")
print(f"Image size : {img.size}")
print(f"Color mode : {img.mode}")
print(
    f"Resolution : "
    f"{img.info.get('dpi', 'Not stored')}"
)
print("============================================")


# ============================================================
# Display figure
# ============================================================

plt.show()
