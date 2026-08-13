import numpy as np
import matplotlib.pyplot as plt

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

pert_low = -np.pi / 20
pert_high = np.pi / 20

# Number of independent channel realisations
num_realisations = 100

# Feedback bit-flip probabilities
p_e_values = [0.0, 0.05, 0.1]


# ============================================================
# Helper function: Generate channel
# ============================================================
def generate_channel(M, var):

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

    direct = (
        h_sr
        + h_st * x * np.exp(1j * theta) * h_tr
    )

    return direct, h_ir, h_si


# ============================================================
# SBF simulation with feedback errors
# ============================================================
def simulate_sbf_with_errors(
    direct,
    h_ir,
    h_si,
    p_e,
    iterations=K
):

    # Initial random IRS phase configuration
    phi = np.random.uniform(
        0,
        2 * np.pi,
        M
    )

    # Initial IRS contribution
    sum_irs = np.sum(
        h_ir
        * np.exp(1j * phi)
        * h_si
    )

    # Initial received signal
    G = (
        direct
        + eta * sum_irs * x
    )

    # Initial SNR
    gamma_best = rho * np.abs(G) ** 2

    gamma_history = np.zeros(iterations)

    gamma_history[0] = gamma_best


    # ========================================================
    # Iterative SBF procedure
    # ========================================================
    for k in range(1, iterations):

        # Random phase perturbation
        eps = np.random.uniform(
            pert_low,
            pert_high,
            M
        )

        # New phase vector
        phi_new = phi + eps

        # IRS contribution with perturbed phase
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

        # New SNR
        gamma_new = rho * np.abs(G_new) ** 2

        # True improvement
        improvement = gamma_new > gamma_best

        # ====================================================
        # Feedback bit-flip error
        # ====================================================
        if np.random.rand() < p_e:
            improvement = not improvement

        # ====================================================
        # Update according to received feedback
        # ====================================================
        if improvement:

            phi = phi_new
            gamma_best = gamma_new

        # Store current accepted SNR
        gamma_history[k] = gamma_best

    return gamma_history


# ============================================================
# Average trajectory over channel realisations
# ============================================================
def average_trajectory(
    p_e,
    num_realisations
):

    all_histories = []

    for _ in range(num_realisations):

        # Generate independent channel
        direct, h_ir, h_si = generate_channel(
            M,
            var
        )

        # Run SBF
        hist = simulate_sbf_with_errors(
            direct,
            h_ir,
            h_si,
            p_e
        )

        all_histories.append(hist)


    # Convert to NumPy array
    all_histories = np.asarray(
        all_histories
    )


    # Average trajectory
    avg = np.mean(
        all_histories,
        axis=0
    )


    # Standard deviation
    std = np.std(
        all_histories,
        axis=0
    )

    return avg, std


# ============================================================
# Run simulations
# ============================================================
print("Simulating SBF with feedback errors...")
print()

trajectories = {}

for p_e in p_e_values:

    print(
        f"  p_e = {p_e:.2f}"
    )

    avg, std = average_trajectory(
        p_e,
        num_realisations
    )

    trajectories[p_e] = (
        avg,
        std
    )

print()
print("Simulation completed.")


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

    # Figure title
    'axes.titlesize': 11,

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
    figsize=(7.0, 4.5)
)


# ============================================================
# Requested colors
# ============================================================
colors = [
    'green',     # p_e = 0.00
    'blue',      # p_e = 0.05
    'red'        # p_e = 0.10
]


# ============================================================
# Line styles
# ============================================================
line_styles = [
    '-',         # p_e = 0.00
    '--',        # p_e = 0.05
    '-.'         # p_e = 0.10
]


# ============================================================
# Requested markers
# ============================================================
markers = [
    'o',         # Circle
    's',         # Square
    '^'          # Triangle
]


# ============================================================
# Plot trajectories
# ============================================================
for idx, p_e in enumerate(p_e_values):

    avg, std = trajectories[p_e]

    k_vals = np.arange(
        len(avg)
    )

    ax.plot(
        k_vals,
        avg,

        # Color
        color=colors[idx],

        # Line style
        linestyle=line_styles[idx],

        # Line width
        linewidth=1.8,

        # Marker
        marker=markers[idx],

        # Marker size
        markersize=4.0,

        # Marker every 10 iterations
        markevery=10,

        # Marker edge
        markeredgewidth=0.8,

        # Legend label
        label=rf'$p_e = {p_e:.2f}$'
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
    r'Best received SNR ($\gamma$)',
    fontsize=11
)


# ============================================================
# Figure title
# ============================================================



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
    linestyle=':',
    linewidth=0.6,
    alpha=0.5
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

# White opaque legend
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
    'sbf_feedback_error_convergence_M128.png'
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
from PIL import Image

img = Image.open(output_file)

print()
print("============================================")
print("PNG figure successfully saved")
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
