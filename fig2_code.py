import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import i0e, i1e
import pandas as pd
from pathlib import Path

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
    "legend.fontsize": 8,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,     # Embed TrueType fonts in PDF
    "ps.fonttype": 42,
})


# ============================================================
# PARAMETERS
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

M = 128              # Number of IRS elements
K = 100              # SBF iterations
num_runs = 800       # Monte-Carlo runs

eta = 1.0
delta = np.pi / 10

x_symbol = 1.0
beta = 0.0

sigma_h_ir = 1.0
sigma_h_si = 1.0

alpha_0 = (
    np.pi / 4.0
    * eta
    * sigma_h_ir
    * sigma_h_si
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def cscg(size=None):
    return (
        rng.standard_normal(size)
        + 1j * rng.standard_normal(size)
    ) / np.sqrt(2.0)


def bessel_ratio(mu):
    return i1e(mu) / i0e(mu)


def solve_mu(target):
    if target <= 0.0:
        return 0.0

    if target >= 1.0:
        raise ValueError(f"Received target = {target}")

    lower = 0.0
    upper = 1.0

    while bessel_ratio(upper) < target:
        upper *= 2.0

    for _ in range(150):
        mid = 0.5 * (lower + upper)
        if bessel_ratio(mid) < target:
            lower = mid
        else:
            upper = mid

    return 0.5 * (lower + upper)


# ============================================================
# MONTE-CARLO SIMULATION
# ============================================================

theta_runs = []
a_runs = []
eq30_moments = []

for run in range(num_runs):

    h_ir = cscg(M)
    h_si = cscg(M)

    h_sr = cscg()
    h_st = cscg()
    h_tr = cscg()

    C = (
        h_sr
        + h_st
        * x_symbol
        * np.exp(1j * beta)
        * h_tr
    )

    a = eta * np.abs(h_ir) * np.abs(h_si)
    channel_phase = np.angle(h_ir) + np.angle(h_si)
    phi = rng.uniform(-np.pi, np.pi, M)
    xi = phi + channel_phase

    S = np.sum(a * np.exp(1j * xi))
    best_value = np.abs(S + C)

    for k in range(K):
        epsilon = rng.choice([-delta, delta], size=M)
        phi_trial = phi + epsilon
        xi_trial = phi_trial + channel_phase

        S_trial = np.sum(a * np.exp(1j * xi_trial))
        trial_value = np.abs(S_trial + C)

        if trial_value > best_value:
            phi = phi_trial
            best_value = trial_value

    xi_best = phi + channel_phase
    S_best = np.sum(a * np.exp(1j * xi_best))
    phi0 = np.angle(S_best)

    theta = xi_best - phi0
    theta = np.angle(np.exp(1j * theta))

    theta_runs.append(theta)
    a_runs.append(a)

    C_rot = C * np.exp(-1j * phi0)
    radicand = max(best_value**2 - np.imag(C_rot)**2, 0.0)

    moment_eq30 = (np.sqrt(radicand) - np.real(C_rot)) / (alpha_0 * M)
    eq30_moments.append(moment_eq30)


# ============================================================
# COMBINE MC SAMPLES
# ============================================================

theta_all = np.concatenate(theta_runs)
a_all = np.concatenate(a_runs)
eq30_moments = np.asarray(eq30_moments)

weights_all = a_all / np.sum(a_all)
analytical_moment = np.mean(eq30_moments)

mu = solve_mu(analytical_moment)


# ============================================================
# ANALYTICAL PDF DISTRIBUTIONS
# ============================================================

x = np.linspace(-np.pi, np.pi, 1000)

expo_cosine = (
    np.exp(mu * (np.cos(x) - 1.0))
    / (2.0 * np.pi * i0e(mu))
)

sigma_gaussian = np.sqrt(np.sum(weights_all * theta_all**2))
gaussian = (
    1.0 / (np.sqrt(2.0 * np.pi) * sigma_gaussian)
    * np.exp(-(x**2) / (2.0 * sigma_gaussian**2))
)

b_laplace = np.sum(weights_all * np.abs(theta_all))
laplacian = (
    1.0 / (2.0 * b_laplace)
    * np.exp(-np.abs(x) / b_laplace)
)


# ============================================================
# IEEE SINGLE-COLUMN FIGURE GENERATION (3.4 in x 3.1 in)
# ============================================================

fig, ax = plt.subplots(figsize=(3.5, 2.5))

# Histogram
ax.hist(
    theta_all,
    bins=36,
    weights=weights_all,
    density=True,
    alpha=0.62,
    edgecolor="white",
    linewidth=0.35,
    label="Empirical"
)

# Von Mises (Exponential-Cosine)
ax.plot(
    x,
    expo_cosine,
    color="green",
    linewidth=1.8,
    label="von Mises"
)

# Gaussian Reference
ax.plot(
    x,
    gaussian,
    color="red",
    linestyle="--",
    linewidth=1.4,
    label="Gaussian"
)

# Laplacian Reference
ax.plot(
    x,
    laplacian,
    color="magenta",
    linestyle="-.",
    linewidth=1.4,
    label="Laplacian"
)

# Bold Axis Labels
ax.set_xlabel(r"Relative Effective Phase $\theta_m$ (rad)", fontweight='bold')
ax.set_ylabel(r"$f_{\theta,w}(\theta_m)$", fontweight='bold')

ax.set_xlim(-np.pi, np.pi)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

# Bold Tick Labels (Axes Values)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight("bold")

# Bold Legend Box and Labels
legend = ax.legend(
    loc="upper left",
    fontsize=8.5,
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

output_dir = Path.cwd() / "figure2_bold_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

pdf_path = output_dir / f"Figure_2_weighted_M{M}_IEEE_bold.pdf"
png_path = output_dir / f"Figure_2_weighted_M{M}_IEEE_bold.png"

fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")

print(f"Bold IEEE Vector PDF saved to: {pdf_path}")
plt.show()
