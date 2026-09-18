# -----------------------------------------------------------------------------
# Gamma approximation validation - Fig. 3
#
# This simulation examines the statistical distribution of the post-selection
# composite channel energy under H1 after SBF phase adjustment.
#
# The normalized channel energy U1/sigma_hsr^2 is evaluated for M = 4 and
# M = 128 IRS elements using 50,000 simulated samples. A Gamma distribution
# is fitted using moment matching and compared with the empirical samples
# through histograms and Q-Q plots.
#
# The purpose of this experiment is to verify the Gamma approximation used
# in the design of the noncoherent detectors.
# -----------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from scipy.stats import kstest

# ============================================================
# 1. OUTPUT / REPRODUCIBILITY
# ============================================================

if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()

OUT = BASE_DIR / "gamma_validation_M4_M128_1x4_paper_output"
OUT.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 50_000
BATCH_SIZE = 2_000
M_VALUES = [4, 128]

# Keep the previously used validation seeds for reproducibility.
SEEDS = {
    4: 1042,
    128: 2042,
}

# ============================================================
# 2. COMMON PHYSICAL PARAMETERS
# ============================================================

x = 1.0
beta = 0.0
eta = 0.8

carrier_frequency_GHz = 2.4

source_to_receiver_distance_m = 20.0
source_to_conventional_tag_distance_m = 10.0
conventional_tag_to_receiver_distance_m = 10.0
source_to_irs_distance_m = 8.0
irs_to_receiver_distance_m = 12.0

# SBF settings
SBF_ITERATIONS = 100
SBF_EPS_LOW = -np.pi / 20.0
SBF_EPS_HIGH = np.pi / 20.0

# ============================================================
# 3. 3GPP InH PATH LOSS
# ============================================================

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


PL_sr = indoor_nlos_path_loss_dB(
    source_to_receiver_distance_m,
    carrier_frequency_GHz,
)

PL_st = indoor_nlos_path_loss_dB(
    source_to_conventional_tag_distance_m,
    carrier_frequency_GHz,
)

PL_tr = indoor_nlos_path_loss_dB(
    conventional_tag_to_receiver_distance_m,
    carrier_frequency_GHz,
)

PL_si = indoor_los_path_loss_dB(
    source_to_irs_distance_m,
    carrier_frequency_GHz,
)

PL_ir = indoor_los_path_loss_dB(
    irs_to_receiver_distance_m,
    carrier_frequency_GHz,
)

sigma_hsr2 = 10.0 ** (-PL_sr / 10.0)
sigma_hst2 = 10.0 ** (-PL_st / 10.0)
sigma_htr2 = 10.0 ** (-PL_tr / 10.0)
sigma_hsi2 = 10.0 ** (-PL_si / 10.0)
sigma_hir2 = 10.0 ** (-PL_ir / 10.0)

# Normalize only for numerical readability / presentation.
u_ref = sigma_hsr2

# ============================================================
# 4. RANDOM CHANNEL GENERATOR
# ============================================================

def cn(rng, shape, variance):
    return np.sqrt(variance / 2.0) * (
        rng.standard_normal(shape)
        + 1j * rng.standard_normal(shape)
    )

# ============================================================
# 5. POST-SBF CHANNEL ENERGY GENERATION
# ============================================================

def generate_post_sbf_batch(rng, batch_size, M):
    h_sr = cn(rng, batch_size, sigma_hsr2)
    h_st = cn(rng, batch_size, sigma_hst2)
    h_tr = cn(rng, batch_size, sigma_htr2)

    h_si = cn(rng, (batch_size, M), sigma_hsi2)
    h_ir = cn(rng, (batch_size, M), sigma_hir2)

    direct_term = (
        h_sr
        + h_st * x * np.exp(1j * beta) * h_tr
    )

    cascaded_term = h_ir * h_si

    # Random phase initialization
    phi = rng.uniform(
        -np.pi,
        np.pi,
        size=(batch_size, M),
    )

    phase_state = np.exp(1j * phi)

    G = (
        direct_term
        + eta
        * np.sum(
            cascaded_term * phase_state,
            axis=1,
        )
        * x
    )

    gain_best = np.abs(G) ** 2

    # SBF phase-selection iterations
    for _ in range(SBF_ITERATIONS):
        epsilon = rng.uniform(
            SBF_EPS_LOW,
            SBF_EPS_HIGH,
            size=(batch_size, M),
        )

        trial_phase_state = (
            phase_state * np.exp(1j * epsilon)
        )

        G_trial = (
            direct_term
            + eta
            * np.sum(
                cascaded_term * trial_phase_state,
                axis=1,
            )
            * x
        )

        gain_trial = np.abs(G_trial) ** 2

        improve = gain_trial > gain_best

        gain_best[improve] = gain_trial[improve]
        phase_state[improve] = trial_phase_state[improve]

    U1 = gain_best
    U1_normalized = U1 / u_ref

    return U1_normalized


def generate_post_sbf_samples(M, total_samples, seed):
    rng = np.random.default_rng(seed)

    collected = []
    completed = 0

    while completed < total_samples:
        n = min(
            BATCH_SIZE,
            total_samples - completed,
        )

        collected.append(
            generate_post_sbf_batch(
                rng,
                n,
                M,
            )
        )

        completed += n

    return np.concatenate(collected)

# ============================================================
# 6. MOMENT-MATCHED GAMMA FIT
# ============================================================

def fit_gamma_moments(samples):
    mean_value = float(np.mean(samples))
    variance_value = float(np.var(samples, ddof=1))

    alpha_g = mean_value**2 / variance_value
    theta_g = variance_value / mean_value

    return (
        mean_value,
        variance_value,
        alpha_g,
        theta_g,
    )

# ============================================================
# 7. GENERATE / FIT M=4 AND M=128
# ============================================================

results = {}

for M in M_VALUES:
    print(
        f"Generating {NUM_SAMPLES:,} post-SBF realizations for M={M}..."
    )

    samples = generate_post_sbf_samples(
        M,
        NUM_SAMPLES,
        SEEDS[M],
    )

    (
        mean_u1,
        var_u1,
        alpha_g,
        theta_g,
    ) = fit_gamma_moments(samples)

    ks_D, ks_p = kstest(
        samples,
        "gamma",
        args=(
            alpha_g,
            0.0,
            theta_g,
        ),
    )

    qq_probabilities = np.linspace(
        0.01,
        0.99,
        99,
    )

    empirical_quantiles = np.quantile(
        samples,
        qq_probabilities,
    )

    theoretical_quantiles = gamma_dist.ppf(
        qq_probabilities,
        a=alpha_g,
        loc=0.0,
        scale=theta_g,
    )

    qq_rmse = float(
        np.sqrt(
            np.mean(
                (
                    empirical_quantiles
                    - theoretical_quantiles
                ) ** 2
            )
        )
    )

    results[M] = {
        "samples": samples,
        "mean": mean_u1,
        "variance": var_u1,
        "alpha_g": alpha_g,
        "theta_g": theta_g,
        "ks_D": float(ks_D),
        "ks_p": float(ks_p),
        "qq_probabilities": qq_probabilities,
        "empirical_quantiles": empirical_quantiles,
        "theoretical_quantiles": theoretical_quantiles,
        "qq_rmse": qq_rmse,
    }

# ============================================================
# 8. PUBLICATION PLOT SETTINGS
# ============================================================

# Four panels arranged as a 2 x 2 publication figure.
FIGSIZE = (10.0, 8.0)

# Keep the same color combination as the original figure.
HIST_COLOR = "#1f77b4"
HIST_ALPHA = 0.55
HIST_EDGE_COLOR = "#1f77b4"
HIST_EDGE_WIDTH = 0.25

# Fitted Gamma curve
GAMMA_COLOR = "green"
GAMMA_LINEWIDTH = 2.0

# Q-Q plot
QQ_POINT_COLOR = "#1f77b4"
QQ_POINT_SIZE = 14
QQ_REF_COLOR = "#1f77b4"
QQ_REF_STYLE = "--"
QQ_REF_WIDTH = 1.5

GRID_STYLE = "--"
GRID_ALPHA = 0.32
GRID_WIDTH = 0.50

# Slightly larger fonts because the panels now have more space
LABEL_FONT = 11
TICK_FONT = 10
LEGEND_FONT = 9
TEXT_FONT = 9
PANEL_FONT = 11

TEXT_BOX_STYLE = dict(
    facecolor="white",
    edgecolor="black",
    boxstyle="square,pad=0.22",
    alpha=0.92,
    linewidth=0.7,
)

OUTPUT_FILE_NAME = "gamma_validation_M4_M128_2x2_paper.png"


# ============================================================
# 9. CREATE 2 x 2 PUBLICATION FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8)
)

axes = axes.flatten()
# Convert 2-D axes array into a simple sequence
axes = axes.flatten()

# Tighten horizontal gap (side-by-side look); vertical gap is set
# generously below because the panel labels are placed under each
# row and must clear the row underneath.
fig.subplots_adjust(wspace=0.06, hspace=0.30)


# ------------------------------------------------------------
# Panel order:
#
# (a) M = 4   : PDF
# (b) M = 4   : Q-Q
# (c) M = 128 : PDF
# (d) M = 128 : Q-Q
# ------------------------------------------------------------

subplot_order = [
    (4, "pdf"),
    (4, "qq"),
    (128, "pdf"),
    (128, "qq"),
]


panel_labels = {
    (4, "pdf"): r"(a)",
    (4, "qq"): r"(b)",
    (128, "pdf"): r"(c)",
    (128, "qq"): r"(d)",
}


# ============================================================
# CREATE PANELS
# ============================================================

for ax, (M, plot_type) in zip(
    axes,
    subplot_order,
):

    samples = results[M]["samples"]
    alpha_g = results[M]["alpha_g"]
    theta_g = results[M]["theta_g"]
    ks_D = results[M]["ks_D"]
    qq_rmse = results[M]["qq_rmse"]


    # ========================================================
    # PDF / HISTOGRAM PANEL
    # ========================================================

    if plot_type == "pdf":

        x_max = float(
            np.quantile(
                samples,
                0.997,
            )
        )

        x_grid = np.linspace(
            0.0,
            x_max,
            1200,
        )

        gamma_pdf = gamma_dist.pdf(
            x_grid,
            a=alpha_g,
            loc=0.0,
            scale=theta_g,
        )


        # ----------------------------------------------------
        # Empirical histogram
        # ----------------------------------------------------

        ax.hist(
            samples,
            bins=80,
            density=True,
            color=HIST_COLOR,
            alpha=HIST_ALPHA,
            edgecolor=HIST_EDGE_COLOR,
            linewidth=HIST_EDGE_WIDTH,
            label="Empirical histogram",
        )


        # ----------------------------------------------------
        # Gamma PDF
        # ----------------------------------------------------

        ax.plot(
            x_grid,
            gamma_pdf,
            color=GAMMA_COLOR,
            linewidth=GAMMA_LINEWIDTH,
            label="Moment-matched Gamma PDF",
        )


        # ----------------------------------------------------
        # Axis labels
        # ----------------------------------------------------

        ax.set_xlabel(
            r"$U_1/\sigma_{h_{sr}}^2$",
            fontsize=LABEL_FONT,
        )

        ax.set_ylabel(
            "Probability Density",
            fontsize=LABEL_FONT,
        )

        ax.set_xlim(
            0.0,
            x_max,
        )


        # ----------------------------------------------------
        # Statistical information
        # ----------------------------------------------------

        text_string = (
            rf"$\alpha_g={alpha_g:.4f}$"
            "\n"
            rf"$\theta_g={theta_g:.4f}$"
            "\n"
            rf"$D_{{\mathrm{{KS}}}}={ks_D:.4f}$"
        )

        ax.text(
            0.97,
            0.96,
            text_string,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=TEXT_FONT,
            bbox=TEXT_BOX_STYLE,
        )


        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        ax.legend(
            loc="center right",
            frameon=True,
            edgecolor="black",
            framealpha=1.0,
            fontsize=LEGEND_FONT,
        )


    # ========================================================
    # Q-Q PANEL
    # ========================================================

    else:

        empirical_quantiles = results[M][
            "empirical_quantiles"
        ]

        theoretical_quantiles = results[M][
            "theoretical_quantiles"
        ]


        lower = float(
            min(
                empirical_quantiles.min(),
                theoretical_quantiles.min(),
            )
        )

        upper = float(
            max(
                empirical_quantiles.max(),
                theoretical_quantiles.max(),
            )
        )


        # ----------------------------------------------------
        # Empirical Q-Q points
        # ----------------------------------------------------

        ax.scatter(
            theoretical_quantiles,
            empirical_quantiles,
            s=QQ_POINT_SIZE,
            color=QQ_POINT_COLOR,
            label="Empirical quantiles",
        )


        # ----------------------------------------------------
        # 45-degree reference
        # ----------------------------------------------------

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle=QQ_REF_STYLE,
            color=QQ_REF_COLOR,
            linewidth=QQ_REF_WIDTH,
            label=r"$45^\circ$ reference",
        )


        # ----------------------------------------------------
        # Axis labels
        # ----------------------------------------------------

        ax.set_xlabel(
            "Theoretical Gamma Quantiles",
            fontsize=LABEL_FONT,
        )

        ax.set_ylabel(
            "Empirical Quantiles",
            fontsize=LABEL_FONT,
        )

        ax.set_xlim(
            lower,
            upper,
        )

        ax.set_ylim(
            lower,
            upper,
        )


        # Keep Q-Q panels approximately square
        ax.set_aspect(
            "equal",
            adjustable="box",
        )


        # ----------------------------------------------------
        # Statistical information
        # ----------------------------------------------------

        text_string = (
            rf"$D_{{\mathrm{{KS}}}}={ks_D:.4f}$"
            "\n"
            rf"Q--Q RMSE$={qq_rmse:.4f}$"
        )

        ax.text(
            0.04,
            0.96,
            text_string,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=TEXT_FONT,
            bbox=TEXT_BOX_STYLE,
        )


        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        ax.legend(
            loc="lower right",
            frameon=True,
            edgecolor="black",
            framealpha=1.0,
            fontsize=LEGEND_FONT,
        )


    # ========================================================
    # COMMON PANEL FORMATTING
    # ========================================================

    ax.grid(
        True,
        linestyle=GRID_STYLE,
        linewidth=GRID_WIDTH,
        alpha=GRID_ALPHA,
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONT,
    )


# ============================================================
# PANEL LABELS — placed after layout is finalized. We use each
# axes' TIGHT bounding box (which includes its xlabel and tick
# labels, not just the plot rectangle) so that panels whose
# xlabel is a tall LaTeX fraction (a, c) get the same real
# clearance as panels with a plain one-line xlabel (b, d). The
# aspect="equal" Q-Q panels are also handled correctly this way
# since the tight bbox reflects their final, shrunk box.
# ============================================================

fig.canvas.draw()
renderer = fig.canvas.get_renderer()

LABEL_GAP_FIGFRAC = 0.012  # small extra gap below the tight bbox (already includes xlabel)

for ax, (M, plot_type) in zip(axes, subplot_order):
    tight_bbox = ax.get_tightbbox(renderer)
    tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())

    x_center = (tight_bbox_fig.x0 + tight_bbox_fig.x1) / 2.0
    y_bottom = tight_bbox_fig.y0

    fig.text(
        x_center,
        y_bottom - LABEL_GAP_FIGFRAC,
        panel_labels[(M, plot_type)],
        ha="center",
        va="top",
        fontsize=PANEL_FONT,
    )


# ============================================================
# 10. SAVE FIGURE
# ============================================================

png_path = OUT / OUTPUT_FILE_NAME

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

print("Figure saved to:")
print(png_path)
