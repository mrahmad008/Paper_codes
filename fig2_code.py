# -----------------------------------------------------------------------------
# Validation of the relative-phase distribution - Fig. 2
#
# This script generates Fig. 2 of the manuscript. It compares the
# amplitude-weighted empirical distribution of the relative effective IRS
# phases, theta_m, obtained from the SBF simulation with the analytical
# phase-distribution model derived for the static-channel case.
#
# The analytical distribution is the zero-mean von Mises
# (exponential-cosine) distribution
#
#   f_theta(theta) = exp(mu*cos(theta)) / (2*pi*I0(mu)),
#
# where the concentration parameter mu is determined from the corrected
# amplitude-weighted phase-moment constraint used in the manuscript.
#
# The constraint includes the realized complex non-IRS component C, so the
# analytical distribution accounts for both the IRS-assisted component and
# the non-IRS contribution to the received signal.
#
# The empirical phase samples are obtained from the relative effective phases
# of the IRS elements after SBF phase adjustment. Their contribution is
# weighted by the corresponding cascaded-channel amplitudes.
#
# Gaussian and Laplacian distributions are also included as reference
# approximations. The purpose of this figure is to compare these distributions
# with the empirical phase data and assess the accuracy of the large-M
# statistical approximation used in the convergence analysis.
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0e, i1e


# ============================================================
# PARAMETERS
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

M = 128              # Change to 64, 128, 256, 512, etc.
K = 100
num_runs = 800

eta = 1.0
delta = np.pi / 10

x_symbol = 1.0
beta = 0.0

# h ~ CN(0,1)
sigma_h_ir = 1.0
sigma_h_si = 1.0

# alpha_0 = E[a_m]
# For h_ir,h_si ~ CN(0,1):
# E[|h|] = sqrt(pi)/2
# Therefore:
# E[|h_ir||h_si|] = pi/4
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
    """
    Generate circularly symmetric complex Gaussian samples
    distributed as CN(0,1).
    """
    return (
        rng.standard_normal(size)
        + 1j * rng.standard_normal(size)
    ) / np.sqrt(2.0)


def bessel_ratio(mu):
    """
    Numerically stable evaluation of I1(mu)/I0(mu).
    """
    return i1e(mu) / i0e(mu)


def solve_mu(target):
    """
    Solve

        I1(mu) / I0(mu) = target

    for mu >= 0 using bisection.
    """

    if target <= 0.0:
        return 0.0

    if target >= 1.0:
        raise ValueError(
            "Bessel-ratio target must satisfy 0 <= target < 1. "
            f"Received target = {target}"
        )

    lower = 0.0
    upper = 1.0

    # Find a sufficiently large upper bound
    while bessel_ratio(upper) < target:
        upper *= 2.0

    # Bisection
    for _ in range(150):

        mid = 0.5 * (lower + upper)

        if bessel_ratio(mid) < target:
            lower = mid
        else:
            upper = mid

    return 0.5 * (lower + upper)


# ============================================================
# STORAGE
# ============================================================

theta_runs = []
a_runs = []

eq30_moments = []


# ============================================================
# MONTE-CARLO SIMULATION
# ============================================================

for run in range(num_runs):

    # --------------------------------------------------------
    # Channel realizations
    # --------------------------------------------------------

    h_ir = cscg(M)
    h_si = cscg(M)

    h_sr = cscg()
    h_st = cscg()
    h_tr = cscg()

    # --------------------------------------------------------
    # Non-IRS received component
    #
    # C = h_sr + h_st x(n) e^{j beta} h_tr
    # --------------------------------------------------------

    C = (
        h_sr
        + h_st
        * x_symbol
        * np.exp(1j * beta)
        * h_tr
    )

    # --------------------------------------------------------
    # IRS cascaded-channel magnitude
    #
    # a_m = eta |h_ir,m| |h_si,m|
    # --------------------------------------------------------

    a = (
        eta
        * np.abs(h_ir)
        * np.abs(h_si)
    )

    # --------------------------------------------------------
    # Cascaded-channel phase
    # --------------------------------------------------------

    channel_phase = (
        np.angle(h_ir)
        + np.angle(h_si)
    )

    # --------------------------------------------------------
    # Initial IRS phase shifts
    # --------------------------------------------------------

    phi = rng.uniform(
        -np.pi,
        np.pi,
        M
    )

    # --------------------------------------------------------
    # Effective received phase
    #
    # xi_m =
    # phi_m
    # + angle(h_ir,m)
    # + angle(h_si,m)
    # --------------------------------------------------------

    xi = (
        phi
        + channel_phase
    )

    # IRS-assisted component
    S = np.sum(
        a
        * np.exp(1j * xi)
    )

    # Complete composite magnitude
    best_value = np.abs(
        S + C
    )

    # ========================================================
    # SBF ITERATIONS
    # ========================================================

    for k in range(K):

        # Random phase perturbation for every IRS element
        epsilon = rng.choice(
            [-delta, delta],
            size=M
        )

        phi_trial = (
            phi
            + epsilon
        )

        xi_trial = (
            phi_trial
            + channel_phase
        )

        S_trial = np.sum(
            a
            * np.exp(1j * xi_trial)
        )

        trial_value = np.abs(
            S_trial + C
        )

        # Accept perturbation only when received magnitude improves
        if trial_value > best_value:

            phi = phi_trial
            best_value = trial_value


    # ========================================================
    # FINAL RELATIVE EFFECTIVE PHASE
    # ========================================================

    xi_best = (
        phi
        + channel_phase
    )

    S_best = np.sum(
        a
        * np.exp(1j * xi_best)
    )

    # Aggregate phase of the IRS-assisted component
    phi0 = np.angle(
        S_best
    )

    # Relative effective phase
    theta = (
        xi_best
        - phi0
    )

    # Wrap theta to [-pi, pi)
    theta = np.angle(
        np.exp(1j * theta)
    )

    theta_runs.append(theta)
    a_runs.append(a)

    # ========================================================
    # EQ. (30) ANALYTICAL FIRST MOMENT
    # ========================================================

    # Rotate the non-IRS component by phi0
    C_rot = (
        C
        * np.exp(-1j * phi0)
    )

    radicand = (
        best_value**2
        - np.imag(C_rot)**2
    )

    # Floating-point safeguard
    radicand = max(
        radicand,
        0.0
    )

    # Eq. (30):
    #
    # [sqrt(y^2 - Im{Ce^-jphi0}^2)
    #  - Re{Ce^-jphi0}]
    # --------------------------------
    #             alpha_0 M
    #
    moment_eq30 = (
        np.sqrt(radicand)
        - np.real(C_rot)
    ) / (
        alpha_0 * M
    )

    eq30_moments.append(
        moment_eq30
    )


# ============================================================
# COMBINE ALL MONTE-CARLO SAMPLES
# ============================================================

theta_all = np.concatenate(
    theta_runs
)

a_all = np.concatenate(
    a_runs
)

eq30_moments = np.asarray(
    eq30_moments
)


# ============================================================
# AMPLITUDE-WEIGHTED EMPIRICAL DISTRIBUTION
#
# The empirical phase samples are weighted globally by a_m:
#
#           a_m
# w_m = ------------
#        sum_all a_m
#
# ============================================================

weights_all = (
    a_all
    / np.sum(a_all)
)


# ============================================================
# DIAGNOSTIC FIRST MOMENTS
# ============================================================

# Ordinary, unweighted first trigonometric moment
unweighted_moment = np.mean(
    np.cos(theta_all)
)


# First moment of the normalized amplitude-weighted
# empirical phase distribution
weighted_moment = np.sum(
    weights_all
    * np.cos(theta_all)
)


# Direct empirical implementation of manuscript Eq. (25):
#
# E_w[cos(theta_m)]
# =
# E[a_m cos(theta_m)] / alpha_0
weighted_moment_eq25 = (
    np.mean(
        a_all
        * np.cos(theta_all)
    )
    / alpha_0
)


# Analytical value obtained from Eq. (30)
analytical_moment = np.mean(
    eq30_moments
)


# ============================================================
# DEPENDENCE DIAGNOSTIC
# ============================================================

corr_a_cos = np.corrcoef(
    a_all,
    np.cos(theta_all)
)[0, 1]


# ============================================================
# PRINT DIAGNOSTIC RESULTS
# ============================================================

print()
print("=" * 65)
print("FIGURE 2: PHASE-DISTRIBUTION VALIDATION")
print("=" * 65)

print(f"M                                     = {M}")
print(f"K                                     = {K}")
print(f"Monte-Carlo runs                      = {num_runs}")

print()

print(
    "Unweighted empirical moment          = "
    f"{unweighted_moment:.10f}"
)

print(
    "Weighted empirical moment            = "
    f"{weighted_moment:.10f}"
)

print(
    "Empirical Eq. (25) weighted moment   = "
    f"{weighted_moment_eq25:.10f}"
)

print(
    "Analytical Eq. (30) moment           = "
    f"{analytical_moment:.10f}"
)

print(
    "Corr(a_m, cos(theta_m))               = "
    f"{corr_a_cos:.10f}"
)

print("=" * 65)


# ============================================================
# VALIDITY CHECK FOR EQ. (30)
# ============================================================

if not (
    0.0 <= analytical_moment < 1.0
):
    raise ValueError(
        "The analytical moment obtained from Eq. (30) "
        "must satisfy 0 <= moment < 1. "
        f"Obtained moment = {analytical_moment}"
    )


# ============================================================
# COMPUTE ANALYTICAL mu FROM EQ. (30)
# ============================================================

mu = solve_mu(
    analytical_moment
)

print()
print(
    "Analytical mu from Eq. (30)           = "
    f"{mu:.10f}"
)

print(
    "I1(mu)/I0(mu)                         = "
    f"{bessel_ratio(mu):.10f}"
)

print("=" * 65)


# ============================================================
# PHASE AXIS
# ============================================================

x = np.linspace(
    -np.pi,
    np.pi,
    1000
)


# ============================================================
# EXPO-COSINE / VON MISES DISTRIBUTION
#
# f_theta(theta)
# =
# exp(mu cos(theta))
# -----------------------
# 2 pi I0(mu)
#
#
# Stable implementation:
#
# I0(mu) = exp(mu) I0e(mu)
#
# Therefore:
#
# exp(mu cos(theta)) / I0(mu)
#
# =
#
# exp(mu(cos(theta)-1)) / I0e(mu)
#
# ============================================================

expo_cosine = (
    np.exp(
        mu
        * (
            np.cos(x)
            - 1.0
        )
    )
    /
    (
        2.0
        * np.pi
        * i0e(mu)
    )
)


# ============================================================
# WEIGHTED GAUSSIAN REFERENCE
# ============================================================

sigma_gaussian = np.sqrt(
    np.sum(
        weights_all
        * theta_all**2
    )
)

gaussian = (
    1.0
    /
    (
        np.sqrt(2.0 * np.pi)
        * sigma_gaussian
    )
    *
    np.exp(
        -(x**2)
        /
        (
            2.0
            * sigma_gaussian**2
        )
    )
)


# ============================================================
# WEIGHTED LAPLACIAN REFERENCE
# ============================================================

b_laplace = np.sum(
    weights_all
    * np.abs(theta_all)
)

laplacian = (
    1.0
    /
    (
        2.0
        * b_laplace
    )
    *
    np.exp(
        -np.abs(x)
        /
        b_laplace
    )
)


# ============================================================
# GOODNESS-OF-FIT: INTEGRATED SQUARED ERROR
# ============================================================

bins = np.linspace(
    -np.pi,
    np.pi,
    37
)

hist_density, bin_edges = np.histogram(
    theta_all,
    bins=bins,
    weights=weights_all,
    density=True
)

bin_centers = (
    bin_edges[:-1]
    + bin_edges[1:]
) / 2.0

bin_width = (
    bin_edges[1]
    - bin_edges[0]
)


# ------------------------------------------------------------
# Expo-Cosine at histogram bin centers
# ------------------------------------------------------------

def von_mises_pdf(theta):

    return (
        np.exp(
            mu
            * (
                np.cos(theta)
                - 1.0
            )
        )
        /
        (
            2.0
            * np.pi
            * i0e(mu)
        )
    )


expo_bins = von_mises_pdf(
    bin_centers
)


# ------------------------------------------------------------
# Gaussian at histogram bin centers
# ------------------------------------------------------------

gaussian_bins = (
    1.0
    /
    (
        np.sqrt(2.0 * np.pi)
        * sigma_gaussian
    )
    *
    np.exp(
        -(bin_centers**2)
        /
        (
            2.0
            * sigma_gaussian**2
        )
    )
)


# ------------------------------------------------------------
# Laplacian at histogram bin centers
# ------------------------------------------------------------

laplacian_bins = (
    1.0
    /
    (
        2.0
        * b_laplace
    )
    *
    np.exp(
        -np.abs(bin_centers)
        /
        b_laplace
    )
)


# ------------------------------------------------------------
# Integrated squared errors
# ------------------------------------------------------------

ISE_expo = np.sum(
    (
        hist_density
        - expo_bins
    )**2
) * bin_width

ISE_gaussian = np.sum(
    (
        hist_density
        - gaussian_bins
    )**2
) * bin_width

ISE_laplacian = np.sum(
    (
        hist_density
        - laplacian_bins
    )**2
) * bin_width


print()
print("=" * 65)
print("INTEGRATED SQUARED ERROR")
print("=" * 65)

print(
    f"Expo-Cosine                        = {ISE_expo:.10f}"
)

print(
    f"Gaussian                           = {ISE_gaussian:.10f}"
)

print(
    f"Laplacian                          = {ISE_laplacian:.10f}"
)

print("=" * 65)


# ============================================================
# FIGURE
# ============================================================

plt.figure(
    figsize=(8.2, 5.7)
)


# ------------------------------------------------------------
# Amplitude-weighted empirical histogram
# ------------------------------------------------------------

plt.hist(
    theta_all,
    bins=36,
    weights=weights_all,
    density=True,
    alpha=0.62,
    edgecolor="white",
    linewidth=0.35,
    label="Empirical"
)


# ------------------------------------------------------------
# Expo-Cosine analytical distribution
# ------------------------------------------------------------

plt.plot(
    x,
    expo_cosine,
    color="green",
    linewidth=2.5,
    label="Expo-Cosine"
)


# ------------------------------------------------------------
# Gaussian reference
# ------------------------------------------------------------

plt.plot(
    x,
    gaussian,
    color="red",
    linestyle="--",
    linewidth=2.0,
    label="Gaussian"
)


# ------------------------------------------------------------
# Laplacian reference
# ------------------------------------------------------------

plt.plot(
    x,
    laplacian,
    color="magenta",
    linestyle="-.",
    linewidth=2.0,
    label="Laplacian"
)


# ============================================================
# FIGURE FORMATTING
# ============================================================

plt.xlabel(
    r"Relative Effective Phase $\theta_m$ (radians)",
    fontsize=12
)

plt.ylabel(
    r"$f_{\theta}(\theta_m)$",
    fontsize=12
)

plt.xlim(
    -np.pi,
    np.pi
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.28
)

plt.legend(
    loc="upper left"
)

plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(
    f"Figure_2_weighted_M{M}.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    f"Figure_2_weighted_M{M}.pdf",
    bbox_inches="tight"
)


# ============================================================
# DISPLAY FIGURE
# ============================================================

plt.show()

