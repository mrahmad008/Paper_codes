# -----------------------------------------------------------------------------
# Tracking sensitivity analysis for SBF - Table 3
#
# This script generates the numerical results reported in Table 3 of the
# manuscript. The sensitivity of the SBF tracking algorithm is evaluated with
# respect to:
#
#   - channel-drift bound, delta_m
#   - discount factor, lambda
#   - SBF perturbation bound, epsilon_m
#   - feedback delay, tau_fb
#
# The tracking performance is measured using:
#
#   1. Mean tracking loss, L_tr
#   2. 95% confidence interval of the tracking loss
#   3. Tracking-outage probability, P_out
#   4. Mean adaptation delay, k_ad
#
# The instantaneous tracking loss is defined as
#
#   L_tr(k) = 10*log10(gamma_CSI(k) / gamma_SBF(k))
#
# where gamma_CSI(k) is the received SNR obtained with the continuous-phase
# perfect-CSI reference and gamma_SBF(k) is the received SNR obtained from the
# retained SBF phase configuration under the same channel state.
#
# For each channel realization, the final 20 iterations are used as the
# evaluation window. The tracking loss is averaged over this window and then
# averaged over 200 independent channel realizations. The 95% confidence
# interval is calculated from the realization-wise mean tracking losses.
#
# A tracking-outage event is defined by L_tr > 6 dB. The adaptation delay is
# the first iteration at which L_tr <= 6 dB is maintained for five consecutive
# iterations. If this condition is not reached within K = 100 iterations,
# the adaptation delay is set to K.
#
# For delta_m and epsilon_m, the values reported in Table 3 represent the
# symmetric bounds of the corresponding uniform distributions.
#
# Unless one parameter is being varied, the remaining parameters are kept at
# their nominal values used in the manuscript.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# ============================================================
# DYNAMIC SBF QUANTITATIVE SENSITIVITY TABLE
# ============================================================
# Purpose:
#   Quantify the dynamic-SBF sensitivity requested by the reviewer
#   without adding another figure.
#
# Sweeps:
#   1) channel-drift bound / variance
#   2) discount factor lambda
#   3) perturbation bound / variance
#   4) feedback delay
#
# Reported metrics:
#   - steady-state mean tracking loss relative to instantaneous
#     continuous perfect-CSI phase alignment
#   - 95% confidence interval of the mean tracking loss
#   - tracking-outage probabilities for 3, 6, and 8 dB thresholds
#     (all three are printed so the final manuscript threshold can
#      be chosen after inspecting the results)
#   - adaptation delay using a 6-dB tracking-loss criterion
#   - percentage of realizations that satisfy that criterion by K
#
# IMPORTANT:
#   The dynamic SBF update rule is kept consistent with the current
#   Fig. 5 implementation:
#       if gamma_trial > gamma_best:
#           accept trial
#       else:
#           gamma_best = lambda * gamma_best
#
#   Tracking metrics use the PHYSICAL SNR produced by the retained
#   IRS phase configuration at the current channel state, rather than
#   the discounted stored comparison value gamma_best.
# ============================================================


# ============================================================
# 1. REPRODUCIBILITY / COMMON SETTINGS
# ============================================================
MASTER_SEED = 42

M = 128
K = 100
NUM_REALIZATIONS = 200

x = 1.0
beta = 0.0
eta = 0.8

# Baseline dynamic-SBF settings
BASE_EPSILON_BOUND = np.pi / 20.0
BASE_DRIFT_BOUND = np.pi / 25.0
BASE_LAMBDA = 0.98
BASE_FEEDBACK_DELAY_S = 0.0

# Steady-state statistics are evaluated over the final 20 iterations
STEADY_STATE_START = 80

# Candidate tracking-outage thresholds.
# We keep several thresholds in the diagnostic output so that the
# manuscript threshold is not chosen blindly.
OUTAGE_THRESHOLDS_DB = [3.0, 6.0, 8.0]

# Operational adaptation-delay definition:
# first iteration at which tracking loss is <= 6 dB for
# ADAPT_CONSECUTIVE consecutive iterations.
ADAPT_THRESHOLD_DB = 6.0
ADAPT_CONSECUTIVE = 5


# ============================================================
# 2. PHYSICAL BASELINE
# ============================================================
carrier_frequency_GHz = 2.4
carrier_frequency_Hz = carrier_frequency_GHz * 1e9

ambient_source_power_dBm = 20.0
ambient_noise_power_dBm = -95.0

rho_dB = ambient_source_power_dBm - ambient_noise_power_dBm
rho = 10.0 ** (rho_dB / 10.0)

# Geometry (m)
d_sr = 20.0
d_st = 10.0
d_tr = 10.0
d_si = 8.0
d_ir = 12.0

# Feedback-delay Doppler mapping
c = 299_792_458.0
wavelength_m = c / carrier_frequency_Hz

# Same velocity previously used for the feedback-delay sweep
velocity_m_per_s = 2.0
doppler_frequency_Hz = velocity_m_per_s / wavelength_m

# Same update interval previously used to interpret drift as
# an equivalent speed.
T_update_seconds = 1.0e-3


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
    los = indoor_los_path_loss_dB(
        distance_m,
        frequency_GHz
    )

    nlos_candidate = (
        17.30
        + 38.3 * np.log10(distance_m)
        + 24.9 * np.log10(frequency_GHz)
    )

    return max(los, nlos_candidate)


PL_sr_dB = indoor_nlos_path_loss_dB(
    d_sr,
    carrier_frequency_GHz
)

PL_st_dB = indoor_nlos_path_loss_dB(
    d_st,
    carrier_frequency_GHz
)

PL_tr_dB = indoor_nlos_path_loss_dB(
    d_tr,
    carrier_frequency_GHz
)

PL_si_dB = indoor_los_path_loss_dB(
    d_si,
    carrier_frequency_GHz
)

PL_ir_dB = indoor_los_path_loss_dB(
    d_ir,
    carrier_frequency_GHz
)

sigma_hsr2 = 10.0 ** (-PL_sr_dB / 10.0)
sigma_hst2 = 10.0 ** (-PL_st_dB / 10.0)
sigma_htr2 = 10.0 ** (-PL_tr_dB / 10.0)
sigma_hsi2 = 10.0 ** (-PL_si_dB / 10.0)
sigma_hir2 = 10.0 ** (-PL_ir_dB / 10.0)


# ============================================================
# 4. SENSITIVITY SWEEPS
# ============================================================

# Fig. 5(a) drift bounds
CHANNEL_DRIFT_BOUNDS = [
    np.pi / 50.0,
    np.pi / 25.0,
    3.0 * np.pi / 50.0,
    2.0 * np.pi / 25.0,
]

# Additional discount-factor sensitivity requested by reviewer
LAMBDA_VALUES = [
    0.90,
    0.95,
    0.98,
    1.00,
]

# Additional perturbation sensitivity requested by reviewer
EPSILON_BOUNDS = [
    np.pi / 40.0,
    np.pi / 20.0,
    np.pi / 10.0,
]

# Fig. 5(c) feedback-delay sweep
FEEDBACK_DELAYS_S = [
    0.0,
    1.0e-3,
    3.0e-3,
    5.0e-3,
]


# ============================================================
# 5. HELPERS
# ============================================================
def complex_gaussian(rng, variance, size=None):
    return np.sqrt(variance / 2.0) * (
        rng.standard_normal(size)
        + 1j * rng.standard_normal(size)
    )


def equivalent_speed_from_drift(
    drift_bound_rad,
    wavelength_m,
    T_update_seconds
):
    return (
        drift_bound_rad
        * wavelength_m
        / (2.0 * np.pi * T_update_seconds)
    )


def uniform_variance(bound):
    """
    For U[-a, a], variance = a^2 / 3.
    """
    return (bound ** 2) / 3.0


def wrap_pi_label(value):
    """
    Human-readable labels for the exact values used here.
    """
    known = {
        round(np.pi / 50.0, 12): r"pi/50",
        round(np.pi / 25.0, 12): r"pi/25",
        round(3.0 * np.pi / 50.0, 12): r"3pi/50",
        round(2.0 * np.pi / 25.0, 12): r"2pi/25",
        round(np.pi / 40.0, 12): r"pi/40",
        round(np.pi / 20.0, 12): r"pi/20",
        round(np.pi / 10.0, 12): r"pi/10",
    }

    return known.get(
        round(float(value), 12),
        f"{value:.6f}"
    )


# ============================================================
# 6. GENERATE ONE COMMON MONTE CARLO REALIZATION
# ============================================================
def generate_trial_inputs(seed):
    """
    The same channel realization and the same normalized random
    sequences are reused across sensitivity cases. This gives a
    paired/fair comparison between parameter values.
    """
    rng = np.random.default_rng(seed)

    h_sr = complex_gaussian(
        rng,
        sigma_hsr2
    )

    h_st = complex_gaussian(
        rng,
        sigma_hst2
    )

    h_tr = complex_gaussian(
        rng,
        sigma_htr2
    )

    h_si = complex_gaussian(
        rng,
        sigma_hsi2,
        M
    )

    h_ir = complex_gaussian(
        rng,
        sigma_hir2,
        M
    )

    C = (
        h_sr
        + h_st
        * x
        * np.exp(1j * beta)
        * h_tr
    )

    cascaded = h_ir * h_si

    cascaded_magnitude = np.abs(
        cascaded
    )

    channel_phase_initial = np.angle(
        cascaded
    )

    # Consistent with the static SBF initialization
    initial_irs_phases = rng.uniform(
        -np.pi,
        np.pi,
        M
    )

    # Unit sequences are scaled by the selected bounds.
    epsilon_unit_sequence = rng.uniform(
        -1.0,
        1.0,
        size=(K - 1, M)
    )

    channel_drift_unit_sequence = rng.uniform(
        -1.0,
        1.0,
        size=(K - 1, M)
    )

    feedback_delay_unit_sequence = rng.uniform(
        -1.0,
        1.0,
        size=(K - 1, M)
    )

    return {
        "C": C,
        "cascaded_magnitude": cascaded_magnitude,
        "channel_phase_initial": channel_phase_initial,
        "initial_irs_phases": initial_irs_phases,
        "epsilon_unit_sequence": epsilon_unit_sequence,
        "channel_drift_unit_sequence":
            channel_drift_unit_sequence,
        "feedback_delay_unit_sequence":
            feedback_delay_unit_sequence,
    }


# ============================================================
# 7. PERFECT-CSI REFERENCE
# ============================================================
def perfect_csi_snr(
    C,
    cascaded_magnitude,
    reflection_magnitude=eta
):
    """
    For the present phase-only channel variation, the instantaneous
    continuous perfect-CSI phase aligns every cascaded IRS component
    with the non-IRS component C.

    phi_m* = angle(C) - angle(h_ir,m h_si,m)

    Since only the channel phases vary in the dynamic model and the
    magnitudes remain fixed, the maximum magnitude is

        |C| + eta * sum_m |h_ir,m h_si,m|.

    Therefore the perfect-CSI SNR is constant over k for a given
    realization, although the required optimal phases vary with k.
    """
    maximum_amplitude = (
        np.abs(C)
        + reflection_magnitude
        * np.sum(cascaded_magnitude)
        * np.abs(x)
    )

    return rho * maximum_amplitude ** 2


# ============================================================
# 8. DYNAMIC SBF FOR ONE REALIZATION / ONE CASE
# ============================================================
def simulate_dynamic_sbf(
    inputs,
    channel_drift_bound=BASE_DRIFT_BOUND,
    epsilon_bound=BASE_EPSILON_BOUND,
    lambda_discount=BASE_LAMBDA,
    feedback_delay_s=BASE_FEEDBACK_DELAY_S,
):
    """
    Dynamic-SBF logic consistent with the current Fig. 5 code.

    Returns:
        actual_gamma_history:
            physical SNR obtained using the retained IRS phase
            configuration at the current channel state.

        stored_gamma_history:
            internal discounted comparison value used by the
            algorithm.

        perfect_csi_history:
            instantaneous continuous perfect-CSI SNR reference.

        tracking_loss_dB:
            10 log10(gamma_CSI / gamma_SBF_actual).
    """

    C = inputs["C"]

    cascaded_magnitude = (
        inputs["cascaded_magnitude"]
    )

    channel_phase = (
        inputs["channel_phase_initial"].copy()
    )

    phi_best = (
        inputs["initial_irs_phases"].copy()
    )

    epsilon_unit_sequence = (
        inputs["epsilon_unit_sequence"]
    )

    channel_drift_unit_sequence = (
        inputs["channel_drift_unit_sequence"]
    )

    feedback_delay_unit_sequence = (
        inputs["feedback_delay_unit_sequence"]
    )

    # Additional phase aging caused by the feedback delay
    feedback_delay_phase_bound = (
        2.0
        * np.pi
        * doppler_frequency_Hz
        * feedback_delay_s
    )

    current_cascaded_channel = (
        cascaded_magnitude
        * np.exp(1j * channel_phase)
    )

    G_initial = (
        C
        + eta
        * np.sum(
            current_cascaded_channel
            * np.exp(1j * phi_best)
        )
        * x
    )

    gamma_best_stored = (
        rho
        * np.abs(G_initial) ** 2
    )

    gamma_csi = perfect_csi_snr(
        C,
        cascaded_magnitude,
        eta
    )

    actual_gamma_history = np.zeros(
        K,
        dtype=float
    )

    stored_gamma_history = np.zeros(
        K,
        dtype=float
    )

    perfect_csi_history = np.full(
        K,
        gamma_csi,
        dtype=float
    )

    actual_gamma_history[0] = (
        gamma_best_stored
    )

    stored_gamma_history[0] = (
        gamma_best_stored
    )

    # --------------------------------------------------------
    # Iterative dynamic SBF
    # --------------------------------------------------------
    for k in range(1, K):

        epsilon = (
            epsilon_bound
            * epsilon_unit_sequence[k - 1]
        )

        phi_trial = (
            phi_best
            + epsilon
        )

        normal_drift = (
            channel_drift_bound
            * channel_drift_unit_sequence[k - 1]
        )

        feedback_delay_aging = (
            feedback_delay_phase_bound
            * feedback_delay_unit_sequence[k - 1]
        )

        channel_phase = (
            channel_phase
            + normal_drift
            + feedback_delay_aging
        )

        current_cascaded_channel = (
            cascaded_magnitude
            * np.exp(1j * channel_phase)
        )

        G_trial = (
            C
            + eta
            * np.sum(
                current_cascaded_channel
                * np.exp(1j * phi_trial)
            )
            * x
        )

        gamma_trial = (
            rho
            * np.abs(G_trial) ** 2
        )

        # Current dynamic-SBF accept/reject rule
        if gamma_trial > gamma_best_stored:

            phi_best = phi_trial

            gamma_best_stored = (
                gamma_trial
            )

        else:

            gamma_best_stored = (
                lambda_discount
                * gamma_best_stored
            )

        # IMPORTANT:
        # Calculate the actual physical SNR of the retained phase
        # configuration at the CURRENT channel state.
        G_actual = (
            C
            + eta
            * np.sum(
                current_cascaded_channel
                * np.exp(1j * phi_best)
            )
            * x
        )

        gamma_actual = (
            rho
            * np.abs(G_actual) ** 2
        )

        actual_gamma_history[k] = (
            gamma_actual
        )

        stored_gamma_history[k] = (
            gamma_best_stored
        )

    # Tracking loss relative to instantaneous perfect CSI
    small = np.finfo(float).tiny

    tracking_loss_dB = (
        10.0
        * np.log10(
            np.maximum(
                perfect_csi_history,
                small
            )
            /
            np.maximum(
                actual_gamma_history,
                small
            )
        )
    )

    # Numerical roundoff could create a tiny negative value.
    tracking_loss_dB = np.maximum(
        tracking_loss_dB,
        0.0
    )

    return {
        "actual_gamma": actual_gamma_history,
        "stored_gamma": stored_gamma_history,
        "perfect_csi_gamma": perfect_csi_history,
        "tracking_loss_dB": tracking_loss_dB,
    }


# ============================================================
# 9. METRICS
# ============================================================
def first_sustained_tracking_iteration(
    tracking_loss_dB,
    threshold_dB=ADAPT_THRESHOLD_DB,
    consecutive=ADAPT_CONSECUTIVE,
):
    """
    Operational adaptation delay:
    first iteration for which the tracking loss stays at or below
    threshold_dB for 'consecutive' successive iterations.

    Returns:
        iteration index, or K if the criterion is not reached.
    """
    max_start = (
        len(tracking_loss_dB)
        - consecutive
        + 1
    )

    for start in range(max_start):

        window = tracking_loss_dB[
            start:start + consecutive
        ]

        if np.all(
            window <= threshold_dB
        ):
            return start

    return K


def confidence_interval_95(values):
    """
    Two-sided 95% Student-t confidence interval for the mean.
    """
    values = np.asarray(
        values,
        dtype=float
    )

    n = values.size

    mean_value = float(
        np.mean(values)
    )

    if n <= 1:
        return (
            mean_value,
            mean_value,
            mean_value
        )

    standard_error = (
        np.std(
            values,
            ddof=1
        )
        / np.sqrt(n)
    )

    critical_value = (
        student_t.ppf(
            0.975,
            df=n - 1
        )
    )

    half_width = (
        critical_value
        * standard_error
    )

    return (
        mean_value,
        mean_value - half_width,
        mean_value + half_width
    )


def calculate_case_metrics(
    case_results
):
    """
    case_results:
        list of simulation dictionaries, one per realization.
    """
    losses = np.stack(
        [
            item["tracking_loss_dB"]
            for item in case_results
        ],
        axis=0
    )

    actual_gamma = np.stack(
        [
            item["actual_gamma"]
            for item in case_results
        ],
        axis=0
    )

    # --------------------------------------------
    # Steady-state tracking loss
    # --------------------------------------------
    steady_losses = losses[
        :,
        STEADY_STATE_START:
    ]

    # One mean tracking-loss number per realization
    per_realization_steady_loss = np.mean(
        steady_losses,
        axis=1
    )

    (
        mean_tracking_loss_dB,
        ci_low_dB,
        ci_high_dB,
    ) = confidence_interval_95(
        per_realization_steady_loss
    )

    # --------------------------------------------
    # Tracking outage probabilities
    # --------------------------------------------
    outage_probabilities = {}

    for threshold in OUTAGE_THRESHOLDS_DB:

        outage_probabilities[threshold] = float(
            np.mean(
                steady_losses
                > threshold
            )
        )

    # --------------------------------------------
    # Adaptation delay
    # --------------------------------------------
    adaptation_delays = np.array(
        [
            first_sustained_tracking_iteration(
                loss,
                threshold_dB=
                    ADAPT_THRESHOLD_DB,
                consecutive=
                    ADAPT_CONSECUTIVE,
            )
            for loss in losses
        ],
        dtype=float
    )

    adaptation_success = (
        adaptation_delays < K
    )

    adaptation_success_rate = float(
        np.mean(
            adaptation_success
        )
    )

    if np.any(
        adaptation_success
    ):
        mean_delay_successful = float(
            np.mean(
                adaptation_delays[
                    adaptation_success
                ]
            )
        )
    else:
        mean_delay_successful = np.nan

    # Restricted/capped mean:
    # non-adapted cases are counted as K.
    mean_delay_capped = float(
        np.mean(
            adaptation_delays
        )
    )

    # --------------------------------------------
    # Final / steady-state SNR diagnostics
    # --------------------------------------------
    mean_final_gamma = float(
        np.mean(
            actual_gamma[:, -1]
        )
    )

    mean_final_gamma_dB = (
        10.0
        * np.log10(
            mean_final_gamma
        )
    )

    mean_steady_gamma = float(
        np.mean(
            actual_gamma[
                :,
                STEADY_STATE_START:
            ]
        )
    )

    mean_steady_gamma_dB = (
        10.0
        * np.log10(
            mean_steady_gamma
        )
    )

    return {
        "mean_tracking_loss_dB":
            mean_tracking_loss_dB,

        "CI95_low_dB":
            ci_low_dB,

        "CI95_high_dB":
            ci_high_dB,

        "outage_3dB":
            outage_probabilities[3.0],

        "outage_6dB":
            outage_probabilities[6.0],

        "outage_8dB":
            outage_probabilities[8.0],

        "adaptation_delay_successful_mean_iter":
            mean_delay_successful,

        "adaptation_delay_capped_mean_iter":
            mean_delay_capped,

        "adapted_by_K_fraction":
            adaptation_success_rate,

        "mean_final_SNR_dB":
            mean_final_gamma_dB,

        "mean_steady_state_SNR_dB":
            mean_steady_gamma_dB,
    }


# ============================================================
# 10. COMMON MONTE CARLO INPUTS
# ============================================================
master_rng = np.random.default_rng(
    MASTER_SEED
)

trial_seeds = master_rng.integers(
    0,
    2**32 - 1,
    size=NUM_REALIZATIONS,
    dtype=np.uint32
)

print(
    "Generating common Monte Carlo realizations..."
)

all_inputs = [
    generate_trial_inputs(
        int(seed)
    )
    for seed in trial_seeds
]

print(
    f"Generated {len(all_inputs)} realizations."
)


# ============================================================
# 11. RUN ONE SENSITIVITY FAMILY
# ============================================================
def evaluate_case(
    family,
    value,
    display_value
):
    """
    Build the simulation parameters for one sensitivity case.
    """

    params = {
        "channel_drift_bound":
            BASE_DRIFT_BOUND,

        "epsilon_bound":
            BASE_EPSILON_BOUND,

        "lambda_discount":
            BASE_LAMBDA,

        "feedback_delay_s":
            BASE_FEEDBACK_DELAY_S,
    }

    if family == "Channel drift":
        params[
            "channel_drift_bound"
        ] = value

    elif family == "Discount factor":
        params[
            "lambda_discount"
        ] = value

    elif family == "Perturbation":
        params[
            "epsilon_bound"
        ] = value

    elif family == "Feedback delay":
        params[
            "feedback_delay_s"
        ] = value

    else:
        raise ValueError(
            f"Unknown family: {family}"
        )

    case_results = [
        simulate_dynamic_sbf(
            inputs,
            **params
        )
        for inputs in all_inputs
    ]

    metrics = calculate_case_metrics(
        case_results
    )

    row = {
        "Parameter": family,
        "Value": display_value,
    }

    # Include the actual variance when the reviewer explicitly
    # refers to drift variance or perturbation variance.
    if family == "Channel drift":

        row[
            "Uniform_variance_rad2"
        ] = uniform_variance(
            value
        )

        row[
            "Equivalent_speed_mps"
        ] = equivalent_speed_from_drift(
            value,
            wavelength_m,
            T_update_seconds
        )

    elif family == "Perturbation":

        row[
            "Uniform_variance_rad2"
        ] = uniform_variance(
            value
        )

        row[
            "Equivalent_speed_mps"
        ] = np.nan

    elif family == "Feedback delay":

        row[
            "Uniform_variance_rad2"
        ] = np.nan

        row[
            "Equivalent_speed_mps"
        ] = np.nan

        row[
            "Feedback_phase_aging_bound_rad"
        ] = (
            2.0
            * np.pi
            * doppler_frequency_Hz
            * value
        )

    else:

        row[
            "Uniform_variance_rad2"
        ] = np.nan

        row[
            "Equivalent_speed_mps"
        ] = np.nan

    row.update(
        metrics
    )

    return row


# ============================================================
# 12. RUN ALL CASES
# ============================================================
rows = []

print()
print(
    "Running channel-drift sensitivity..."
)

for value in CHANNEL_DRIFT_BOUNDS:

    rows.append(
        evaluate_case(
            "Channel drift",
            value,
            wrap_pi_label(value)
        )
    )


print(
    "Running discount-factor sensitivity..."
)

for value in LAMBDA_VALUES:

    rows.append(
        evaluate_case(
            "Discount factor",
            value,
            f"{value:.2f}"
        )
    )


print(
    "Running perturbation sensitivity..."
)

for value in EPSILON_BOUNDS:

    rows.append(
        evaluate_case(
            "Perturbation",
            value,
            wrap_pi_label(value)
        )
    )


print(
    "Running feedback-delay sensitivity..."
)

for value in FEEDBACK_DELAYS_S:

    rows.append(
        evaluate_case(
            "Feedback delay",
            value,
            f"{1e3 * value:.0f} ms"
        )
    )


# ============================================================
# 13. RESULTS TABLE
# ============================================================
results_df = pd.DataFrame(
    rows
)

# Put important manuscript-facing columns first
preferred_columns = [
    "Parameter",
    "Value",
    "Uniform_variance_rad2",
    "Equivalent_speed_mps",
    "Feedback_phase_aging_bound_rad",
    "mean_tracking_loss_dB",
    "CI95_low_dB",
    "CI95_high_dB",
    "outage_3dB",
    "outage_6dB",
    "outage_8dB",
    "adaptation_delay_successful_mean_iter",
    "adaptation_delay_capped_mean_iter",
    "adapted_by_K_fraction",
    "mean_steady_state_SNR_dB",
    "mean_final_SNR_dB",
]

for col in preferred_columns:
    if col not in results_df.columns:
        results_df[col] = np.nan

results_df = results_df[
    preferred_columns
]


# ============================================================
# 14. OUTPUT
# ============================================================
OUTPUT_DIR = (
    Path.cwd()
    / "dynamic_sbf_sensitivity_table"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

csv_path = (
    OUTPUT_DIR
    / "dynamic_sbf_sensitivity_full.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# A compact candidate table using 6-dB tracking outage
compact_df = results_df[
    [
        "Parameter",
        "Value",
        "mean_tracking_loss_dB",
        "CI95_low_dB",
        "CI95_high_dB",
        "outage_6dB",
        "adaptation_delay_capped_mean_iter",
        "adapted_by_K_fraction",
    ]
].copy()

compact_csv_path = (
    OUTPUT_DIR
    / "dynamic_sbf_sensitivity_compact_candidate.csv"
)

compact_df.to_csv(
    compact_csv_path,
    index=False
)


# ============================================================
# 15. PRINT FORMATTED RESULTS
# ============================================================
pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)

pd.set_option(
    "display.precision",
    5
)

print()
print("=" * 100)
print("DYNAMIC SBF SENSITIVITY RESULTS")
print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)

print("=" * 100)

print()
print(
    "Metric definitions:"
)

print(
    f"  Steady-state window       : "
    f"k = {STEADY_STATE_START} to {K - 1}"
)

print(
    "  Tracking loss            : "
    "10*log10(gamma_CSI/gamma_SBF_actual)"
)

print(
    "  95% CI                   : "
    "Student-t CI across per-realization steady-state mean losses"
)

print(
    "  Outage probabilities     : "
    "fraction of steady-state samples exceeding 3, 6, or 8 dB tracking loss"
)

print(
    f"  Adaptation delay         : "
    f"first k with loss <= {ADAPT_THRESHOLD_DB:.1f} dB "
    f"for {ADAPT_CONSECUTIVE} consecutive iterations"
)

print(
    "  Capped adaptation delay  : "
    f"non-adapted realizations are assigned K={K}"
)

print()
print(f"Full CSV    : {csv_path}")
print(f"Compact CSV : {compact_csv_path}")
print("=" * 100)
