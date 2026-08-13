import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from PIL import Image


# ============================================================
# Parameters
# ============================================================

M = 4                          # number of IRS elements
ETA = 1.0
THETA = 0.0
X = 1.0
VAR = 1.0                      # channel variance (0 dB)

# SNR range
SNR_dB = np.arange(-10, 15, 2)     # -10 to 14 dB
SNR_lin = 10 ** (SNR_dB / 10)

# Number of channel realisations
N_CH = 5000


# ============================================================
# Codebooks for N = 4, 6, 8
# ============================================================

codebook_4 = np.array([
    [0, 0, 0, 0],
    [0, np.pi, 0, np.pi],
    [0, 0, np.pi, np.pi],
    [0, np.pi, np.pi, 0]
])

codebook_6 = np.array([
    [0, 0, 0, 0],
    [0, np.pi, 0, np.pi],
    [0, 0, np.pi, np.pi],
    [0, np.pi, np.pi, 0],
    [0, np.pi/2, 0, np.pi/2],
    [0, 0, np.pi/2, np.pi/2]
])

codebook_8 = np.array([
    [0, 0, 0, 0],
    [0, np.pi, 0, np.pi],
    [0, 0, np.pi, np.pi],
    [0, np.pi, np.pi, 0],
    [0, np.pi/2, 0, np.pi/2],
    [0, 0, np.pi/2, np.pi/2],
    [0, np.pi/2, np.pi, 3*np.pi/2],
    [0, 3*np.pi/2, np.pi, np.pi/2]
])

codebooks = {
    4: codebook_4,
    6: codebook_6,
    8: codebook_8
}


# ============================================================
# Helper functions
# ============================================================

def compute_G_all(phi, direct, h_ir, h_si):
    """
    Compute received complex channel gain for a batch
    of phase configurations.
    """

    irs_sum = np.sum(
        h_ir
        * np.exp(1j * phi)
        * h_si,
        axis=1
    )

    return direct + ETA * irs_sum


def mps_best_phi_batch(
    h_ir,
    h_si,
    direct,
    codebook
):
    """
    Select the codeword producing the maximum received
    SNR for each channel realisation.
    """

    N_vec = codebook.shape[0]

    phi_all = codebook[None, :, :]

    h_ir_exp = h_ir[:, None, :]

    h_si_exp = h_si[:, None, :]

    irs_sum = np.sum(
        h_ir_exp
        * np.exp(1j * phi_all)
        * h_si_exp,
        axis=2
    )

    G_all = (
        direct[:, None]
        + ETA * irs_sum
    )

    gamma_all = np.abs(G_all) ** 2

    best_idx = np.argmax(
        gamma_all,
        axis=1
    )

    return codebook[best_idx, :]


def generate_channels(
    N_ch,
    M,
    var
):
    """
    Generate independent Rayleigh fading channels.
    """

    h_sr = np.sqrt(var / 2) * (
        np.random.randn(N_ch)
        + 1j * np.random.randn(N_ch)
    )

    h_st = np.sqrt(var / 2) * (
        np.random.randn(N_ch)
        + 1j * np.random.randn(N_ch)
    )

    h_tr = np.sqrt(var / 2) * (
        np.random.randn(N_ch)
        + 1j * np.random.randn(N_ch)
    )

    h_si = np.sqrt(var / 2) * (
        np.random.randn(N_ch, M)
        + 1j * np.random.randn(N_ch, M)
    )

    h_ir = np.sqrt(var / 2) * (
        np.random.randn(N_ch, M)
        + 1j * np.random.randn(N_ch, M)
    )

    direct = (
        h_sr
        + h_st
        * X
        * np.exp(1j * THETA)
        * h_tr
    )

    return direct, h_ir, h_si


# ============================================================
# Generate channels once
# ============================================================

print("Generating channels ...")

direct, h_ir, h_si = generate_channels(
    N_CH,
    M,
    VAR
)


# ============================================================
# Compute BER for each codebook size
# ============================================================

BER_results = {}


for N in [4, 6, 8]:

    print(
        f"Computing gains for N = {N} ..."
    )

    # Select best MPS phase vector
    phi_mps = mps_best_phi_batch(
        h_ir,
        h_si,
        direct,
        codebooks[N]
    )

    # Compute received channel gain
    G_mps = compute_G_all(
        phi_mps,
        direct,
        h_ir,
        h_si
    )

    # Received channel power
    gamma_mps = np.abs(G_mps) ** 2


    # --------------------------------------------------------
    # BER calculation
    # --------------------------------------------------------

    BER = np.zeros(
        len(SNR_lin)
    )

    for i, snr in enumerate(SNR_lin):

        BER[i] = np.mean(
            0.5
            * erfc(
                np.sqrt(
                    snr * gamma_mps
                )
            )
        )

    BER_results[N] = BER


# ============================================================
# Manual scaling to create clear gaps
#
# NOTE:
# These factors are retained exactly from your original code.
# ============================================================

scale_N6 = 0.5
scale_N8 = 0.2


BER_scaled = {

    4: BER_results[4],

    6: BER_results[6]
       * scale_N6,

    8: BER_results[8]
       * scale_N8
}


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
    figsize=(8, 6)
)


# ============================================================
# Original color and marker scheme
#
# N = 4 -> RED + TRIANGLE
# N = 6 -> BLUE + SQUARE
# N = 8 -> GREEN + CIRCLE
# ============================================================

style_map = {

    4: {
        'color': 'red',
        'marker': '^',
        'label': 'N = 4'
    },

    6: {
        'color': 'blue',
        'marker': 's',
        'label': 'N = 6'
    },

    8: {
        'color': 'green',
        'marker': 'o',
        'label': 'N = 8'
    }
}


# ============================================================
# Plot BER curves
# ============================================================

for N in [4, 6, 8]:

    style = style_map[N]

    ax.semilogy(
        SNR_dB,
        BER_scaled[N],

        # Preserve original colors
        color=style['color'],

        # Preserve original markers
        marker=style['marker'],

        # Preserve original line width
        linewidth=2.5,

        # Preserve original marker size
        markersize=8,

        # Marker at every point
        markevery=1,

        # Legend
        label=style['label']
    )


# ============================================================
# Axis labels
# ============================================================

ax.set_xlabel(
    r'Average Transmit SNR $\rho$ (dB)',
    fontsize=11,
    fontweight='normal'
)

ax.set_ylabel(
    'Bit Error Rate (BER)',
    fontsize=11,
    fontweight='normal'
)


# ============================================================
# Axis limits
# ============================================================

ax.set_ylim(
    1e-6,
    1
)

ax.set_xlim(
    -11,
    16
)


# ============================================================
# Major and minor ticks
# ============================================================

ax.minorticks_on()

ax.tick_params(
    axis='both',
    which='major',
    direction='in',
    length=4,
    width=0.8
)

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

# Major grid
ax.grid(
    True,
    which='major',
    linestyle='--',
    linewidth=0.7,
    alpha=0.6
)

# Minor grid
ax.grid(
    True,
    which='minor',
    linestyle=':',
    linewidth=0.5,
    alpha=0.4
)


# ============================================================
# Legend
# ============================================================

legend = ax.legend(
    fontsize=10,
    loc='upper right',
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
    'mps_ber_N4_6_8_gap.png'
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
