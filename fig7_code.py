import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from tqdm import tqdm

# ============================================================
# Parameters for M=4
# ============================================================
M4 = 4
VAR4 = 1.0
num_runs = 3000

# PARAFAC parameters
K = 50
T = 4
L = 2

# SNR range
SNR_dB = np.arange(-10, 21, 2)
SNR_lin = 10**(SNR_dB/10)

np.random.seed(42)

# ============================================================
# Codebook for MPS (N=8)
# ============================================================
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

# ============================================================
# Helper functions
# ============================================================
def generate_channel(M, var):
    h_sr = np.sqrt(var/2) * (np.random.randn() + 1j * np.random.randn())
    h_st = np.sqrt(var/2) * (np.random.randn() + 1j * np.random.randn())
    h_tr = np.sqrt(var/2) * (np.random.randn() + 1j * np.random.randn())
    h_si = np.sqrt(var/2) * (np.random.randn(M) + 1j * np.random.randn(M))
    h_ir = np.sqrt(var/2) * (np.random.randn(M) + 1j * np.random.randn(M))
    direct = h_sr + h_st * np.exp(1j * 0) * h_tr
    return direct, h_ir, h_si

def compute_G(phi, direct, h_ir, h_si):
    return direct + np.sum(h_ir * np.exp(1j * phi) * h_si)

def mps_gain(direct, h_ir, h_si):
    best_gamma = -np.inf
    for phi in codebook_8:
        G = compute_G(phi, direct, h_ir, h_si)
        gamma = np.abs(G)**2
        if gamma > best_gamma:
            best_gamma = gamma
    return best_gamma

def run_sbf(direct, h_ir, h_si, M, iterations=100):
    phi = np.random.uniform(-np.pi, np.pi, M)
    G = compute_G(phi, direct, h_ir, h_si)
    gamma_best = np.abs(G)**2

    for _ in range(iterations):
        eps = np.random.uniform(-np.pi/20, np.pi/20, M)
        phi_new = phi + eps
        G_new = compute_G(phi_new, direct, h_ir, h_si)
        gamma_new = np.abs(G_new)**2
        if gamma_new > gamma_best:
            gamma_best = gamma_new
            phi = phi_new
    return gamma_best

def perfect_csi_gain(direct, h_ir, h_si):
    ang_direct = np.angle(direct)
    ang_irs = np.angle(h_ir * h_si)
    phi_opt = ang_direct - ang_irs
    G_opt = compute_G(phi_opt, direct, h_ir, h_si)
    return np.abs(G_opt)**2

def parafac_effective_snr(snr_lin, K, T, L):
    scale = 1.0 / (K * T * L)
    return snr_lin / (1 + scale)

# ============================================================
# Simulate M=4
# ============================================================
print("Simulating M=4...")
gamma_mps_arr = []
gamma_sbf_arr = []
gamma_csi_arr = []

for _ in range(num_runs):
    direct, h_ir, h_si = generate_channel(M4, VAR4)
    gamma_mps_arr.append(mps_gain(direct, h_ir, h_si))
    gamma_sbf_arr.append(run_sbf(direct, h_ir, h_si, M4, iterations=100))
    gamma_csi_arr.append(perfect_csi_gain(direct, h_ir, h_si))

gamma_mps_arr = np.array(gamma_mps_arr)
gamma_sbf_arr = np.array(gamma_sbf_arr)
gamma_csi_arr = np.array(gamma_csi_arr)

# ============================================================
# Simulate M=128 (with reduced variance to keep BER visible)
# ============================================================
print("Simulating M=128...")
M128 = 128
VAR128 = 10.0 / M128  # moderate array gain

gamma_sbf_128_arr = []
gamma_csi_128_arr = []

num_runs_128 = 500  # fewer runs for speed (increase for smoother curves)

for _ in range(num_runs_128):
    direct, h_ir, h_si = generate_channel(M128, VAR128)
    gamma_sbf_128_arr.append(run_sbf(direct, h_ir, h_si, M128, iterations=100))
    gamma_csi_128_arr.append(perfect_csi_gain(direct, h_ir, h_si))

gamma_sbf_128_arr = np.array(gamma_sbf_128_arr)
gamma_csi_128_arr = np.array(gamma_csi_128_arr)

# ============================================================
# Compute BER for each SNR
# ============================================================
BER_mps = np.zeros(len(SNR_lin))
BER_sbf = np.zeros(len(SNR_lin))
BER_csi = np.zeros(len(SNR_lin))
BER_sbf_128 = np.zeros(len(SNR_lin))
BER_csi_128 = np.zeros(len(SNR_lin))

for idx, snr in enumerate(tqdm(SNR_lin, desc="SNR sweep")):
    sigma_w2 = 1 / snr

    # M=4
    snr_eff_mps = gamma_mps_arr / sigma_w2
    BER_mps[idx] = np.mean(0.5 * erfc(np.sqrt(snr_eff_mps)))

    snr_eff_sbf = gamma_sbf_arr / sigma_w2
    BER_sbf[idx] = np.mean(0.5 * erfc(np.sqrt(snr_eff_sbf)))

    snr_eff_csi = gamma_csi_arr / sigma_w2
    BER_csi[idx] = np.mean(0.5 * erfc(np.sqrt(snr_eff_csi)))

    # M=128
    snr_eff_sbf_128 = gamma_sbf_128_arr / sigma_w2
    BER_sbf_128[idx] = np.mean(0.5 * erfc(np.sqrt(snr_eff_sbf_128)))

    snr_eff_csi_128 = gamma_csi_128_arr / sigma_w2
    BER_csi_128[idx] = np.mean(0.5 * erfc(np.sqrt(snr_eff_csi_128)))

# PARAFAC (analytical)
BER_parafac = np.array([0.5 * erfc(np.sqrt(parafac_effective_snr(snr, K, T, L))) for snr in SNR_lin])

# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(8, 6))

# M=4 curves
plt.semilogy(SNR_dB, BER_parafac, 'k-d', linewidth=2.5, markersize=8, label='PARAFAC (CSI Estimation)')
plt.semilogy(SNR_dB, BER_mps, 'r-^', linewidth=2.5, markersize=8, label='MPS (N=8)')
plt.semilogy(SNR_dB, BER_sbf, 'b-s', linewidth=2.5, markersize=8, label='SBF (M=4)')
plt.semilogy(SNR_dB, BER_csi, 'g-o', linewidth=2.5, markersize=8, label='Perfect CSI (M=4)')

# M=128 curves
plt.semilogy(SNR_dB, BER_sbf_128, 'b--s', linewidth=2.5, markersize=8, label='SBF (M=128)')
plt.semilogy(SNR_dB, BER_csi_128, 'g--o', linewidth=2.5, markersize=8, label='Perfect CSI (M=128)')

# Professional grid (major + minor)
plt.grid(True, which='major', linestyle='--', linewidth=0.8, color='gray', alpha=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', alpha=0.5)
plt.minorticks_on()

plt.xlabel(r'Average Transmit SNR $\rho$ (dB)', fontsize=14)
plt.ylabel('Bit Error Rate (BER)', fontsize=14)
plt.legend(fontsize=9, loc='upper right')
plt.ylim([1e-6, 1])
plt.xlim([-11, 21])
plt.tight_layout()
plt.savefig('figure7_with_M128.png', dpi=600)
plt.show()

# ============================================================
# Print values at 10 dB
# ============================================================
idx10 = np.where(SNR_dB == 10)[0][0]
print("\n" + "="*60)
print("BER at SNR = 10 dB")
print("="*60)
print(f"PARAFAC:        {BER_parafac[idx10]:.2e}")
print(f"MPS (M=4):      {BER_mps[idx10]:.2e}")
print(f"SBF (M=4):      {BER_sbf[idx10]:.2e}")
print(f"Perfect CSI (M=4): {BER_csi[idx10]:.2e}")
print(f"SBF (M=128):    {BER_sbf_128[idx10]:.2e}")
print(f"Perfect CSI (M=128): {BER_csi_128[idx10]:.2e}")
