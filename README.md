# Simulation Codes for IRS-Assisted Backscatter Wireless Communication

This repository contains the Python-based simulation codes used to generate the simulation results presented in our study on Phase-shift Adjustments of Intelligent Reflecting Surface Elements in IRS-Assisted Backscatter Wireless Communication Networks. The repository is provided to support the reproducibility of the results reported in the manuscript.

## Overview 

The considered system includes a transmitter, a conventional backscatter tag (Tag 1), an IRS-assisted backscatter tag (Tag 2), and a receiver.

Two feedback-based methods are evaluated for IRS phase-shift adjustment:

- **Matrix of Phase-Shifts (MPS):** The receiver selects the phase-shift vector that provides the highest measured received SNR from a predefined MPS and feeds back its index.

- **Single-Bit Feedback (SBF):** The IRS phase shifts are iteratively perturbed, and the receiver returns one feedback bit indicating whether the measured received SNR has improved.

The repository also includes simulations for the proposed direct and indirect noncoherent detectors.

## Simulation Setup

A Python-based simulation framework was developed to evaluate the proposed IRS phase-adjustment algorithms under the considered 3GPP Indoor Hotspot (InH) channel conditions. The following table summarizes the simulation parameters used in the numerical evaluations:

| Parameter | Symbol | Value | Unit |
| :--- | :---: | :---: | :---: |
| Channel Bandwidth | - | 20 | MHz |
| Noise Spectral Density | - | -174 | dBm/Hz |
| Receiver Noise Figure | - | 6 | dB |
| Carrier Frequency | $f_c$ | 2.4 | GHz |
| Transmit Power | - | 20 | dBm |
| Receiver Noise Power | - | −95 | dBm |
| Transmitter–Receiver Distance | - | 20 | m |
| Transmitter–Backscatter Tag 1 Distance | - | 10 | m |
| Backscatter Tag 1–Receiver Distance | - | 10 | m |
| Transmitter–IRS Distance | - | 8 | m |
| IRS–Receiver Distance | - | 12 | m |
| Transmitter–Receiver Path Loss | - | 76.60 | dB |
| Transmitter–Backscatter Tag 1 Path Loss | - | 65.07 | dB |
| Backscatter Tag 1–Receiver Path Loss | - | 65.07 | dB |
| Transmitter–IRS Path Loss | - | 55.63 | dB |
| IRS–Receiver Path Loss | - | 58.67 | dB |
| Reflection Coefficient | $\eta$ | 0.8 | – |
| Number of IRS Elements | $M$ | Varied | – |
| Total Number of Iterations of SBF Algorithm | $K$ | 100 | – |
| SBF Perturbation Bound | $\varepsilon_m$ | $\frac{\pi}{20}$ | rad |
| Channel Drift Bound | $\delta_m$ | $\frac{\pi}{25}$ | rad |
| Discount Factor | $\lambda$ | 0.98 | – |
| Number of Observations | $N_s$ | 20 | – |


The number of IRS elements, channel realizations, phase errors, feedback
conditions, and other parameters are varied according to the experiment.
The corresponding values are stated in the individual Python files and
figure descriptions.

## Reproducibility: Figure and Table Mapping

This repository contains the Python scripts used to generate the numerical results presented in the manuscript, Phase-Shift Adjustments of Intelligent Reflecting Surface Elements in IRS-Assisted Backscatter Wireless Communication Networks.

The following table maps each manuscript figure and simulation-generated table to its corresponding Python script and random seed(s).

| Manuscript Item | Description | Python Script | Random Seed(s) |
|---|---|---|---|
| Figure 2 | Convergence analysis for static channels | [`fig2_code.py`](fig2_code.py) | 42 |
| Figure 3 | Validation of the Gamma approximation for the post-selection channel energy | [`fig3_code.py`](fig3_code.py) | M = 4: 1042; M = 128: 2042 |
| Figure 4 | SBF convergence under static channel conditions | [`fig4_code.py`](fig4_code.py) | 42 |
| Figure 5 | SBF convergence under time-varying channel conditions: effects of channel drift, phase-setting errors, feedback delay, and reflection coefficient | [`fig5_code.py`](fig5_code.py) | 42 |
| Figure 6 | Convergence behavior for different numbers of IRS elements | [`fig6_code.py`](fig6_code.py) | 42 |
| Table 3 | SBF tracking analysis under time-varying channels | [`Table3_code.py`](Table3_code.py) | 42 |
| Figure 7 | BER performance of the MPS-based algorithm | [`fig7_code.py`](fig7_code.py) | 42 |
| Figure 8 | Comparison of MPS and SBF with QPSA and perfect-CSI benchmarks | [`fig8_code.py`](fig8_code.py) | 42; bootstrap seed: 123 |
| Figure 9 | Effect of feedback bit errors on SBF | [`fig9_code.py`](fig9_code.py) | 42 |
| Figure 10 | Performance of the direct and indirect noncoherent detectors | [`fig10_code.py`](fig10_code.py) | Training: 2042; U1 testing: 7042; U0 testing: 8042; observation: 9042 |
| Figure 11 | Detector sensitivity to SNR-estimation and channel-statistics mismatch | [`fig11_code.py`](fig11_code.py) | Training: 2042; U1 testing: 7042; U0 testing: 8042; observation: 9042 |

### Reproducibility Notes

- All random seeds are fixed to support reproducible numerical experiments.
- Figures 10 and 11 use separate random seeds for training, testing, and observation generation.
- Figure 8 uses an independent random seed for its paired-bootstrap confidence-interval analysis.
- The scripts should be executed using the simulation parameters specified in the manuscript and corresponding Python files.

Comments have been added to the Python files to explain the purpose of each simulation, the main parameters, and the corresponding results in the manuscript.

## Dependencies & Environment

All simulation scripts in this repository are written in Python and rely on standard scientific computing libraries.

### Core Dependencies

| Package / Tool | Verified Version | Purpose in Repository |
| :--- | :---: | :--- |
| **Python** | `3.13` | Execution runtime |
| **NumPy** | `2.1.3` | Matrix operations, complex phase math, and random channel generation |
| **SciPy** | `1.16.3` | Gamma distribution fitting (`fig3_code.py`) and numerical integration (`fig10_code.py`, `fig11_code.py`) |
| **Matplotlib** | `3.10.0` | Generating performance curves, BER figures, and convergence plots |
| **pandas** | `2.2.3` | Used to organize and save results in CSV files|


### Quick Installation

Install all required packages in a single step via `pip`:

```bash
pip install "numpy>=2.1.3" "scipy>=1.16.3" "matplotlib>=3.10.0" "pandas>=2.2.3"
```
