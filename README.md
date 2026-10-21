# Simulation Codes for IRS-Assisted Backscatter Wireless Communication

This repository contains the Python simulation codes used to generate the
numerical results presented in our study on feedback-based phase-shift
adjustment and noncoherent detection for Intelligent Reflecting Surface
(IRS)-assisted backscatter wireless communication (BackWComm) networks.

The repository is provided to support the reproducibility of the results
reported in the manuscript.

## Overview

The considered system includes a transmitter, a conventional backscatter
tag (Tag 1), an IRS-assisted backscatter tag (Tag 2), and a receiver.

Two feedback-based methods are evaluated for IRS phase-shift adjustment:

- **Matrix of Phase-Shifts (MPS):** the receiver selects the phase-shift
  vector that provides the highest measured SNR from a predefined codebook
  and feeds back its index.

- **Single-Bit Feedback (SBF):** the IRS phases are iteratively perturbed,
  and the receiver returns one feedback bit indicating whether the measured
  SNR has improved.

The repository also includes simulations for the proposed direct and
indirect noncoherent detectors.

## Simulation Setup

The simulations follow the system model and parameter settings described
in the manuscript.

The main simulation setting uses:

- Carrier frequency: **2.4 GHz**
- 3GPP Indoor Hotspot (InH) propagation scenario
- Transmit power: **20 dBm**
- Receiver noise power: **-95 dBm**
- Reflection coefficient: **0.8**
- SBF iterations: **100**
- SBF perturbation bound: **π/20**
- Channel-drift bound: **π/25**
- Discount factor: **0.98**

The number of IRS elements, channel realizations, phase errors, feedback
conditions, and other parameters are varied according to the experiment.
The corresponding values are stated in the individual Python files and
figure descriptions.

## Repository Contents

## Reproducibility: Figure and Table Mapping

This repository contains the Python scripts used to generate the numerical results presented in the manuscript, *Phase Adjustments of Intelligent Reflecting Surface Elements in IRS-Assisted Backscatter Wireless Communication Networks*.

The following table maps each manuscript figure and simulation-generated table to its corresponding Python script and random seed(s).

| Manuscript Item | Description | Python Script | Random Seed(s) |
|---|---|---|---|
| Figure 2 | Convergence analysis for static channels | `fig2_code.py` | 42 |
| Figure 3 | Validation of the Gamma approximation for the post-selection channel energy | `fig3_code.py` | M = 4: 1042; M = 128: 2042 |
| Figure 4 | SBF convergence under static channel conditions | `fig4_code.py` | 42 |
| Figure 5 | SBF convergence under time-varying channel conditions: effects of channel drift, phase-setting errors, feedback delay, and reflection coefficient | `fig5_code.py` | 42 |
| Figure 6 | Convergence behavior for different numbers of IRS elements | `fig6_code.py` | 42 |
| Table 3 | SBF tracking analysis under time-varying channels | `Table3_code.py` | 42 |
| Figure 7 | BER performance of the MPS-based algorithm | `fig7_code.py` | 42 |
| Figure 8 | Comparison of MPS and SBF with QPSA and perfect-CSI benchmarks | `fig8_code.py` | Master seed: 42; bootstrap seed: 123 |
| Figure 9 | Effect of feedback bit errors on SBF | `fig9_code.py` | 42 |
| Figure 10 | Performance of the direct and indirect noncoherent detectors | `fig10_code.py` | Training: 2042; U1 testing: 7042; U0 testing: 8042; observation: 9042 |
| Figure 11 | Detector sensitivity to SNR-estimation and channel-statistics mismatch | `fig11_code.py` | Training: 2042; U1 testing: 7042; U0 testing: 8042; observation: 9042 |

### Reproducibility Notes

- All random seeds are fixed to support reproducible numerical experiments.
- Figures 10 and 11 use separate random seeds for training, testing, and observation generation.
- Figure 8 uses an independent random seed for its paired-bootstrap confidence-interval analysis.
- The scripts should be executed using the simulation parameters specified in the manuscript and corresponding Python files.

Comments have been added to the Python files to explain the purpose of each simulation, the main parameters, and the corresponding results in the manuscript.


