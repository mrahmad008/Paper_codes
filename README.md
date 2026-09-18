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

The Python scripts reproduce the main numerical experiments reported in
the manuscript, including:

- validation of the Gamma approximation for the post-selection channel
  energy;
- SBF convergence under static channel conditions;
- SBF tracking under time-varying channels;
- effects of channel drift, phase-setting errors, feedback delay, and
  reflection coefficient;
- convergence behavior for different numbers of IRS elements;
- BER performance of the MPS-based algorithm;
- comparison of MPS and SBF with QPSA and perfect-CSI benchmarks;
- effect of feedback bit errors on SBF;
- performance of the direct and indirect noncoherent detectors; and
- detector sensitivity to SNR-estimation and channel-statistics mismatch.

Comments have been added to the Python files to explain the purpose of
each simulation, the main parameters, and the corresponding results in
the manuscript.

## Running the Simulations

The simulations were developed in Python.

Run the required Python file directly, for example:

```bash
python filename.py
