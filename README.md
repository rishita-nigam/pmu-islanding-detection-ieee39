# ⚡ PMU-Based Islanding Detection — IEEE 39-Bus New England System

> Topology-aware islanding detection and stability analysis using Phasor
> Measurement Unit (PMU) data from the IEEE 39-bus New England power system.

---

## Overview

This project detects **power island formation** in real-time using PMU
measurements. When a section of the grid disconnects from the main network,
the system automatically identifies the islanded buses, classifies their
stability, and assesses whether the island can sustain itself with local
generation.

The pipeline combines signal processing, unsupervised ML, and network
topology to produce a physics-grounded assessment.

---

## How It Works

**1. Dataset Generation (MATLAB/Simulink)**
The IEEE 39-bus New England network (`NE39bus2_PQ.slx`) is simulated under
fault and islanding scenarios. Per-bus PMU measurements are exported as CSV:
`Voltage`, `Frequency`, `Voltage_Angle`.

**2. Stability Check**
- Detects voltage drops exceeding threshold (default 3% drop)
- Computes ROCOF (Rate of Change of Frequency) per bus
- Flags the system as STABLE or UNSTABLE

**3. Islanding Detection (if Unstable)**
- Isolation Forest scores each bus for anomaly level
- Topology-aware clustering via **Spectral Clustering** using the IEEE
  39-bus adjacency matrix (falls back to KMeans if topology unavailable)
- Elbow method selects optimal K; forced K=2 for final island vs main-grid split

**4. Island Viability Assessment**
- Checks if the detected island contains at least one generator bus (30–39)
- Reports ROCOF and voltage drop location relative to island boundary
- Labels island as: Viable / Non-Viable / Marginal

---

## Project Structure
