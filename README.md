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
pmu-islanding-ieee39/
│
├── pmu_analyzer.py          # Main analysis script
├── NE39bus2_PQ.slx          # MATLAB/Simulink simulation model
├── pmu_fault_dataset.csv    # PMU dataset (bus-level measurements)
│
├── requirements.txt
└── README.md
---

## Output

After running, the `output/` folder contains:

| File | Description |
|------|-------------|
| `pmu_analysis_plots.png` | Voltage/frequency signal plots with event markers |
| `pmu_pre_during_post_comparison_bars.png` | Bus-by-bus pre/during/post comparison |
| `pmu_anomaly_clusters.png` | Anomaly scores with island vs main-grid clusters |
| `pmu_analysis_summary.json` | Full structured JSON report |

---

## Getting Started

```bash
git clone https://github.com/yourusername/pmu-islanding-ieee39.git
cd pmu-islanding-ieee39
pip install -r requirements.txt
python pmu_analyzer.py
```

The script reads `pmu_fault_dataset.csv` from the same directory.
To override settings, create a `config.json` (all defaults are in the script).

---

## Requirements

- Python 3.10+
- MATLAB R2021a+ (for dataset regeneration only)

---

## Developed at VIT as part of a Power Systems & ML research project.
