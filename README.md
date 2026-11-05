# Nested Array RV-SBL DOA Estimation

High-resolution Direction of Arrival (DOA) estimation using **Real-Valued Sparse Bayesian Learning (RV-SBL)** with Continuous Off-Grid Refinement for Nested Sensor Arrays.

## 🔬 Overview
This repository contains a modern Python implementation of advanced sensor array processing algorithms derived from M.Sc. research in Telecommunications/Signal Processing. It demonstrates the migration of complex MATLAB-based tensor/matrix operations to an efficient Python/NumPy/SciPy production-ready workflow.

### Key Features
- **Nested Array Topologies:** Leverages difference coarrays to achieve increased degrees of freedom.
- **Real-Valued Transformation:** Converts complex data models into a real-valued domain to reduce computational complexity.
- **Sparse Bayesian Learning (SBL):** Implements Expectation-Maximization (EM) based sparse signal recovery.
- **Off-Grid Refinement:** Continuously updates grid points to eliminate grid-mismatch errors.

## 🛠️ Requirements
- Python 3.10+
- NumPy
- SciPy

Install dependencies via:
```bash
pip install numpy scipy
