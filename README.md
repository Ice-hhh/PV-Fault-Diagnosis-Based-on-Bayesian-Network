# PV Fault Diagnosis Based on Bayesian Network

This repository contains photovoltaic power prediction experiments for the distributed PV forecasting dataset. The original work was stored only as notebooks, so this version adds a reproducible command-line pipeline, explicit dependencies, and documentation.

## Project Structure

```text
.
├── data/                                  # Competition CSV files
├── src/pv_fault_diagnosis/
│   ├── data.py                            # Data loading, feature engineering, submission writer
│   └── reproduce.py                       # Train/evaluate/predict entry point
├── GPR_final.ipynb                        # Original Gaussian Process notebook
├── pymc3_final_linear-Copy2.ipynb         # Original PyMC3 notebook
├── requirements.txt
└── README.md
```

## Environment

Python 3.10 or newer is recommended for the reproducible script.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer conda:

```bash
conda create -n pv-bayes python=3.10
conda activate pv-bayes
pip install -r requirements.txt
```

## Reproduce

Run a fast smoke test first:

```bash
PYTHONPATH=src python -m pv_fault_diagnosis.reproduce --quick --model ridge
```

Run the default Bayesian ridge baseline on all rows:

```bash
PYTHONPATH=src python -m pv_fault_diagnosis.reproduce
```

The script writes:

- `outputs/submission.csv`
- `outputs/model.joblib`

The generated submission has the same identifying columns as the test power file plus `p1` to `p96` predictions.

## Notes on the Original Notebooks

The notebooks are kept as historical experiment records. Several reproducibility issues were fixed in the script version:

- Windows-only paths such as `data\xxx.csv` were replaced by portable path handling.
- Hard-coded placeholder output `xxx.csv` was replaced with configurable output paths.
- Daily `p1` to `p96` values are expanded from each row's own date instead of one global start date.
- Train and test features are aligned before scaling so both matrices have the same columns.
- The PyMC3/Theano notebook depends on old packages that do not install cleanly on modern Python versions; the command-line baseline uses scikit-learn's `BayesianRidge` wrapped for 96-output prediction.

## Data

The `data/` directory is expected to contain the following GBK-encoded CSV files:

- `A榜-训练集_分布式光伏发电预测_基本信息.csv`
- `A榜-训练集_分布式光伏发电预测_气象变量数据.csv`
- `A榜-训练集_分布式光伏发电预测_实际功率数据.csv`
- `A榜-测试集_分布式光伏发电预测_基本信息.csv`
- `A榜-测试集_分布式光伏发电预测_气象变量数据.csv`
- `A榜-测试集_分布式光伏发电预测_实际功率数据.csv`
