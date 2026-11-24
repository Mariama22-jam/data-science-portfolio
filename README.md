# Mariama Jammeh: statistics, machine learning and Monte Carlo portfolio

[![CI](https://github.com/Mariama22-jam/data-science-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/Mariama22-jam/data-science-portfolio/actions/workflows/ci.yml)
[![Full run](https://github.com/Mariama22-jam/data-science-portfolio/actions/workflows/full-run.yml/badge.svg)](https://github.com/Mariama22-jam/data-science-portfolio/actions/workflows/full-run.yml)

Three reproducible projects that extend my two MSc theses (AIMS: machine learning
for the Cox model; Gdańsk: dynamic Monte Carlo of polymer adsorption) and add a
new applied-ML study. Every number in this repository is produced by code in this
repository from public data, and is regenerated automatically by GitHub Actions.

| Project | Question | Methods | Data |
|---|---|---|---|
| [`survival-ml-pbc`](survival-ml-pbc/) | How much do penalised and ensemble survival models improve on a Cox model, under honest validation? | Cox, ridge, LASSO, elastic net, componentwise boosting, boosted trees, random survival forest; repeated CV, Uno's C, integrated Brier score, calibration, Schoenfeld tests | Mayo Clinic PBC trial (n = 312) |
| [`polymer-monte-carlo`](polymer-monte-carlo/) | Where does a grafted polymer adsorb onto an attractive wall, and does the sampler reproduce known scaling? | Self-avoiding walks on the cubic lattice; pivot, reptation and local moves; exact-enumeration tests; blocking errors; finite-size scaling | Simulated (numba) |
| [`heart-failure-ml`](heart-failure-ml/) | Do modern models beat logistic regression for heart-failure mortality on a small clinical dataset? | Logistic regression, gradient boosting, random forest, small neural net; calibration slope/intercept; SHAP; leakage check | UCI Heart Failure Clinical Records (n = 299) |

## Reproduce everything

```bash
git clone https://github.com/Mariama22-jam/data-science-portfolio.git
cd data-science-portfolio
make install     # pip install each project's pinned requirements (Python 3.13)
make test        # unit tests, including exact-enumeration checks of the sampler
make quick       # smoke runs, about 1 minute
make all         # full runs that regenerate every results/ folder
```

## Automation

* **CI** (`.github/workflows/ci.yml`) runs on every push and pull request: unit
  tests plus a quick run of each project. Outputs are attached to the run as artifacts.
* **Full run** (`.github/workflows/full-run.yml`) runs the complete analyses when
  analysis code on `main` changes, monthly, or on demand. It commits the
  regenerated `results/` folders back to the repository, so tables and figures
  never drift from the code.

## Repository layout

```
├── survival-ml-pbc/        data/ src/ tests/ results/
├── polymer-monte-carlo/    polymc/ scripts/ tests/ results/
├── heart-failure-ml/       data/ src/ tests/ results/
├── .github/workflows/      ci.yml, full-run.yml
└── Makefile
```

## Contact

Email: [jammehmariama657@gmail.com](mailto:jammehmariama657@gmail.com)
