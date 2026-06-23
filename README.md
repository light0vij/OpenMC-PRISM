# OpenMC-PRISM

**Physics-informed Reactor-assembly optimization via Interchangeable AI Search Methods**

A self-directed project exploring genetic algorithms and Bayesian optimization for nuclear fuel assembly layout design, built on [OpenMC](https://openmc.org/), with a geometry based on the Gundremmingen-A boiling water reactor. The problem — optimizing the enrichment layout of a BWR fuel assembly — was motivated by Radaideh et al. (2021), though the methods here are entirely independent: where that work applied deep reinforcement learning, this project pursues GA, Bayesian optimization, and a hybrid of the two, within a modular architecture that cleanly separates the physics environment from interchangeable optimizer agents.

---

## Overview

The central problem is the *fuel assembly loading pattern*: given an N×N grid of fuel rods with two enrichment levels (low: 1.87 wt% U-235, high: 2.53 wt%), find the spatial arrangement that drives k∞ toward a target value while keeping the Power Peaking Factor (PPF) below a safety threshold. The framework is designed to generalize to any N×N BWR assembly; the notebooks demonstrate it on the 6×6 case from the benchmark. The search space even for a 6×6 assembly runs into the millions of combinations, making exhaustive simulation impossible.

This project attacks that problem with three complementary strategies:

- **Genetic Algorithm (GA)** — population-based evolutionary search with tournament selection, two-point crossover, and optional diversity injection.
- **Bayesian Optimization (BO)** — a Matérn-5/2 Gaussian Process surrogate guided by Expected Improvement acquisition, seeded by Latin Hypercube Sampling.
- **Hybrid Memetic (GA→BO)** — the GA finds a promising region of design space; BO then refines the best candidates.

### Objective functions

Both optimizers minimize a composite score that balances reactivity and thermal safety. The GA minimizes a fitness function (lower is better):

$$F_{\text{GA}} = \alpha_k \left| k_{\text{target}} - k_{\infty} \right| + \alpha_{\text{PPF}} \max\left(0,\ \text{PPF} - \text{PPF}_{\text{target}}\right)^2 + w_{\text{adj}} \cdot N_{\text{adj}}$$

The BO maximizes a sign-flipped equivalent score:

$$\text{Score}_{\text{BO}} = -\alpha_k \left| k_{\text{target}} - k_{\infty} \right| - \alpha_{\text{PPF}} \max\left(0,\ \text{PPF} - \text{PPF}_{\text{target}}\right)^2$$

Both are evaluated at the same operating targets: **k∞ target = 1.25**, **PPF target = 1.41**. The goal is therefore not to maximize k∞ but to hit the reactivity target with the flattest possible flux profile.

### Physics heuristics

All three search methods support toggleable **physics heuristics** as reward shaping — penalty terms grounded in classical nuclear engineering practice:

- **Corner masking** — the four corner positions of a BWR assembly face wide water gaps on two sides, which significantly increases local moderation and thermal flux. Placing a high-enrichment rod there amplifies this effect further, driving up the PPF. A penalty is added for each such corner violation.
- **Adjacency penalty** — packing high-enrichment rods next to each other concentrates fission power in a small region, creating a hot-spot that is the typical cause of PPF limit exceedances. A penalty is added for each directly-adjacent high-enrichment pair (N_adj in the GA fitness above).

These constraints are implemented as pure integer counts in `heuristics.py` (environment layer) and as weighted penalty terms in `GAConfig` / `RunConfig` (optimizer layer). Each penalty can be switched on or off independently via config toggles, enabling direct A/B comparisons between unconstrained and physics-informed search runs.

Half-diagonal symmetry reduces the effective variable count from N² to N(N+1)/2 unique positions, making the problem tractable for larger grids without sacrificing physical completeness.

---

## Project Structure

```
openmc-prism/
│
├── openmc_bo/                  ← Optimizer package — model-agnostic, never edit for new models
│   ├── __init__.py
│   ├── config.py               ← RunConfig dataclass: all tunable BO parameters & penalty weights
│   ├── lhs_sampler.py          ← Latin Hypercube Sampling for initial design
│   ├── gp_surrogate.py         ← Matérn-5/2 GP + Expected Improvement acquisition
│   ├── bo_loop.py              ← Bayesian Optimisation loop + smoke-test helper
│   └── results.py              ← CSV export, convergence plots, design report
│
├── openmc_ga/                  ← Optimizer package — model-agnostic, never edit for new models
│   ├── __init__.py
│   ├── ga_config.py            ← GAConfig dataclass: chromosome helpers & penalty weights
│   ├── ga_population.py        ← Population initialisation with symmetry-aware constraints
│   ├── ga_fitness.py           ← Fitness evaluation and Hamming diversity metric
│   ├── ga_operators.py         ← Crossover, mutation, and elitism operators
│   ├── ga_loop.py              ← Main GA loop (tournament/roulette selection, convergence check)
│   └── ga_results.py           ← CSV export and convergence plots
│
│   > Both openmc_bo and openmc_ga are installable Python packages (see pyproject.toml).
│   > They are pure algorithm: zero environment imports. Any nuclear model can be plugged in
│   > by passing an evaluate_fn callable via functools.partial.
│
├── environments/               ← Physics layer — the OpenMC model lives here
│   ├── __init__.py
│   ├── env.py                  ← OpenMC BWR NxN model; exposes evaluate_for_ga / evaluate_for_bo
│   ├── heuristics.py           ← Physics constraint counters (corner violations, adjacency pairs)
│   ├── mapping.py              ← Symmetry mapping and enrichment grid utilities
│   ├── bwr_vis.py              ← Assembly visualisation helpers
│   └── bwr_report.py           ← Per-run design report generation
│
├── Notebooks/                  ← One notebook per optimizer / study
│   ├── BO_Optimisation_BWR_NxN.ipynb
│   ├── GA_Optimisation_BWR_NxN.ipynb
│   └── GA_BO_Memetic_BWR_NXN.ipynb
│
├── Results/                    ← Auto-created; CSVs and PNGs written here at runtime
│   ├── BWR_6x6_HP-BO/
│   ├── BWR_6x6_HP-GA/
│   └── BWR_6x6_HP-GA-BO/
│
├── Dockerfile                  ← Builds the ghcr.io/light0vij/openmc-prism image
├── docker-compose.yml          ← Runs JupyterLab locally on port 8888
├── executedownload.sh          ← Downloads ENDF/B-VIII.0 nuclear data on first container start
├── pyproject.toml              ← Package metadata for openmc_bo and openmc_ga
├── requirements.txt            ← Python dependencies baked into the image
├── .dockerignore
├── .gitignore
└── README.md
```

### Architecture note

`openmc_bo` and `openmc_ga` deliberately contain **no imports from `environments`**. The physics model is injected at the notebook level via `functools.partial`, making the optimizer packages reusable for any fuel assembly geometry or even unrelated combinatorial problems.

---

## Installation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running.

### Quick start

**1. Pull the image**

```bash
docker pull ghcr.io/light0vij/openmc-prism:latest
```

**2. Run JupyterLab**

```bash
docker run -p 8888:8888 -v /YOUR_FOLDER_PATH:/nuclear_data ghcr.io/light0vij/openmc-prism:latest
```

Replace `/YOUR_FOLDER_PATH` with an absolute path to a local folder where the nuclear data should be stored (e.g. `/home/user/nuclear_data` on Linux/macOS, or `C:\nuclear_data` on Windows).

The image is kept intentionally lean (~3 GB) through two design choices:

- **OpenMC is installed via [shimwell's pre-built wheels](https://github.com/shimwell/openmc_wheels)** rather than compiled from source, saving around 30 minutes of build time.
- **Nuclear data is not baked into the image.** The ENDF/B-VIII.0 cross-section library (~2 GB) is downloaded separately at container startup via `executedownload.sh` and saved to the mounted folder. On the first run this download happens automatically; on every subsequent run the script detects that the data is already present and skips it — so startup is near-instant after the first time. Bundling the data would push the image to ~11 GB.

**3. Open the notebooks**

Navigate to `http://localhost:8888` in your browser and open any notebook under `Notebooks/`.

---

## License

MIT — see `LICENSE` if present, or use freely for research and educational purposes.

---

## References

Romano, P. K., Horelik, N. E., Herman, B. R., Nelson, A. G., Forget, B., & Smith, K. (2015). OpenMC: A State-of-the-Art Monte Carlo Code for Research and Development. *Annals of Nuclear Energy*, 82, 90–97. [https://doi.org/10.1016/j.anucene.2014.07.048](https://doi.org/10.1016/j.anucene.2014.07.048)

Radaideh, M. I., Forget, B., & Shirvan, K. (2021). Physics-informed reinforcement learning optimization of nuclear assembly design. *Nuclear Engineering and Design*, 372, 110966. [https://doi.org/10.1016/j.nucengdes.2020.110966](https://doi.org/10.1016/j.nucengdes.2020.110966)

---

## Planned Extensions

### Reinforcement Learning

A third optimizer agent — Proximal Policy Optimization (PPO) with action masking — is planned. The plug-and-play architecture means the existing `environments/env.py` can be adapted to a Gymnasium-compatible environment with minimal changes, while `openmc_bo` and `openmc_ga` remain untouched.

### Gadolinium Rod Poisoning

Current assemblies use only two enrichment levels. A natural extension is to add gadolinium-bearing fuel rods as a third rod type, introducing a burnable absorber that controls excess reactivity early in the fuel cycle.

### Depletion Verification

Simulating neutron flux over an 18–24 month fuel cycle for every candidate layout during optimization would be computationally prohibitive. The planned workflow separates this into two phases:

**Step 1 — BOC Optimization.** The GA, BO, or Hybrid Memetic optimizer runs as-is, evaluating layouts at Beginning of Cycle (Day 0). The optimizer's identify arrangements that best satisfy the k∞ and PPF objectives with fresh fuel.

**Step 2 — Depletion Verification.** The top 3–5 layouts from Step 1 are passed to a dedicated script that runs a full OpenMC depletion calculation to verify that gadolinium burns away correctly and that k_eff stays above 1.0 for the entire cycle.

This two-step strategy — *optimize the BOC state, verify with depletion* — keeps the AI search tractable while ensuring the winning designs are physically viable over their full operational lifetime.
