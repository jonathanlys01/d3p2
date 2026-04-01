# Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty

<p align="left">
<a href="https://arxiv.org/abs/2512.21336">
<img src="https://img.shields.io/badge/arXiv-2512.21336-red.svg"></a>
</p>

This repository contains the code for *[Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty](https://arxiv.org/abs/2512.21336)*.

## Installation

Create a conda environment with the required dependencies:

```bash
conda env create -f requirements.yaml
conda activate dentropy
```

## Quick Start

### Basic Usage

Run sampling experiments with different methods:

```bash
# Random
python main.py sampling=random

# Best-of-N sampling
python main.py sampling=bon

# SMC sampling
python main.py sampling=smc

# Greedy sampling
python main.py sampling=greedy
```

## Sampling Methods

### Random

**Usage:**
```bash
python main.py sampling=random \
  sampling.steps=128 \
  sampling.num_sample_batches=16 \
  seed=42
```

**Parameters:**
- `sampling.steps`: Number of diffusion steps (default: 128)
- `sampling.num_sample_batches`: Number of runs to execute (default: 8)

---

### Best-of-N

**Usage:**
```bash
python main.py sampling=bon \
  sampling.num_particles=8 \
  sampling.steps=128 \
  seed=42
```

**Parameters:**
- `sampling.num_particles`: Number of samples to generate (N) (default: 8)
- `sampling.steps`: Diffusion steps (default: 128)
- `sampling.num_sample_batches`: Number of runs (default: 8)

---

### Sequential Monte Carlo

**Usage:**
```bash
python main.py sampling=smc \
  smc.num_particles=8 \
  smc.resample_interval=50 \
  smc.lambda_weight=5.0 \
  seed=42
```

**Parameters:**
- `smc.num_particles`: Number of particles to maintain (default: 8)
- `smc.resample_interval`: Steps between resampling (default: 50)
- `smc.lambda_weight`: Temperature parameter for potential function (default: 5.0)
- `smc.potential_type`: Potential function type, 'max' or 'mean' (default: 'max')

---

### Greedy

**Usage:**
```bash
python main.py sampling=greedy \
  greedy.num_candidates=8 \
  greedy.beam_size=1 \
  seed=42
```

**Parameters:**
- `greedy.num_candidates`: Number of candidates per beam at each step (default: 8)
- `greedy.beam_size`: Number of beams to maintain (default: 1)
  - `beam_size=1`: Pure greedy search
  - `beam_size>1`: Beam search

---

## Release Progress

- ✅ Implementation of Denoising Entropy
- ✅ Implementation of Best-of-N and SMC
- ✅ Evaluation on MDLM
- ❌ Evaluation on LLaDA

## Acknowledgements

This repository is built upon: [MDLM](https://github.com/kuleshov-group/mdlm)

## Citation

```bibtex
@article{chen2025optimizing,
  title={Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty},
  author={Chen, Ziyu and Jiang, Xinbei and Sun, Peng and Lin, Tao},
  journal={arXiv preprint arXiv:2512.21336},
  year={2025}
}
```
