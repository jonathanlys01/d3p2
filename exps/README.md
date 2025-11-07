# Experiments

## Embeddings

Check the CKA between the learned embeddings from MDLM and LLADA against a specific baseline (BERT-like model). Measures
- CKA
- Cosine similarity to check collapse of embeddings

This should guide the choice of the subsample parameters. And give an estimate of the best correlation possible between the diversity metric and the method.

## Sweeps

Hyperparameter sweeps (optuna TPE, not grid) for D6P3 models:
- Varying weights for dpp terms: `w_interaction`, `w_split` (no quality is also tested)
- Parameters of the subsample steps: `start` and `end`

Free (non conditioned) generation for MDLM 
LLADA models needs to start from a "prompt", mask random sequences from Fineweb (or other dataset) and generate the rest.

## Conditioned generation

LLADA model with a true prompt (at the beginning of the sequence). Possible to use a QA dataset like truthfulqa or similar.


## Metrics
- Diversity metrics: DPP logdet, Distinct n-grams, Entropy of n-grams, average cosine distance between embeddings of samples
- Quality metrics: Perplexity (LLM)
- Cosine alignment between best answer and ground truth (if available)
- Mauve against reference when sampling from fineweb.

## Baselines

IID baseline
Quality only baseline (no diversity)

