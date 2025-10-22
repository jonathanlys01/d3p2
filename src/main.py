"""
Main 5D3P experiment script.
(Distributed DPP Sampling for Discrete Diffusion Models)
"""

from config import Config
from diffusion import DDM
from utils import seed_all


def main():
    config = Config()
    seed_all(config.seed)

    model = DDM(config)

    texts = []

    for i in range(config.n_runs):
        print(f"Sampling batch {i + 1}/{config.n_runs}...")
        samples = model.sample(num_steps=config.num_steps)
        print(samples)
        texts.extend(model.tokenizer.batch_decode(samples, skip_special_tokens=True))

    print("\nGenerated Samples:")
    for i, text in enumerate(texts):
        print(f"Sample {i + 1}: {text}")

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
