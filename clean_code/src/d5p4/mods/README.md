# Modified Samplers

This folder contains experimental or modified variants of the core MDLM and LLaDA modeling and single-run code.

- `profile/`: profiler-oriented LLaDA sampler and trace runner.
- `esmc/`: SMC-flavored and Entropy-based SMC (ESMC) MDLM sampler variants.

The stable release-facing samplers remain at the package root:

- `d5p4.diffusion_llada`
- `d5p4.diffusion_mdlm`
- `d5p4.single_run_llada`
- `d5p4.single_run_mdlm`

Run modified entrypoints with module paths, for example:

```bash
python -m d5p4.mods.esmc.single_run_smc_mdlm
python -m d5p4.mods.profile.profile_llada_trace
```

