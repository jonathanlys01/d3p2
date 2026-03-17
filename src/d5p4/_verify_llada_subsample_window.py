"""Print the effective LLaDA subsampling schedule for a given config.

This script does not load the model. It only mirrors the control flow in
`diffusion_llada.py` to show when subsampling is active.

Example:
    python -m d5p4.verify_llada_subsample_window \
        model=llada gen_length=128 block_length=32 llada_steps=128 \
        n_groups=3 group_size=2 subsample_start=28 subsample_end=31
"""

from d5p4.config import Config


def main() -> None:
    cfg = Config()
    assert cfg.model == "llada", "Use this script with model=llada"

    num_blocks = cfg.gen_length // cfg.block_length
    steps_per_block = cfg.llada_steps // num_blocks
    batch_size = cfg.n_groups * cfg.group_size

    print("LLaDA subsampling schedule")
    print("=" * 32)
    print(f"gen_length       : {cfg.gen_length}")
    print(f"block_length     : {cfg.block_length}")
    print(f"num_blocks       : {num_blocks}")
    print(f"llada_steps      : {cfg.llada_steps}")
    print(f"steps_per_block  : {steps_per_block}")
    print(f"n_groups         : {cfg.n_groups}")
    print(f"group_size       : {cfg.group_size}")
    print(f"batch_size       : {batch_size}")
    print(f"subsample_start  : {cfg.subsample_start}")
    print(f"subsample_end    : {cfg.subsample_end}")
    print()

    total_subsample_events = 0
    total_expand_events = 0

    for block in range(num_blocks):
        print(f"Block {block}: token range [{block * cfg.block_length}, {(block + 1) * cfg.block_length - 1}]")
        active_steps: list[int] = []
        for step in range(steps_per_block):
            active = cfg.subsample_start <= step <= cfg.subsample_end
            marker = "subsample" if active else "keep-all"
            print(f"  step {step:>2}: {marker}")
            if active:
                active_steps.append(step)
                total_subsample_events += 1
                total_expand_events += cfg.n_groups * cfg.group_size
        if active_steps:
            print(f"  active step indices in this block: {active_steps}")
        else:
            print("  active step indices in this block: []")
        print()

    print("Summary")
    print("=" * 32)
    print(f"subsample events across full sample : {total_subsample_events}")
    print(f"selected parents per event          : {cfg.n_groups}")
    print(f"expanded branches per event         : {cfg.n_groups * cfg.group_size}")
    print()
    print("Interpretation:")
    print("- The subsampling window resets for every block.")
    print("- There is no config-only way to target only the final block.")
    print("- If you want a global step schedule, the sampler logic must change.")


if __name__ == "__main__":
    main()
