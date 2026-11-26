"""B4: the open-ended MDLM wave, its held-out prompts, generation helpers, featurization and paired summaries."""

import io
import json
import lzma
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

import d5p4.waves.__main__ as cli
from d5p4.data import owt_heldout
from d5p4.data.owt_heldout import build_open_chunks, iter_documents, pack_chunks
from d5p4.mdlm_open import copy_mismatches, initial_state, kept_rows, prompt_seed
from d5p4.waves import b4, get_wave, open_fronts
from d5p4.waves.open_eval import FEATURES, REFERENCE_DIR, flatten_run, token_entropy
from d5p4.waves.spec import DONE_MARKER


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "src" / "d5p4" / "_default.yaml"


# Wave -------------------------------------------------------------------------------------------------------------


def test_b4_order_core_points_first_then_second_half_then_multiplicative():
    runs = get_wave("b4")

    assert len(runs) == 18 * 3 * 2 + 4 * 3 * 2
    first = [run.tag for run in runs[:18]]
    assert first[:6] == [
        "ind_t0p9_seed0_s0of2",
        "ind_t1_seed0_s0of2",
        "ind_t1p05_seed0_s0of2",
        "ind_t1p1_seed0_s0of2",
        "ind_t1p2_seed0_s0of2",
        "gbs_t1_seed0_s0of2",
    ]
    assert first[6] == "dbs_a1_t1_seed0_s0of2"
    assert first[-1] == "d5p4_w1000_t1_seed0_s0of2"
    assert all(run.tag.endswith("_s0of2") for run in runs[:54])
    assert all(run.tag.endswith("_s1of2") for run in runs[54:108])
    assert all(run.tag.startswith("d5p4x_") for run in runs[108:])
    assert {run.overrides["seed"] for run in runs[:54]} == {0, 1, 2}


def test_b4_pools_match_and_every_run_is_mdlm_open_generation():
    for run in get_wave("b4"):
        overrides = run.all_overrides()
        assert run.task.family == "open"
        assert run.task.entry == "mdlm_open.py"
        assert overrides["n_groups"] * overrides["group_size"] == 32
        assert overrides["model"] == "mdlm"
        assert overrides["mdlm_steps"] == overrides["subsample_end"] == 1024
        if run.method == "baseline":
            assert overrides["group_size"] == 1
        else:
            assert (overrides["n_groups"], overrides["group_size"]) == (8, 4)


@pytest.mark.parametrize("run", get_wave("b4")[:18] + get_wave("b4")[108:112], ids=lambda run: run.tag)
def test_b4_runs_build_valid_configs(run):
    config = cli.build_config(run, DEFAULT_CONFIG)

    assert config.method == run.method
    assert config.batch_size == 32
    assert config.qa_num_shards == 2
    assert config.open_num_prompts == 128
    if run.tag.startswith("d5p4x"):
        assert config._kernel_method == "multiplicative"
    elif run.tag.startswith("d5p4"):
        assert config._kernel_method == "additive"


def test_b4_smoke_limit_caps_prompts(tmp_path):
    command = cli.run_command(get_wave("b4")[1], tmp_path, DEFAULT_CONFIG, None, limit=4)

    assert command[1].endswith("src/d5p4/mdlm_open.py")
    assert "open_num_prompts=4" in command
    assert "qa_shard_index=0" in command


# Held-out prompts -------------------------------------------------------------------------------------------------


def _member(name: str, payload: bytes) -> tuple[tarfile.TarInfo, io.BytesIO]:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    return info, io.BytesIO(payload)


def _owt_archive(path: Path, members: list[list[str]]) -> None:
    """An OpenWebText-style subset archive: .xz members, each an xz-compressed tar of .txt documents."""
    with tarfile.open(path, "w") as outer:
        for m, docs in enumerate(members):
            inner_bytes = io.BytesIO()
            with tarfile.open(fileobj=inner_bytes, mode="w") as inner:
                for d, doc in enumerate(docs):
                    inner.addfile(*_member(f"{m:03d}-{d}.txt", doc.encode()))
            outer.addfile(*_member(f"openwebtext/subset-{m}_data.xz", lzma.compress(inner_bytes.getvalue())))


class CharTokenizer:
    """One token per character; BOS = EOS = 0 as in GPT-2."""

    bos_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_iter_documents_reads_the_last_members_in_archive_order(tmp_path):
    archive = tmp_path / "subset.tar"
    _owt_archive(archive, [["a0", "a1"], ["b0"], ["c0", "c1"]])

    assert list(iter_documents(str(archive), 2)) == ["b0", "c0", "c1"]
    with pytest.raises(ValueError):
        list(iter_documents(str(archive), 4))


def test_pack_chunks_frames_wrapped_documents_like_mdlm():
    chunks = pack_chunks([[1, 2, 3], [4, 5], [6, 7, 8, 9]], bos_id=0, eos_id=99, count=2, block_length=6)

    # documents + EOS: 1 2 3 99 4 5 99 6 7 8 9 99, cut into blocks of 4 and framed by BOS / EOS
    assert chunks == [[0, 1, 2, 3, 99, 99], [0, 4, 5, 99, 6, 99]]
    with pytest.raises(ValueError):
        pack_chunks([[1, 2]], bos_id=0, eos_id=99, count=2, block_length=6)


def test_open_chunks_split_prompts_and_references_independently_of_the_prompt_count():
    docs = ["x" * 30] * 20
    full = build_open_chunks(
        docs, CharTokenizer(), num_prompts=3, num_references=2, prompt_tokens=2, block_length=10, prompt_pool=4
    )
    smoke = build_open_chunks(
        docs, CharTokenizer(), num_prompts=1, num_references=2, prompt_tokens=2, block_length=10, prompt_pool=4
    )

    assert len(full.prompts) == 3 and len(smoke.prompts) == 1
    assert full.references == smoke.references
    assert all(len(c) == 10 for c in full.prompts + full.references)
    assert full.prefix(0) == full.prompts[0][:3]
    assert full.continuation(full.references[0]) == full.references[0][3:]


def test_load_open_chunks_reads_the_configured_archive(tmp_path, monkeypatch):
    archive = tmp_path / "subset.tar"
    _owt_archive(archive, [["z" * 500] * 4, ["y" * 500] * 30])
    config = cli.build_config(
        get_wave("b4")[0],
        DEFAULT_CONFIG,
        {"open_data_path": str(archive), "open_heldout_members": 1, "open_num_prompts": 2, "open_num_references": 3},
    )
    monkeypatch.setattr(owt_heldout, "PROMPT_POOL", 4)
    monkeypatch.setattr(owt_heldout, "build_open_chunks", lambda *a, **kw: (list(a[0]), kw))

    documents, kwargs = owt_heldout.load_open_chunks(config, CharTokenizer())

    assert documents == ["y" * 500] * 30
    assert kwargs == {"num_prompts": 2, "num_references": 3, "prompt_tokens": 32, "block_length": 1024}


# Generation helpers -----------------------------------------------------------------------------------------------


def test_search_runs_keep_one_copy_per_group_and_independent_runs_the_pool():
    assert kept_rows(32, 4) == [0, 4, 8, 12, 16, 20, 24, 28]
    assert kept_rows(32, 1) == list(range(32))


def test_copy_mismatches_counts_groups_whose_copies_differ():
    tokens = torch.tensor([[1, 2], [1, 2], [3, 4], [3, 5]])
    assert copy_mismatches(tokens, 2) == 1
    assert copy_mismatches(tokens, 1) == 0


def test_initial_state_keeps_the_prefix_and_masks_the_rest():
    x = initial_state([50256, 7, 8], batch_size=2, length=6, mask_index=50257)
    assert x.tolist() == [[50256, 7, 8, 50257, 50257, 50257]] * 2


def test_prompt_seeds_differ_across_seeds_and_prompts():
    seeds = {prompt_seed(seed, prompt) for seed in range(3) for prompt in range(1024)}
    assert len(seeds) == 3 * 1024


# MDLM without flash-attn ------------------------------------------------------------------------------------------


def test_torch_rotary_matches_the_flash_attn_convention():
    from d5p4.mdlm_ref import modeling_mdlm

    torch.manual_seed(0)
    rotary = modeling_mdlm.Rotary(8)
    qkv = torch.randn(2, 5, 3, 2, 8)
    cos, sin = rotary(qkv)
    out = modeling_mdlm.apply_rotary_pos_emb_torch(qkv.clone(), cos, sin)

    # flash_attn.layers.rotary, non-interleaved: (x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin) on q and k only
    c, s = cos[0, :, 0, 0, :4][None, :, None], sin[0, :, 0, 0, :4][None, :, None]
    for slot in (0, 1):
        x1, x2 = qkv[:, :, slot, :, :4], qkv[:, :, slot, :, 4:]
        expected = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
        assert torch.allclose(out[:, :, slot], expected, atol=1e-6)
    assert torch.equal(out[:, :, 2], qkv[:, :, 2])


def test_mdlm_forward_runs_without_flash_attn(monkeypatch):
    from d5p4.mdlm_ref import modeling_mdlm
    from d5p4.mdlm_ref.configuration_mdlm import MDLMConfig

    monkeypatch.setattr(modeling_mdlm, "HAS_FLASH_ATTN", False)
    monkeypatch.setattr(modeling_mdlm, "SUPPORTS_FLASH", False)
    config = MDLMConfig(vocab_size=50, model_length=16, hidden_dim=32, cond_dim=8, n_blocks=2, n_heads=2)
    model = modeling_mdlm.MDLM(config).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 50, (3, 16)), return_dict=True, output_hidden_states=True)

    assert out.logits.shape == (3, 16, 50)
    assert len(out.hidden_states) == 3
    assert torch.isfinite(out.logits).all()


# Featurization and summaries --------------------------------------------------------------------------------------


def test_token_entropy():
    assert token_entropy([]) == 0.0
    assert token_entropy([5, 5, 5]) == 0.0
    assert token_entropy([1, 2, 3, 4]) == pytest.approx(np.log(4))


def test_flatten_run_keeps_prompt_and_slot():
    texts, prompts, slots = flatten_run({"dataset_indices": [4, 6], "text_samples": [["a", "b"], ["c", "d"]]})
    assert texts == ["a", "b", "c", "d"]
    assert prompts.tolist() == [4, 4, 6, 6]
    assert slots.tolist() == [0, 1, 0, 1]


def test_independent_pools_are_split_into_k_sets_and_mauve_uses_the_first_k():
    rng = np.random.default_rng(0)
    slots = np.array([3, 0, 2, 1, 5, 4, 7, 6])
    sets, mauve_rows = open_fronts.prompt_sets(slots, 2, independent=True, rng=rng)
    assert sorted(np.concatenate(sets).tolist()) == list(range(8))
    assert all(len(s) == 2 for s in sets)
    assert sorted(slots[mauve_rows].tolist()) == [0, 1]

    sets, mauve_rows = open_fronts.prompt_sets(np.array([1, 0]), 2, independent=False, rng=rng)
    assert len(sets) == 1 and sorted(mauve_rows.tolist()) == [0, 1]


def test_self_cosine_is_the_mean_off_diagonal_similarity():
    e = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert open_fronts.self_cosine(e) == pytest.approx(2 / 6)
    assert open_fronts.self_cosine(e[:1]) == 0.0


def test_holm_and_bootstrap_p():
    assert open_fronts.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert open_fronts.bootstrap_p(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.25)
    assert open_fronts.bootstrap_p(np.array([-1.0, 1.0, 2.0, 3.0])) == pytest.approx(0.5)


def _fake_open_root(root: Path, runs, n_prompts=12, dim=8):
    """Result files and features for `runs`: D5P4 sets are more diverse, every method has the same perplexity."""
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(150, dim)).astype(np.float32)
    ref_dir = root / "eval" / runs[0].task.name / REFERENCE_DIR
    ref_dir.mkdir(parents=True)
    np.savez(
        ref_dir / FEATURES, nll=np.full(150, 3.0), entropy=np.full(150, 5.0), tokens=np.full(150, 900), mauve=reference
    )
    for run in runs:
        k = run.task.k
        per_prompt = run.task.pool_size if run.method == "baseline" else k
        texts = [[f"{run.tag} prompt {q} text {s} " + "word " * s for s in range(per_prompt)] for q in range(n_prompts)]
        out = run.results_dir(root)
        out.mkdir(parents=True)
        (out / "exp-1.json").write_text(json.dumps({"dataset_indices": list(range(n_prompts)), "text_samples": texts}))
        (out / DONE_MARKER).write_text("{}")
        n = n_prompts * per_prompt
        spread = 1.0 if run.tag.startswith("d5p4") else 0.2
        jina = rng.normal(size=(n, 4)) * spread + np.array([1.0, 0, 0, 0])
        jina /= np.linalg.norm(jina, axis=1, keepdims=True)
        feature_dir = root / "eval" / run.task.name / run.tag
        feature_dir.mkdir(parents=True)
        np.savez(
            feature_dir / FEATURES,
            nll=rng.normal(3.0, 0.1, n),
            entropy=np.full(n, 5.0),
            tokens=np.full(n, 900),
            mauve=rng.normal(size=(n, dim)).astype(np.float32),
            jina=jina.astype(np.float32),
            prompt=np.repeat(np.arange(n_prompts), per_prompt),
            slot=np.tile(np.arange(per_prompt), n_prompts),
        )


def test_fronts_command_writes_open_summaries_and_comparisons(tmp_path, monkeypatch):
    from dataclasses import replace

    from d5p4.waves.spec import d5p4, diverse_beam, independent, seeded

    task = replace(b4.MDLM_OWT, n_groups=2, group_size=2)
    runs = [
        seeded(run, seed)
        for seed in (0, 1)
        for run in (
            independent(task, 1.0),
            independent(task, 1.2),
            diverse_beam(task, 10.0),
            d5p4(task, 10.0),
            d5p4(task, 100.0),
        )
    ]
    monkeypatch.setattr(cli, "get_wave", lambda name: runs)
    _fake_open_root(tmp_path, runs)

    assert cli.main(["fronts", "--wave", "fake", "--root", str(tmp_path), "--mauve-samples", "5"]) == 0

    import csv

    points = list(csv.DictReader((tmp_path / "fronts" / "open_points.csv").open()))
    assert [p["tag"] for p in points] == [
        "d5p4_w100_t1",
        "d5p4_w10_t1",
        "dbs_a10_t1",
        "ind_t1",
        "ind_t1p2",
        "human_reference",
    ]
    by_tag = {p["tag"]: p for p in points}
    assert float(by_tag["ind_t1"]["self_cos_minus_ref"]) == 0.0
    assert float(by_tag["d5p4_w10_t1"]["self_cos"]) < float(by_tag["dbs_a10_t1"]["self_cos"])
    assert 0.0 < float(by_tag["d5p4_w10_t1"]["mauve"]) <= 1.0
    per_prompt = list(csv.DictReader((tmp_path / "fronts" / "open_per_prompt.csv").open()))
    assert len(per_prompt) == len(runs) * 12

    comparisons = list(csv.DictReader((tmp_path / "fronts" / "open_comparisons.csv").open()))
    statistics = {c["statistic"] for c in comparisons}
    assert statistics == {"hv_cos", "hv_bleu", "max_mauve", "max_mauve_star"}
    assert {c["baseline"] for c in comparisons} == {"ind", "dbs"}
    cos_vs_dbs = next(c for c in comparisons if c["statistic"] == "hv_cos" and c["baseline"] == "dbs")
    assert float(cos_vs_dbs["difference"]) > 0
    assert float(cos_vs_dbs["p_holm"]) >= float(cos_vs_dbs["p_bootstrap"])
