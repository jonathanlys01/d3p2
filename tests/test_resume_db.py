import os
import tempfile

import pytest
import torch

from d5p4.config import Config
from d5p4.resume_db import (
    ResumableRunStore,
    ResumeLockError,
    experiment_hash,
    make_work_items,
    manifest_hash,
    run_generator_loop,
    semantic_config_dict,
)


def _cfg(tmpdir: str, **kwargs):
    return Config(
        disable_sys_args=True,
        model="llada",
        results_dir=os.path.join(tmpdir, "results-a"),
        resume_db_dir=os.path.join(tmpdir, "shared-resume"),
        resume_db_keep_completed=False,
        **kwargs,
    )


def test_semantic_config_excludes_node_local_paths():
    cfg_a = Config(
        disable_sys_args=True,
        model="mdlm",
        cache_dir="/node-a/cache",
        results_dir="/node-a/results",
        resume_db_dir="/shared/resume",
    )
    cfg_b = Config(
        disable_sys_args=True,
        model="mdlm",
        cache_dir="/node-b/cache",
        results_dir="/node-b/results",
        resume_db_dir="/other/shared/resume",
        quiet=True,
    )

    assert semantic_config_dict(cfg_a) == semantic_config_dict(cfg_b)


def test_experiment_hash_uses_workflow_config_and_manifest_only():
    cfg_a = Config(disable_sys_args=True, model="llada", cache_dir="/node-a/cache")
    cfg_b = Config(disable_sys_args=True, model="llada", cache_dir="/node-b/cache")
    items = make_work_items(2, prefix="prompt", prompts=["a", "b"])
    work_hash = manifest_hash(items)

    assert experiment_hash("prompt_generation:llada", cfg_a, work_hash) == experiment_hash(
        "prompt_generation:llada",
        cfg_b,
        work_hash,
    )
    assert experiment_hash("other", cfg_a, work_hash) != experiment_hash("prompt_generation:llada", cfg_a, work_hash)


def test_store_roundtrips_tokens_and_decoded_payload():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg(tmpdir)
        items = make_work_items(1, prefix="prompt", prompts=["hello"])
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

        with ResumableRunStore(
            config=cfg,
            workflow_id="prompt_generation:llada",
            mode="prompt_generation",
            work_items=items,
        ) as store:
            store.record_generated(item_index=0, token_ids=tokens, prompt_len=1, internal_scores=[0.5])
            store.record_decoded(
                item_index=0,
                decoded=["abc"],
                eval_decoded=["abc"],
                result={"accuracy": 1.0},
            )
            loaded = store.get_generation(0)
            assert loaded is not None
            torch.testing.assert_close(loaded["tokens"], tokens)
            assert loaded["prompt_len"] == 1
            assert loaded["internal_scores"] == [0.5]
            assert loaded["decoded"] == ["abc"]
            assert loaded["result"] == {"accuracy": 1.0}
            assert store.generated_indices() == {0}


def test_store_uses_configured_shared_db_dir_and_releases_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg(tmpdir)
        items = make_work_items(1, prefix="run")
        store = ResumableRunStore(
            config=cfg,
            workflow_id="prompt_generation:llada",
            mode="prompt_generation",
            work_items=items,
        )
        store.open()
        db_path = store.db_path
        lock_path = store.lock_path

        assert str(db_path).startswith(cfg.resume_db_dir)
        assert db_path.exists()
        assert lock_path.exists()

        store.release("result.json")

        assert not db_path.exists()
        assert not lock_path.exists()


def test_store_lock_rejects_second_live_owner():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg(tmpdir)
        items = make_work_items(1, prefix="run")
        first = ResumableRunStore(
            config=cfg,
            workflow_id="prompt_generation:llada",
            mode="prompt_generation",
            work_items=items,
        )
        second = ResumableRunStore(
            config=cfg,
            workflow_id="prompt_generation:llada",
            mode="prompt_generation",
            work_items=items,
        )
        first.open()
        try:
            with pytest.raises(ResumeLockError):
                second.open()
        finally:
            first.close()


def test_generator_loop_works_without_resume_store():
    class DummyModel:
        distributed_utils = None

        def _preprocess_prompt(self, _prompt: str):
            return torch.tensor([[1, 2]], dtype=torch.long)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg(tmpdir, resume_runs=False)
        output = run_generator_loop(
            config=cfg,
            model=DummyModel(),
            prompts=["hello"],
            workflow_id="prompt_generation:llada",
            sample_fn=lambda _prompt: (torch.tensor([[1, 2, 3]], dtype=torch.long), None),
            decode_fn=lambda _prompt, _tokens: ["decoded"],
        )

        assert output["generations"] == [["decoded"]]
        assert output["results"] is None
        assert not os.path.exists(cfg.resume_db_dir)
