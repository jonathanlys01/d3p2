"""Small SQLite cache for resumable prompt/task generation runs."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import socket
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch

from d5p4.config import Config
from d5p4.utils import print


SCHEMA_VERSION = 1

HASH_EXCLUDED_CONFIG_KEYS = {
    "disable_sys_args",
    "interactive",
    "minimal_log",
    "quiet",
    "standalone_job",
    "cache_dir",
    "results_dir",
    "resume_runs",
    "resume_db_dir",
    "resume_db_timeout_s",
    "resume_db_keep_completed",
}


class ResumeLockError(RuntimeError):
    """Raised when another live process owns the resume DB lock."""


@dataclass
class ResumeRunState:
    store: ResumableRunStore | None
    completed_indices: set[int]
    unique_id: uuid.UUID


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    raise TypeError(f"config must be a dict or dataclass instance, got {type(config).__name__}.")


def semantic_config_dict(config: Any) -> dict[str, Any]:
    return {key: value for key, value in _config_to_dict(config).items() if key not in HASH_EXCLUDED_CONFIG_KEYS}


def manifest_hash(items: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(_json_dumps(list(items)).encode("utf-8")).hexdigest()


def experiment_hash(workflow_id: str, config: Any, work_manifest_hash: str) -> str:
    payload = {
        "workflow_id": workflow_id,
        "config": semantic_config_dict(config),
        "manifest_hash": work_manifest_hash,
    }
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def default_resume_dir(config: Any) -> Path:
    resume_db_dir = getattr(config, "resume_db_dir", None)
    if resume_db_dir:
        return Path(os.path.expandvars(os.path.expanduser(str(resume_db_dir))))
    return Path(str(getattr(config, "results_dir"))) / "resume"


def tensor_to_blob(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def tensor_from_blob(blob: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)


def owner_metadata() -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "started_at": time.time(),
    }


class ResumableRunStore:
    """Persistent token rows for one prompt/task generation experiment."""

    def __init__(
        self,
        *,
        config: Any,
        workflow_id: str,
        mode: str,
        work_items: list[dict[str, Any]],
        write_lock: bool = True,
    ) -> None:
        self.config = config
        self.workflow_id = workflow_id
        self.mode = mode
        self.work_items = work_items
        self.work_manifest_hash = manifest_hash(work_items)
        self.experiment_hash = experiment_hash(workflow_id, config, self.work_manifest_hash)
        self.db_dir = default_resume_dir(config)
        self.db_path = self.db_dir / f"{self.experiment_hash}.sqlite3"
        self.lock_path = self.db_dir / f"{self.experiment_hash}.lock"
        self.timeout_s = float(getattr(config, "resume_db_timeout_s", 60.0))
        self.write_lock = write_lock
        self._conn: sqlite3.Connection | None = None
        self._lock_file: Any | None = None
        self.run_uuid: str | None = None

    def __enter__(self) -> ResumableRunStore:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ResumableRunStore is not open.")
        return self._conn

    def open(self) -> None:
        self.db_dir.mkdir(parents=True, exist_ok=True)
        if self.write_lock:
            self._acquire_lock()
        self._conn = sqlite3.connect(self.db_path, timeout=self.timeout_s)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.execute(f"PRAGMA busy_timeout={int(self.timeout_s * 1000)}")
        self._create_schema()
        if self.write_lock:
            self._initialize_run()
        else:
            row = self.conn.execute(
                "SELECT run_uuid FROM runs WHERE experiment_hash = ?",
                (self.experiment_hash,),
            ).fetchone()
            self.run_uuid = None if row is None else str(row["run_uuid"])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def _acquire_lock(self) -> None:
        self._lock_file = open(self.lock_path, "a+")  # noqa: SIM115 - open file holds flock.
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            self._lock_file = None
            raise ResumeLockError(f"Another live job owns resumable run {self.experiment_hash}.") from exc

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                experiment_hash TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                model TEXT NOT NULL,
                config_json TEXT NOT NULL,
                work_manifest_json TEXT NOT NULL,
                manifest_hash TEXT NOT NULL,
                run_uuid TEXT NOT NULL,
                result_path TEXT,
                owner_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS generations (
                experiment_hash TEXT NOT NULL,
                item_index INTEGER NOT NULL,
                token_ids_blob BLOB NOT NULL,
                prompt_len INTEGER,
                internal_scores_json TEXT,
                sequence_scores_json TEXT,
                decoded_json TEXT,
                eval_decoded_json TEXT,
                selected_indices_json TEXT,
                result_json TEXT,
                generated_at REAL NOT NULL,
                decoded_at REAL,
                PRIMARY KEY (experiment_hash, item_index)
            );
            """,
        )
        self.conn.commit()

    def _initialize_run(self) -> None:
        now = time.time()
        with self.conn:
            row = self.conn.execute(
                "SELECT * FROM runs WHERE experiment_hash = ?",
                (self.experiment_hash,),
            ).fetchone()
            if row is None:
                self.run_uuid = str(uuid.uuid4())
                self.conn.execute(
                    """
                    INSERT INTO runs (
                        experiment_hash, schema_version, status, workflow_id, mode, model,
                        config_json, work_manifest_json, manifest_hash, run_uuid,
                        owner_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.experiment_hash,
                        SCHEMA_VERSION,
                        "running",
                        self.workflow_id,
                        self.mode,
                        str(getattr(self.config, "model", "")),
                        _json_dumps(_config_to_dict(self.config)),
                        _json_dumps(self.work_items),
                        self.work_manifest_hash,
                        self.run_uuid,
                        _json_dumps(owner_metadata()),
                        now,
                        now,
                    ),
                )
            else:
                if int(row["schema_version"]) != SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Unsupported resume DB schema {row['schema_version']} for {self.db_path}; "
                        f"expected {SCHEMA_VERSION}.",
                    )
                if str(row["manifest_hash"]) != self.work_manifest_hash:
                    raise RuntimeError(f"Manifest hash mismatch for existing resume DB {self.db_path}.")
                self.run_uuid = str(row["run_uuid"])
                self.conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, owner_json = ?, updated_at = ?
                    WHERE experiment_hash = ?
                    """,
                    ("running", _json_dumps(owner_metadata()), now, self.experiment_hash),
                )

    def generated_indices(self) -> set[int]:
        rows = self.conn.execute(
            "SELECT item_index FROM generations WHERE experiment_hash = ?",
            (self.experiment_hash,),
        ).fetchall()
        return {int(row["item_index"]) for row in rows}

    def record_generated(
        self,
        *,
        item_index: int,
        token_ids: torch.Tensor,
        prompt_len: int | None = None,
        internal_scores: list[float] | None = None,
        sequence_scores: list[float] | None = None,
    ) -> None:
        now = time.time()
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO generations (
                    experiment_hash, item_index, token_ids_blob, prompt_len,
                    internal_scores_json, sequence_scores_json, generated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.experiment_hash,
                    item_index,
                    tensor_to_blob(token_ids),
                    prompt_len,
                    _json_dumps(internal_scores) if internal_scores is not None else None,
                    _json_dumps(sequence_scores) if sequence_scores is not None else None,
                    now,
                ),
            )
            self.conn.execute(
                "UPDATE runs SET updated_at = ? WHERE experiment_hash = ?",
                (now, self.experiment_hash),
            )

    def record_decoded(
        self,
        *,
        item_index: int,
        decoded: list[str],
        result: dict[str, Any] | None = None,
        eval_decoded: list[str] | None = None,
        selected_indices: list[int] | None = None,
    ) -> None:
        now = time.time()
        with self.conn:
            self.conn.execute(
                """
                UPDATE generations
                SET decoded_json = ?, eval_decoded_json = ?, selected_indices_json = ?,
                    result_json = ?, decoded_at = ?
                WHERE experiment_hash = ? AND item_index = ?
                """,
                (
                    _json_dumps(decoded),
                    _json_dumps(eval_decoded) if eval_decoded is not None else None,
                    _json_dumps(selected_indices) if selected_indices is not None else None,
                    _json_dumps(result) if result is not None else None,
                    now,
                    self.experiment_hash,
                    item_index,
                ),
            )
            self.conn.execute(
                "UPDATE runs SET updated_at = ? WHERE experiment_hash = ?",
                (now, self.experiment_hash),
            )

    def get_generation(self, item_index: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM generations WHERE experiment_hash = ? AND item_index = ?",
            (self.experiment_hash, item_index),
        ).fetchone()
        if row is None:
            return None
        return {
            "tokens": tensor_from_blob(row["token_ids_blob"]),
            "prompt_len": row["prompt_len"],
            "internal_scores": json.loads(row["internal_scores_json"]) if row["internal_scores_json"] else None,
            "sequence_scores": json.loads(row["sequence_scores_json"]) if row["sequence_scores_json"] else None,
            "decoded": json.loads(row["decoded_json"]) if row["decoded_json"] else None,
            "eval_decoded": json.loads(row["eval_decoded_json"]) if row["eval_decoded_json"] else None,
            "selected_indices": json.loads(row["selected_indices_json"]) if row["selected_indices_json"] else None,
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
        }

    def complete(self, result_path: str | None = None) -> None:
        now = time.time()
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, result_path = ?, updated_at = ?
            WHERE experiment_hash = ?
            """,
            ("complete", result_path, now, self.experiment_hash),
        )
        self.conn.commit()

    def release(self, result_path: str | None = None) -> None:
        self.complete(result_path)
        keep_completed = bool(getattr(self.config, "resume_db_keep_completed", False))
        db_path = self.db_path
        lock_path = self.lock_path
        self.close()
        if keep_completed:
            return
        for path in (db_path, lock_path):
            with suppress(FileNotFoundError):
                path.unlink()


def make_work_items(
    count: int,
    *,
    prefix: str = "item",
    prompts: list[str] | None = None,
    references: list[Any] | None = None,
    metadata: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    items = []
    for index in range(count):
        item_metadata = {} if metadata is None else metadata[index]
        item = {
            "item_index": index,
            "item_key": item_metadata.get("item_key", f"{prefix}:{index}"),
            "metadata": item_metadata,
        }
        if prompts is not None:
            item["prompt"] = prompts[index]
        if references is not None:
            item["references"] = references[index]
        items.append(item)
    return items


def open_resumable_run(  # noqa: PLR0913
    *,
    config: Any,
    workflow_id: str,
    work_items: list[dict[str, Any]],
    distributed_utils: Any | None,
    master: bool,
    mode: str = "prompt_generation",
) -> ResumeRunState:
    store = None
    completed_indices: set[int] = set()
    if bool(getattr(config, "resume_runs", False)) and master:
        store = ResumableRunStore(
            config=config,
            workflow_id=workflow_id,
            mode=mode,
            work_items=work_items,
        )
        store.open()
        completed_indices = store.generated_indices()
        assert store.run_uuid is not None
        unique_id = uuid.UUID(store.run_uuid)
    else:
        unique_id = uuid.uuid4()

    completed_indices, unique_id_str = sync_resume_state(completed_indices, str(unique_id), distributed_utils)
    return ResumeRunState(
        store=store,
        completed_indices=completed_indices,
        unique_id=uuid.UUID(unique_id_str),
    )


def release_resumable_run(
    *,
    config: Any,
    workflow_id: str,
    work_items: list[dict[str, Any]],
    result_path: str | None,
    mode: str = "prompt_generation",
) -> None:
    if not bool(getattr(config, "resume_runs", False)):
        return
    release_store = ResumableRunStore(
        config=config,
        workflow_id=workflow_id,
        mode=mode,
        work_items=work_items,
    )
    release_store.open()
    release_store.release(result_path)


def sync_resume_state(
    completed_indices: set[int],
    run_uuid: str,
    distributed_utils: Any | None,
) -> tuple[set[int], str]:
    if distributed_utils is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return completed_indices, run_uuid

    obj_list = [(sorted(completed_indices), run_uuid)] if distributed_utils.rank == 0 else [None]
    torch.distributed.broadcast_object_list(obj_list, src=0)
    assert obj_list[0] is not None
    completed, synced_uuid = obj_list[0]
    return set(completed), synced_uuid


def sync_resume_item(item: Any, distributed_utils: Any | None) -> Any:
    if distributed_utils is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return item

    obj_list = [item] if distributed_utils.rank == 0 else [None]
    torch.distributed.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def run_generator_loop(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    config: Config,
    model: Any,
    prompts: list[str],
    references: list[Any] | None = None,
    metadata: list[dict[str, Any]] | None = None,
    workflow_id: str,
    prefix: str = "item",
    mode: str = "prompt_generation",
    sample_fn: Callable[[str], Any],
    decode_fn: Callable[[str, Any], list[str]],
    score_fn: Callable[..., Any] | None = None,
    post_process_fn: Callable[..., Any] | None = None,
    verbose_log_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """A generic, resumable sampling and scoring engine for distributed/single runs.

    Coordinates SQLite caching on rank 0 and DDP broadcasts to worker ranks.
    """
    master = model.distributed_utils is None or model.distributed_utils.rank == 0

    work_items = make_work_items(
        len(prompts),
        prefix=prefix,
        prompts=prompts,
        references=references,
        metadata=metadata,
    )

    resume_state = open_resumable_run(
        config=config,
        workflow_id=workflow_id,
        work_items=work_items,
        distributed_utils=model.distributed_utils,
        master=master,
        mode=mode,
    )

    store = resume_state.store
    completed_indices = resume_state.completed_indices
    unique_id = resume_state.unique_id
    print(f"Experiment ID: {unique_id}")

    results = []
    generations_all = []
    internal_scores_all = []
    eval_generations_all = []
    selected_indices_all = []

    try:
        for i, prompt_item in enumerate(prompts):
            prompt = sync_resume_item(prompt_item, model.distributed_utils)

            if i in completed_indices:
                if not master:
                    continue

                assert store is not None
                generation = store.get_generation(i)
                assert generation is not None

                raw_samples = generation["tokens"]
                scores = generation["internal_scores"]

                decoded = generation["decoded"]
                if decoded is None:
                    decoded = decode_fn(prompt, raw_samples)

                result = generation["result"]
                if result is None and score_fn is not None:
                    result = score_fn(i, prompt, decoded)
                    store.record_decoded(
                        item_index=i,
                        decoded=decoded,
                        result=result,
                        eval_decoded=generation["eval_decoded"],
                        selected_indices=generation["selected_indices"],
                    )

                eval_decoded = generation["eval_decoded"]
                sel_indices = generation["selected_indices"]
                if post_process_fn is not None and (eval_decoded is None or sel_indices is None):
                    eval_decoded, sel_indices = post_process_fn(i, prompt, decoded, scores)
                    store.record_decoded(
                        item_index=i,
                        decoded=decoded,
                        result=result,
                        eval_decoded=eval_decoded,
                        selected_indices=sel_indices,
                    )
            else:
                if master:
                    log_suffix = ""
                    if references is not None and i < len(references) and references[i]:
                        ref_val = references[i][0] if isinstance(references[i], list) else references[i]
                        log_suffix = f"  (ref={ref_val!r})"
                    elif metadata is not None and i < len(metadata):
                        meta = metadata[i]
                        if "gold_answer" in meta:
                            log_suffix = f"  (gold={meta['gold_answer']!r})"
                        elif "task_id" in meta:
                            log_suffix = f"  (task={meta['task_id']!r})"
                        elif "item_key" in meta:
                            log_suffix = f"  (item={meta['item_key']!r})"
                    print(f"Sampling {i + 1}/{len(prompts)}{log_suffix}...", progress=True)

                raw_samples, internal_scores = sample_fn(prompt)

                if not master:
                    continue

                prompt_len = model._preprocess_prompt(prompt).shape[1]

                scores = None
                if internal_scores is not None:
                    if torch.is_tensor(internal_scores):
                        scores = [float(s) for s in internal_scores.detach().cpu().tolist()]
                    else:
                        scores = internal_scores

                if store is not None:
                    store.record_generated(
                        item_index=i,
                        token_ids=raw_samples,
                        prompt_len=prompt_len,
                        internal_scores=scores,
                    )

                decoded = decode_fn(prompt, raw_samples)

                result = None
                if score_fn is not None:
                    result = score_fn(i, prompt, decoded)

                eval_decoded = None
                sel_indices = None
                if post_process_fn is not None:
                    eval_decoded, sel_indices = post_process_fn(i, prompt, decoded, scores)

                if store is not None:
                    store.record_decoded(
                        item_index=i,
                        decoded=decoded,
                        result=result,
                        eval_decoded=eval_decoded,
                        selected_indices=sel_indices,
                    )

            if master:
                if result is not None:
                    results.append(result)
                generations_all.append(decoded)
                if scores is not None:
                    internal_scores_all.append(scores)
                if eval_decoded is not None:
                    eval_generations_all.append(eval_decoded)
                if sel_indices is not None:
                    selected_indices_all.append(sel_indices)

                if verbose_log_fn is not None:
                    verbose_log_fn(i, result, decoded)
    finally:
        if store is not None:
            store.close()

    return {
        "unique_id": unique_id,
        "results": results if results else None,
        "generations": generations_all,
        "internal_scores": internal_scores_all if internal_scores_all else None,
        "eval_generations": eval_generations_all if eval_generations_all else None,
        "selected_indices": selected_indices_all if selected_indices_all else None,
        "work_items": work_items,
    }
