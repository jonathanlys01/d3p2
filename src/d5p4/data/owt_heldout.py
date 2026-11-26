"""Held-out OpenWebText chunks for open-ended MDLM generation: prompts and MAUVE references.

mdlm-owt (Sahoo et al., 2024) trains on `openwebtext` train[:-100000] and validates on its last 100,000 documents.
The Hugging Face loader yields documents subset by subset (urlsf_subset00 to urlsf_subset20) and, inside a subset, in
the archive order of its .xz members. urlsf_subset20, the last subset, holds about 235k documents in 612 members of
about 385 documents each, so its last `open_heldout_members` members (32 by default, about 12k documents) lie well
inside the held-out 100k. Nothing here was seen in training.

Documents are packed the way MDLM packs its training data (`wrap`): the GPT-2 tokens of each document followed by EOS,
concatenated, cut into blocks of 1022 tokens, each framed as [BOS] + block + [EOS] (1024 tokens, BOS = EOS for GPT-2).
The first `PROMPT_POOL` chunks (in document order) are prompts: a run uses the first `open_num_prompts` of them,
keeping BOS and the first `open_prompt_tokens` tokens and filling the rest. The next `open_num_references` chunks are
the MAUVE reference: their text after the same number of tokens, so references and generated continuations have the
same length and framing. The reference set does not depend on how many prompts a run uses (smoke runs included).
"""

from __future__ import annotations

import io
import lzma
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from d5p4.config import Config


BLOCK_LENGTH = 1024
PROMPT_POOL = 1024


@dataclass(frozen=True)
class OpenChunks:
    """Token ids of the prompt chunks and the reference chunks, each [BOS] + 1022 tokens + [EOS]."""

    prompts: list[list[int]]
    references: list[list[int]]
    prompt_tokens: int

    def prefix(self, index: int) -> list[int]:
        """BOS and the first `prompt_tokens` tokens of prompt `index`: the positions generation keeps."""
        return self.prompts[index][: 1 + self.prompt_tokens]

    def continuation(self, chunk: list[int]) -> list[int]:
        """The tokens after the prefix: what generation produces, and what the reference text is made of."""
        return chunk[1 + self.prompt_tokens :]


def iter_documents(tar_path: str, members: int) -> Iterator[str]:
    """Documents of the last `members` .xz members of an OpenWebText subset archive, in loader order."""
    with tarfile.open(tar_path) as outer:
        xz_members = [m for m in outer.getmembers() if m.name.endswith(".xz")]
        if members > len(xz_members):
            raise ValueError(f"{tar_path} has {len(xz_members)} .xz members, fewer than the {members} requested.")
        for member in xz_members[len(xz_members) - members :]:
            handle = outer.extractfile(member)
            assert handle is not None
            with tarfile.open(fileobj=io.BytesIO(lzma.decompress(handle.read()))) as inner:
                for doc in inner.getmembers():
                    if not doc.isfile():
                        continue
                    doc_handle = inner.extractfile(doc)
                    assert doc_handle is not None
                    yield doc_handle.read().decode("utf-8", errors="replace")


def pack_chunks(
    documents: Iterable[list[int]],
    bos_id: int,
    eos_id: int,
    count: int,
    block_length: int = BLOCK_LENGTH,
) -> list[list[int]]:
    """MDLM `wrap` packing: documents + EOS, concatenated, cut into [BOS] + (block_length - 2) + [EOS] chunks."""
    content = block_length - 2
    chunks: list[list[int]] = []
    buffer: list[int] = []
    for tokens in documents:
        buffer.extend(tokens)
        buffer.append(eos_id)
        while len(buffer) >= content and len(chunks) < count:
            chunks.append([bos_id, *buffer[:content], eos_id])
            buffer = buffer[content:]
        if len(chunks) >= count:
            return chunks
    raise ValueError(f"Only {len(chunks)} chunks of {block_length} tokens available, {count} requested.")


def build_open_chunks(  # noqa: PLR0913
    documents: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    num_prompts: int,
    num_references: int,
    prompt_tokens: int,
    block_length: int = BLOCK_LENGTH,
    prompt_pool: int = PROMPT_POOL,
) -> OpenChunks:
    assert 0 <= prompt_tokens < block_length - 2, f"prompt_tokens must be in [0, {block_length - 2})"
    assert 0 < num_prompts <= prompt_pool, f"num_prompts must be in [1, {prompt_pool}]"
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    assert isinstance(bos_id, int) and isinstance(eos_id, int)
    tokenized = (tokenizer(doc, add_special_tokens=False)["input_ids"] for doc in documents)
    chunks = pack_chunks(tokenized, bos_id, eos_id, prompt_pool + num_references, block_length)
    return OpenChunks(
        prompts=chunks[:num_prompts],
        references=chunks[prompt_pool:],
        prompt_tokens=prompt_tokens,
    )


def load_open_chunks(config: Config, tokenizer: PreTrainedTokenizerBase) -> OpenChunks:
    """The prompt and reference chunks of `config` (deterministic: no shuffling, no seed)."""
    return build_open_chunks(
        iter_documents(config.open_data_path, config.open_heldout_members),
        tokenizer,
        num_prompts=config.open_num_prompts,
        num_references=config.open_num_references,
        prompt_tokens=config.open_prompt_tokens,
        block_length=config.sequence_length,
    )
