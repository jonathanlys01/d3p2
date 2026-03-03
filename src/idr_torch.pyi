"""
Local type stub: idr-torch exposes runtime attributes dynamically; declare common
module-level values here so static checkers stop requiring repeated type ignores.
"""

from typing import Any

is_master: bool
rank: int
local_rank: int
world_size: int
size: int

def __getattr__(name: str) -> Any: ...
