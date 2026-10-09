"""One node per clinical phase.

Each module exposes `node_<phase>(state, llm=None)`. The split exists so adding a
phase means adding a file and nothing else — graph.py wires them by name.
"""

from dental.nodes import (
    closing,
    complaint,
    diagnosis,
    examination,
    history,
    investigations,
    management,
)

__all__ = [
    "closing",
    "complaint",
    "diagnosis",
    "examination",
    "history",
    "investigations",
    "management",
]
