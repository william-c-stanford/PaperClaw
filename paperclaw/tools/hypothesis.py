"""Hypothesis-map edit tools — let the chat agent add/update/remove hypotheses.

Operate on ``.hypothesis_map.json`` in the idea workspace (``base_dir``), so the
user can grow and refine the hypothesis map conversationally. The map is a tree:
each node has id / statement / rationale / test / status / children.

Registered in ``tools/__init__.py`` and offered to the idea chat (and pipeline)
tool loop. Three tools keep each action explicit for the model:
  hypothesis_add     — add a root or child hypothesis
  hypothesis_update  — edit a node's text or status
  hypothesis_remove  — delete a node (and its subtree)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

_FILE = ".hypothesis_map.json"
_STATUSES = ("untested", "supported", "refuted", "inconclusive")


def _load(base_dir: Path) -> dict:
    p = base_dir / _FILE
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("nodes", [])
                return data
        except json.JSONDecodeError:
            pass
    return {"nodes": []}


def _save(base_dir: Path, data: dict) -> None:
    data["ideaId"] = base_dir.name  # folder name is the idea id
    data["generatedAt"] = data.get("generatedAt") or time.time()
    (base_dir / _FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _find(nodes: list, node_id: str):
    """Return (node, sibling_list_containing_it) or (None, None)."""
    for n in nodes:
        if n.get("id") == node_id:
            return n, nodes
        found, lst = _find(n.get("children") or [], node_id)
        if found:
            return found, lst
    return None, None


def _next_id(siblings: list, parent_id: str) -> str:
    """Next hierarchical id at this level: roots H1, H2…; children H1.1, H1.2…."""
    nums = []
    for s in siblings:
        sid = s.get("id", "")
        if parent_id:
            if sid.startswith(parent_id + "."):
                tail = sid[len(parent_id) + 1:].split(".")[0]
                if tail.isdigit():
                    nums.append(int(tail))
        else:
            m = re.fullmatch(r"H(\d+)", sid)
            if m:
                nums.append(int(m.group(1)))
    n = (max(nums) + 1) if nums else 1
    return f"{parent_id}.{n}" if parent_id else f"H{n}"


# ── add ───────────────────────────────────────────────────────────────────────

ADD_SCHEMA: dict[str, Any] = {
    "name": "hypothesis_add",
    "description": (
        "Add a hypothesis to this idea's hypothesis map. Omit parent_id to add a "
        "ROOT hypothesis; pass an existing node id as parent_id to add it as a "
        "child (sub-hypothesis). Returns the new node's id."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "statement": {"type": "string", "description": "The hypothesis — a precise, falsifiable claim."},
            "rationale": {"type": "string", "description": "Optional: why it matters / how it relates."},
            "test": {"type": "string", "description": "Optional: one-line way to test it / decision criterion."},
            "parent_id": {"type": "string", "description": "Optional: id of the parent node to nest under."},
        },
        "required": ["statement"],
    },
}


def add(base_dir: Path, inputs: dict[str, Any]) -> str:
    statement = (inputs.get("statement") or "").strip()
    if not statement:
        return "Error: 'statement' is required."
    data = _load(base_dir)
    parent_id = (inputs.get("parent_id") or "").strip()
    if parent_id:
        parent, _ = _find(data["nodes"], parent_id)
        if not parent:
            return f"Error: no hypothesis with id {parent_id!r}."
        siblings = parent.setdefault("children", [])
        where = f"under {parent_id}"
    else:
        siblings = data["nodes"]
        where = "as a root"
    node = {
        "id": _next_id(siblings, parent_id),
        "statement": statement,
        "rationale": (inputs.get("rationale") or None),
        "test": (inputs.get("test") or None),
        "status": "untested",
        "children": [],
    }
    siblings.append(node)
    _save(base_dir, data)
    return f"Added hypothesis {where} with id '{node['id']}'."


# ── update ────────────────────────────────────────────────────────────────────

UPDATE_SCHEMA: dict[str, Any] = {
    "name": "hypothesis_update",
    "description": (
        "Edit an existing hypothesis node by id: change its statement, rationale, "
        "test, or status (untested | supported | refuted | inconclusive)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Node id to edit."},
            "statement": {"type": "string"},
            "rationale": {"type": "string"},
            "test": {"type": "string"},
            "status": {"type": "string", "enum": list(_STATUSES)},
        },
        "required": ["id"],
    },
}


def update(base_dir: Path, inputs: dict[str, Any]) -> str:
    node_id = (inputs.get("id") or "").strip()
    data = _load(base_dir)
    node, _ = _find(data["nodes"], node_id)
    if not node:
        return f"Error: no hypothesis with id {node_id!r}."
    for field in ("statement", "rationale", "test"):
        if inputs.get(field) is not None:
            node[field] = inputs[field] or None
    if inputs.get("status"):
        if inputs["status"] not in _STATUSES:
            return f"Error: status must be one of {_STATUSES}."
        node["status"] = inputs["status"]
    _save(base_dir, data)
    return f"Updated hypothesis '{node_id}'."


# ── remove ────────────────────────────────────────────────────────────────────

REMOVE_SCHEMA: dict[str, Any] = {
    "name": "hypothesis_remove",
    "description": "Delete a hypothesis node (and all its children) from the map by id.",
    "input_schema": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Node id to remove."}},
        "required": ["id"],
    },
}


def remove(base_dir: Path, inputs: dict[str, Any]) -> str:
    node_id = (inputs.get("id") or "").strip()
    data = _load(base_dir)
    node, lst = _find(data["nodes"], node_id)
    if not node:
        return f"Error: no hypothesis with id {node_id!r}."
    lst.remove(node)
    _save(base_dir, data)
    return f"Removed hypothesis '{node_id}'."
