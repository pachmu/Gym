# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build the IHEval Gym datasets from the upstream ``ytyz1307zzh/IHEval`` repo.

Iterates the eight single-turn IHEval tasks, loads every ``input_data.json``
under each task directory (``aligned/``, ``conflict/``, ``reference/``), and
writes:

* ``data/test.jsonl``    — all rows across all tasks.
* ``data/example.jsonl`` — a small mixed sample for smoke testing (committed).

Each row becomes a Gym task. The prompt (system + user instruction) goes into
``responses_create_params.input``; tool-use tasks additionally carry the
function schema in ``responses_create_params.tools`` and the canned tool-call
trajectory as Responses-API ``function_call`` / ``function_call_output`` input
items. The nemo-evaluator native driver translates those into the upstream
chat-completions tool turn at seed time (see ``_tool_trajectory``). The gold
answer and routing metadata travel in ``verifier_metadata``.

Source: on first run the upstream repo is downloaded as a zip into
``$XDG_CACHE_HOME/byob_iheval`` (or ``~/.cache/byob_iheval``). Set
``IHEVAL_REPO_DIR`` to point at an existing checkout instead.

Usage::

    python resources_servers/iheval/prepare_iheval.py
    python resources_servers/iheval/prepare_iheval.py --example-only
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOG = logging.getLogger(__name__)

_GITHUB_ZIP = "https://github.com/ytyz1307zzh/IHEval/archive/refs/heads/main.zip"
_ZIP_TOP_DIR = "IHEval-main"

_DATA_DIR = Path(__file__).resolve().parent / "data"
_AGENT = "iheval_simple_agent"

# (domain, task) pairs. ``task`` doubles as the verifier's scorer key.
# ``multi-turn`` rule-following is included: its ``conversation_history`` is
# pre-canned in the data (the assistant turns are fixed, not model-generated),
# and scoring grades only the final response with the same IFEval checker as
# ``single-turn`` — so it is a single generation over a pre-filled context.
_TASKS: Tuple[Tuple[str, str], ...] = (
    ("task-execution", "verb-extract"),
    ("task-execution", "translation"),
    ("task-execution", "lang-detect"),
    ("safety", "system-prompt-extract"),
    ("safety", "user-prompt-hijack"),
    ("tool-use", "slack-user"),
    ("tool-use", "get-webpage"),
    ("rule-following", "single-turn"),
    ("rule-following", "multi-turn"),
)

# Rows sampled (in order) for the committed example.jsonl smoke-test dataset.
_EXAMPLE_PER_TASK = {
    "verb-extract": 1,
    "lang-detect": 1,
    "system-prompt-extract": 1,
    "get-webpage": 1,
    "single-turn": 1,
    "multi-turn": 1,
}


# ── Upstream source ──────────────────────────────────────────────────────


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    cache = Path(base) / "byob_iheval"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _repo_root() -> Path:
    override = os.environ.get("IHEVAL_REPO_DIR")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "benchmark").is_dir():
            raise FileNotFoundError(f"IHEVAL_REPO_DIR missing 'benchmark/': {root}")
        return root

    cache = _cache_dir()
    target = cache / _ZIP_TOP_DIR
    sentinel = target / "benchmark"
    if sentinel.is_dir():
        return target

    LOG.info("Downloading IHEval source from %s", _GITHUB_ZIP)
    with urllib.request.urlopen(_GITHUB_ZIP, timeout=120) as resp:
        archive = resp.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        zf.extractall(cache)
    if not sentinel.is_dir():
        raise FileNotFoundError(f"IHEval extraction failed; missing {sentinel}")
    return target


def _iter_rows(root: Path, domain: str, task: str) -> List[Dict[str, Any]]:
    """Load every ``input_data.json`` under a task, tagging its setting."""
    task_root = root / "benchmark" / domain / task
    if not task_root.is_dir():
        raise FileNotFoundError(f"IHEval task directory missing: {task_root}")
    rows: List[Dict[str, Any]] = []
    for path in sorted(task_root.rglob("input_data.json")):
        setting = str(path.parent.relative_to(task_root))
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for row in data:
            row = dict(row)
            row["_setting"] = setting
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No input_data.json under {task_root}")
    return rows


# ── Row → Gym task conversion ────────────────────────────────────────────


def _tool_schema(definition: Dict[str, Any]) -> Dict[str, Any]:
    """Convert IHEval's flat tool definition to a Responses-API function tool.

    Upstream stores ``parameters`` as ``{<arg>: {description, type}}`` with no
    ``type: object`` wrapper; the Responses API expects a JSON-Schema object.
    Every IHEval tool has all positional args required.
    """
    raw_params = definition.get("parameters", {}) or {}
    properties = {name: dict(spec) for name, spec in raw_params.items()}
    return {
        "type": "function",
        "name": definition.get("name", ""),
        "description": definition.get("description", ""),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(raw_params.keys()),
            "additionalProperties": False,
        },
        # ``strict`` is a required field on the Responses-API FunctionToolParam.
        # We keep it False: the canned trajectory already supplies the call, so
        # constrained decoding of arguments is unnecessary and would reject the
        # loosely-typed IHEval parameter schemas on some backends.
        "strict": False,
    }


def _tool_trajectory(tool: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Canned ``[function_call, function_call_output]`` Responses-API items.

    The canonical Responses-API encoding (kept here so the gym-native model
    server, whose input schema has no ``role: "tool"``, accepts the row). The
    nemo-evaluator ``gym://...protocol=native`` driver's ``messages_from_rcp``
    translates these typed items into the upstream chat-completions form — an
    ``assistant`` message with ``tool_calls`` followed by a ``role: "tool"``
    result whose ``content`` carries the tool output — matching
    ``run_vllm_model.py::prepare_tool_call``.
    """
    call = tool.get("call", {}) or {}
    ret = tool.get("return", {}) or {}
    call_id = str(call.get("id") or "call_iheval_0")
    return [
        {
            "type": "function_call",
            "call_id": call_id,
            "name": call.get("name", ""),
            "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
        },
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": str(ret.get("content", "")),
        },
    ]


def _history_messages(history: List[str]) -> List[Dict[str, Any]]:
    """Alternating user/assistant turns from a multi-turn ``conversation_history``.

    Mirrors ``run_model.py``: even indices are prior user turns, odd indices are
    the (pre-canned) assistant replies. These precede the final instruction.
    """
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": str(msg)} for i, msg in enumerate(history)]


def _to_task(row: Dict[str, Any], domain: str, task: str) -> Dict[str, Any]:
    instruction = str(row.get("instruction", ""))
    system = row.get("system")
    tool = row.get("tool") or None
    history = row.get("conversation_history") or []

    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    if history:
        messages.extend(_history_messages(history))
    messages.append({"role": "user", "content": instruction})

    params: Dict[str, Any] = {"input": messages}
    if tool:
        messages.extend(_tool_trajectory(tool))
        params["tools"] = [_tool_schema(tool.get("definition", {}))]

    # Routing/gold fields live at the ROW TOP LEVEL (not nested under a
    # ``verifier_metadata`` object), mirroring the rolemrc / ragtruth servers.
    # The nemo-evaluator ``gym://...protocol=native`` driver forwards a row's
    # top-level SCALAR fields onto the ``/verify`` request but drops nested
    # objects — so ``answer`` (a dict/list for safety, rule-following and
    # get-webpage) must be JSON-ENCODED to a string to survive, matching the
    # legacy byob port's ``json.dumps(answer)``. verify() json-decodes it. See
    # app.py ``IHEvalRunRequest`` / ``_decode_answer``.
    return {
        "responses_create_params": params,
        "id": row.get("id"),
        "task": task,
        "domain": domain,
        "setting": row.get("_setting", ""),
        "instruction": instruction,
        "answer": json.dumps(row.get("answer"), ensure_ascii=False),
        "agent_ref": {"type": "responses_api_agents", "name": _AGENT},
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"IHEval: wrote {len(rows)} rows -> {path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-only", action="store_true", help="Only (re)write data/example.jsonl.")
    args = parser.parse_args(argv)

    root = _repo_root()
    all_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []
    for domain, task in _TASKS:
        rows = _iter_rows(root, domain, task)
        tasks = [_to_task(r, domain, task) for r in rows]
        all_rows.extend(tasks)
        for t in tasks[: _EXAMPLE_PER_TASK.get(task, 0)]:
            example_rows.append(t)
        print(f"IHEval: {domain}/{task}: {len(tasks)} rows")

    _write_jsonl(_DATA_DIR / "example.jsonl", example_rows)
    if not args.example_only:
        _write_jsonl(_DATA_DIR / "test.jsonl", all_rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
