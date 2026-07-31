"""Pluggable retrieval strategies - the engine-submission lane of the koth
ladder.

A *strategy* is real ranking code: given the query and the candidate hits a
backend retrieved, it returns the ids in the order the reader should see them.
This is where a contributor competes on algorithm (a new fusion, a learned
reranker, a novel signal) rather than on config coefficients (the kit lane).

Two ways a strategy runs, with very different trust:

- **shipped / merged** (trusted): resolved from ``retrieval.strategy`` in
  config.yaml as a dotted import path, loaded in-process. It is trusted
  because a human reviewed and merged it - the review gate, applied to code.
- **a competition submission** (untrusted): a file under ``contrib/strategies/``
  loaded by the benchmark and run via :func:`run_sandboxed`, which executes it
  in a separate ``python -I`` process under resource limits and an audit hook
  that blocks network, subprocess, and filesystem writes.

Honesty about the sandbox boundary: the audit hook + rlimits stop casual and
accidental misbehaviour and raise the bar against a hostile one, but no
in-process Python guard is a true boundary - a determined attacker who can
introspect the interpreter (native ctypes/cffi, or pure-Python gc/frame walking
to reach and neutralise the hook object) can defeat it. The *real* boundary is
the same one ditto leans on, in two parts. First, least privilege at scoring
time: the scoring job runs on an ephemeral CI runner with a read-only token
and no secrets, and the write-capable job that arms auto-merge on a dethrone
checks out nothing and executes no challenger code. Second, quarantine after
merge: a winning strategy auto-merges into ``contrib/`` (ladder branch only),
where it is executed exclusively through this sandbox by both gates and the
bench - never imported in-process - and the ledger/ratchet scripts treat it
as text. Trusted, importable shipping (promotion into ``vouch.strategies``)
still happens only through a human-reviewed PR. So a full escape during
scoring can at worst forge its own verdict and land code in the quarantine
it already runs in; it cannot reach a write token, secrets, or an import
path. The sandbox is defence in depth; the trust root is least privilege
plus the quarantine plus the human gate on shipped defaults.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# a retrieval hit as the reorder stages pass it: (kind, id, summary, score).
Hit = tuple[str, str, str, float]

DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MEM_MB = 2048
DEFAULT_CPU_S = 15

# audit events refused inside the sandbox child. network, process spawning,
# native dlopen, and filesystem-mutating calls outside of `open` (delete,
# rename, mkdir, chmod, ...) are blocked outright; writing through `open`
# itself is blocked by inspecting its mode (reads - needed to import numpy
# etc. - are allowed).
_BLOCKED_EXACT = frozenset({
    "os.system",
    "os.exec",
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "os.spawn",
    "subprocess.Popen",
    "ctypes.dlopen",
    "ctypes.dlsym",
    "ctypes.call_function",
    "ctypes.cdata",
    # filesystem mutation that doesn't go through `open` at all — the mode/
    # flags check below only catches a write-mode open call.
    "os.remove",
    "os.rmdir",
    "os.rename",  # also fires for os.replace
    "os.mkdir",
    "os.link",
    "os.symlink",
    "os.chmod",
    "os.truncate",
})
_BLOCKED_PREFIXES = ("socket.", "urllib.", "ftplib.", "http.")


@dataclass(frozen=True)
class Candidate:
    """What a strategy sees for one retrieved artifact - data only.

    A strategy never receives the KB, the filesystem, or a network handle;
    just these fields, so it cannot reach anything outside the ranking task.
    """

    kind: str
    id: str
    summary: str
    score: float


@runtime_checkable
class RetrievalStrategy(Protocol):
    """The interface a submission implements.

    ``rank`` returns candidate ids best-first. It is treated as *ordering
    only*: ids not in the input are ignored, and any input id the strategy
    omits is appended in its original order, so a strategy can reorder and
    drop-from-the-top but can never fabricate a result or smuggle one in.
    """

    def rank(
        self, query: str, candidates: list[Candidate], *, limit: int
    ) -> list[str]: ...


def apply_ordering(ordered_ids: list[str], hits: list[Hit]) -> list[Hit]:
    """Reorder ``hits`` by ``ordered_ids``, preserving the hit set.

    Mirrors the discipline of the built-in rerank stage: the strategy may move
    artifacts but not add or remove them. Unknown ids are dropped; hits the
    strategy did not mention keep their relative order at the tail.
    """
    by_id: dict[str, Hit] = {}
    for hit in hits:
        by_id.setdefault(hit[1], hit)
    seen: set[str] = set()
    ordered: list[Hit] = []
    for hid in ordered_ids:
        match = by_id.get(hid)
        if match is not None and hid not in seen:
            ordered.append(match)
            seen.add(hid)
    for hit in hits:
        if hit[1] not in seen:
            ordered.append(hit)
            seen.add(hit[1])
    return ordered


def load_from_path(path: str | Path) -> RetrievalStrategy:
    """Import a strategy from a .py file and return its strategy object.

    The module must expose either a ``STRATEGY`` object with a ``rank`` method
    or a module-level ``rank`` function. Loading runs module top-level code, so
    only call this on trusted (merged) files or inside the sandbox child.
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"vouch_strategy_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load strategy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _strategy_from_module(module)


def load_dotted(dotted: str) -> RetrievalStrategy:
    """Import a trusted, shipped strategy by dotted module path (config hook)."""
    module = importlib.import_module(dotted)
    return _strategy_from_module(module)


def _strategy_from_module(module: Any) -> RetrievalStrategy:
    candidate = getattr(module, "STRATEGY", None)
    if candidate is not None and hasattr(candidate, "rank"):
        return candidate  # type: ignore[no-any-return]
    fn = getattr(module, "rank", None)
    if callable(fn):
        rank_fn = fn

        class _FnStrategy:
            def rank(
                self, query: str, candidates: list[Candidate], *, limit: int
            ) -> list[str]:
                return list(rank_fn(query, candidates, limit=limit))

        return _FnStrategy()
    raise ValueError(
        "strategy module must define STRATEGY (with .rank) or a rank() function"
    )


class SandboxProxy:
    """A strategy handle whose ``rank`` runs the real code in a sandbox child.

    Built by the benchmark for an untrusted submission. Each ``rank`` call
    spawns a fresh ``python -I`` process, so a submission cannot keep state
    between calls or wear down the limits over time. A crash, a timeout, or a
    limit breach yields an empty ordering, which :func:`apply_ordering` turns
    into "keep the backend's order" - a broken strategy simply fails to
    improve rather than taking down the run.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        mem_mb: int = DEFAULT_MEM_MB,
        cpu_s: int = DEFAULT_CPU_S,
    ) -> None:
        self.path = str(Path(path).resolve())
        self.timeout_s = timeout_s
        self.mem_mb = mem_mb
        self.cpu_s = cpu_s
        self.failures = 0

    def rank(
        self, query: str, candidates: list[Candidate], *, limit: int
    ) -> list[str]:
        ordered = run_sandboxed(
            self.path,
            query,
            candidates,
            limit=limit,
            timeout_s=self.timeout_s,
            mem_mb=self.mem_mb,
            cpu_s=self.cpu_s,
        )
        if ordered is None:
            self.failures += 1
            return []
        return ordered


def run_sandboxed(
    path: str,
    query: str,
    candidates: list[Candidate],
    *,
    limit: int,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    mem_mb: int = DEFAULT_MEM_MB,
    cpu_s: int = DEFAULT_CPU_S,
) -> list[str] | None:
    """Run a strategy file in a locked-down child; return ordered ids or None.

    None means the child failed (crash, timeout, limit, or malformed output) -
    the caller treats that as "no reordering", never as an error that aborts
    scoring.
    """
    payload = {
        "path": path,
        "query": query,
        "limit": limit,
        "mem_bytes": mem_mb * 1024 * 1024,
        "cpu_seconds": cpu_s,
        "candidates": [
            {"kind": c.kind, "id": c.id, "summary": c.summary, "score": c.score}
            for c in candidates
        ],
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-m", "vouch.strategy", "--child"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            # a minimal env; -I already ignores PYTHON* vars and the cwd.
            env={"PATH": "/usr/bin:/bin"},
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        result = json.loads(proc.stdout)
        ordered = result["ordered"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(ordered, list):
        return None
    return [str(x) for x in ordered]


# --- sandbox child ---------------------------------------------------------


def _install_audit_hook() -> None:
    import os

    # Bind the guard sets into closure cells that hold the original frozenset
    # objects. The hook reads them via LOAD_DEREF, not from module globals, so
    # untrusted top-level code cannot disarm the block by reassigning
    # ``__main__._BLOCKED_EXACT`` (the child runs as __main__). This is still
    # in-process defence in depth - see the module docstring: a determined
    # attacker who introspects the interpreter can defeat any pure-Python
    # hook, which is why the real boundary is the ephemeral runner and the
    # no-auto-merge rule, not this function.
    blocked_exact = _BLOCKED_EXACT
    blocked_prefixes = _BLOCKED_PREFIXES

    def _hook(event: str, args: tuple[Any, ...]) -> None:
        if event in blocked_exact or event.startswith(blocked_prefixes):
            raise PermissionError(f"blocked in strategy sandbox: {event}")
        if event == "open":
            # args = (path, mode, flags); refuse any write/append/create intent.
            mode = args[1] if len(args) > 1 else ""
            if isinstance(mode, str) and any(c in mode for c in "wax+"):
                raise PermissionError("filesystem writes are blocked in the sandbox")
            flags = args[2] if len(args) > 2 else 0
            if isinstance(flags, int) and (
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND)
            ):
                raise PermissionError("filesystem writes are blocked in the sandbox")

    sys.addaudithook(_hook)


def _apply_rlimits(mem_bytes: int, cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:  # non-unix; the audit hook still applies
        return
    # cpu seconds is a hard ceiling backing the parent's wall-clock timeout;
    # nofile caps fd pressure. RLIMIT_AS is deliberately generous - numpy and
    # friends reserve large virtual space even at small RSS, and a too-tight
    # AS limit kills honest strategies, so wall-clock + cpu are the real caps.
    for res, soft in (
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_AS, mem_bytes),
        (resource.RLIMIT_NOFILE, 64),
    ):
        try:
            hard = resource.getrlimit(res)[1]
            cap = soft if hard == resource.RLIM_INFINITY else min(soft, hard)
            resource.setrlimit(res, (cap, hard))
        except (ValueError, OSError):
            pass


def _child_main() -> int:
    import os

    payload = json.load(sys.stdin)
    mem_bytes = int(payload["mem_bytes"])
    cpu_seconds = int(payload["cpu_seconds"])
    path = payload["path"]
    query = payload["query"]
    limit = int(payload["limit"])
    candidates = [
        Candidate(kind=c["kind"], id=c["id"], summary=c["summary"], score=c["score"])
        for c in payload["candidates"]
    ]
    # Isolate the result channel from the strategy's stdout: a strategy that
    # prints (a debug line, a library banner, a progress bar) must not corrupt
    # the JSON the parent parses. Keep a private dup of the real stdout for the
    # result, then point fd 1 at /dev/null so anything the strategy writes is
    # discarded. Both opens happen before the write-blocking audit hook arms.
    result_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    # limits and the audit hook are armed BEFORE the untrusted module is loaded.
    _apply_rlimits(mem_bytes, cpu_seconds)
    _install_audit_hook()
    strategy = load_from_path(path)
    ordered = strategy.rank(query, candidates, limit=limit)
    valid = {c.id for c in candidates}
    ordered_ids = [str(x) for x in ordered if str(x) in valid]
    os.write(result_fd, json.dumps({"ordered": ordered_ids}).encode("utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--child":
        return _child_main()
    print("usage: python -I -m vouch.strategy --child  (reads a job on stdin)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
