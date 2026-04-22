#!/usr/bin/env python3
"""
Batch-run all benchmarks through the *_nf trainers (DDQN, DQN, MAB-EP, MAB-UCB).

Each trainer is invoked exactly as (cwd = MapTune root)::

  python3 batched_DDQN_nf.py      <SAMPLE_GATE> benchmarks/<file> <genlib>
  python3 batched_DQN_nf.py       <SAMPLE_GATE> benchmarks/<file> <genlib>
  python3 batched_MAB_EP_nf.py    <SAMPLE_GATE> benchmarks/<file> <genlib>
  python3 batched_MAB_UCB_nf.py   <SAMPLE_GATE> benchmarks/<file> <genlib>

Those map to the trainers' ``sys.argv[-3]``, ``[-2]``, ``[-1]`` (sample_gate,
design path, genlib path). Default sample_gate is 65; default genlib is
``7nm.genlib`` under MapTune root unless ``--genlib`` overrides it (absolute
paths allowed).

Prerequisites:
  - ``abc`` on PATH (both trainers shell out to it).
  - PyTorch / gymnasium env working (same as a manual single-design run).

Working directory for child processes is the MapTune project root (parent of this
``scripts/`` directory). Design paths are passed as ``benchmarks/<name>.bench`` /
``benchmarks/<name>.blif`` so outputs match existing ``gen_newlibs/...`` layout.

No dry-run mode: this script always executes the trainer subprocesses.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MAPTUNE_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_GLOBS = ("benchmarks/*.bench", "benchmarks/*.blif")


def _discover_designs(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in BENCHMARK_GLOBS:
        paths.extend(root.glob(pattern))
    # De-duplicate (if any overlap) and sort by name for deterministic order
    unique = {p.resolve(): p for p in paths}
    return sorted(unique.values(), key=lambda p: p.name.lower())


def _ensure_dirs(root: Path) -> None:
    (root / "temp_blifs").mkdir(parents=True, exist_ok=True)
    (root / "gen_newlibs").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)


# (cli_key, script filename, log filename pattern with {base} = design stem)
_TRAINERS: tuple[tuple[str, str, str], ...] = (
    ("ddqn", "batched_DDQN_nf.py", "{base}_ddqn.log"),
    ("dqn", "batched_DQN_nf.py", "{base}_dqn.log"),
    ("mab_ep", "batched_MAB_EP_nf.py", "{base}_mab_ep.log"),
    ("mab_ucb", "batched_MAB_UCB_nf.py", "{base}_mab_ucb.log"),
)


def _trainers_for_only(only: str) -> list[tuple[str, str, str]]:
    if only == "all":
        return list(_TRAINERS)
    if only == "both":
        return [t for t in _TRAINERS if t[0] in ("ddqn", "dqn")]
    return [t for t in _TRAINERS if t[0] == only]


def _run_trainer(
    *,
    root: Path,
    python_exe: str,
    script: str,
    sample_gate: int,
    design_relpath: str,
    genlib: str,
    log_path: Path,
) -> int:
    cmd = [python_exe, script, str(sample_gate), design_relpath, genlib]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"$ {' '.join(cmd)}\n")
        logf.flush()
        # subprocess requires a real fd for stdout= (has fileno); stream via PIPE.
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
            logf.flush()
        return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run batched DDQN/DQN/MAB-EP/MAB-UCB *_nf trainers on all "
            "benchmarks/*.bench and *.blif."
        )
    )
    parser.add_argument(
        "--sample-gate",
        type=int,
        default=65,
        help="Passed as argv[-3] to each trainer (default: 65).",
    )
    parser.add_argument(
        "--genlib",
        default="7nm.genlib",
        help="Cell library path relative to MapTune root, or absolute (default: 7nm.genlib).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for trainers (default: sys.executable).",
    )
    parser.add_argument(
        "--only",
        choices=("ddqn", "dqn", "mab_ep", "mab_ucb", "both", "all"),
        default="mab_ep",
        help=(
            "Which trainer(s): ddqn, dqn, mab_ep, mab_ucb; "
            "both = DDQN then DQN; all = all four (default: mab_ep)."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "If a subprocess fails, log and continue with remaining designs. "
            "Default: exit immediately with that subprocess exit code."
        ),
    )
    args = parser.parse_args()

    root = MAPTUNE_ROOT
    genlib_arg = args.genlib
    if not Path(genlib_arg).is_absolute():
        genlib_arg = str((root / genlib_arg).resolve())

    designs = _discover_designs(root)
    if not designs:
        print(f"No benchmarks found under {root} ({', '.join(BENCHMARK_GLOBS)}).", file=sys.stderr)
        return 1

    _ensure_dirs(root)
    python_exe = args.python
    failed: list[tuple[str, int]] = []
    trainers = _trainers_for_only(args.only)
    label_by_key = {
        "ddqn": "DDQN",
        "dqn": "DQN",
        "mab_ep": "MAB-EP",
        "mab_ucb": "MAB-UCB",
    }

    for design_path in designs:
        try:
            rel = str(design_path.relative_to(root))
        except ValueError:
            rel = str(design_path)
        base = design_path.stem
        print(f"\n=== {rel} ===", flush=True)

        for key, script, log_pat in trainers:
            rc = _run_trainer(
                root=root,
                python_exe=python_exe,
                script=script,
                sample_gate=args.sample_gate,
                design_relpath=rel,
                genlib=genlib_arg,
                log_path=root / "logs" / log_pat.format(base=base),
            )
            if rc != 0:
                failed.append((f"{rel} ({label_by_key[key]})", rc))
                if not args.continue_on_error:
                    return rc

    if failed:
        print("\nFailures:", file=sys.stderr)
        for label, code in failed:
            print(f"  {label}: exit {code}", file=sys.stderr)
        return 1

    print("\nAll runs completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
