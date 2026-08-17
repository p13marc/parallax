#!/usr/bin/env python3
"""Decode-performance matrix runner for parallax-player (#192).

Runs the player once per (config × repeat) in interleaved order — the
A/B/A protocol from the #189/#190 rounds, which cancels thermal drift —
and samples per-thread CPU from /proc over a steady-state window inside
each run. Reports, per config: dav1d-worker cores, total player cores,
RSS, arena memfd bytes, drops, and the final position.

Usage:
    scripts/decode_matrix.py --file ~/Videos/boss_pokemon.webm \
        --config "baseline:" \
        --config "t4:--dav1d-threads 4" \
        --config "ahead16:--decode-ahead 16" \
        --duration 60 --warmup 10 --repeat 2 --out /tmp/matrix

Config syntax: "name:extra player flags". The player must support
--exit-after and print the final "stats: position_ns=… dropped=…" line.
Stdlib only; Linux only (reads /proc).
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

CLK_TCK = os.sysconf("SC_CLK_TCK")


def thread_cpu_ticks(pid: int) -> dict[int, tuple[str, int]]:
    """{tid: (comm, utime+stime ticks)} for every live thread of pid."""
    out = {}
    task_dir = Path(f"/proc/{pid}/task")
    try:
        tids = [int(t.name) for t in task_dir.iterdir()]
    except FileNotFoundError:
        return out
    for tid in tids:
        try:
            stat = (task_dir / str(tid) / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        # comm is parenthesized and may contain spaces; split around it.
        rparen = stat.rfind(")")
        comm = stat[stat.find("(") + 1 : rparen]
        fields = stat[rparen + 2 :].split()
        # fields[11] = utime (stat field 14), fields[12] = stime (15).
        out[tid] = (comm, int(fields[11]) + int(fields[12]))
    return out


def vm_rss_kb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    return 0


def memfd_bytes(pid: int) -> int:
    """Total size of the parallax arena memfds, deduped by inode.

    Filtered by name: Mesa/pipewire/winit also use memfds (llvmpipe's
    sparse 1 GB "state table"s would dominate the number).
    """
    seen = set()
    total = 0
    fd_dir = Path(f"/proc/{pid}/fd")
    try:
        for fd in fd_dir.iterdir():
            try:
                if "/memfd:parallax-shared-arena" not in os.readlink(fd):
                    continue
                st = os.stat(fd)
                if st.st_ino not in seen:
                    seen.add(st.st_ino)
                    total += st.st_size
            except OSError:
                continue
    except FileNotFoundError:
        pass
    return total


def classify(comm: str) -> str:
    """Bucket a thread name; dav1d names its workers 'dav1d-worker'."""
    if comm.startswith("dav1d"):
        return "dav1d"
    if comm.startswith("tokio"):
        return "tokio"
    return "other"


def run_once(player, file, extra_flags, duration, warmup, quiet):
    cmd = [player, str(file), f"--exit-after={duration}", *extra_flags]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else None,
        text=True,
    )
    pid = proc.pid
    window = max(duration - warmup - 2, 5)

    time.sleep(warmup)
    t0 = time.monotonic()
    start = thread_cpu_ticks(pid)
    time.sleep(window)
    end = thread_cpu_ticks(pid)
    wall = time.monotonic() - t0
    rss_kb = vm_rss_kb(pid)
    arena = memfd_bytes(pid)

    stdout, _ = proc.communicate(timeout=duration + 30)
    stats = {"position_ns": 0, "dropped": -1}
    m = re.search(r"stats: position_ns=(\d+) dropped=(\d+)", stdout or "")
    if m:
        stats = {"position_ns": int(m.group(1)), "dropped": int(m.group(2))}

    cores = defaultdict(float)
    threads = defaultdict(int)
    for tid, (comm, ticks1) in end.items():
        ticks0 = start.get(tid, (comm, 0))[1]
        group = classify(comm)
        cores[group] += (ticks1 - ticks0) / CLK_TCK / wall
        threads[group] += 1
    return {
        "dav1d_cores": round(cores["dav1d"], 3),
        "tokio_cores": round(cores["tokio"], 3),
        "other_cores": round(cores["other"], 3),
        "total_cores": round(sum(cores.values()), 3),
        "dav1d_threads": threads["dav1d"],
        "rss_mb": round(rss_kb / 1024, 1),
        "arena_mb": round(arena / (1024 * 1024), 1),
        "window_s": round(wall, 1),
        "position_s": round(stats["position_ns"] / 1e9, 1),
        "dropped": stats["dropped"],
        "exit_code": proc.returncode,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--player", default="target/release/parallax-player")
    ap.add_argument("--file", required=True)
    ap.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="NAME:FLAGS",
        help='e.g. "t4-ahead16:--dav1d-threads 4 --decode-ahead 16"; repeatable',
    )
    ap.add_argument("--duration", type=int, default=60, help="seconds per run")
    ap.add_argument("--warmup", type=int, default=10, help="seconds before sampling")
    ap.add_argument("--repeat", type=int, default=2, help="runs per config, interleaved")
    ap.add_argument("--gap", type=int, default=3, help="idle seconds between runs")
    ap.add_argument("--out", default=None, help="prefix for .csv/.md result files")
    ap.add_argument("--loud", action="store_true", help="pass player stderr through")
    args = ap.parse_args()

    configs = []
    for spec in args.config:
        name, _, flags = spec.partition(":")
        configs.append((name.strip(), flags.split()))

    rows = []
    # Interleaved order: A B C A B C … so drift hits every config equally.
    for rep in range(args.repeat):
        for name, flags in configs:
            label = f"{name}#{rep + 1}"
            print(f"[{label}] {' '.join(flags) or '(defaults)'} …", flush=True)
            r = run_once(
                args.player, args.file, flags, args.duration, args.warmup, not args.loud
            )
            r = {"config": name, "rep": rep + 1, **r}
            rows.append(r)
            print(
                f"[{label}] dav1d {r['dav1d_cores']} cores ({r['dav1d_threads']} thr) · "
                f"total {r['total_cores']} · rss {r['rss_mb']} MB · "
                f"arena {r['arena_mb']} MB · drops {r['dropped']} · "
                f"pos {r['position_s']}s",
                flush=True,
            )
            time.sleep(args.gap)

    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    table = "\n".join(md)
    print("\n" + table)

    if args.out:
        with open(f"{args.out}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        Path(f"{args.out}.md").write_text(table + "\n")
        print(f"\nwrote {args.out}.csv and {args.out}.md")


if __name__ == "__main__":
    sys.exit(main())
