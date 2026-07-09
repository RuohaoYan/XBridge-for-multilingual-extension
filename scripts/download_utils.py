#!/usr/bin/env python3
"""Shared download helpers with real-time progress."""

import os
import sys
import time
import urllib.error
import urllib.request


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}PB"


def download_url(
    url: str,
    dest: str,
    timeout: int = 600,
    chunk_size: int = 1024 * 1024,
    resume: bool = True,
    label: str = "",
) -> str:
    """Download URL to dest with live progress (size, %, speed, ETA). Supports resume."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    if os.path.isfile(dest) and os.path.getsize(dest) > 0 and not resume:
        os.remove(dest)

    # Resume from .part or existing dest
    start = 0
    write_path = part
    if resume:
        if os.path.isfile(part):
            start = os.path.getsize(part)
            write_path = part
        elif os.path.isfile(dest):
            start = os.path.getsize(dest)
            write_path = dest

    headers = {"User-Agent": "Mozilla/5.0"}
    if start > 0:
        headers["Range"] = f"bytes={start}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and start > 0:
            os.replace(part, dest) if os.path.isfile(part) else None
            print(f"[download] already complete -> {dest}")
            return dest
        raise

    content_range = resp.headers.get("Content-Range", "")
    if start > 0 and content_range:
        # bytes START-END/TOTAL
        total = int(content_range.split("/")[-1])
    else:
        total = int(resp.headers.get("Content-Length", 0)) + start

    tag = label or os.path.basename(dest)
    print(f"[download] {tag}")
    print(f"  url: {url}")
    if start > 0:
        print(f"  resume from {_human_size(start)}")

    done = start
    t0 = time.time()
    last_print = t0
    mode = "ab" if start > 0 else "wb"

    with open(write_path, mode) as f:
        while True:
            buf = resp.read(chunk_size)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            now = time.time()
            if now - last_print >= 0.2 or (total and done >= total):
                elapsed = max(now - t0, 1e-6)
                speed = (done - start) / elapsed
                if total > 0:
                    pct = done * 100.0 / total
                    remain = max(total - done, 0)
                    eta = remain / speed if speed > 0 else 0
                    line = (
                        f"\r  {_human_size(done)} / {_human_size(total)} "
                        f"({pct:5.1f}%)  {_human_size(speed)}/s  ETA {eta:5.0f}s"
                    )
                else:
                    line = f"\r  {_human_size(done)}  {_human_size(speed)}/s"
                sys.stdout.write(line)
                sys.stdout.flush()
                last_print = now

    sys.stdout.write("\n")
    sys.stdout.flush()

    if write_path == part:
        os.replace(part, dest)
    elif write_path != dest:
        os.replace(write_path, dest)

    if total > 0 and os.path.getsize(dest) < total:
        raise IOError(
            f"Incomplete download: {dest} ({os.path.getsize(dest)} < {total}). "
            f"Re-run the same command to resume."
        )

    print(f"  saved -> {dest} ({_human_size(os.path.getsize(dest))})")
    return dest


def iter_lines_with_progress(path: str, desc: str = "convert", every: int = 50000):
    """Yield lines from text file, print progress every N lines."""
    n = 0
    t0 = time.time()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            if n % every == 0:
                elapsed = max(time.time() - t0, 1e-6)
                sys.stdout.write(f"\r  [{desc}] {n:,} lines  {n/elapsed:,.0f} lines/s")
                sys.stdout.flush()
            yield line
    if n:
        sys.stdout.write(f"\r  [{desc}] {n:,} lines done\n")
        sys.stdout.flush()
