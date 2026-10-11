"""Shared test helpers: compile+run emitted C++, torch availability."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

# Where compiled test binaries go. Unset means the system temp dir.
SCRATCH = os.environ.get("HLS_SCRATCH") or None


def cxx() -> str | None:
    for c in ("clang++", "g++"):
        if shutil.which(c):
            return c
    return None


def compile_cpp(source: str, workdir: str | None = None) -> str:
    """Compile a full C++ translation unit (with main). Return path to binary.

    Pragmas are plain `#pragma HLS ...`; a non-HLS compiler ignores them. We treat
    unknown-pragma warnings as harmless but reject real errors.
    """
    compiler = cxx()
    assert compiler, "no C++ compiler (clang++/g++) available"
    if SCRATCH:
        os.makedirs(SCRATCH, exist_ok=True)
    workdir = workdir or tempfile.mkdtemp(prefix="hlsest_", dir=SCRATCH)
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, "net.cpp")
    binp = os.path.join(workdir, "net.bin")
    with open(src, "w") as f:
        f.write(source)
    proc = subprocess.run(
        [compiler, "-O0", "-std=c++17", "-Wno-unknown-pragmas", src, "-o", binp],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"C++ compile failed:\n{proc.stderr}"
    return binp


def run_binary(binp: str, input_ints: list[int]) -> list[int]:
    """Feed whitespace-separated ints on stdin, parse ints from stdout."""
    stdin = " ".join(str(int(x)) for x in input_ints)
    proc = subprocess.run(
        [binp], input=stdin, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, f"binary crashed:\n{proc.stderr}"
    return [int(tok) for tok in proc.stdout.split()]
