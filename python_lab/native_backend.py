"""Native C++ backend helpers for the arithmetic learning platform."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NATIVE_DIR = ROOT / "native"
BUILD_DIR = NATIVE_DIR / "build"
SOURCE = NATIVE_DIR / "src" / "main.cpp"


def executable_path() -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    return BUILD_DIR / f"arithmetic_benchmark{suffix}"


def find_compiler() -> tuple[str, str] | None:
    for compiler in ("g++", "clang++"):
        path = shutil.which(compiler)
        if path:
            return compiler, path
    cl = shutil.which("cl")
    if cl:
        return "cl", cl
    return None


def find_vsdevcmd() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"),
        Path(r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe"),
    ]
    for vswhere in candidates:
        if not vswhere.exists():
            continue
        completed = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
        )
        install_path = completed.stdout.strip()
        if completed.returncode == 0 and install_path:
            devcmd = Path(install_path) / "Common7" / "Tools" / "VsDevCmd.bat"
            if devcmd.exists():
                return devcmd

    fallback_paths = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
    ]
    for path in fallback_paths:
        if path.exists():
            return path
    return None


def build_with_vsdevcmd(devcmd: Path, exe: Path) -> tuple[Path | None, str]:
    build_script = BUILD_DIR / "build_msvc.bat"
    asm_path = BUILD_DIR / "arithmetic_benchmark.asm"
    build_script.write_text(
        "\n".join(
            [
                "@echo off",
                f'call "{devcmd}" -arch=x64',
                f'cl /std:c++17 /O2 /EHsc /FAcs /Fa"{asm_path}" "{SOURCE}" /Fe:"{exe}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cmd_exe = r"C:\Windows\System32\cmd.exe" if platform.system() == "Windows" else "cmd.exe"
    completed = subprocess.run(
        [cmd_exe, "/d", "/c", str(build_script)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = (completed.stdout + "\n" + completed.stderr).strip()
        return None, f"Native build failed with MSVC developer environment: {details}"
    return exe, f"Built native backend with MSVC developer environment: {devcmd}"


def build_native_backend() -> tuple[Path | None, str]:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    exe = executable_path()
    compiler = find_compiler()
    if compiler is None:
        devcmd = find_vsdevcmd()
        if devcmd is not None:
            return build_with_vsdevcmd(devcmd, exe)
        return None, "No C++ compiler found on PATH. Install MSVC Build Tools, MinGW g++, or LLVM clang++."

    compiler_name, compiler_path = compiler
    if compiler_name == "cl":
        cmd = [
            compiler_path,
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/FAcs",
            str(SOURCE),
            f"/Fe:{exe}",
        ]
    else:
        cmd = [
            compiler_path,
            "-std=c++17",
            "-O3",
            "-S",
            str(SOURCE),
            "-o",
            str(BUILD_DIR / "arithmetic_benchmark.s"),
        ]
        asm_completed = subprocess.run(cmd, capture_output=True, text=True)
        if asm_completed.returncode != 0:
            details = (asm_completed.stdout + "\n" + asm_completed.stderr).strip()
            return None, f"Native assembly generation failed with {compiler_name}: {details}"
        cmd = [
            compiler_path,
            "-std=c++17",
            "-O3",
            str(SOURCE),
            "-o",
            str(exe),
        ]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        details = (completed.stdout + "\n" + completed.stderr).strip()
        return None, f"Native build failed with {compiler_name}: {details}"
    return exe, f"Built native backend with {compiler_name}: {compiler_path}"


def write_input_csv(path: Path, inputs: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b"])
        writer.writerows(inputs)


def run_native_trial(
    exe: Path,
    operation: str,
    numeric_type: str,
    input_path: Path,
    experiment_id: str,
    input_class: str,
    input_set_id: str,
    trial_id: int,
    seed: int,
    passes: int,
) -> dict:
    cmd = [
        str(exe),
        "--operation",
        operation,
        "--numeric-type",
        numeric_type,
        "--input",
        str(input_path),
        "--experiment-id",
        experiment_id,
        "--input-class",
        input_class,
        "--input-set-id",
        input_set_id,
        "--trial-id",
        str(trial_id),
        "--seed",
        str(seed),
        "--passes",
        str(passes),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    output = completed.stdout.strip().splitlines()[-1]
    row = json.loads(output)
    if completed.returncode != 0:
        raise RuntimeError(row.get("error", completed.stderr.strip()))
    return row
