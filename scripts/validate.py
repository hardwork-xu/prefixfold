#!/usr/bin/env python3
"""Execute local acceptance and preserve sanitized evidence / 执行本地验收并保留脱敏证据。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
UV = ROOT / ".venv/bin/uv"
LOGS = ROOT / "results/validation"


def stamp() -> str:
    return datetime.now(UTC).isoformat()


def sanitize(value: str) -> str:
    for key, replacement in ((str(ROOT), "<repo>"), (str(Path.home()), "<home>")):
        value = value.replace(key, replacement)
    return value


def source() -> dict[str, Any]:
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [ROOT / "benchmark.py", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "file_sha256": hashes,
    }


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    identity = source()
    result: dict[str, Any] = {
        "schema_version": 1,
        "started_at": stamp(),
        "source": identity,
        "checks": checks,
        "note": "Local acceptance; remote CI recorded separately / 本地验收，远端 CI 单独记录",
    }
    destination = ROOT / "results/acceptance.json"

    def save() -> None:
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    def run(
        name: str,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path = ROOT,
        timeout: int = 180,
    ) -> None:
        started = stamp()
        try:
            proc = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            code, output = proc.returncode, proc.stdout
            status = "passed" if code == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            code, status = None, "failed"
            output = f"timeout / 超时: {exc.timeout}s"
        relative_log = f"results/validation/{name}.log"
        (ROOT / relative_log).write_text(sanitize(output))
        checks.append(
            {
                "name": name,
                "command": [sanitize(item) for item in command],
                "cwd": str(cwd.relative_to(ROOT)),
                "started_at": started,
                "finished_at": stamp(),
                "status": status,
                "exit_code": code,
                "summary": sanitize(output[-1800:]),
                "evidence": relative_log,
                "source": identity,
            }
        )
        print(f"{name}: {status}", flush=True)
        save()

    run("lock", [str(UV), "lock", "--check"])
    run(
        "api",
        [
            str(PY),
            "-c",
            "from prefixfold import SharedPrefix, DecodeBatch, AttentionPlan, attend; "
            "print('API import OK / API 导入通过')",
        ],
    )
    run("demo", [str(PY), "-m", "prefixfold", "demo"])
    run("example", [str(PY), "examples/decode.py"])
    run("documented_examples", [str(PY), "scripts/check_examples.py"])
    run("unit_tests", [str(PY), "-m", "pytest", "-q", "-m", "not integration"])
    run("integration_tests", [str(PY), "-m", "pytest", "-q", "-m", "integration"])
    run("lint", [str(PY), "-m", "ruff", "check", "."])
    run("format", [str(PY), "-m", "ruff", "format", "--check", "."])
    run("types", [str(PY), "-m", "mypy", "src/prefixfold"])
    run(
        "analysis",
        [
            str(PY),
            "scripts/analyze.py",
            "--input",
            "results/benchmark.json",
            "--output-dir",
            "results",
            "--update-readme",
        ],
    )
    run("build", [str(PY), "-m", "build"])
    artifacts = sorted((ROOT / "dist").glob("prefixfold-0.1.0*"))
    run("package_metadata", [str(PY), "-m", "twine", "check", *[str(p) for p in artifacts]])
    fresh = ROOT / "work/clean-install"
    if fresh.exists():
        # This directory belongs exclusively to this verifier, never user files.
        # 此目录仅属于本验收程序，不清理其他目录。
        shutil.rmtree(fresh)
    fresh.parent.mkdir(exist_ok=True)
    run("clean_environment", [str(PY), "-m", "venv", str(fresh)])
    wheels = sorted((ROOT / "dist").glob("prefixfold-*.whl"))
    run(
        "clean_install",
        [str(UV), "pip", "install", "--python", str(fresh / "bin/python"), *map(str, wheels)],
    )
    run(
        "installed_wheel_demo",
        [str(fresh / "bin/python"), "-I", "-m", "prefixfold", "demo"],
        cwd=fresh,
    )
    run("docs", [str(PY), "scripts/check_docs.py"])
    run(
        "citation",
        [str(UV), "tool", "run", "--from", "cffconvert==2.0.0", "cffconvert", "--validate"],
    )
    run("privacy_and_delivery", [str(PY), "scripts/audit.py"])
    run("git_diff", ["git", "diff", "--check", "HEAD"])
    benchmark = json.loads((ROOT / "results/benchmark.json").read_text())
    current = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in benchmark["source"]["file_sha256"]
    }
    matching = current == benchmark["source"]["file_sha256"]
    success = matching and all(
        v["status"] == "passed"
        for case in benchmark["workloads"]
        for v in case["variants"].values()
    )
    checks.append(
        {
            "name": "benchmark",
            "command": benchmark["command"],
            "status": "passed" if success else "failed",
            "exit_code": 0 if success else 1,
            "summary": "Existing executed run, source hashes checked / 核验此前实际实验的源码哈希",
            "evidence": "results/benchmark.json",
            "source": benchmark["source"],
            "run_id": benchmark["run_id"],
            "record_kind": "previously_executed_run",
        }
    )
    if shutil.which("docker"):
        run("docker_build", ["docker", "build", "-t", "prefixfold:acceptance", "."], timeout=600)
        run("docker_run", ["docker", "run", "--rm", "prefixfold:acceptance"])
    else:
        for name, cmd in (
            ("docker_build", "docker build -t prefixfold:acceptance ."),
            ("docker_run", "docker run --rm prefixfold:acceptance"),
        ):
            checks.append(
                {
                    "name": name,
                    "command": cmd,
                    "status": "not_run",
                    "exit_code": None,
                    "reason": "docker executable not installed / 未安装 Docker",
                    "source": identity,
                    "evidence": "research/ENVIRONMENT.json",
                }
            )
    result["finished_at"] = stamp()
    result["counts"] = {
        state: sum(c["status"] == state for c in checks)
        for state in ("passed", "failed", "skipped", "not_run")
    }
    save()
    print(json.dumps(result["counts"]))
    return int(any(c["status"] == "failed" for c in checks))


if __name__ == "__main__":
    raise SystemExit(main())
