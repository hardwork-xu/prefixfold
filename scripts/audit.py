#!/usr/bin/env python3
"""Audit intended public files without echoing sensitive matches / 审查拟公开文件，不回显敏感内容。

Heuristic inspection supplements manual review; it is not a proof that every secret is absent.
启发式扫描用于辅助人工审阅，不能证明不存在任何秘密。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".git",
    "work",
    "dist",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
MAX_FILE_BYTES = 10 * 1024 * 1024
EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
)
RULES = (
    ("absolute-home-path", re.compile(r"/(?:Users|home)/[^\s/\"'<>]+(?:/|\b)")),
    ("windows-home-path", re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\s\\/\"'<>]+", re.IGNORECASE)),
    ("private-key-material", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    (
        "github-token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    ),
    ("service-api-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "credential-assignment",
        re.compile(
            r"(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*[\"'][A-Za-z0-9+/=_-]{16,}[\"']",
            re.IGNORECASE,
        ),
    ),
)


def private_email(value: str) -> bool:
    """Permit GitHub privacy addresses only / 仅放行 GitHub 隐私邮箱。"""
    return not value.lower().endswith("@users.noreply.github.com")


def text_rules(text: str, *, core: bool = False) -> set[str]:
    """Return rule identifiers, never matched content / 仅返回规则标识，不返回匹配内容。"""
    findings = {name for name, pattern in RULES if pattern.search(text)}
    if any(private_email(match[0]) for match in EMAIL.finditer(text)):
        findings.add("private-email-address")
    if core:
        if re.search(r"\b(?:TODO|FIXME)\b", text):
            findings.add("core-placeholder-comment")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            findings.add("core-python-syntax-error")
        else:
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if any(
                    isinstance(decorator, ast.Name)
                    and decorator.id == "abstractmethod"
                    or isinstance(decorator, ast.Attribute)
                    and decorator.attr == "abstractmethod"
                    for decorator in node.decorator_list
                ):
                    continue
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    body = body[1:]
                if len(body) == 1 and (
                    isinstance(body[0], ast.Pass)
                    or isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and body[0].value.value is Ellipsis
                    or isinstance(body[0], ast.Raise)
                    and isinstance(body[0].exc, (ast.Name, ast.Call))
                    and (
                        isinstance(body[0].exc, ast.Name)
                        and body[0].exc.id == "NotImplementedError"
                        or isinstance(body[0].exc, ast.Call)
                        and isinstance(body[0].exc.func, ast.Name)
                        and body[0].exc.func.id == "NotImplementedError"
                    )
                ):
                    findings.add("core-placeholder-function")
    return findings


def intended_files(root: Path) -> tuple[list[Path], bool]:
    """List intended files, with archive fallback / 列出拟交付文件，源码包使用目录扫描。"""
    listed = None
    if shutil.which("git") is not None:
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if git_root.returncode == 0 and Path(git_root.stdout.strip()).resolve() == root.resolve():
            listed = subprocess.run(
                ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                cwd=root,
                capture_output=True,
                check=False,
            )
    git_available = listed is not None and listed.returncode == 0
    paths = (
        [root / item.decode("utf-8") for item in listed.stdout.split(b"\0") if item]
        if git_available and listed is not None
        else list(root.rglob("*"))
    )
    selected = [
        path
        for path in paths
        if (path.is_file() or path.is_symlink())
        and not any(
            part in EXCLUDED or part.startswith(".venv") for part in path.relative_to(root).parts
        )
    ]
    return sorted(set(selected)), git_available


def audit(root: Path) -> dict:
    """Scan public files and commit identities / 扫描拟公开文件与提交身份。"""
    root = root.resolve()
    findings = []
    files, has_git = intended_files(root)
    text_count = 0
    for path in files:
        relative = str(path.relative_to(root))
        rules = set()
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            findings.append({"path": relative, "rule": "external-symlink"})
            continue
        try:
            size = path.stat().st_size
        except OSError:
            findings.append({"path": relative, "rule": "unreadable-file"})
            continue
        if size > MAX_FILE_BYTES:
            findings.append({"path": relative, "rule": "file-exceeds-10-mib"})
            continue
        if path.name in {".env", "id_rsa", "id_ed25519"} or path.suffix.lower() in {
            ".pem",
            ".key",
            ".p12",
            ".pfx",
        }:
            rules.add("private-credential-file")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = None
        except OSError:
            rules.add("unreadable-file")
            text = None
        if text is not None:
            text_count += 1
            rules.update(
                text_rules(text, core=relative.startswith("src/") and path.suffix == ".py")
            )
        findings.extend({"path": relative, "rule": rule} for rule in sorted(rules))
    commits = 0
    if has_git:
        log = subprocess.run(
            ["git", "log", "--format=%H%x00%ae%x00%ce"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        for line in log.stdout.splitlines():
            parts = line.split("\0")
            if len(parts) != 3:
                continue
            commits += 1
            for role, email in zip(("author", "committer"), parts[1:], strict=True):
                if private_email(email):
                    findings.append(
                        {"path": f"git:{parts[0][:12]}", "rule": f"private-{role}-email"}
                    )
    return {
        "status": "passed" if not findings else "failed",
        "message": "Public-file privacy audit / 公开文件隐私审查",
        "files_checked": len(files),
        "text_files_checked": text_count,
        "commits_checked": commits,
        "findings": findings,
        "scope": (
            "Heuristic content and commit-email scan; no sensitive matches printed / "
            "启发式内容及提交邮箱扫描，不输出敏感匹配内容"
        ),
        "git_history_status": "checked" if has_git else "not_available_in_source_archive",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root / 仓库目录")
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
