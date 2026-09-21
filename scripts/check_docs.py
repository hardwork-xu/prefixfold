#!/usr/bin/env python3
"""Check local documentation contracts / 检查本地文档契约。

This checks structure, links and numeric alignment; semantic translation review remains manual.
本脚本检查结构、链接与数值对齐；翻译语义仍需人工审阅。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", "work", "dist", "__pycache__", "node_modules"}
REQUIRED_DOCS = {
    "RESEARCH.md",
    "ARCHITECTURE.md",
    "EXPERIMENTS.md",
    "DEVELOPMENT.md",
    "WALKTHROUGH.md",
    "RELEASE.md",
    "RESUME.md",
}
BEGIN, END = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+['\"][^\n]*?['\"])?\s*\)")
REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)", re.MULTILINE)


def issue(path: str, rule: str) -> dict[str, str]:
    """Return a compact finding without file contents / 返回不含文件内容的精简问题记录。"""
    return {"path": path, "rule": rule}


def prose_and_fences(text: str) -> tuple[str, bool]:
    """Remove fenced/inline code and detect unclosed fences / 去除代码并检查围栏闭合。"""
    opening: tuple[str, int] | None = None
    prose = []
    for line in text.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if opening is None:
            if marker:
                opening = (marker[1][0], len(marker[1]))
            else:
                prose.append(line)
        elif marker and marker[1][0] == opening[0] and len(marker[1]) >= opening[1]:
            if not marker[2].strip():
                opening = None
    return re.sub(r"`+[^`\n]*`+", "", "\n".join(prose)), opening is None


def link_targets(text: str) -> list[str]:
    """Extract inline/image and reference destinations / 提取行内、图片与引用链接目标。"""
    prose, _ = prose_and_fences(text)
    return [
        match[1].strip("<>")
        for pattern in (INLINE_LINK, REFERENCE_LINK)
        for match in pattern.finditer(prose)
    ]


def markdown_issues(path: Path, root: Path) -> list[dict[str, str]]:
    """Validate one Markdown file locally without fetching URLs / 离线检查单个 Markdown 文件。"""
    relative = str(path.relative_to(root))
    text = path.read_text(encoding="utf-8")
    issues = []
    if not prose_and_fences(text)[1]:
        issues.append(issue(relative, "unclosed-code-fence"))
    for target in link_targets(text):
        parsed = urlsplit(target)
        if parsed.scheme in {"http", "https", "mailto"} or target.startswith("#"):
            continue
        if parsed.scheme or parsed.netloc:
            issues.append(issue(relative, "nonportable-link-scheme"))
            continue
        local = unquote(parsed.path)
        if not local:
            continue
        if Path(local).is_absolute():
            issues.append(issue(relative, "absolute-local-link"))
            continue
        resolved = (path.parent / local).resolve()
        if not resolved.is_relative_to(root.resolve()):
            issues.append(issue(relative, "link-outside-repository"))
        elif not resolved.exists():
            missing = issue(relative, "missing-local-link-target")
            missing["target"] = str(resolved.relative_to(root.resolve()))
            issues.append(missing)
    return issues


def target_issues(targets: dict, config: dict) -> list[dict[str, str]]:
    """Require predeclared targets and benchmark settings to agree / 检查预设目标与基准配置一致。"""
    issues = []
    path = "research/targets.json ↔ configs/benchmark.json"
    try:
        case = next(item for item in config["cases"] if item["name"] == config["primary_case"])
        for key in ("batch", "heads", "prefix", "suffix", "dim"):
            if case[key] != targets["primary_workload"][key]:
                issues.append(issue(path, f"primary-workload-mismatch:{key}"))
        for key in ("seed", "warmup", "repetitions"):
            if config[key] != targets["protocol"][key]:
                issues.append(issue(path, f"measurement-protocol-mismatch:{key}"))
        for key in ("tile_tokens", "workspace_bytes"):
            if config["plan"][key] != targets["protocol"][key]:
                issues.append(issue(path, f"attention-plan-mismatch:{key}"))
        pairs = (
            ("kv_reduction_fraction", "primary_engineering_target"),
            ("torch_sdpa_speedup", "secondary_latency_target"),
        )
        for config_key, target_key in pairs:
            if config["targets"][config_key] != targets[target_key]["minimum"]:
                issues.append(issue(path, f"performance-target-mismatch:{config_key}"))
        if targets.get("frozen_before_benchmark") is not True:
            issues.append(issue(path, "targets-not-declared-frozen"))
    except (KeyError, TypeError, StopIteration):
        issues.append(issue(path, "missing-target-or-config-field"))
    return issues


def check(root: Path) -> dict:
    """Check structure, links, tables and citations / 检查双语结构、链接、表格与引用。"""
    root = root.resolve()
    issues = []
    markdown = [
        path
        for path in root.rglob("*.md")
        if not any(
            part in EXCLUDED or part.startswith((".venv", ".pytest", ".mypy", ".ruff"))
            for part in path.relative_to(root).parts
        )
    ]
    for path in markdown:
        issues.extend(markdown_issues(path, root))
    for filename, other in (("README.md", "README_zh.md"), ("README_zh.md", "README.md")):
        path = root / filename
        if not path.is_file():
            issues.append(issue(filename, "missing-readme"))
            continue
        targets = {
            unquote(urlsplit(target).path).removeprefix("./")
            for target in link_targets(path.read_text())
        }
        if other not in targets:
            issues.append(issue(filename, "missing-language-switch-link"))
    en = {str(path.relative_to(root / "docs/en")) for path in (root / "docs/en").rglob("*.md")}
    zh = {str(path.relative_to(root / "docs/zh")) for path in (root / "docs/zh").rglob("*.md")}
    for filename in sorted(en ^ zh):
        issues.append(issue(f"docs/*/{filename}", "missing-language-counterpart"))
    for filename in sorted(REQUIRED_DOCS - (en & zh)):
        issues.append(issue(f"docs/*/{filename}", "missing-required-document"))
    for language, filename in (("en", "README.md"), ("zh", "README_zh.md")):
        path, table = root / filename, root / "results" / f"table_{language}.md"
        if not path.exists():
            continue
        text = path.read_text()
        if BEGIN in text or END in text:
            if (
                text.count(BEGIN) != 1
                or text.count(END) != 1
                or text.index(BEGIN) >= text.index(END)
            ):
                issues.append(issue(filename, "malformed-generated-table-markers"))
            elif not table.exists():
                issues.append(issue(filename, "generated-table-file-missing"))
            elif text.split(BEGIN, 1)[1].split(END, 1)[0].strip() != table.read_text().strip():
                issues.append(issue(filename, "generated-table-out-of-sync"))
    target_path, config_path = root / "research/targets.json", root / "configs/benchmark.json"
    try:
        issues.extend(
            target_issues(json.loads(target_path.read_text()), json.loads(config_path.read_text()))
        )
    except (OSError, json.JSONDecodeError):
        issues.append(issue("research/targets.json", "unreadable-targets-or-config"))
    cff = root / "CITATION.cff"
    try:
        metadata = yaml.safe_load(cff.read_text())
        if not isinstance(metadata, dict) or any(
            not metadata.get(key) for key in ("cff-version", "message", "title", "authors")
        ):
            issues.append(issue("CITATION.cff", "missing-required-citation-metadata"))
        elif not isinstance(metadata["authors"], list) or not all(
            isinstance(author, dict) and (author.get("name") or author.get("family-names"))
            for author in metadata["authors"]
        ):
            issues.append(issue("CITATION.cff", "invalid-citation-authors"))
    except (OSError, yaml.YAMLError):
        issues.append(issue("CITATION.cff", "unreadable-citation"))
    return {
        "status": "passed" if not issues else "failed",
        "message": "Documentation structural check / 文档结构检查",
        "markdown_files_checked": len(markdown),
        "paired_documents": len(en & zh),
        "findings": issues,
        "scope": (
            "Local targets and bilingual structure; translation meaning and remote URLs "
            "need separate review / 本地目标与双语结构；翻译语义及远端链接需另行审查"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root / 仓库目录")
    args = parser.parse_args()
    result = check(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
