"""Documentation and privacy helpers / 文档与隐私检查辅助逻辑。"""

import json
from pathlib import Path

import pytest

from scripts.audit import private_email, text_rules
from scripts.check_docs import link_targets, markdown_issues, prose_and_fences, target_issues


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_code_fences_ignore_sample_links(fence):
    text = f"{fence}text\n[fake](missing.md)\n{fence}\n[real](existing.md)"
    assert prose_and_fences(text)[1]
    assert link_targets(text) == ["existing.md"]


def test_unclosed_code_fence_is_reported(tmp_path):
    path = tmp_path / "page.md"
    path.write_text("```python\nprint(1)\n")
    assert {item["rule"] for item in markdown_issues(path, tmp_path)} == {"unclosed-code-fence"}


def test_link_resolution_handles_images_references_and_spaces(tmp_path):
    (tmp_path / "space name.md").write_text("content")
    (tmp_path / "figure.svg").write_text("<svg/>")
    path = tmp_path / "page.md"
    path.write_text(
        "[page](<space name.md>) ![plot](figure.svg) [another][ref]\n"
        "[ref]: space%20name.md#section\n[remote](https://example.org/) [anchor](#here)"
    )
    assert markdown_issues(path, tmp_path) == []


def test_missing_and_outside_links_are_reported(tmp_path):
    path = tmp_path / "page.md"
    path.write_text("[missing](no.md) [outside](../outside.md)")
    assert {item["rule"] for item in markdown_issues(path, tmp_path)} == {
        "missing-local-link-target",
        "link-outside-repository",
    }


def test_frozen_target_mismatch_is_reported():
    root = Path(__file__).resolve().parents[1]
    targets = json.loads((root / "research/targets.json").read_text())
    config = json.loads((root / "configs/benchmark.json").read_text())
    assert target_issues(targets, config) == []
    config["targets"]["kv_reduction_fraction"] = 0.1
    assert (
        target_issues(targets, config)[0]["rule"]
        == "performance-target-mismatch:kv_reduction_fraction"
    )


def test_privacy_rule_never_returns_matched_content():
    private_path = "/" + "Users" + "/private-person/project/file.py"
    result = text_rules(private_path)
    assert result == {"absolute-home-path"}
    assert private_path not in json.dumps(sorted(result))


def test_public_noreply_and_private_email_differ():
    public_address = "maintainer" + "@users.noreply.github.com"
    private_address = "private-person" + "@" + "example.org"
    assert not private_email(public_address)
    assert private_email(private_address)
    assert text_rules(public_address) == set()
    assert text_rules(private_address) == {"private-email-address"}


def test_secrets_are_detected_but_ordinary_token_fields_are_allowed():
    token = "ghp" + "_" + "a" * 36
    assert text_rules(token) == {"github-token"}
    assert text_rules('tile_tokens = 1024\n{"tokens": 40}') == set()


@pytest.mark.parametrize("body", ["pass", "...", "raise NotImplementedError()"])
def test_core_placeholder_detection(body):
    assert "core-placeholder-function" in text_rules(f"def compute():\n    {body}\n", core=True)


@pytest.mark.parametrize("decorator", ["abstractmethod", "abc.abstractmethod"])
def test_supported_abstract_declaration_is_not_placeholder(decorator):
    text = f"@{decorator}\ndef compute():\n    raise NotImplementedError()\n"
    assert text_rules(text, core=True) == set()
