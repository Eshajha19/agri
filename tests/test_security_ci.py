from __future__ import annotations

from pathlib import Path

from scripts.security_ci import PolicyConfig, evaluate_security_policy, scan_repo_for_secrets, scan_requirements_for_policy


def test_scan_repo_detects_secret_in_file(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "sample.py").write_text("token = 'sample-secret-12345'\n", encoding="utf-8")

    findings = scan_repo_for_secrets(repo_root, PolicyConfig())

    assert findings
    assert findings[0].rule_id.startswith("secret-scan:")


def test_requirements_policy_flags_forbidden_sources(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "requirements.txt").write_text("git+https://example.com/repo.git\n", encoding="utf-8")

    policy = PolicyConfig(requirement_files=["requirements.txt"])
    findings = scan_requirements_for_policy(repo_root, policy)

    assert len(findings) == 1
    assert findings[0].rule_id == "dependency:forbidden-source"


def test_requirements_policy_flags_blocked_dependency(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "requirements-dev.txt").write_text("unsafe-package-example==1.0.0\n", encoding="utf-8")

    policy = PolicyConfig(requirement_files=["requirements-dev.txt"], blocked_dependencies=["unsafe-package-example"])
    findings = scan_requirements_for_policy(repo_root, policy)

    assert len(findings) == 1
    assert findings[0].rule_id == "dependency:blocked-package"


def test_policy_evaluation_returns_no_findings_for_clean_repo(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "requirements.txt").write_text("fastapi==0.0.0\n", encoding="utf-8")

    policy = PolicyConfig(requirement_files=["requirements.txt"])
    findings = evaluate_security_policy(repo_root, policy)

    assert findings == []
