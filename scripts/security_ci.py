"""Security CI helper for secret scanning, dependency policy enforcement, and reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from security_hygiene import scan_text_for_secrets


DEFAULT_SCAN_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".md", ".ini"}
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "tests",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "logs",
}
DEFAULT_REQUIREMENT_FILES = ["requirements.txt", "requirements-dev.txt"]
DEFAULT_FORBIDDEN_REQUIREMENT_PATTERNS = [
    r"^\s*-e\s+",
    r"^\s*git\+",
    r"^\s*https?://",
    r"^\s*file:",
    r"^\s*--extra-index-url",
    r"^\s*--find-links",
]


@dataclass(slots=True)
class PolicyConfig:
    scan_extensions: list[str] = field(default_factory=lambda: sorted(DEFAULT_SCAN_EXTENSIONS))
    excluded_dirs: list[str] = field(default_factory=lambda: sorted(DEFAULT_EXCLUDED_DIRS))
    requirement_files: list[str] = field(default_factory=lambda: list(DEFAULT_REQUIREMENT_FILES))
    forbidden_requirement_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_FORBIDDEN_REQUIREMENT_PATTERNS))
    blocked_dependencies: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SecurityFinding:
    category: str
    severity: str
    path: str
    message: str
    rule_id: str


def load_policy(policy_path: str | Path | None) -> PolicyConfig:
    if policy_path is None:
        return PolicyConfig()

    raw = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    return PolicyConfig(
        scan_extensions=list(raw.get("scan_extensions", DEFAULT_SCAN_EXTENSIONS)),
        excluded_dirs=list(raw.get("excluded_dirs", DEFAULT_EXCLUDED_DIRS)),
        requirement_files=list(raw.get("requirement_files", DEFAULT_REQUIREMENT_FILES)),
        forbidden_requirement_patterns=list(raw.get("forbidden_requirement_patterns", DEFAULT_FORBIDDEN_REQUIREMENT_PATTERNS)),
        blocked_dependencies=[str(name).lower() for name in raw.get("blocked_dependencies", [])],
    )


def _should_scan_file(path: Path, policy: PolicyConfig) -> bool:
    if path.name.startswith(".") and path.suffix not in {".yml", ".yaml", ".json", ".toml"}:
        return False
    return path.suffix.lower() in set(policy.scan_extensions)


def _iter_repo_files(root: Path, policy: PolicyConfig) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in policy.excluded_dirs for part in path.parts):
            continue
        if _should_scan_file(path, policy):
            yield path


def scan_repo_for_secrets(root: str | Path, policy: PolicyConfig | None = None) -> list[SecurityFinding]:
    policy = policy or PolicyConfig()
    root_path = Path(root)
    findings: list[SecurityFinding] = []

    for file_path in _iter_repo_files(root_path, policy):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for finding in scan_text_for_secrets(text):
            findings.append(
                SecurityFinding(
                    category=finding.category,
                    severity="high",
                    path=str(file_path.relative_to(root_path)),
                    message=f"Matched {finding.category} secret pattern",
                    rule_id=f"secret-scan:{finding.category}",
                )
            )

    return findings


def _parse_requirement_name(line: str) -> str:
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)", line)
    return match.group(1).lower() if match else ""


def scan_requirements_for_policy(root: str | Path, policy: PolicyConfig | None = None) -> list[SecurityFinding]:
    policy = policy or PolicyConfig()
    root_path = Path(root)
    findings: list[SecurityFinding] = []
    forbidden_patterns = [re.compile(pattern) for pattern in policy.forbidden_requirement_patterns]

    for relative_path in policy.requirement_files:
        req_path = root_path / relative_path
        if not req_path.exists():
            continue

        for line_number, raw_line in enumerate(req_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("-r "):
                continue

            for pattern in forbidden_patterns:
                if pattern.search(line):
                    findings.append(
                        SecurityFinding(
                            category="dependency-policy",
                            severity="high",
                            path=f"{relative_path}:{line_number}",
                            message=f"Forbidden requirement syntax: {line}",
                            rule_id="dependency:forbidden-source",
                        )
                    )
                    break

            dependency_name = _parse_requirement_name(line)
            if dependency_name and dependency_name in policy.blocked_dependencies:
                findings.append(
                    SecurityFinding(
                        category="dependency-sca",
                        severity="high",
                        path=f"{relative_path}:{line_number}",
                        message=f"Blocked dependency present: {dependency_name}",
                        rule_id="dependency:blocked-package",
                    )
                )

    return findings


def evaluate_security_policy(root: str | Path, policy: PolicyConfig | None = None) -> list[SecurityFinding]:
    policy = policy or PolicyConfig()
    findings = []
    findings.extend(scan_repo_for_secrets(root, policy))
    findings.extend(scan_requirements_for_policy(root, policy))
    return findings


def _format_findings(findings: list[SecurityFinding]) -> str:
    if not findings:
        return "No security policy violations found."
    lines = ["Security policy violations:"]
    for finding in findings:
        lines.append(f"- [{finding.severity}] {finding.path} :: {finding.rule_id} :: {finding.message}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Security CI helper")
    parser.add_argument("command", choices=["scan", "policy"], help="Which check to run")
    parser.add_argument("--root", default=".", help="Repository root to scan")
    parser.add_argument("--policy", default=None, help="Path to JSON policy file")
    parser.add_argument("--output", default=None, help="Optional JSON report output")
    args = parser.parse_args(argv)

    policy = load_policy(args.policy)
    if args.command == "scan":
        findings = scan_repo_for_secrets(args.root, policy)
    else:
        findings = evaluate_security_policy(args.root, policy)

    report = {
        "command": args.command,
        "root": str(Path(args.root).resolve()),
        "findings": [asdict(finding) for finding in findings],
        "violation_count": len(findings),
    }

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(_format_findings(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
