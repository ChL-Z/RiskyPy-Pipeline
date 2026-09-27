# BANDIT-REFINE: security repair of LLM-generated code driven by Bandit findings.
# The model generates code for a task, Bandit analyzes the code, and when Bandit
# reports findings the same model receives the code and the findings and is asked
# to repair the implementation. Code without findings is kept as generated.
# The module is model-agnostic: the caller provides the generation function.

import os
import tempfile

from bandit.core import config as bandit_config
from bandit.core import manager as bandit_manager

REPAIR_PROMPT = """Bandit, a static analysis tool for Python, reported the following security issues in the code below.
Repair the implementation so that it no longer has these security issues, while keeping its functionality.

BANDIT FINDINGS:
{findings}

CODE:
```python
{code}
```"""


def bandit_issues(code):
    """Run Bandit with its default configuration on a code and return the reported issues."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        manager = bandit_manager.BanditManager(bandit_config.BanditConfig(), "file", False)
        manager.discover_files([path], True)
        manager.run_tests()
        return manager.get_issue_list()
    finally:
        os.unlink(path)


def format_findings(code, issues):
    """Describe each issue as its test id and name, its message, and the reported line of the code."""
    lines = code.splitlines()
    findings = []
    for issue in issues:
        line = lines[issue.lineno - 1].strip() if 0 < issue.lineno <= len(lines) else ""
        findings.append(f"Issue: [{issue.test_id}:{issue.test}] {issue.text}\nLine {issue.lineno}: {line}")
    return "\n\n".join(findings)


def run(tasks, generate_code):
    """Return the final code of each task, in the order of `tasks`.
    generate_code maps a list of prompts to a list of extracted codes."""
    codes = generate_code(tasks)
    to_repair = []
    for i, code in enumerate(codes):
        issues = bandit_issues(code)
        if issues:
            to_repair.append((i, REPAIR_PROMPT.format(findings=format_findings(code, issues), code=code)))
    if to_repair:
        repaired = generate_code([prompt for _, prompt in to_repair])
        for (i, _), code in zip(to_repair, repaired):
            codes[i] = code
    return codes
