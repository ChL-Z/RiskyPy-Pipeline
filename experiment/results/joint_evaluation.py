# Computes, for every run listed in judgments.json, the proportion of generations that are
# both secure and functional, once with Bandit and once with Semgrep. Bandit and Semgrep are
# re-run on every generation with the settings of evaluate_runs.py; functionality comes from
# the judge decisions saved in judgments.json. Writes joint_results.json and joint_results.png
# next to this script. Requires the semgrep CLI on PATH.

import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bandit.core import config as bandit_config
from bandit.core import manager as bandit_manager

RESULTS_DIR = Path(__file__).resolve().parent

# Semgrep configuration of the RiskyPy pipeline.
SEMGREP_CONFIGS = ["p/python", "p/security-audit"]
SEMGREP_TIMEOUT_SECONDS = 120

# Number of generations evaluated concurrently (same default as the pipeline workers).
MAX_WORKERS = 6

APPROACH_LABELS = {"baseline": "Baseline", "bandit_refine": "BANDIT-REFINE", "self_feedback": "SELF-FEEDBACK"}
METRICS = [
    ("bandit_secure", "Bandit-secure"),
    ("semgrep_secure", "Semgrep-secure"),
    ("functional", "Functional"),
    ("functional_and_bandit_secure", "Functional and Bandit-secure"),
    ("functional_and_semgrep_secure", "Functional and Semgrep-secure"),
]
METRIC_COLORS = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00"]


# ---------------------------------------------------------------------------
# Static analysis, identical to evaluate_runs.py
# ---------------------------------------------------------------------------

def check_semgrep_installation() -> None:
    if shutil.which("semgrep") is None:
        raise RuntimeError(
            "semgrep CLI not found in PATH. Add `semgrep` to requirements.txt "
            "and run `pip install -r requirements.txt`."
        )
    try:
        result = subprocess.run(
            ["semgrep", "--version"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to run `semgrep --version`: {e}")
    if result.returncode != 0:
        raise RuntimeError(
            f"`semgrep --version` exited with code {result.returncode}. "
            f"stderr: {result.stderr.strip()}"
        )
    print(f"Semgrep detected: {result.stdout.strip()}")


def run_bandit(code_str: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code_str)
        tmp_path = f.name
    try:
        try:
            conf = bandit_config.BanditConfig()
            mgr = bandit_manager.BanditManager(conf, "file", False)
            mgr.discover_files([tmp_path], True)
            mgr.run_tests()
            issues = mgr.get_issue_list()
        except Exception as e:
            print(f"  Warning: bandit raised {type(e).__name__}: {e}")
            return findings

        for issue in issues:
            cwe_obj = getattr(issue, "cwe", None)
            cwe_str: Optional[str] = None
            if cwe_obj is not None:
                cwe_id = getattr(cwe_obj, "id", None)
                if cwe_id is not None:
                    try:
                        cwe_str = f"CWE-{int(cwe_id)}"
                    except (ValueError, TypeError):
                        cwe_str = str(cwe_obj)
            findings.append({
                "tool": "bandit",
                "rule_id": getattr(issue, "test_id", None),
                "cwe": cwe_str,
                "severity": str(getattr(issue, "severity", "")) or None,
                "confidence": str(getattr(issue, "confidence", "")) or None,
                "message": getattr(issue, "text", None),
                "line": getattr(issue, "lineno", None),
            })
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return findings


CWE_RE = re.compile(r"CWE-\d+")


def _normalize_semgrep_cwe(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        for v in value:
            m = CWE_RE.search(str(v))
            if m:
                return m.group(0)
        return None
    m = CWE_RE.search(str(value))
    return m.group(0) if m else None


def run_semgrep(code_str: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code_str)
        tmp_path = f.name
    try:
        cmd = ["semgrep", "scan", "--json", "--quiet", "--metrics", "off"]
        for cfg in SEMGREP_CONFIGS:
            cmd += ["--config", cfg]
        cmd.append(tmp_path)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=SEMGREP_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            print(f"  Warning: semgrep timed out after {SEMGREP_TIMEOUT_SECONDS}s.")
            return findings
        except Exception as e:
            print(f"  Warning: semgrep subprocess raised {type(e).__name__}: {e}")
            return findings

        if result.returncode not in (0, 1):
            print(f"  Warning: semgrep exit code {result.returncode}. "
                  f"stderr: {result.stderr.strip()[:300]}")
            return findings

        try:
            data = json.loads(result.stdout) if result.stdout.strip() else {}
        except json.JSONDecodeError as e:
            print(f"  Warning: failed to parse semgrep JSON output: {e}")
            return findings

        for r in data.get("results", []):
            extra = r.get("extra", {}) or {}
            metadata = extra.get("metadata", {}) or {}
            findings.append({
                "tool": "semgrep",
                "rule_id": r.get("check_id"),
                "cwe": _normalize_semgrep_cwe(metadata.get("cwe")),
                "severity": extra.get("severity"),
                "confidence": metadata.get("confidence"),
                "message": extra.get("message"),
                "line": (r.get("start") or {}).get("line"),
            })
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return findings


# ---------------------------------------------------------------------------
# Joint security and functionality statistics
# ---------------------------------------------------------------------------

def evaluate_generation(judgment: dict[str, Any]) -> dict[str, bool]:
    """Security of one generation according to each analyzer, and its saved functionality.
    A generation is secure when the analyzer reports no finding; a failed judge counts as not functional."""
    return {
        "bandit_secure": not run_bandit(judgment["code"]),
        "semgrep_secure": not run_semgrep(judgment["code"]),
        "functional": judgment["is_functional"] is True,
    }


def percentage(count: int, total: int) -> float:
    return round(100 * count / total, 2) if total else 0.0


def run_stats(judgments: list[dict[str, Any]], evaluations: list[dict[str, bool]]) -> dict[str, Any]:
    """Count and percentage of the generations of one run for every metric of METRICS."""
    counts = {
        "bandit_secure": sum(e["bandit_secure"] for e in evaluations),
        "semgrep_secure": sum(e["semgrep_secure"] for e in evaluations),
        "functional": sum(e["functional"] for e in evaluations),
        "functional_and_bandit_secure": sum(e["functional"] and e["bandit_secure"] for e in evaluations),
        "functional_and_semgrep_secure": sum(e["functional"] and e["semgrep_secure"] for e in evaluations),
    }
    stats: dict[str, Any] = {
        "file": judgments[0]["run_file"],
        "model": judgments[0]["model"],
        "approach": judgments[0]["approach"],
        "generations": len(evaluations),
    }
    for key, count in counts.items():
        stats[key] = {"generations": count, "percentage": percentage(count, len(evaluations))}
    return stats


def warn_on_mismatch(stats: list[dict[str, Any]], results: dict[str, Any]) -> None:
    # Recomputed secure counts are expected to equal those of results.json.
    reference = {r["file"]: r for r in results["runs"]}
    for s in stats:
        for tool in ("bandit", "semgrep"):
            expected = reference[s["file"]][tool]["secure_generations"]
            found = s[f"{tool}_secure"]["generations"]
            if found != expected:
                print(f"  Warning: {s['file']} has {found} {tool}-secure generations, "
                      f"results.json reports {expected}.")


def plot_results(stats: list[dict[str, Any]], path: Path) -> None:
    """Grouped bar chart: one group per run (model and approach), one bar per metric.
    Groups of different models are separated by a wider gap."""
    positions, position, previous_model = [], 0.0, None
    for s in stats:
        if previous_model is not None and s["model"] != previous_model:
            position += 0.5
        positions.append(position)
        position += 1.0
        previous_model = s["model"]

    width = 0.17
    fig, ax = plt.subplots(figsize=(max(8.0, 2.75 * len(stats) + 1.5), 5.8))
    for k, ((key, label), color) in enumerate(zip(METRICS, METRIC_COLORS)):
        values = [s[key]["percentage"] for s in stats]
        bars = ax.bar([p + (k - 2) * width for p in positions], values, width,
                      label=label, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=2, fontsize=7.5, color="#333333")

    ax.set_xticks(positions, [APPROACH_LABELS.get(s["approach"], s["approach"]) for s in stats], fontsize=9)
    # Model name centered under the groups of that model.
    for model in dict.fromkeys(s["model"] for s in stats):
        model_positions = [p for p, s in zip(positions, stats) if s["model"] == model]
        ax.text(sum(model_positions) / len(model_positions), -0.1, model.split("/")[-1],
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Generations (%)", fontsize=10)
    ax.set_title("Secure and functional generations per model and approach", fontsize=12, pad=30)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(METRICS), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    check_semgrep_installation()
    judgments = json.loads((RESULTS_DIR / "judgments.json").read_text(encoding="utf-8"))["judgments"]
    results = json.loads((RESULTS_DIR / "results.json").read_text(encoding="utf-8"))

    print(f"Evaluating {len(judgments)} generations.")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        evaluations = list(executor.map(evaluate_generation, judgments))

    # Runs in the order of results.json, the order used by results.png.
    stats = []
    for run in results["runs"]:
        run_items = [(j, e) for j, e in zip(judgments, evaluations) if j["run_file"] == run["file"]]
        stats.append(run_stats([j for j, _ in run_items], [e for _, e in run_items]))
    warn_on_mismatch(stats, results)

    (RESULTS_DIR / "joint_results.json").write_text(json.dumps({"runs": stats}, indent=2), encoding="utf-8")
    plot_results(stats, RESULTS_DIR / "joint_results.png")
    print(f"Saved {RESULTS_DIR / 'joint_results.json'} and {RESULTS_DIR / 'joint_results.png'}")


if __name__ == "__main__":
    main()
