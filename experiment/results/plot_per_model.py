# Creates one grouped bar chart per model from a joint results JSON file.
# Usage: python plot_per_model.py [path/to/joint_results.json]
# Each chart is saved as <model_name>.png in the current directory.

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

METRICS = [
    ("bandit_secure", "Bandit-secure", "#0072B2"),
    ("semgrep_secure", "Semgrep-secure", "#E69F00"),
    ("functional", "Functional", "#009E73"),
    ("functional_and_bandit_secure", "Functional and Bandit-secure", "#56B4E9"),
    ("functional_and_semgrep_secure", "Functional and Semgrep-secure", "#D55E00"),
]

APPROACH_LABELS = {
    "baseline": "Baseline",
    "bandit_refine": "BANDIT-REFINE",
    "self_feedback": "SELF-FEEDBACK",
}

BAR_WIDTH = 0.17


def plot_model(model_name, runs):
    # Draws the bars of every metric for each approach of one model and saves the figure.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(runs))

    for i, (key, label, color) in enumerate(METRICS):
        values = [run[key]["percentage"] for run in runs]
        offset = (i - (len(METRICS) - 1) / 2) * BAR_WIDTH
        bars = ax.bar(x + offset, values, BAR_WIDTH, label=label, color=color,
                      edgecolor="white", linewidth=1)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3,
                     fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([APPROACH_LABELS.get(run["approach"], run["approach"]) for run in runs])
    ax.set_xlabel(model_name, fontsize=12, fontweight="bold", labelpad=15)
    ax.set_ylabel("Generations (%)", fontsize=12)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, linestyle="--", color="#d0d0d0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("Secure and functional generations per model and approach", fontsize=14, pad=35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(METRICS),
              frameon=False, fontsize=10)

    fig.savefig(f"{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "joint_results.json"
    with open(path, encoding="utf-8") as f:
        runs = json.load(f)["runs"]

    # Group runs by model while keeping the order in which they appear in the file.
    runs_by_model = {}
    for run in runs:
        runs_by_model.setdefault(run["model"].split("/")[-1], []).append(run)

    for model_name, model_runs in runs_by_model.items():
        plot_model(model_name, model_runs)


if __name__ == "__main__":
    main()
