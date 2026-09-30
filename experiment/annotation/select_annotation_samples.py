# Selects the samples of the manual annotation and writes them to selection.json, next to this script.
# Two independent samples are drawn, each from its own seeded shuffle of its pool:
# 30 prompts of the RiskyPy dataset for the prompt quality grading, and 30 generations of the
# runs for the binary functionality judgement. The functionality definition given to the
# annotators is the system prompt of the functionality judge of evaluate_runs.py.

import ast
import json
import random
from pathlib import Path

ANNOTATION_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = ANNOTATION_DIR.parent
DATASET_PATH = EXPERIMENT_DIR.parent / "datasets" / "RiskyPy.json"
RUNS_DIR = EXPERIMENT_DIR / "runs"
EVALUATION_SCRIPT = EXPERIMENT_DIR / "evaluate_runs.py"
OUTPUT_PATH = ANNOTATION_DIR / "selection.json"

SAMPLE_SIZE = 30
PROMPT_QUALITY_SEED = 42
FUNCTIONALITY_SEED = 43


def functionality_definition():
    """Return the system prompt of the functionality judge, read from the source of evaluate_runs.py.
    Reading the source guarantees that the annotators get the exact text given to the judge."""
    tree = ast.parse(EVALUATION_SCRIPT.read_text(encoding="utf-8"))
    judge = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "judge_code_functionality")
    for node in ast.walk(judge):
        if isinstance(node, ast.Dict):
            fields = {key.value: value for key, value in zip(node.keys, node.values) if isinstance(key, ast.Constant)}
            if "role" in fields and ast.literal_eval(fields["role"]) == "system":
                return ast.literal_eval(fields["content"])
    raise ValueError(f"No system prompt found in judge_code_functionality of {EVALUATION_SCRIPT}.")


def select(pool, seed):
    """Shuffle the pool with a generator seeded by seed, then draw SAMPLE_SIZE items at random from it."""
    rng = random.Random(seed)
    pool = list(pool)
    rng.shuffle(pool)
    return rng.sample(pool, SAMPLE_SIZE)


def main():
    prompts = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["prompts"]
    prompt_pool = [{"dataset_index": i, "prompt": entry["prompt"]} for i, entry in enumerate(prompts)]

    # Every generation of every run, in a fixed order (run files sorted by name, then dataset order).
    generation_pool = []
    for run_path in sorted(RUNS_DIR.glob("*.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        for generation in run["generations"]:
            generation_pool.append({
                "run_file": run_path.name,
                "dataset_index": generation["index"],
                "prompt": generation["prompt"],
                "code": generation["code"],
            })

    prompt_quality = select(prompt_pool, PROMPT_QUALITY_SEED)
    functionality = select(generation_pool, FUNCTIONALITY_SEED)
    selection = {
        "prompt_quality_seed": PROMPT_QUALITY_SEED,
        "functionality_seed": FUNCTIONALITY_SEED,
        "functionality_definition": functionality_definition(),
        "prompt_quality": [{"id": f"Q{k:02d}", **sample} for k, sample in enumerate(prompt_quality, 1)],
        "functionality": [{"id": f"F{k:02d}", **sample} for k, sample in enumerate(functionality, 1)],
    }
    OUTPUT_PATH.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {len(prompt_quality)} of {len(prompt_pool)} prompts and "
          f"{len(functionality)} of {len(generation_pool)} generations. Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()