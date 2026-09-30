# RiskyPy

RiskyPy generates Python coding tasks for evaluating security and functionality together. It combines risky programming domains, checks prompt quality, uses two models to assess task difficulty and vulnerability exposure, and removes similar prompts. The repository includes a dataset of 73 prompts and experiments with two code-generation models.

## Folders

| Folder | Contents |
| --- | --- |
| `pipeline/` | Dataset generation script, domain definitions, and dependencies. |
| `datasets/` | Generated prompts, canary code, and static-analysis findings in `RiskyPy.json`. |
| `experiment/` | Model generation and evaluation scripts and dependencies. |
| `experiment/BANDIT-REFINE/` | One code-repair pass using Bandit findings. |
| `experiment/SELF-FEEDBACK/` | Security review and code rewriting by the generating model. |
| `experiment/runs/` | Saved code generations for each model and approach. |
| `experiment/results/` | Evaluation statistics, functionality judgments, and the results chart. |
| `experiment/annotation/` | Manual annotation app, sample selection, annotations, and agreement analysis. |
| `colabrunner/` | Google Colab notebook for running the model-generation experiments. |

## Run

Commands below run from the repository root in a Python environment. API steps require network access and the corresponding API keys. Semgrep downloads its rules from the registry.

### Generate the dataset

```bash
python -m pip install -r pipeline/requirements.txt
export OPENAI_API_KEY="<key>"
export OPENROUTER_API_KEY="<key>"
python pipeline/RiskyPyPipeline.py
```

This writes `datasets/RiskyPy.json`, replacing any existing dataset.

### Generate and evaluate model outputs

```bash
python -m pip install -r experiment/requirements.txt
python experiment/run_experiment.py
export OPENAI_API_KEY="<key>"
python experiment/evaluate_runs.py
```

Generation requires a CUDA GPU. It runs Qwen3-4B and Seed-Coder-8B with the baseline, BANDIT-REFINE, and SELF-FEEDBACK approaches. Existing run files are skipped. Move the included files out of `experiment/runs/` before generating fresh runs.

Evaluation can run directly on the included runs without a GPU. It uses Bandit, Semgrep, and an OpenAI functionality judge, and replaces the files in `experiment/results/`.

The Colab notebook provides an alternative for model generation. Evaluation runs separately.

### Manual annotation

```bash
python -m pip install -r experiment/annotation/requirements.txt
python -m streamlit run experiment/annotation/annotate.py
python experiment/annotation/analyze_annotations.py
```

The app uses the included `selection.json`. `select_annotation_samples.py` creates a new selection when needed. Annotation analysis also requires the experiment dependencies.
