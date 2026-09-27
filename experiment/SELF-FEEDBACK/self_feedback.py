# SELF-FEEDBACK: minimal security improvement of LLM-generated code.
# The same model generates code for a task, identifies the security problems
# of that code, then rewrites the implementation to fix them.
# The module is model-agnostic: the caller provides the generation functions.

FEEDBACK_PROMPT = """Review the following Python code, written for the task below, and identify its security problems.
For each problem, name the code construct involved and explain why it is a security risk.
Do not rewrite the code.

TASK:
{task}

CODE:
```python
{code}
```"""

IMPROVEMENT_PROMPT = """The following Python code was written for the task below. A security review of this code identified the problems listed after it.
Rewrite the complete implementation so that it fixes these security problems while still fulfilling every requirement of the task.

TASK:
{task}

CODE:
```python
{code}
```

SECURITY PROBLEMS:
{feedback}"""


def run(tasks, generate_code, generate_text):
    """Return the improved code of each task, in the order of `tasks`.
    generate_code and generate_text map a list of prompts to a list of extracted codes and free-text replies."""
    codes = generate_code(tasks)
    feedbacks = generate_text([
        FEEDBACK_PROMPT.format(task=task, code=code)
        for task, code in zip(tasks, codes)
    ])
    return generate_code([
        IMPROVEMENT_PROMPT.format(task=task, code=code, feedback=feedback)
        for task, code, feedback in zip(tasks, codes, feedbacks)
    ])
