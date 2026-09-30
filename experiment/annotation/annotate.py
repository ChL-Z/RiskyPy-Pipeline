# Streamlit application for the manual annotation of the samples listed in selection.json.
# Part 1: grade the quality of each prompt from 0 to 10.
# Part 2: judge whether each generated code is functional according to the given definition.
# Every answer is saved at once to annotations_<annotator>.json, next to this file.
# Run with: python -m streamlit run annotate.py --server.address localhost

import json
import re
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
SELECTION_PATH = APP_DIR / "selection.json"
NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
GRADES = list(range(11))
FUNCTIONALITY_CHOICES = {"Functional": True, "Not functional": False}
PARTS = {"prompt_quality": "Part 1: Prompt quality", "functionality": "Part 2: Functionality"}
ANSWER_FIELDS = {"prompt_quality": "grade", "functionality": "is_functional"}

# Criteria of a good prompt, taken from the requirements that the RiskyPy pipeline imposes on its prompts.
PROMPT_QUALITY_DEFINITION = """\
The better a prompt meets all of the following criteria, the higher its grade:

- **Realistic:** a plausible, small, utility-style task that a developer could actually face.
- **Consistent:** the requirements do not contradict each other, the task fits in a single Python file of roughly 40 to 200 lines, and the prompt explicitly asks for a single-file Python implementation.
- **Precise:** the requirements are concrete and unambiguous, with clearly specified inputs and outputs.
- **Approachable:** a small code generation model (1B to 8B parameters) could produce a somewhat functional solution. No deep architectural design or complex algorithm.
- **Silent about security:** it does not mention security, vulnerabilities, risks, exploits or sanitization, nor an expected number of lines."""


@st.cache_data
def load_selection():
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def output_path(annotator):
    return APP_DIR / f"annotations_{annotator}.json"


def load_answers(selection, annotator):
    """Return the answers saved in the output file of the annotator, or empty answers if it does not exist.
    Stops the app if the output file does not list the same samples as selection.json."""
    answers = {"prompt_quality": {}, "functionality": {}, "definition_acknowledged": False}
    path = output_path(annotator)
    if not path.exists():
        return answers
    saved = json.loads(path.read_text(encoding="utf-8"))
    for part, field in ANSWER_FIELDS.items():
        if [e["id"] for e in saved[part]] != [s["id"] for s in selection[part]]:
            st.error(f"{path.name} does not contain the samples of selection.json.")
            st.stop()
        answers[part] = {e["id"]: e[field] for e in saved[part] if e[field] is not None}
    answers["definition_acknowledged"] = saved["functionality_definition_acknowledged"]
    return answers


def save_answers():
    """Write every sample of the selection with its current answer (null when not answered yet).
    The file is written to a temporary file first, then renamed, so it is never left half written."""
    selection, answers = load_selection(), st.session_state.answers
    output = {
        "annotator": st.session_state.annotator,
        "prompt_quality": [
            {"id": s["id"], "dataset_index": s["dataset_index"], "grade": answers["prompt_quality"].get(s["id"])}
            for s in selection["prompt_quality"]
        ],
        "functionality_definition_acknowledged": answers["definition_acknowledged"],
        "functionality": [
            {"id": s["id"], "run_file": s["run_file"], "dataset_index": s["dataset_index"],
             "is_functional": answers["functionality"].get(s["id"])}
            for s in selection["functionality"]
        ],
    }
    path = output_path(st.session_state.annotator)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def record_answer(part, sample_id, widget_key):
    value = st.session_state[widget_key]
    st.session_state.answers[part][sample_id] = FUNCTIONALITY_CHOICES[value] if part == "functionality" else value
    save_answers()


def acknowledge_definition():
    st.session_state.answers["definition_acknowledged"] = True
    save_answers()


def move(part, step):
    st.session_state[f"position_{part}"] += step


def start_screen(selection):
    st.title("RiskyPy manual annotation")
    name = st.text_input("Annotator name (letters, digits, hyphens and underscores only)")
    if st.button("Start"):
        if not NAME_PATTERN.fullmatch(name):
            st.error("The name must contain only letters, digits, hyphens and underscores.")
            return
        st.session_state.answers = load_answers(selection, name)
        st.session_state.annotator = name
        # Each part starts at its first sample without an answer.
        for part in PARTS:
            unanswered = [k for k, s in enumerate(selection[part]) if s["id"] not in st.session_state.answers[part]]
            st.session_state[f"position_{part}"] = unanswered[0] if unanswered else 0
        save_answers()
        st.rerun()


def sidebar(selection):
    """Show the annotator, the part selector, the progress of each part and the output file.
    Returns the key of the selected part."""
    answers = st.session_state.answers
    st.sidebar.write(f"Annotator: **{st.session_state.annotator}**")
    part = st.sidebar.radio("Part", list(PARTS), format_func=PARTS.get, key="part")
    for key, label in PARTS.items():
        st.sidebar.write(f"{label}: {len(answers[key])} of {len(selection[key])} answered")
    st.sidebar.write("Answers are saved automatically to:")
    st.sidebar.code(str(output_path(st.session_state.annotator)), language="text", wrap_lines=True)
    return part


def scroll_to_top(sample_id):
    # The page keeps its scroll position when the displayed sample changes, so a new sample would
    # open at the bottom. The sample id makes the script differ per sample, so it only runs on a change.
    st.html(f"<script>/* {sample_id} */ document.querySelector('[data-testid=\"stMain\"]').scrollTo(0, 0);</script>",
            unsafe_allow_javascript=True)


def navigation(part, count):
    position = st.session_state[f"position_{part}"]
    previous_column, next_column = st.columns(2)
    previous_column.button("Previous", on_click=move, args=(part, -1), disabled=position == 0, width="stretch")
    next_column.button("Next", on_click=move, args=(part, 1), disabled=position == count - 1, width="stretch")


def prompt_quality_page(selection):
    samples = selection["prompt_quality"]
    position = st.session_state.position_prompt_quality
    sample = samples[position]
    scroll_to_top(sample["id"])
    st.header(PARTS["prompt_quality"])
    st.write("Grade the quality of the prompt below from 0 (lowest quality) to 10 (highest quality), "
             "according to the definition of a good prompt.")
    st.subheader("Definition of a good prompt")
    st.container(border=True).markdown(PROMPT_QUALITY_DEFINITION)
    st.subheader(f"Prompt {sample['id']} ({position + 1} of {len(samples)})")
    st.code(sample["prompt"], language="text", wrap_lines=True)
    grade = st.session_state.answers["prompt_quality"].get(sample["id"])
    widget_key = f"grade_{sample['id']}"
    st.radio("Grade", GRADES, index=None if grade is None else GRADES.index(grade), horizontal=True,
             key=widget_key, on_change=record_answer, args=("prompt_quality", sample["id"], widget_key))
    navigation("prompt_quality", len(samples))


def definition_screen(definition):
    st.header(PARTS["functionality"])
    st.write("Read the definition below carefully before starting this part. For every generation of this "
             "part, decide whether the code is functional strictly according to this definition, and only "
             "according to it. The definition stays displayed above each generation.")
    st.subheader("Definition of a functional generation")
    st.code(definition, language="text", wrap_lines=True)
    st.button("I have read the definition and I will apply it strictly", on_click=acknowledge_definition)


def functionality_page(selection):
    samples = selection["functionality"]
    position = st.session_state.position_functionality
    sample = samples[position]
    scroll_to_top(sample["id"])
    st.header(PARTS["functionality"])
    st.subheader("Definition of a functional generation")
    st.code(selection["functionality_definition"], language="text", wrap_lines=True)
    st.subheader(f"Generation {sample['id']} ({position + 1} of {len(samples)})")
    st.write("**Task**")
    st.code(sample["prompt"], language="text", wrap_lines=True)
    st.write("**Generated code**")
    st.code(sample["code"], language="python", line_numbers=True)
    choices = list(FUNCTIONALITY_CHOICES)
    answer = st.session_state.answers["functionality"].get(sample["id"])
    widget_key = f"functional_{sample['id']}"
    st.radio("According to the definition above, is this generation functional?", choices,
             index=None if answer is None else choices.index("Functional" if answer else "Not functional"),
             horizontal=True, key=widget_key, on_change=record_answer,
             args=("functionality", sample["id"], widget_key))
    navigation("functionality", len(samples))


def main():
    st.set_page_config(page_title="RiskyPy annotation", layout="wide")
    selection = load_selection()
    if "annotator" not in st.session_state:
        start_screen(selection)
        return
    part = sidebar(selection)
    if part == "prompt_quality":
        prompt_quality_page(selection)
    elif not st.session_state.answers["definition_acknowledged"]:
        definition_screen(selection["functionality_definition"])
    else:
        functionality_page(selection)


main()