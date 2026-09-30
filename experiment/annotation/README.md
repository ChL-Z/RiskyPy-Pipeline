# RiskyPy annotation app

This application is used to annotate two independent sets of samples:

- **Part 1: Prompt quality.** 30 code generation prompts. Give each prompt a grade from 0 (lowest quality) to 10 (highest quality).
- **Part 2: Functionality.** 30 pairs of a task and the Python code generated for it. Decide whether each code is functional or not. Before starting this part, read the definition of a functional generation shown by the app, and apply it strictly: judge only according to this definition. The definition stays displayed above each generation.

## Files

- `annotate.py`: the application.
- `selection.json`: the samples to annotate. Do not modify it.
- `requirements.txt`: the Python dependencies.

## Installation

Python 3.10 or newer is required. In a terminal opened in this folder:

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

In a terminal opened in this folder, with the virtual environment activated:

```bash
python -m streamlit run annotate.py --server.address localhost
```

- Start the app through `python` as shown above, not with `streamlit run annotate.py` directly: some computers block the `streamlit` launcher (on Windows, for example, with a message about the Device Guard policy), while `python` is allowed.
- Do not start it with `python annotate.py`: this only prints warnings in the terminal and does not open the app.
- The app opens in your web browser (at http://localhost:8501).
- If Streamlit asks for an email address the first time it starts, leave it empty and press Enter. This is an optional newsletter sign-up from Streamlit.
- If Windows Firewall asks whether to allow Python to access public and private networks, click **Cancel**. The app only communicates with your own browser, on your own computer, which works without this permission.
- To stop the app, close the browser tab and press Ctrl+C in the terminal.

## Annotating

1. Enter your annotator name and click **Start**. Use the same name every time you open the app, so that you continue your own annotations.
2. Choose the part to annotate in the left sidebar. The sidebar also shows how many samples of each part you have answered.
3. Answer each sample, and move between samples with **Previous** and **Next**. You can go back and change an answer at any time.

Every answer is saved immediately to the file `annotations_<your name>.json`, in this folder. You can stop at any time and continue later: when you start again with the same name, the app loads your answers and opens each part at its first sample without an answer.

## Sending your annotations

When both parts are complete (30 of 30 answered in each part), send the file `annotations_<your name>.json` to the person in charge of the study.