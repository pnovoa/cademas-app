## CADEMAS-ML

Streamlit application for cooperative and context-aware decision support.

### Run Locally

Create and activate a virtual environment (Python 3.11):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

H2O model loading requires Java 17+. On macOS: `brew install openjdk@17`.

Run the app:

```bash
streamlit run app/app_v1.py
```

### Example Inputs

A ready-to-load example is available in `example_attrition/`:

- Model configuration: `example_attrition/models/model_definitions.json`
- Context configurations: `example_attrition/context/*.json`
- MOJO models: `example_attrition/models/*.zip`
- Dataset: `example_attrition/data/cases_atttrition.csv`

For Streamlit Community Cloud, use `app/app_v1.py` as the main file.
