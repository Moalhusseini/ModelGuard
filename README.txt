# ModelGuard — Local Streamlit Runtime

## 1. Scope
This package contains the complete ModelGuard project snapshot and a standalone local Streamlit runtime.
The runtime uses the trained XGBoost pipeline and the existing Monitoring, Governance, and MLflow records.
The Jupyter notebooks are retained as research/development documentation; they are not required to run the Streamlit application.

## 2. Requirements
- Windows 10/11
- Python 3.12.x (the project notebook records Python 3.12.10)
- Ollama installed
- Local Ollama model `qwen2.5:1.5b`

Install the model once:
```powershell
ollama pull qwen2.5:1.5b
```

## 3. Run
Double-click `start_modelguard.bat`.

The launcher creates `.venv` if needed, installs the runtime dependencies, checks Ollama and the model tag, then starts Streamlit.
Alternatively:
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## 4. Customer churn prediction
1. Expand **Customer prediction input**.
2. Paste one JSON object with exactly the 19 model input features.
3. Do not include `customerID` or `Churn`.
4. Ask a prediction question in the chat.

The app validates missing and unexpected feature names before calling the agent. The runtime also validates the record against the loaded model's expected feature schema.

## 5. What the agent does
- Deterministic routing for the registered ModelGuard tool domains, with local LLM planning as fallback.
- Python tools retrieve Monitoring, Governance, MLflow, and model evidence.
- Deterministic synthesis adds comparisons and summaries for selected domains.
- Audit logs record tool execution and validation outcomes.

## 6. Important limitations
- This is a local research prototype, not a production deployment.
- System status currently checks required file presence; it does not prove full operational readiness.
- Monitoring analyses use stored datasets and are not live production monitoring.
- Batch 1 and Batch 2 are random partitions, not chronological production windows.
- The churn score is not established as a calibrated probability.
- Individual feature attribution (e.g. local SHAP values) is not implemented. Customer profile signals are descriptive and must not be presented as causal explanations.
- Governance incident records describe scenarios, not confirmed incidents.
- Human review remains necessary for consequential business decisions.

## 7. Project folders
Keep `artifacts`, `monitoring`, `governance`, `memory`, `logs`, `mlruns`, `mlflow.db`, and the source files together in the same project root. Do not move the model file independently from its `artifacts` folder.

## 8. Troubleshooting
- If Ollama is unavailable: open a terminal and run `ollama serve`.
- If the model is missing: run `ollama pull qwen2.5:1.5b`.
- If Python packages fail to install: confirm Python 3.12 is installed, then run the installation commands in section 3.
- If a prediction fails: expand the result's **Execution details** to inspect the planned tools, tool execution, reflection, and validation records.
