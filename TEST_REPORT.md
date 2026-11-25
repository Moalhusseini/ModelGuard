# ModelGuard package verification report

## Package checks performed in the build environment
- Python syntax compilation: `app.py` and `modelguard_agent.py` passed.
- Model artifact loading: succeeded.
- Expected input schema: 19 features found.
- Customer prediction smoke test: succeeded for the included sample record; model score 0.5839, threshold 0.56, label `Churn`.
- Prediction orchestration test: deterministic routing selected `predict_customer_churn`; tool execution succeeded; reflection approved; validation released the response.
- Tool/report smoke checks: risk register, incident management, governance controls, system status, batch performance, model information, and drift analysis returned successfully.
- Analytical summaries verified in generated output for incident priorities, lifecycle status, batch deltas, and model comparisons.

## Environment caveats
- The build environment did not have Streamlit or MLflow installed, so the Streamlit UI and MLflow connector were not launched end-to-end here.
- The serialized model was created with scikit-learn 1.9.1. The build environment had a different scikit-learn version and emitted version warnings, although model loading and prediction succeeded. The packaged `requirements.txt` pins the versions recorded in the project notebook.
- Ollama and `qwen2.5:1.5b` are local user-machine dependencies and cannot be verified from this build environment.
- Run `smoke_test.py` inside the project's Python 3.12 environment after installing requirements.

## Important analytical limitations retained
- Churn score is not claimed to be calibrated.
- No local SHAP/per-customer feature attribution is implemented; profile signals are descriptive only.
- Monitoring uses stored datasets, not live production streams.
- Batch 1 and Batch 2 are random partitions, not chronological time windows.
- System status checks file presence, not full operational readiness.
- Governance incident records are scenarios, not confirmed real incidents.
