"""Offline smoke checks for the ModelGuard local runtime.

Run from the project root:
    .venv\\Scripts\\python.exe smoke_test.py

This script does not call Ollama or modify model/data artifacts.
"""
import json
from pathlib import Path

import modelguard_agent as mg


def main():
    root = Path(__file__).resolve().parent
    required = [
        root / "artifacts" / "final_xgboost_model.joblib",
        root / "artifacts" / "model_results.json",
        root / "monitoring" / "performance_baseline.json",
        root / "governance" / "risk_register.json",
        root / "governance" / "incident_management.json",
        root / "mlflow.db",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required project files: " + ", ".join(missing))

    if mg.xgb_model is None or len(mg.EXPECTED_FEATURES) != 19:
        raise SystemExit("XGBoost model did not load with the expected 19-feature schema.")

    sample = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 53.85,
        "TotalCharges": 108.15,
    }
    prediction = mg.predict_customer_churn(sample)
    assert prediction["prediction_label"] in ("Churn", "No Churn")
    assert 0 <= prediction["churn_score"] <= 1

    checks = {
        "risk_register": len(mg.get_risk_register()),
        "incident_scenarios": len(mg.get_incident_management()),
        "lifecycle_controls": len(mg.get_governance_controls()["lifecycle_controls"]),
        "system_available": mg.get_system_status()["available"],
        "batch_records": len(mg.get_batch_performance()),
        "drift_features": mg.get_drift_analysis()["number_of_features"],
    }
    print("Model prediction:", json.dumps(prediction, ensure_ascii=False))
    print("Project checks:", json.dumps(checks, ensure_ascii=False))
    print("Registered tools:", len(mg.TOOL_REGISTRY))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
