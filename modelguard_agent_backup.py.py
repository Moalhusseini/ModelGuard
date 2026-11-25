"""ModelGuard agent runtime extracted from Notebook 05.

Run from the project root. The project data folders (artifacts, monitoring,
governance, memory, logs, mlruns, and mlflow.db) must remain in place.
"""
import json
import time
import uuid
import inspect
from datetime import datetime
from pathlib import Path
from typing import List, Literal

import joblib
import mlflow
import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field
from scipy.stats import ks_2samp, chi2_contingency
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
GOVERNANCE_DIR = PROJECT_ROOT / "governance"
REPORTS_DIR = PROJECT_ROOT / "reports"
MEMORY_DIR = PROJECT_ROOT / "memory"
LOG_DIR = PROJECT_ROOT / "logs"

for directory in (REPORTS_DIR, MEMORY_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
AGENT_NAME = "ModelGuard Agent"
MODEL_NAME = "qwen2.5:1.5b"
OLLAMA_URL = "http://localhost:11434"
GENERATE_URL = f"{OLLAMA_URL}/api/generate"
TAGS_URL = f"{OLLAMA_URL}/api/tags"
MODEL_TEMPERATURE = 0
MODEL_TIMEOUT = 180
MAX_ITERATIONS = 5
session = requests.Session()

PERSISTENT_MEMORY_FILE = MEMORY_DIR / "persistent_memory.json"
AUDIT_LOG_FILE = LOG_DIR / "agent_audit_log.jsonl"

if not PERSISTENT_MEMORY_FILE.exists():
    PERSISTENT_MEMORY_FILE.write_text(
        json.dumps({"system_facts": [], "verified_observations": []}, indent=4),
        encoding="utf-8"
    )

SESSION_ID = str(uuid.uuid4())
short_term_memory = {
    "session_id": SESSION_ID,
    "created_at": datetime.now().isoformat(),
    "user_requests": [],
    "execution_plans": [],
    "tool_observations": []
}



# Data, monitoring, governance and MLflow tools


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.name}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.name}")

    return pd.read_csv(path)

def get_model_information():
    results = load_json(ARTIFACTS_DIR / "model_results.json")

    return {
        "model_results": results.get("xgboost"),
        "baseline_model": results.get("logistic_regression"),
        "experimental_model": results.get("xgboost_undersampling")
    }

def get_performance_baseline():
    return load_json(MONITORING_DIR / "performance_baseline.json")

def get_batch_performance():
    df = load_csv(MONITORING_DIR / "batch_performance_report.csv")

    return df.to_dict(orient="records")

def get_drift_analysis():
    reference = load_csv(MONITORING_DIR / "reference_data.csv")
    comparison = load_csv(MONITORING_DIR / "comparison_data.csv")

    numerical_features = [
        "tenure", "MonthlyCharges", "TotalCharges"
    ]

    categorical_features = [
        column for column in reference.columns
        if column not in numerical_features
    ]

    alpha = 0.05
    adjusted_alpha = alpha / len(reference.columns)

    results = []

    for feature in numerical_features:
        test = ks_2samp(
            reference[feature].dropna(),
            comparison[feature].dropna()
        )

        results.append({
            "feature": feature,
            "test": "Kolmogorov-Smirnov",
            "statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "adjusted_alpha": adjusted_alpha,
            "drift_detected": bool(test.pvalue < adjusted_alpha)
        })

    for feature in categorical_features:
        categories = sorted(
            set(reference[feature].dropna().unique()) |
            set(comparison[feature].dropna().unique()),
            key=str
        )

        table = pd.DataFrame({
            "Reference": [
                (reference[feature] == category).sum()
                for category in categories
            ],
            "Comparison": [
                (comparison[feature] == category).sum()
                for category in categories
            ]
        }, index=categories).T

        chi2, p_value, _, _ = chi2_contingency(table)

        results.append({
            "feature": feature,
            "test": "Chi-Square",
            "statistic": float(chi2),
            "p_value": float(p_value),
            "adjusted_alpha": adjusted_alpha,
            "drift_detected": bool(p_value < adjusted_alpha)
        })

    return {
        "analysis_type": "Reference vs Comparison Data",
        "number_of_features": len(results),
        "drifted_features": [
            item["feature"] for item in results
            if item["drift_detected"]
        ],
        "results": results
    }

def get_system_profile():
    return load_json(GOVERNANCE_DIR / "ai_system_profile.json")

def get_risk_register():
    return load_json(GOVERNANCE_DIR / "risk_register.json")

def get_impact_assessment():
    return load_json(GOVERNANCE_DIR / "impact_assessment.json")

def get_incident_management():
    return load_json(GOVERNANCE_DIR / "incident_management.json")

def get_governance_controls():
    files = {
        "lifecycle_controls": "lifecycle_controls.json",
        "ai_management_structure": "ai_management_structure.json",
        "framework_assessment": "framework_assessment.json",
        "responsible_ai_principles": "responsible_ai_principles.json"
    }

    return {
        name: load_json(GOVERNANCE_DIR / filename)
        for name, filename in files.items()
    }

def get_mlflow_client():
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient(tracking_uri=TRACKING_URI)

def get_mlflow_experiments():
    client = get_mlflow_client()
    experiments = client.search_experiments()

    return [
        {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "lifecycle_stage": experiment.lifecycle_stage
        }
        for experiment in experiments
    ]

def get_mlflow_run_details(run_id=None):
    client = get_mlflow_client()

    experiment = client.get_experiment_by_name(
        "ModelGuard_Churn_Prediction"
    )

    if experiment is None:
        raise ValueError("ModelGuard MLflow experiment not found.")

    if run_id is not None:
        run = client.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags
        }

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=100
    )

    return [
        {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags
        }
        for run in runs
    ]

def get_system_status():
    required_files = {
        "Model Results": ARTIFACTS_DIR / "model_results.json",
        "Performance Baseline": MONITORING_DIR / "performance_baseline.json",
        "Reference Data": MONITORING_DIR / "reference_data.csv",
        "Comparison Data": MONITORING_DIR / "comparison_data.csv",
        "Batch Performance": MONITORING_DIR / "batch_performance_report.csv",
        "System Profile": GOVERNANCE_DIR / "ai_system_profile.json",
        "Risk Register": GOVERNANCE_DIR / "risk_register.json",
        "Impact Assessment": GOVERNANCE_DIR / "impact_assessment.json",
        "Incident Management": GOVERNANCE_DIR / "incident_management.json",
        "MLflow Database": PROJECT_ROOT / "mlflow.db"
    }

    status = {
        name: "Available" if path.exists() else "Missing"
        for name, path in required_files.items()
    }

    return {
        "components": status,
        "available": sum(value == "Available" for value in status.values()),
        "missing": sum(value == "Missing" for value in status.values())
    }


TOOL_REGISTRY = {
    "get_model_information": get_model_information,
    "get_performance_baseline": get_performance_baseline,
    "get_batch_performance": get_batch_performance,
    "get_drift_analysis": get_drift_analysis,
    "get_system_profile": get_system_profile,
    "get_risk_register": get_risk_register,
    "get_impact_assessment": get_impact_assessment,
    "get_incident_management": get_incident_management,
    "get_governance_controls": get_governance_controls,
    "get_mlflow_experiments": get_mlflow_experiments,
    "get_mlflow_run_details": get_mlflow_run_details,
    "get_system_status": get_system_status,
}

TOOL_DESCRIPTIONS = {
    "get_model_information": "Retrieve model information, configurations, and evaluation results.",
    "get_performance_baseline": "Retrieve the baseline performance metrics of the churn prediction model.",
    "get_batch_performance": "Compare model performance across different evaluation batches.",
    "get_drift_analysis": "Analyze statistical data drift between reference and comparison datasets.",
    "get_system_profile": "Retrieve the AI system profile, purpose, and deployment status.",
    "get_risk_register": "Retrieve identified AI system risks, scores, and mitigation measures.",
    "get_impact_assessment": "Retrieve the impact assessment for affected stakeholders.",
    "get_incident_management": "Retrieve incident scenarios, responses, and responsibilities.",
    "get_governance_controls": "Retrieve AI governance, lifecycle, and framework controls.",
    "get_mlflow_experiments": "Retrieve MLflow experiments.",
    "get_mlflow_run_details": "Retrieve MLflow run metrics, parameters, and tags.",
    "get_system_status": "Check the availability of ModelGuard system components.",
}


# Memory functions


def load_persistent_memory():
    """Load verified information from persistent memory."""
    with open(PERSISTENT_MEMORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def save_persistent_memory(memory_data):
    """Save persistent memory to JSON."""
    with open(PERSISTENT_MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory_data, file, indent=4, ensure_ascii=False)

def add_short_term_memory(category, content):
    """Store information during the current session."""
    if category not in short_term_memory:
        raise ValueError(f"Unknown memory category: {category}")

    short_term_memory[category].append({
        "timestamp": datetime.now().isoformat(),
        "content": content
    })

def add_persistent_memory(category, content, source):
    """Store verified information without creating duplicates."""

    allowed_categories = ["system_facts", "verified_observations"]

    if category not in allowed_categories:
        raise ValueError("Invalid persistent memory category.")

    if not content or not source:
        raise ValueError("Content and source are required.")

    memory_data = load_persistent_memory()

    # Check for duplicate information
    for entry in memory_data[category]:
        if entry["content"] == content and entry["source"] == source:
            print("Information already exists in persistent memory.")
            return

    memory_data[category].append({
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "source": source
    })

    save_persistent_memory(memory_data)

def retrieve_memory(query):
    """Retrieve relevant entries using simple keyword matching."""
    memory_data = load_persistent_memory()
    query_words = set(query.lower().split())

    results = []

    for category, entries in memory_data.items():
        for entry in entries:
            searchable_text = entry["content"].lower()
            if any(word in searchable_text for word in query_words):
                results.append({
                    "category": category,
                    **entry
                })

    return results


# Local LLM planning
PLANNING_TOOLS = dict(TOOL_DESCRIPTIONS)



def reason_and_plan(user_request):

    prompt = f"""
You are the reasoning and planning component of ModelGuard.

Analyze the user request and create a structured execution plan.

User request:
{user_request}

Available tools:
{json.dumps(PLANNING_TOOLS, ensure_ascii=False)}

Return valid JSON only, using this structure:

{{
    "reasoning": {{
        "objective": "Main purpose of the request",
        "required_information": [
            "Information needed to answer the request"
        ]
    }},
    "plan": [
        {{
            "step": 1,
            "tool_name": "exact_tool_name",
            "purpose": "Purpose of this tool"
        }}
    ]
}}

Rules:
- Select only tools from the available tools list.
- Select the minimum number of tools needed to answer the request.
- Use only directly relevant tools.
- Do not retrieve additional information unless necessary.
- Do not include MLflow tools unless the user asks about experiments or model tracking.
- Organize steps in a logical order.
- Maximum 5 tools per plan.
- Do not execute tools.
- Do not invent tool names.
- If no tool is required, return an empty plan.
- Do not include private chain-of-thought.
"""

    try:
        response = session.post(
            GENERATE_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": MODEL_TEMPERATURE}
            },
            timeout=MODEL_TIMEOUT
        )

        response.raise_for_status()
        result = json.loads(response.json()["response"])

        reasoning = result.get("reasoning", {})
        plan = result.get("plan", [])

        if not isinstance(reasoning, dict) or not isinstance(plan, list):
            raise ValueError("Invalid reasoning or plan structure.")

        # Validate and normalize the execution plan.
        validated_plan = []
        seen_tools = set()

        for step in plan:
            tool_name = step.get("tool_name")

            if tool_name not in PLANNING_TOOLS:
                continue

            if tool_name in seen_tools:
                continue

            seen_tools.add(tool_name)

            validated_plan.append({
                "step": len(validated_plan) + 1,
                "tool_name": tool_name,
                "purpose": str(step.get("purpose", ""))
            })
        
        # Limit the plan to a maximum of five tools.
        validated_plan = validated_plan[:5]
        
        return {
            "status": "success",
            "reasoning": reasoning,
            "plan": validated_plan
        }

    except Exception as error:
        return {
            "status": "error",
            "error": str(error),
            "reasoning": {},
            "plan": []
        }

def reason_and_plan_with_memory(user_request):
    """Create a tool plan using deterministic routing, with LLM fallback."""
    memory_context = retrieve_memory(user_request)
    add_short_term_memory("user_requests", user_request)
    request = user_request.lower()

    # Deterministic routing prevents unrelated tools from being selected.
    routes = [
        (("governance", "governance risk", "governance risks", "risk register", "main risks", "identified risks", "ai risks"), [
            ("get_risk_register", "Retrieve the documented AI risk register."),
            ("get_system_profile", "Retrieve the AI system purpose and context."),
            ("get_impact_assessment", "Retrieve the documented stakeholder impact assessment.")]),
        (("impact assessment", "stakeholder impact", "affected stakeholders"), [
            ("get_impact_assessment", "Retrieve the stakeholder impact assessment."),
            ("get_system_profile", "Retrieve the AI system profile.")]),
        (("incident", "incident management", "incident scenario"), [
            ("get_incident_management", "Retrieve documented incident scenarios and responses.")]),
        (("governance control", "lifecycle control", "responsible ai", "ai principles", "management system", "framework assessment", "nist", "oecd", "iso 42001", "audit checklist", "compliance control"), [
            ("get_governance_controls", "Retrieve governance frameworks, principles, and lifecycle controls."),
            ("get_system_profile", "Retrieve the AI system profile and deployment status.")]),
        (("mlflow", "experiment tracking", "mlflow experiment", "model runs", "tracking runs"), [
            ("get_mlflow_experiments", "Retrieve MLflow experiment records."),
            ("get_mlflow_run_details", "Retrieve run metrics, parameters, and tags.")]),
        (("system status", "system health", "component status", "is the system", "availability"), [
            ("get_system_status", "Check the availability of ModelGuard components.")]),
        (("batch performance", "performance across batches", "compare batches"), [
            ("get_batch_performance", "Compare model performance across evaluation batches.")]),
        (("data drift", "feature drift", "drift detection", "drift analysis", "drifted"), [
            ("get_drift_analysis", "Analyze feature distribution drift.")]),
        (("performance", "accuracy", "precision", "recall", "f1", "roc-auc", "pr-auc", "model metrics"), [
            ("get_performance_baseline", "Retrieve the model performance baseline.")]),
        (("which model", "what model", "model used", "model information", "model configuration", "model comparison"), [
            ("get_model_information", "Retrieve model configurations and experiment results.")]),
        (("predict customer", "customer prediction", "predict churn", "churn prediction", "will this customer churn", "churn probability", "probability for this customer", "provided customer", "customer record"), [
            ("predict_customer_churn", "Predict churn for the supplied customer record.")]),
    ]

    selected = []
    selected_names = set()
    for keywords, tools in routes:
        if any(keyword in request for keyword in keywords):
            for name, purpose in tools:
                if name in TOOL_REGISTRY and name not in selected_names:
                    selected.append((name, purpose))
                    selected_names.add(name)

    if selected:
        plan = [
            {"step": index, "tool_name": name, "purpose": purpose}
            for index, (name, purpose) in enumerate(selected, start=1)
        ][:5]
        reasoning_result = {
            "status": "success",
            "reasoning": {
                "objective": "Retrieve evidence relevant to the user's request.",
                "required_information": [purpose for _, purpose in selected]
            },
            "plan": plan
        }
    else:
        if memory_context:
            context_text = "\n".join(
                f"- {item['content']} (Source: {item['source']})"
                for item in memory_context
            )
            augmented_request = (
                f"User request: {user_request}\n\n"
                f"Verified memory context:\n{context_text}\n\n"
                "Use this context only when relevant. Do not treat it as a replacement for current tool results."
            )
        else:
            augmented_request = user_request
        reasoning_result = reason_and_plan(augmented_request)

    add_short_term_memory("execution_plans", reasoning_result)
    return {
        "user_request": user_request,
        "retrieved_memory": memory_context,
        "planning_result": reasoning_result
    }


def execute_tool(tool_name, tool_args=None):

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")

    if tool_args is None:
        tool_args = {}

    if not isinstance(tool_args, dict):
        raise TypeError("Tool arguments must be provided as a dictionary.")

    tool_function = TOOL_REGISTRY[tool_name]

    start_time = time.perf_counter()

    try:
        # Validate arguments against the function signature
        signature = inspect.signature(tool_function)
        signature.bind(**tool_args)

        # Execute the tool with the provided arguments
        result = tool_function(**tool_args)

        execution_time = time.perf_counter() - start_time

        return {
            "tool_name": tool_name,
            "status": "success",
            "execution_time": round(execution_time, 4),
            "result": result
        }

    except Exception as error:
        return {
            "tool_name": tool_name,
            "status": "error",
            "execution_time": round(
                time.perf_counter() - start_time, 4
            ),
            "error": str(error)
        }


# Customer churn model
MODEL_PATH = ARTIFACTS_DIR / "final_xgboost_model.joblib"
xgb_model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
EXPECTED_FEATURES = list(xgb_model.feature_names_in_) if xgb_model is not None else []
CHURN_THRESHOLD = 0.56



def predict_customer_churn(customer_data):
    """
    Predict customer churn using the trained XGBoost pipeline.
    """

    if xgb_model is None:
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    # Convert input to DataFrame
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])

    # Validate input type
    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("Input must be a dictionary or pandas DataFrame.")

    # Validate required features
    missing_features = [
        feature for feature in EXPECTED_FEATURES
        if feature not in customer_data.columns
    ]

    extra_features = [
        feature for feature in customer_data.columns
        if feature not in EXPECTED_FEATURES
    ]

    if missing_features or extra_features:
        raise ValueError({
            "missing_features": missing_features,
            "unexpected_features": extra_features
        })

    # Preserve the original feature order
    customer_data = customer_data[EXPECTED_FEATURES].copy()

    # Generate model score
    churn_score = float(
        xgb_model.predict_proba(customer_data)[0][1]
    )

    # Apply the selected threshold
    prediction = int(churn_score >= CHURN_THRESHOLD)

    # These are descriptive input signals, not causal explanations of the model.
    row = customer_data.iloc[0].to_dict()
    profile_signals = {
        "tenure_months": int(row["tenure"]) if pd.notna(row.get("tenure")) else None,
        "contract_type": str(row.get("Contract")),
        "internet_service": str(row.get("InternetService")),
        "monthly_charges": float(row["MonthlyCharges"]) if pd.notna(row.get("MonthlyCharges")) else None,
    }

    return {
        "prediction": prediction,
        "prediction_label": (
            "Churn" if prediction == 1 else "No Churn"
        ),
        "churn_score": round(churn_score, 4),
        "decision_threshold": CHURN_THRESHOLD,
        "profile_signals": profile_signals,
        "explanation_limit": (
            "No local feature-attribution method (such as SHAP) is implemented; "
            "profile signals are descriptive and must not be interpreted as causal drivers."
        )
    }


TOOL_REGISTRY["predict_customer_churn"] = predict_customer_churn
TOOL_DESCRIPTIONS["predict_customer_churn"] = (
    "Predict customer churn using the trained XGBoost model. "
    "Requires structured customer data containing all 19 features."
)
PLANNING_TOOLS["predict_customer_churn"] = TOOL_DESCRIPTIONS["predict_customer_churn"]




class FinalResponse(BaseModel):
    status: Literal["success", "partial", "error"]
    answer: str = Field(min_length=1)
    key_findings: List[str]
    evidence: List[str]
    limitations: List[str]


class ReflectionOutput(BaseModel):
    status: Literal["approved", "needs_correction", "rejected"]
    evidence_supported: bool
    identified_issues: List[str]
    correction_required: bool



def run_react_agent(user_request, execution_plan, tool_inputs=None):

    observations = []
    execution_log = []
    used_tools = []

    if tool_inputs is None:
        tool_inputs = {}

    if not isinstance(tool_inputs, dict):
        return {
            "status": "error",
            "error": "Tool inputs must be provided as a dictionary.",
            "iterations": 0,
            "used_tools": [],
            "observations": [],
            "execution_log": [],
            "final_response": None
        }

    # 1. Validate the execution plan
    if not isinstance(execution_plan, list) or not execution_plan:
        return {
            "status": "error",
            "error": "The execution plan is empty or invalid.",
            "iterations": 0,
            "used_tools": [],
            "observations": [],
            "execution_log": [],
            "final_response": None
        }

    planned_tools = []

    for step in execution_plan:

        if not isinstance(step, dict):
            return {
                "status": "error",
                "error": "Invalid execution plan step."
            }

        tool_name = step.get("tool_name")

        if tool_name not in TOOL_REGISTRY:
            return {
                "status": "error",
                "error": f"Unknown tool in plan: {tool_name}"
            }

        if tool_name in planned_tools:
            return {
                "status": "error",
                "error": f"Repeated tool in plan: {tool_name}"
            }

        planned_tools.append(tool_name)

    # 2. Execute tools with their required inputs
    for iteration, step in enumerate(execution_plan, start=1):

        tool_name = step["tool_name"]

        tool_args = tool_inputs.get(tool_name, {})

        tool_result = execute_tool(
            tool_name,
            tool_args=tool_args
        )

        success = tool_result.get("status") == "success"

        used_tools.append(tool_name)

        execution_log.append({
            "iteration": iteration,
            "tool_name": tool_name,
            "purpose": step.get("purpose", ""),
            "execution_status": tool_result.get("status"),
            "execution_time": tool_result.get("execution_time"),
            "error": tool_result.get("error")
        })

        if not success:
            return {
                "status": "error",
                "error": tool_result.get("error"),
                "iterations": iteration,
                "used_tools": used_tools,
                "observations": observations,
                "execution_log": execution_log,
                "final_response": None
            }

        observations.append({
            "iteration": iteration,
            "tool_name": tool_name,
            "status": "success",
            "result": tool_result.get("result")
        })

    # 3. Collect evidence from observations
    evidence = {
        item["tool_name"]: item["result"]
        for item in observations
    }

    # Store generated reports
    response_sections = []

    # 4. Generate a prediction response when prediction evidence exists
    prediction = evidence.get("predict_customer_churn")

    if prediction is not None:

        final_response = f"""
### Customer Churn Prediction

**Prediction:** {prediction["prediction_label"]}

**Churn Score:** {prediction["churn_score"]:.4f}

**Decision Threshold:** {prediction["decision_threshold"]:.2f}

#### Interpretation

The XGBoost model classified the customer as
{"likely to churn" if prediction["prediction"] == 1 else "not likely to churn"}
based on the selected decision threshold.

The churn score represents the model's estimated
probability score and is not necessarily calibrated.

#### Limitation

This prediction is based on the supplied customer
record and the trained model. It should support,
not replace, business judgment.
"""

        response_sections.append(final_response)

    # 5. Generate a monitoring response when performance
    # and drift evidence are both available
    performance = evidence.get("get_performance_baseline")
    drift = evidence.get("get_drift_analysis")

    if performance is not None and drift is not None:

        metrics = performance.get("metrics")

        required_metrics = [
            "Accuracy", "Precision", "Recall",
            "F1-Score", "ROC-AUC", "PR-AUC"
        ]

        if not isinstance(metrics, dict):
            return {
                "status": "incomplete",
                "error": "Performance metrics are missing or invalid.",
                "iterations": len(execution_log),
                "used_tools": used_tools,
                "observations": observations,
                "execution_log": execution_log,
                "evidence": evidence,
                "final_response": None
            }

        missing_metrics = [
            metric for metric in required_metrics
            if metric not in metrics
        ]

        if missing_metrics:
            return {
                "status": "incomplete",
                "error": f"Missing metrics: {missing_metrics}",
                "iterations": len(execution_log),
                "used_tools": used_tools,
                "observations": observations,
                "execution_log": execution_log,
                "evidence": evidence,
                "final_response": None
            }

        for metric in required_metrics:

            value = metrics[metric]

            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 <= value <= 1
            ):
                return {
                    "status": "incomplete",
                    "error": f"Invalid metric value: {metric}",
                    "iterations": len(execution_log),
                    "used_tools": used_tools,
                    "observations": observations,
                    "execution_log": execution_log,
                    "evidence": evidence,
                    "final_response": None
                }

        number_of_features = drift.get("number_of_features")
        drifted_features = drift.get("drifted_features")

        if (
            not isinstance(number_of_features, int)
            or isinstance(number_of_features, bool)
            or number_of_features < 1
            or not isinstance(drifted_features, list)
            or len(drifted_features) > number_of_features
        ):
            return {
                "status": "incomplete",
                "error": "Drift analysis results are invalid.",
                "iterations": len(execution_log),
                "used_tools": used_tools,
                "observations": observations,
                "execution_log": execution_log,
                "evidence": evidence,
                "final_response": None
            }

        metric_text = "\n".join(
            f"- **{name}:** {metrics[name]:.4f}"
            for name in required_metrics
        )

        drift_text = (
            "Statistically significant drift was detected in: "
            + ", ".join(drifted_features) + "."
            if drifted_features
            else (
                f"No statistically significant drift was detected "
                f"across the {number_of_features} analyzed features."
            )
        )

        monitoring_response = f"""
### Churn Prediction Model Assessment

**Model:** {performance.get("model", "Not specified")}

**Evaluation Dataset:** {performance.get("evaluation_dataset", "Not specified")}

**Classification Threshold:** {performance.get("threshold", "Not specified")}

#### Baseline Performance Metrics

{metric_text}

#### Drift Analysis

**Analysis Type:** {drift.get("analysis_type", "Not specified")}

**Number of Analyzed Features:** {number_of_features}

**Number of Drifted Features:** {len(drifted_features)}

{drift_text}

#### Limitations

The drift analysis compares reference and comparison
datasets. It does not represent live production monitoring.
"""

        response_sections.append(monitoring_response)

    # 6. Combine available reports without inventing evidence
    if response_sections:

        final_response = "\n\n".join(response_sections)

    else:

        final_response = (
            "The requested tools were executed successfully. "
            "However, the available evidence is not sufficient "
            "to generate a specialized report."
        )

    return {
        "status": "completed",
        "iterations": len(execution_log),
        "used_tools": used_tools,
        "observations": observations,
        "execution_log": execution_log,
        "evidence": evidence,
        "final_response": final_response
    }

def reflect_on_response(response_data, react_result):
    """Validate structured claims and evidence coverage across all tool domains."""
    issues = []
    supported_claims = 0
    checked_claims = 0
    evidence = react_result.get("evidence", {})

    if react_result.get("status") != "completed":
        issues.append("Agent execution was not completed.")

    def get_claim(prefix):
        for item in response_data.key_findings:
            if item.lower().startswith(prefix.lower() + ":"):
                return item.split(":", 1)[1].strip()
        return None

    def validate_text_claim(prefix, expected):
        nonlocal checked_claims, supported_claims
        reported = get_claim(prefix)
        if reported is not None:
            checked_claims += 1
            if reported.lower() == str(expected).lower():
                supported_claims += 1
            else:
                issues.append(f"{prefix} does not match evidence.")

    def validate_numeric_claim(prefix, expected, tolerance=0.0001):
        nonlocal checked_claims, supported_claims
        reported = get_claim(prefix)
        if reported is not None:
            checked_claims += 1
            try:
                if abs(float(reported) - float(expected)) <= tolerance:
                    supported_claims += 1
                else:
                    issues.append(f"{prefix} does not match evidence.")
            except (ValueError, TypeError):
                issues.append(f"{prefix} has an invalid numerical value.")

    prediction = evidence.get("predict_customer_churn")
    if isinstance(prediction, dict):
        validate_text_claim("Prediction", prediction.get("prediction_label"))
        validate_numeric_claim("Churn Score", prediction.get("churn_score"))
        validate_numeric_claim("Decision Threshold", prediction.get("decision_threshold"))

    performance = evidence.get("get_performance_baseline")
    if isinstance(performance, dict):
        metrics = performance.get("metrics", {})
        for label in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]:
            if label in metrics:
                validate_numeric_claim(label, metrics[label])

    drift = evidence.get("get_drift_analysis")
    if isinstance(drift, dict):
        if drift.get("number_of_features") is not None:
            validate_numeric_claim("Analyzed Features", drift["number_of_features"], tolerance=0)
        if isinstance(drift.get("drifted_features"), list):
            validate_numeric_claim("Drifted Features", len(drift["drifted_features"]), tolerance=0)

    # Verify that every successfully executed tool is represented in the response metadata.
    expected_sources = sorted(evidence.keys())
    validate_numeric_claim("Evidence Source Count", len(expected_sources), tolerance=0)
    validate_text_claim("Evidence Sources", ", ".join(expected_sources))

    if evidence and not response_data.evidence:
        issues.append("Tool evidence exists but no evidence sources were listed in the response.")

    if not evidence:
        issues.append("No tool evidence was available for reflection.")

    evidence_supported = checked_claims > 0 and supported_claims == checked_claims and not issues
    correction_required = bool(issues)

    return ReflectionOutput(
        status="needs_correction" if correction_required else "approved",
        evidence_supported=evidence_supported,
        identified_issues=issues,
        correction_required=correction_required
    )


def _modelguard_render_value(value, indent=0):
    """Render tool evidence as readable Markdown without inventing content."""
    prefix = "  " * indent
    if isinstance(value, dict):
        if not value:
            return prefix + "- No records returned."
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}- **{key.replace('_', ' ').title()}:**")
                lines.append(_modelguard_render_value(item, indent + 1))
            else:
                lines.append(f"{prefix}- **{key.replace('_', ' ').title()}:** {item}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return prefix + "- No records returned."
        lines = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}- **Record {index}:**")
                lines.append(_modelguard_render_value(item, indent + 1))
            else:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines)
    return prefix + str(value)


def build_structured_response(react_result):
    """Build a grounded report for every registered ModelGuard tool domain."""
    evidence = react_result.get("evidence", {})
    key_findings = []
    evidence_sources = []
    limitations = []
    sections = []

    if not isinstance(evidence, dict):
        evidence = {}

    tool_titles = {
        "predict_customer_churn": "Customer Churn Prediction",
        "get_model_information": "Model Information and Experiment Results",
        "get_performance_baseline": "Model Performance Baseline",
        "get_batch_performance": "Batch Performance Comparison",
        "get_drift_analysis": "Feature Drift Analysis",
        "get_system_profile": "AI System Profile",
        "get_risk_register": "AI Governance Risk Register",
        "get_impact_assessment": "Stakeholder Impact Assessment",
        "get_incident_management": "Incident Management",
        "get_governance_controls": "AI Governance Controls and Frameworks",
        "get_mlflow_experiments": "MLflow Experiments",
        "get_mlflow_run_details": "MLflow Run Details",
        "get_system_status": "ModelGuard System Status",
    }

    # Preserve the domain-specific findings already used by existing checks.
    prediction = evidence.get("predict_customer_churn")
    if isinstance(prediction, dict):
        key_findings.extend([
            f"Prediction: {prediction.get('prediction_label', 'Not available')}",
            f"Churn Score: {prediction.get('churn_score', 'Not available')}",
            f"Decision Threshold: {prediction.get('decision_threshold', 'Not available')}"
        ])
        limitations.append("The churn score is a model score and is not necessarily calibrated.")

    performance = evidence.get("get_performance_baseline")
    if isinstance(performance, dict):
        metrics = performance.get("metrics", {})
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]:
            if isinstance(metrics, dict) and metric in metrics:
                key_findings.append(f"{metric}: {metrics[metric]}")

    drift = evidence.get("get_drift_analysis")
    if isinstance(drift, dict):
        number_features = drift.get("number_of_features")
        drifted = drift.get("drifted_features", [])
        if number_features is not None:
            key_findings.append(f"Analyzed Features: {number_features}")
        if isinstance(drifted, list):
            key_findings.append(f"Drifted Features: {len(drifted)}")
        limitations.append("Drift analysis compares stored reference and comparison datasets; it is not live production monitoring.")

    # Domain-specific summaries for governance, MLflow, batches, and system status.
    risk_data = evidence.get("get_risk_register")
    if risk_data is not None:
        risk_records = risk_data if isinstance(risk_data, list) else None
        if isinstance(risk_data, dict):
            for value in risk_data.values():
                if isinstance(value, list):
                    risk_records = value
                    break
        if risk_records is not None:
            key_findings.append(f"Governance Risks Recorded: {len(risk_records)}")

    impact_data = evidence.get("get_impact_assessment")
    if isinstance(impact_data, list):
        key_findings.append(f"Impact Assessment Records: {len(impact_data)}")

    incident_data = evidence.get("get_incident_management")
    if isinstance(incident_data, list):
        key_findings.append(f"Incident Scenarios Recorded: {len(incident_data)}")

    experiments = evidence.get("get_mlflow_experiments")
    if isinstance(experiments, list):
        key_findings.append(f"MLflow Experiments Found: {len(experiments)}")

    runs = evidence.get("get_mlflow_run_details")
    if isinstance(runs, list):
        key_findings.append(f"MLflow Runs Found: {len(runs)}")
    elif isinstance(runs, dict) and "run_id" in runs:
        key_findings.append("MLflow Run Details: 1 run")

    batch_data = evidence.get("get_batch_performance")
    if isinstance(batch_data, list):
        key_findings.append(f"Performance Batches Analyzed: {len(batch_data)}")

    system_status = evidence.get("get_system_status")
    if isinstance(system_status, dict):
        if "available" in system_status:
            key_findings.append(f"Available System Components: {system_status['available']}")
        if "missing" in system_status:
            key_findings.append(f"Missing System Components: {system_status['missing']}")

    for tool_name, result in evidence.items():
        title = tool_titles.get(tool_name, tool_name.replace("_", " ").title())
        sections.append(f"### {title}\n\n{_modelguard_render_value(result)}")
        evidence_sources.append(tool_name)

    # Evidence metadata is checked by Reflection for every tool domain.
    key_findings.append(f"Evidence Source Count: {len(evidence_sources)}")
    key_findings.append("Evidence Sources: " + ", ".join(sorted(evidence_sources)))

    if react_result.get("status") != "completed":
        limitations.append("The agent execution did not complete successfully.")

    # Deterministic analytical synthesis. Calculations are performed in Python;
    # statements are limited to fields present in the retrieved project evidence.
    analysis_sections = []

    if isinstance(prediction, dict):
        score = prediction.get("churn_score")
        threshold = prediction.get("decision_threshold")
        profile = prediction.get("profile_signals", {})
        parts = [
            f"Model classification: **{prediction.get('prediction_label', 'Unknown')}**.",
            f"Estimated model score: **{score}**; decision threshold: **{threshold}**."
        ]
        tenure = profile.get("tenure_months")
        contract = profile.get("contract_type")
        if tenure is not None:
            parts.append(f"Observed customer tenure: **{tenure} months**.")
        if contract:
            parts.append(f"Observed contract type: **{contract}**.")
        if tenure is not None and tenure < 12:
            parts.append("Descriptive retention consideration: the customer is in the first year of tenure; consider reviewing onboarding and early-life customer experience.")
        if contract and contract.lower() == "month-to-month":
            parts.append("Descriptive retention consideration: review whether an appropriate longer-term plan or contract incentive is suitable; do not assume this feature caused the prediction.")
        parts.append("Feature-attribution limitation: this runtime does not calculate local SHAP or other per-customer contribution values. The profile signals above are not an explanation of model causality.")
        parts.append("The score is not known to be calibrated; use it as a model score, not a guaranteed probability.")
        analysis_sections.append("### Prediction interpretation and decision support\n\n" + "\n\n".join(parts))

    model_info = evidence.get("get_model_information")
    if isinstance(model_info, dict):
        xgb = model_info.get("model_results") or {}
        lr = model_info.get("baseline_model") or {}
        exp = model_info.get("experimental_model") or {}
        comparison_lines = []
        xgb_test = xgb.get("test_results", {}) if isinstance(xgb, dict) else {}
        lr_tests = lr.get("test_results", []) if isinstance(lr, dict) else []
        lr_at_06 = next((r for r in lr_tests if isinstance(r, dict) and float(r.get("Threshold", -1)) == 0.6), None)
        if xgb_test and lr_at_06:
            comparison_lines.append(
                f"- Held-out test F1: XGBoost **{xgb_test.get('F1-score')}** vs Logistic Regression (threshold 0.60) **{lr_at_06.get('F1-score')}**."
            )
            comparison_lines.append(
                f"- Held-out test recall: XGBoost **{xgb_test.get('Recall')}** vs Logistic Regression (threshold 0.60) **{lr_at_06.get('Recall')}**."
            )
            comparison_lines.append("These are comparable held-out test results from the recorded model-results file.")
        xgb_cv = xgb.get("cross_validation", {}) if isinstance(xgb, dict) else {}
        exp_cv = exp.get("cross_validation", {}) if isinstance(exp, dict) else {}
        if xgb_cv and exp_cv:
            xgb_f1 = xgb_cv.get("f1")
            undersampling_f1 = (exp_cv.get("F1-score") or {}).get("Mean Score")
            if xgb_f1 is not None and undersampling_f1 is not None:
                comparison_lines.append(
                    f"- Cross-validation F1: XGBoost **{xgb_f1:.4f}** vs undersampling experiment **{undersampling_f1:.4f}**."
                )
                comparison_lines.append("The undersampling result is cross-validation only; it must not be ranked directly against held-out test metrics.")
        if comparison_lines:
            analysis_sections.append("### Model comparison\n\n" + "\n".join(comparison_lines))

    batches = evidence.get("get_batch_performance")
    if isinstance(batches, list) and len(batches) >= 3:
        base = next((r for r in batches if str(r.get("Batch", "")).lower() == "initial baseline"), None)
        b1 = next((r for r in batches if str(r.get("Batch", "")).lower() == "batch 1"), None)
        b2 = next((r for r in batches if str(r.get("Batch", "")).lower() == "batch 2"), None)
        if base and b1 and b2:
            metrics_to_compare = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
            lines = []
            for metric in metrics_to_compare:
                if metric in b1 and metric in b2:
                    delta = float(b2[metric]) - float(b1[metric])
                    lines.append(f"- {metric}: Batch 2 minus Batch 1 = **{delta:+.4f}**.")
            lines.append("These batches are random partitions of the held-out data, not chronological production windows; differences do not establish temporal degradation.")
            analysis_sections.append("### Batch comparison\n\n" + "\n".join(lines))

    incidents = evidence.get("get_incident_management")
    if isinstance(incidents, list) and incidents:
        groups = {}
        for item in incidents:
            priority = str(item.get("Priority", "Unspecified")).title()
            groups.setdefault(priority, []).append(item)
        order = ["Critical", "High", "Medium", "Low", "Unspecified"]
        lines = ["The register contains documented scenarios, not evidence that incidents have occurred."]
        for priority in order:
            if priority in groups:
                labels = [f"{x.get('Incident_ID', '')} – {x.get('Incident', '')}" for x in groups[priority]]
                lines.append(f"- **{priority}:** " + "; ".join(labels))
        analysis_sections.append("### Incident priority overview\n\n" + "\n".join(lines))

    controls = evidence.get("get_governance_controls")
    if isinstance(controls, dict):
        lifecycle = controls.get("lifecycle_controls")
        if isinstance(lifecycle, list):
            counts = {}
            for item in lifecycle:
                status = str(item.get("Implementation_Status", "Unspecified"))
                counts[status] = counts.get(status, 0) + 1
            lines = [f"- **{status}:** {count}" for status, count in counts.items()]
            gaps = [
                f"{item.get('Lifecycle_Stage')}: {item.get('Implementation_Status')}"
                for item in lifecycle
                if str(item.get("Implementation_Status", "")).lower() not in ("completed", "prototype completed")
            ]
            if gaps:
                lines.append("Recorded incomplete/planned stages: " + "; ".join(gaps) + ".")
            lines.append("A status of Prototype Completed indicates a prototype, not a live operational control.")
            analysis_sections.append("### Lifecycle implementation summary\n\n" + "\n".join(lines))

    status_data = evidence.get("get_system_status")
    if isinstance(status_data, dict):
        analysis_sections.append(
            "### Operational status limitation\n\n"
            f"The availability check reports {status_data.get('available', 0)} present files/components and "
            f"{status_data.get('missing', 0)} missing items. This check tests filesystem presence only; "
            "it does not establish that each component is operational, correctly configured, secure, or production-ready."
        )

    answer = "\n\n".join(sections + analysis_sections) if sections or analysis_sections else (
        "No tool evidence was returned. The request cannot be answered from the available project records."
    )

    return FinalResponse(
        status="success" if react_result.get("status") == "completed" and evidence else "partial",
        answer=answer,
        key_findings=key_findings,
        evidence=evidence_sources,
        limitations=limitations
    )


def evaluate_agent_response(
    response,
    evidence,
    required_evidence=None
):
    issues = []

    # Combine response text for claim checking
    response_text = " ".join(
        [response.answer]
        + response.key_findings
        + response.limitations
    ).lower()

    # Check deployment claims
    actual_status = str(
        evidence.get("deployment_status", "Unknown")
    ).lower()

    if (
        "deployed in production" in response_text
        and actual_status != "deployed"
    ):
        issues.append("Unsupported deployment claim")

    # Check automatic retraining claims
    if "automatically retrains" in response_text:
        issues.append("Unsupported automatic retraining claim")

    # Check required evidence availability
    if required_evidence:
        for item in required_evidence:
            if item not in evidence:
                uncertainty_phrases = [
                    "insufficient evidence",
                    "cannot determine",
                    "not available",
                    "unknown"
                ]

                if not any(
                    phrase in response_text
                    for phrase in uncertainty_phrases
                ):
                    issues.append(
                        f"Insufficient evidence for: {item}"
                    )

    return {
        "passed": len(issues) == 0,
        "issues": issues
    }

def release_agent_response(
    response,
    react_result,
    evidence,
    required_evidence=None
):
    issues = []

    # 1. Check factual claims and evidence availability
    factual_check = evaluate_agent_response(
        response,
        evidence,
        required_evidence
    )
    issues.extend(factual_check["issues"])

    # 2. Validate numerical claims against tool results
    reflection = reflect_on_response(
        response,
        react_result
    )
    issues.extend(reflection.identified_issues)

    # 3. Remove duplicate issues
    issues = list(dict.fromkeys(issues))

    # 4. Block invalid responses
    if issues:
        return {
            "released": False,
            "response": None,
            "message": (
                "The response was blocked because it failed "
                "the validation checks."
            ),
            "issues": issues
        }

    # 5. Release only validated responses
    return {
        "released": True,
        "response": response.model_dump(),
        "message": "The response passed validation and is approved for delivery.",
        "issues": []
    }

def log_agent_execution(
    user_request,
    planning_result,
    react_result,
    reflection_result,
    final_response,
    validation_result=None
):
    # Extract planned tools
    plan_data = planning_result.get("plan", [])

    tool_names = []
    for step in plan_data:
        if isinstance(step, dict):
            tool_names.append(step.get("tool_name"))
        elif hasattr(step, "model_dump"):
            tool_names.append(step.model_dump().get("tool_name"))

    # Extract tool execution records
    execution_records = [
        {
            "tool_name": item.get("tool_name"),
            "status": item.get("execution_status", item.get("status")),
            "execution_time": item.get("execution_time")
        }
        for item in react_result.get("execution_log", [])
    ]

    # Convert reflection result
    reflection_data = (
        reflection_result.model_dump()
        if hasattr(reflection_result, "model_dump")
        else reflection_result
    )

    # Handle approved and blocked responses
    if final_response is None:
        response_data = None
    elif hasattr(final_response, "model_dump"):
        response_data = final_response.model_dump()
    else:
        response_data = final_response

    # Extract validation information
    validation_data = validation_result or {}

    if response_data is not None:
        response_status = response_data.get("status", "unknown")
    elif validation_data.get("released") is False:
        response_status = "blocked"
    else:
        response_status = "not_generated"

    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "agent": AGENT_NAME,
        "model": MODEL_NAME,
        "user_request": user_request,
        "planned_tools": tool_names,
        "execution_status": react_result.get("status"),
        "iterations": react_result.get("iterations"),
        "used_tools": react_result.get("used_tools", []),
        "tool_execution": execution_records,
        "reflection": reflection_data,
        "validation": {
            "released": validation_data.get("released"),
            "issues": validation_data.get("issues", [])
        },
        "final_response_status": response_status
    }

    with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record

def run_modelguard_agent(user_request, tool_inputs=None):

    planning_result = None
    react_result = None
    reflection_result = None
    structured_response = None
    release_result = None

    try:
        # 1. Memory and Planning
        planning_result = reason_and_plan_with_memory(user_request)

        planning_output = planning_result["planning_result"]
        execution_plan = planning_output["plan"]

        if planning_output["status"] != "success":
            raise ValueError("Planning was not successful.")

        # 2. Validate planned tools
        planned_tools = [step["tool_name"] for step in execution_plan]

        if not planned_tools:
            raise ValueError("The execution plan is empty.")

        unknown_tools = [
            name for name in planned_tools
            if name not in TOOL_REGISTRY
        ]

        if unknown_tools:
            raise ValueError(f"Unknown tools: {unknown_tools}")

        # 3. ReAct Execution
        react_result = run_react_agent(
            user_request=user_request,
            execution_plan=execution_plan,
            tool_inputs=tool_inputs
        )

        # 4. Structured Response
        structured_response = build_structured_response(react_result)

        # 5. Reflection
        reflection_result = reflect_on_response(
            structured_response,
            react_result
        )

        # 6. Response Validation Gate
        release_result = release_agent_response(
            response=structured_response,
            react_result=react_result,
            evidence=react_result.get("evidence", {})
        )

        # 7. Audit Logging
        audit_record = log_agent_execution(
            user_request=user_request,
            planning_result=planning_result["planning_result"],
            react_result=react_result,
            reflection_result=reflection_result,
            final_response=release_result["response"],
            validation_result=release_result
        )

        return {
            "status": "success" if release_result["released"] else "blocked",
            "response": release_result["response"],
            "reflection": reflection_result.model_dump(),
            "validation": release_result,
            "audit_record": audit_record
        }

    except Exception as error:
        error_record = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "agent": AGENT_NAME,
            "model": MODEL_NAME,
            "user_request": user_request,
            "planned_tools": (
                [step.get("tool_name") for step in execution_plan]
                if "execution_plan" in locals() and isinstance(execution_plan, list)
                else []
            ),
            "execution_status": "error",
            "error": str(error),
            "final_response_status": "error"
        }
        try:
            with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as file:
                file.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return {
            "status": "error",
            "response": None,
            "error": str(error),
            "audit_record": error_record
        }
