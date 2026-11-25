import json
import requests
import streamlit as st

from modelguard_agent import (
    AGENT_NAME,
    MODEL_NAME,
    OLLAMA_URL,
    EXPECTED_FEATURES,
    run_modelguard_agent,
)

st.set_page_config(
    page_title="ModelGuard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ModelGuard")
st.caption("AI Lifecycle Management & Governance Agent")

with st.sidebar:
    st.subheader("System status")
    try:
        check = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        check.raise_for_status()
        st.success("Ollama API: Connected")
        models = [item.get("name", "") for item in check.json().get("models", [])]
        if MODEL_NAME in models or any(name.startswith(MODEL_NAME.split(":")[0] + ":") for name in models):
            st.success(f"Local model available: {MODEL_NAME}")
        else:
            st.warning(f"Could not confirm model tag: {MODEL_NAME}")
    except requests.RequestException:
        st.error("Ollama: Not connected")

    st.write(f"**Agent:** {AGENT_NAME}")
    st.write(f"**Local model:** {MODEL_NAME}")
    st.caption(
        "ModelGuard is a local research prototype. "
        "Its outputs support analysis and do not replace human review."
    )

st.markdown(
    "Ask about model performance, feature drift, governance risks, "
    "MLflow experiments, system status, or customer churn prediction."
)

with st.expander("Customer prediction input (optional)", expanded=False):
    st.write(
        "Paste one customer record as a JSON object containing exactly "
        "the model's 19 input features. Do not include customerID or Churn."
    )
    customer_json = st.text_area(
        "Customer data (JSON)",
        height=220,
        placeholder='{"gender": "Female", "SeniorCitizen": 0, "...": "..."}',
        key="customer_json",
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("details"):
            with st.expander("Execution details"):
                st.json(message["details"])

prompt = st.chat_input("Ask ModelGuard a question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tool_inputs = None
    request_text = prompt.lower()
    prediction_intent = any(
        phrase in request_text
        for phrase in (
            "predict", "prediction", "churn probability",
            "will this customer", "customer churn"
        )
    )

    if customer_json.strip():
        try:
            parsed_customer = json.loads(customer_json)
            if not isinstance(parsed_customer, dict):
                raise ValueError("Customer input must be a JSON object.")

            if prediction_intent:
                missing = [name for name in EXPECTED_FEATURES if name not in parsed_customer]
                extra = [name for name in parsed_customer if name not in EXPECTED_FEATURES]
                if missing or extra:
                    details = []
                    if missing:
                        details.append("Missing features: " + ", ".join(missing))
                    if extra:
                        details.append("Unexpected fields: " + ", ".join(extra))
                    raise ValueError(" | ".join(details))

                tool_inputs = {"predict_customer_churn": {"customer_data": parsed_customer}}
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Customer JSON validation failed: {exc}")
            st.stop()

    with st.chat_message("assistant"):
        with st.spinner("ModelGuard is planning and analyzing..."):
            result = run_modelguard_agent(
                user_request=prompt,
                tool_inputs=tool_inputs,
            )

        status = result.get("status", "unknown")
        response = result.get("response")

        if status == "success" and response:
            st.markdown(response.get("answer", "No answer returned."))

            findings = response.get("key_findings", [])
            if findings:
                with st.expander("Key findings", expanded=True):
                    for finding in findings:
                        st.write(f"- {finding}")

            evidence = response.get("evidence", [])
            if evidence:
                with st.expander("Evidence sources"):
                    for item in evidence:
                        st.write(f"- {item}")

            limitations = response.get("limitations", [])
            if limitations:
                with st.expander("Limitations"):
                    for item in limitations:
                        st.write(f"- {item}")

            details = {
                "reflection": result.get("reflection"),
                "validation": result.get("validation"),
                "audit_record": result.get("audit_record"),
            }
            with st.expander("Execution details"):
                st.json(details)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("answer", ""),
                "details": details,
            })

        elif status == "blocked":
            st.warning("The response was blocked by the validation gate.")
            # Show the full audit trail, including planned tools and tool errors.
            st.json(result)
            st.session_state.messages.append({
                "role": "assistant",
                "content": "The response was blocked by the validation gate.",
                "details": result,
            })

        else:
            error_text = result.get("error", "An unknown error occurred.")
            st.error(f"ModelGuard could not complete the request: {error_text}")
            st.json(result)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"ModelGuard error: {error_text}",
                "details": result,
            })
