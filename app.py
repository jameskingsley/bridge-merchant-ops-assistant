import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.assistant import MerchantOpsAssistant


def get_client_and_config() -> tuple[OpenAI, str, str, Path]:
    load_dotenv(override=True)
    api_key = os.getenv("WEKOYA_API_KEY", "").strip()
    if not api_key or api_key == "wk_live_your_key_here":
        raise RuntimeError("Set a valid WEKOYA_API_KEY in .env before running Streamlit.")

    base_url = os.getenv("WEKOYA_BASE_URL", "https://wekoya-api.wekoya.tech/v1").strip()
    default_chat = os.getenv("CHAT_MODEL", "auto").strip()
    embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2").strip()
    log_path = Path(os.getenv("COST_LOG_PATH", "logs/request_costs.jsonl").strip())
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, default_chat, embedding_model, log_path


def init_session(chat_model: str, embedding_model: str, log_path: Path, client: OpenAI) -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = "merchant-ops-demo-01"
    if "active_chat_model" not in st.session_state:
        st.session_state.active_chat_model = chat_model
    if "assistant" not in st.session_state or st.session_state.active_chat_model != chat_model:
        st.session_state.active_chat_model = chat_model
        st.session_state.assistant = MerchantOpsAssistant(
            client=client,
            chat_model=chat_model,
            embedding_model=embedding_model,
            cost_log_path=str(log_path),
            session_id=st.session_state.session_id,
        )
    if "history" not in st.session_state:
        st.session_state.history = []


def handle_prompt(prompt: str, log_path: Path) -> None:
    assistant: MerchantOpsAssistant = st.session_state.assistant
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Checking policies, catalog, and tools via Bridge..."):
        outcome = assistant.ask(prompt)

    st.session_state.history.append({
        "user": prompt,
        "answer": outcome["answer"],
        "turn_cost_ngn": outcome["turn_cost_ngn"],
        "tool_events": outcome["tool_events"],
        "last_log": outcome["last_log"],
    })

    with st.chat_message("assistant"):
        st.write(outcome["answer"])
        last_log = outcome.get("last_log") or {}
        st.caption(
            f"Turn cost: NGN {outcome['turn_cost_ngn']:.6f} | "
            f"Prompt tokens: {last_log.get('prompt_tokens', 0)} | "
            f"Cached tokens: {last_log.get('cached_tokens', 0)}"
        )
        if outcome.get("tool_events"):
            with st.expander("Tool Calls"):
                st.json(outcome["tool_events"])


def main() -> None:
    st.set_page_config(
        page_title="NaijaMerchant Ops Assistant",
        page_icon="NG",
        layout="wide",
    )
    st.title("NaijaMerchant Ops Assistant")
    st.caption(
        "Built on Bridge (https://wekoya-api.wekoya.tech/v1) with gemini-embedding-2, "
        "tool calling, session prompt caching (prompt_cache_key + X-Session-Id), and per-request NGN cost logs."
    )

    try:
        client, default_chat, embedding_model, log_path = get_client_and_config()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Configuration")
        selected_model = st.text_input("Chat Model", value=default_chat, help="Use 'auto' or a model ID from GET /v1/models")
        session_input = st.text_input(
            "Session ID (Cache Key)",
            value=st.session_state.get("session_id", "merchant-ops-demo-01"),
        )

        if st.button("Apply / New Session"):
            st.session_state.session_id = session_input.strip() or "merchant-ops-demo-01"
            st.session_state.history = []
            st.session_state.assistant = MerchantOpsAssistant(
                client=client,
                chat_model=selected_model.strip() or "auto",
                embedding_model=embedding_model,
                cost_log_path=str(log_path),
                session_id=st.session_state.session_id,
            )
            st.rerun()

        init_session(selected_model.strip() or "auto", embedding_model, log_path, client)

        st.divider()
        st.header("Naira Cost Telemetry")
        if log_path.exists():
            rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            total_ngn = sum(float(r.get("cost_ngn", 0.0)) for r in rows)
            st.metric("Total Logged Spend", f"NGN {total_ngn:.4f}")
            st.dataframe(rows[-10:], use_container_width=True)
        else:
            st.info("No cost log entries yet.")

    st.markdown("**Sample Merchant Queries:**")
    c1, c2, c3 = st.columns(3)
    queued_prompt: str | None = None
    if c1.button("Check ORD-4022 & transfer/delivery rules"):
        queued_prompt = "Check order ORD-4022 and explain what policy applies to its payment status and interstate delivery."
    if c2.button("Check SKU-PKG-012 stock & reorder rule"):
        queued_prompt = "Check stock for SKU-PKG-012 and tell me what our inventory policy says about reordering."
    if c3.button("7-day return & refund policy"):
        queued_prompt = "A customer in Lagos asks to return a damaged item 5 days after delivery. What is our policy and refund timeline?"

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            last_log = turn.get("last_log") or {}
            st.caption(
                f"Turn cost: NGN {turn['turn_cost_ngn']:.6f} | "
                f"Prompt tokens: {last_log.get('prompt_tokens', 0)} | "
                f"Cached tokens: {last_log.get('cached_tokens', 0)}"
            )
            if turn.get("tool_events"):
                with st.expander("Tool Calls"):
                    st.json(turn["tool_events"])

    typed_prompt = st.chat_input("Ask about orders, SKU stock, bank transfer reconciliation, or merchant policies...")
    final_prompt = typed_prompt or queued_prompt
    if final_prompt:
        handle_prompt(final_prompt, log_path)
        st.rerun()


if __name__ == "__main__":
    main()
