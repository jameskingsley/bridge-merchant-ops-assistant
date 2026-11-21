# NaijaMerchant Ops Assistant

A Merchant Operations & Policy Q&A assistant built for Nigerian retail, energy, and logistics merchants on the **Bridge** API (https://wekoya-api.wekoya.tech/v1) using the OpenAI Python SDK and a Streamlit interface.

**Live Streamlit Demo:** [https://bridge-merchant-ops-assistant-636mebtfptzco99d3kgq63.streamlit.app/](https://bridge-merchant-ops-assistant-636mebtfptzco99d3kgq63.streamlit.app/)

## What It Does

- **Semantic Policy & Catalog Retrieval (gemini-embedding-2):** Embeds merchant operational rules (POL-001 through POL-004) and product catalog items (SKU-PWR-100, SKU-INV-500, SKU-PKG-012, SKU-RTR-200) into 768-dimensional unit-normalized vectors and ranks matches via NumPy cosine similarity.
- **Multi-Turn Tool Calling:** Executes search_merchant_kb, check_order_status (covering payment states, bank transfer session references, cutoff times, and dispatch/delivery status), and check_sku_stock (covering NGN pricing, stock counts, and reorder threshold flags).
- **Session Prompt Caching:** Places a static operational system prompt first and sends prompt_cache_key alongside the X-Session-Id header so multi-turn conversations reuse warm cache prefixes when eligible.
- **Per-Request Naira (NGN) Cost Telemetry:** Reads model pricing from /v1/models (with fallback NGN rates), accounts for cached_tokens and embedding tokens, writes structured entries to logs/request_costs.jsonl, and surfaces spend and token metrics in the Streamlit sidebar and chat turns.
- **Resilience & Timeout Fallback:** Defaults to a fast tool-capable chat model (gemini-3.8-flash) with bounded output tokens and automatic fallback to claude-haiku-5-5 if a request hits a 504 gateway error or timeout.

## Model Choices & Trade-Offs

- **Chat Model (gemini-3.8-flash with claude-haiku-5-5 fallback):** Pinning gemini-3.8-flash avoids multi-step tool-calling latency spikes and gateway timeouts common on slower reasoning routes, while keeping per-turn Naira costs low. Falling back to claude-haiku-5-5 on 504 / timeout errors improves reliability during peak traffic without failing the user turn.
- **Embedding Model (gemini-embedding-2 at 768 dimensions):** Truncating embeddings from 3,072 to 768 dimensions reduces memory footprint and retrieval overhead while maintaining strong semantic accuracy across Nigerian merchant policies and SKUs at low NGN token cost.
- **In-Memory Vector Search vs. External Vector DB:** An in-memory NumPy index built from data/merchant_rulebook.json and data/merchant_catalog.json keeps setup lightweight and deterministic for small-to-medium merchant datasets with zero external vector database latency or cost.
- **Session Prompt Caching (prompt_cache_key + X-Session-Id):** Reduces prompt token spend on follow-up turns within an active merchant session once the shared prefix meets Bridge's caching threshold, at the cost of keeping the system prompt static and passing a consistent session identifier.

## Naira Cost Tracking

Every embedding call (kb_index, kb_query_embedding) and chat completion round (chat_completion) is recorded with:

- 	imestamp (UTC ISO-8601)
- session_id
- operation and model
- prompt_tokens, cached_tokens, and completion_tokens
- cost_ngn and currency ("NGN")

In the Streamlit app, the sidebar displays **Total Logged Spend** in NGN and recent log records, while each assistant response shows that turn's NGN cost alongside prompt and cached token counts.

## Project Structure

`	ext
bridge-merchant-ops-assistant/
├── app.py                        # Streamlit UI, session state, cost telemetry, and sample queries
├── data/
│   ├── merchant_catalog.json     # Sample merchant SKUs, NGN prices, stock quantities, and reorder points
│   └── merchant_rulebook.json    # Merchant policies (POL-001 to POL-004)
├── src/
│   ├── __init__.py
│   ├── assistant.py              # Tool-calling loop, prompt caching headers/keys, and 504 fallback
│   ├── cost_tracker.py           # NGN rate lookup and JSONL request cost logger
│   ├── retriever.py              # gemini-embedding-2 (768d) indexer and cosine similarity search
│   └── tools.py                  # Tool schemas and implementations for KB, orders, and stock
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md




Local Setup & Running

PowerShell

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item  .env
# Set WEKOYA_API_KEY and optional model/log overrides in .env
streamlit run app.py
Streamlit Community Cloud Configuration
In App settings -> Secrets on Streamlit Community Cloud, configure:

Ini, TOML
WEKOYA_API_KEY = "your_bridge_api_key"
WEKOYA_BASE_URL = "[https://wekoya-api.wekoya.tech/v1](https://wekoya-api.wekoya.tech/v1)"
CHAT_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "gemini-embedding-2"
COST_LOG_PATH = "logs/request_costs.jsonl"



