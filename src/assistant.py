import uuid
from typing import Any

from openai import APITimeoutError, InternalServerError, OpenAI

from src.cost_tracker import NairaCostTracker
from src.retriever import MerchantKnowledgeBase
from src.tools import MerchantToolRegistry

SYSTEM_PROMPT = """You are NaijaMerchant Ops Assistant, a practical operations and policy copilot for Nigerian retail, energy, and logistics merchants.

How to answer:
1. Ground policy and catalog claims in the merchant rulebook and product catalog via search_merchant_kb.
2. Use check_order_status when asked about a specific order ID, payment/bank transfer session reference, or delivery ETA.
3. Use check_sku_stock when asked about SKU stock levels, prices in NGN, or restock/reorder status.
4. Keep answers concise, accurate, and actionable for store operators and support agents. State NGN amounts clearly (e.g., NGN 28,500) and mention relevant policy IDs (such as POL-001 through POL-004) when citing rules.
5. Never invent order IDs, SKUs, session references, or policy rules. If information is missing, say so clearly."""


class MerchantOpsAssistant:
    """Runs multi-turn tool-calling merchant ops queries with prompt caching and NGN cost tracking."""

    def __init__(
        self,
        client: OpenAI,
        chat_model: str = "gemini-3.8-flash",
        embedding_model: str = "gemini-embedding-2",
        cost_log_path: str = "logs/request_costs.jsonl",
        session_id: str | None = None,
        fallback_model: str = "claude-haiku-5-5",
    ) -> None:
        self.client = client
        self.chat_model = chat_model
        self.fallback_model = fallback_model
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        self.tracker = NairaCostTracker(client=client, log_path=cost_log_path)
        self.kb = MerchantKnowledgeBase(client=client, model=embedding_model)
        self.kb.build_index()
        self.tracker.record_embedding(
            session_id=self.session_id,
            operation="kb_index",
            model=self.kb.model,
            tokens=self.kb.last_embedding_tokens,
        )

        self.tools = MerchantToolRegistry(
            kb=self.kb,
            on_embedding_call=self._log_query_embedding,
        )
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _log_query_embedding(self, model: str, tokens: int) -> None:
        self.tracker.record_embedding(
            session_id=self.session_id,
            operation="kb_query_embedding",
            model=model,
            tokens=tokens,
        )

    def _create_completion(self, model_name: str) -> Any:
        return self.client.chat.completions.create(
            model=model_name,
            messages=self.messages,
            tools=self.tools.schemas(),
            max_tokens=700,
            prompt_cache_key=self.session_id,
            extra_headers={"X-Session-Id": self.session_id},
        )

    def ask(self, user_message: str, max_tool_rounds: int = 5) -> dict[str, Any]:
        self.messages.append({"role": "user", "content": user_message})
        turn_cost_ngn = 0.0
        tool_events: list[dict[str, Any]] = []
        final_text = ""
        last_log: dict[str, Any] = {}

        for _ in range(max_tool_rounds):
            active_model = self.chat_model
            try:
                reply = self._create_completion(active_model)
            except (InternalServerError, APITimeoutError):
                active_model = self.fallback_model if self.chat_model != self.fallback_model else "gemini-3.8-flash"
                reply = self._create_completion(active_model)

            last_log = self.tracker.record_chat(
                session_id=self.session_id,
                operation="chat_completion",
                model=active_model,
                usage=getattr(reply, "usage", None),
            )
            turn_cost_ngn += float(last_log["cost_ngn"])

            message = reply.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []

            if not tool_calls:
                final_text = message.content or ""
                self.messages.append({"role": "assistant", "content": final_text})
                break

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            self.messages.append(assistant_msg)

            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args = tc.function.arguments
                tool_output = self.tools.execute(tool_name, tool_args)
                tool_events.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "output": tool_output,
                })
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                })
        else:
            final_text = "Reached maximum tool lookup steps without a final response."
            self.messages.append({"role": "assistant", "content": final_text})

        return {
            "session_id": self.session_id,
            "answer": final_text,
            "turn_cost_ngn": round(turn_cost_ngn, 6),
            "tool_events": tool_events,
            "last_log": last_log,
        }
