import json
from typing import Any

from src.retriever import MerchantKnowledgeBase

SAMPLE_ORDERS: dict[str, dict[str, Any]] = {
    "ORD-4021": {
        "order_id": "ORD-4021",
        "customer_city": "Lagos (Lekki)",
        "item_sku": "SKU-PWR-100",
        "qty": 2,
        "total_ngn": 57000,
        "payment_status": "paid_transfer",
        "session_ref": "999001261121041122",
        "fulfillment_status": "dispatched",
        "confirmed_at_wat": "2026-11-21 11:20 WAT",
        "eta": "Today before 6:00 PM WAT",
    },
    "ORD-4022": {
        "order_id": "ORD-4022",
        "customer_city": "Enugu",
        "item_sku": "SKU-INV-500",
        "qty": 1,
        "total_ngn": 145000,
        "payment_status": "pending_reconciliation",
        "session_ref": "999001261121048890",
        "fulfillment_status": "held_payment_check",
        "confirmed_at_wat": "2026-11-21 15:45 WAT",
        "eta": "Pending bank transfer clearance; interstate shipping 2-4 business days after confirmation",
    },
    "ORD-4023": {
        "order_id": "ORD-4023",
        "customer_city": "Abuja",
        "item_sku": "SKU-RTR-200",
        "qty": 1,
        "total_ngn": 36000,
        "payment_status": "paid_transfer",
        "session_ref": "999001261121052301",
        "fulfillment_status": "packed_next_day",
        "confirmed_at_wat": "2026-11-21 16:10 WAT",
        "eta": "Next business day dispatch; 2-4 business days interstate transit",
    },
}


class MerchantToolRegistry:
    """Defines tool schemas and executes merchant operations tools."""

    def __init__(self, kb: MerchantKnowledgeBase, on_embedding_call: Any = None) -> None:
        self.kb = kb
        self.on_embedding_call = on_embedding_call

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_merchant_kb",
                    "description": (
                        "Search the merchant rulebook (dispatch cutoffs, bank transfer reconciliation, "
                        "returns/refunds, reorder thresholds) and product catalog using semantic embeddings."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Merchant policy or product question to search for.",
                            },
                            "source_type": {
                                "type": "string",
                                "enum": ["policy", "catalog"],
                                "description": "Optional filter: 'policy' or 'catalog'. Omit to search both.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top matching entries to return (1 to 5).",
                                "default": 3,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_order_status",
                    "description": "Look up a customer order's payment, transfer session reference, and dispatch/ETA status.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "Order ID such as ORD-4021, ORD-4022, or ORD-4023.",
                            }
                        },
                        "required": ["order_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_sku_stock",
                    "description": "Check current stock level, price in NGN, and reorder alert status for a product SKU.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sku": {
                                "type": "string",
                                "description": "Product SKU such as SKU-PWR-100, SKU-INV-500, SKU-PKG-012, or SKU-RTR-200.",
                            }
                        },
                        "required": ["sku"],
                    },
                },
            },
        ]

    def execute(self, name: str, raw_arguments: str) -> str:
        try:
            args = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON arguments from model."})

        if name == "search_merchant_kb":
            query = str(args.get("query", "")).strip()
            if not query:
                return json.dumps({"error": "Missing required 'query' parameter."})
            source_type = args.get("source_type")
            top_k = max(1, min(5, int(args.get("top_k", 3) or 3)))
            matches = self.kb.search(query=query, top_k=top_k, source_type=source_type)
            if callable(self.on_embedding_call):
                self.on_embedding_call(self.kb.model, self.kb.last_embedding_tokens)
            return json.dumps({"matches": matches})

        if name == "check_order_status":
            order_id = str(args.get("order_id", "")).strip().upper()
            order = SAMPLE_ORDERS.get(order_id)
            if not order:
                return json.dumps({"error": f"Order '{order_id}' not found.", "known_orders": list(SAMPLE_ORDERS.keys())})
            return json.dumps({"order": order})

        if name == "check_sku_stock":
            sku = str(args.get("sku", "")).strip().upper()
            self.kb._load_entries()
            for entry in self.kb.entries:
                if entry["source_type"] == "catalog" and entry["id"].upper() == sku:
                    meta = entry["metadata"]
                    needs_reorder = int(meta["stock_qty"]) <= int(meta["reorder_point"])
                    return json.dumps({
                        "sku": meta["sku"],
                        "name": meta["name"],
                        "price_ngn": meta["price_ngn"],
                        "stock_qty": meta["stock_qty"],
                        "reorder_point": meta["reorder_point"],
                        "needs_reorder": needs_reorder,
                        "notes": meta["notes"],
                    })
            return json.dumps({"error": f"SKU '{sku}' not found in catalog."})

        return json.dumps({"error": f"Unknown tool '{name}'."})
