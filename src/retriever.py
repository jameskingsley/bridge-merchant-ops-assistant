import json
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


class MerchantKnowledgeBase:
    """Embeds and searches merchant policies and catalog items via Bridge."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gemini-embedding-2",
        dimensions: int = 768,
        rulebook_path: str = "data/merchant_rulebook.json",
        catalog_path: str = "data/merchant_catalog.json",
    ) -> None:
        self.client = client
        self.model = model
        self.dimensions = dimensions
        self.rulebook_path = Path(rulebook_path)
        self.catalog_path = Path(catalog_path)
        self.entries: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None
        self.last_embedding_tokens: int = 0

    def _load_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        rules = json.loads(self.rulebook_path.read_text(encoding="utf-8"))
        for rule in rules:
            text = (
                f"[POLICY {rule['id']} | {rule['category']}] {rule['title']}: "
                f"{rule['content']}"
            )
            entries.append({
                "source_type": "policy",
                "id": rule["id"],
                "title": rule["title"],
                "text": text,
                "metadata": rule,
            })

        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        for item in catalog:
            text = (
                f"[CATALOG {item['sku']} | {item['category']}] {item['name']} — "
                f"Price: NGN {item['price_ngn']:,}, Stock: {item['stock_qty']} units, "
                f"Reorder threshold: {item['reorder_point']} units. {item['notes']}"
            )
            entries.append({
                "source_type": "catalog",
                "id": item["sku"],
                "title": item["name"],
                "text": text,
                "metadata": item,
            })

        return entries

    def build_index(self) -> int:
        self.entries = self._load_entries()
        texts = [entry["text"] for entry in self.entries]

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            extra_body={"task_type": "RETRIEVAL_DOCUMENT"},
        )

        self.last_embedding_tokens = (
            response.usage.total_tokens
            if getattr(response, "usage", None) and getattr(response.usage, "total_tokens", None)
            else sum(max(1, len(t) // 4) for t in texts)
        )

        vectors = [item.embedding for item in response.data]
        self._embeddings = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._embeddings = self._embeddings / np.clip(norms, 1e-12, None)
        return len(self.entries)

    def search(self, query: str, top_k: int = 3, source_type: str | None = None) -> list[dict[str, Any]]:
        if self._embeddings is None or not self.entries:
            self.build_index()

        q_resp = self.client.embeddings.create(
            model=self.model,
            input=[query],
            dimensions=self.dimensions,
            extra_body={"task_type": "RETRIEVAL_QUERY"},
        )
        q_tokens = (
            q_resp.usage.total_tokens
            if getattr(q_resp, "usage", None) and getattr(q_resp.usage, "total_tokens", None)
            else max(1, len(query) // 4)
        )
        self.last_embedding_tokens += q_tokens

        q_vec = np.asarray(q_resp.data[0].embedding, dtype=np.float32)
        q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)

        scores = self._embeddings @ q_vec
        ranked_indices = np.argsort(scores)[::-1]

        results: list[dict[str, Any]] = []
        for idx in ranked_indices:
            entry = self.entries[int(idx)]
            if source_type and entry["source_type"] != source_type:
                continue
            results.append({
                "score": round(float(scores[int(idx)]), 4),
                "source_type": entry["source_type"],
                "id": entry["id"],
                "title": entry["title"],
                "text": entry["text"],
                "metadata": entry["metadata"],
            })
            if len(results) >= top_k:
                break

        return results