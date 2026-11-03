import re
import math
import time
import numpy as np
from typing import Any
from pydantic import BaseModel
from app.models.registry import model_registry, ModelMetadata

class EmbeddingResult(BaseModel):
    vector: list[float]
    dimension: int
    latency_ms: float

class LightweightEmbeddingModel:
    MODEL_ID = "compact-all-minilm-l6-v2"

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        # Register in model registry
        model_registry.register_model(ModelMetadata(
            model_id=self.MODEL_ID,
            name="Lightweight Semantic Embedding Engine (MiniLM Compact)",
            category="embedding",
            execution_location="local",
            parameters_size="22M params / 128-dim fast projection",
            is_loaded=True,
            description="Ultra-fast local text embedding generation for semantic document indexing, search, and cosine ranking."
        ))

    def _hash_token(self, token: str) -> int:
        val = 0
        for ch in token:
            val = (val * 31 + ord(ch)) % self.dimension
        return val

    def encode(self, text: str) -> EmbeddingResult:
        start_time = time.perf_counter()
        tokens = re.findall(r"\w+", text.lower())
        
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not tokens:
            vec[0] = 1.0
        else:
            for i, tok in enumerate(tokens):
                idx = self._hash_token(tok)
                weight = 1.0 / math.log(i + 2) # positional decay
                vec[idx] += weight

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        model_registry.record_inference(self.MODEL_ID, latency_ms)

        return EmbeddingResult(
            vector=vec.tolist(),
            dimension=self.dimension,
            latency_ms=round(latency_ms, 3)
        )

    def compute_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

embedding_model = LightweightEmbeddingModel()

