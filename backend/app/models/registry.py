import time
from typing import Any
from pydantic import BaseModel, Field
from app.core.logging import logger

class ModelMetadata(BaseModel):
    model_id: str
    name: str
    category: str # "classification", "embedding", "reasoning"
    execution_location: str # "local", "browser_webgpu", "cloud"
    parameters_size: str # e.g. "66M (DistilBERT)", "384-dim (MiniLM)"
    is_loaded: bool = False
    average_latency_ms: float = 0.0
    total_inferences: int = 0
    description: str

class ModelRegistry:
    def __init__(self):
        self._models: dict[str, ModelMetadata] = {}
        self._instances: dict[str, Any] = {}

    def register_model(self, metadata: ModelMetadata, loader_func: Any = None):
        self._models[metadata.model_id] = metadata
        if loader_func:
            self._instances[metadata.model_id] = loader_func
        logger.info(f"Registered model: {metadata.name} [{metadata.execution_location}]")

    def list_models(self) -> list[ModelMetadata]:
        return list(self._models.values())

    def get_model(self, model_id: str) -> ModelMetadata | None:
        return self._models.get(model_id)

    def record_inference(self, model_id: str, latency_ms: float):
        if model_id in self._models:
            meta = self._models[model_id]
            prev_total = meta.total_inferences
            meta.average_latency_ms = (
                (meta.average_latency_ms * prev_total + latency_ms) / (prev_total + 1)
            )
            meta.total_inferences += 1

model_registry = ModelRegistry()

