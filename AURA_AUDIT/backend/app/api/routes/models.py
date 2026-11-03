from fastapi import APIRouter
from pydantic import BaseModel
from app.models.registry import model_registry, ModelMetadata
from app.models.intent_classifier import intent_classifier, IntentResult
from app.models.embeddings import embedding_model, EmbeddingResult

router = APIRouter(prefix="/models", tags=["Models"])

class ClassifyIntentRequest(BaseModel):
    text: str

class EmbedRequest(BaseModel):
    text: str

@router.get("", response_model=list[ModelMetadata])
def list_models():
    return model_registry.list_models()

@router.post("/classify-intent", response_model=IntentResult)
def classify_intent(request: ClassifyIntentRequest):
    return intent_classifier.classify(request.text)

@router.post("/embed", response_model=EmbeddingResult)
def generate_embedding(request: EmbedRequest):
    return embedding_model.encode(request.text)

