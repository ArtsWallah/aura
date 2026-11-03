from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings, validate_service_environment
from app.core.logging import logger
from app.agent.service import task_execution_service
from app.api.routes import health, tools, models, tasks, files, artifacts, settings as settings_routes

# Ensure tools and models are loaded
import app.tools
import app.models

@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_service_environment()
    logger.info("Initializing AURA Platform backend services...")
    settings.init_directories()
    await task_execution_service.start()
    yield
    await task_execution_service.stop()
    logger.info("Shutting down AURA Platform backend services...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous Understanding, Reasoning & Analytics platform with specialized small models, typed tool registry, and privacy-aware execution.",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def sanitized_request_validation_error(request: Request, exc: RequestValidationError):
    """Return validation locations/categories without values or raw messages."""
    safe_sources = {"body", "query", "path", "header", "cookie"}
    category_map = {
        "missing": "missing_value",
        "string_type": "invalid_type",
        "int_type": "invalid_type",
        "float_type": "invalid_type",
        "bool_type": "invalid_type",
        "list_type": "invalid_type",
        "dict_type": "invalid_type",
        "model_type": "invalid_type",
        "literal_error": "invalid_choice",
        "int_parsing": "invalid_integer",
        "float_parsing": "invalid_number",
        "bool_parsing": "invalid_boolean",
        "json_invalid": "invalid_json",
        "greater_than": "out_of_range",
        "greater_than_equal": "out_of_range",
        "less_than": "out_of_range",
        "less_than_equal": "out_of_range",
        "too_short": "invalid_length",
        "too_long": "invalid_length",
    }
    safe_errors = []
    for error in exc.errors():
        raw_location = error.get("loc", ())
        source = raw_location[0] if raw_location and raw_location[0] in safe_sources else "request"
        safe_errors.append({
            "location": [source, "field"] if len(raw_location) > 1 else [source],
            "category": category_map.get(error.get("type"), "invalid_value"),
        })
    return JSONResponse(
        status_code=422,
        content={"detail": safe_errors, "message": "Request validation failed."},
    )


@app.middleware("http")
async def local_single_user_boundary(request: Request, call_next):
    """This application supports one local OS user; it has no remote user model."""
    host = request.client.host if request.client else ""
    if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return JSONResponse(status_code=403, content={"detail": "AURA accepts local loopback connections only."})
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") not in {item.rstrip("/") for item in settings.CORS_ORIGINS}:
        return JSONResponse(status_code=403, content={"detail": "Origin is not allowed."})
    protected_endpoint = request.url.path == "/api/settings" or (
        request.method in {"POST", "PUT", "PATCH", "DELETE"}
        and request.url.path in {"/api/tools/execute", "/api/artifacts/migrate-legacy"}
    )
    if protected_endpoint and origin is None:
        return JSONResponse(status_code=403, content={"detail": "A configured frontend Origin is required."})
    return await call_next(request)

# Mount API Routers
app.include_router(health.router) # /health
app.include_router(health.router, prefix=settings.API_V1_STR) # /api/health
app.include_router(tools.router, prefix=settings.API_V1_STR) # /api/tools
app.include_router(models.router, prefix=settings.API_V1_STR) # /api/models
app.include_router(tasks.router, prefix=settings.API_V1_STR) # /api/tasks
app.include_router(files.router, prefix=settings.API_V1_STR) # /api/files
app.include_router(artifacts.router, prefix=settings.API_V1_STR)
app.include_router(settings_routes.router, prefix=settings.API_V1_STR) # /api/settings

# Artifacts are served through task-scoped routes that validate each artifact
# against the completed task's generated artifact list.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)

