from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger
from app.agent.service import task_execution_service
from app.api.routes import health, tools, models, tasks, files, artifacts, settings as settings_routes

# Ensure tools and models are loaded
import app.tools
import app.models

@asynccontextmanager
async def lifespan(app: FastAPI):
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

# Mount static directories for charts and reports
# Artifacts are served only through the authenticated-at-rest decryption route.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)

