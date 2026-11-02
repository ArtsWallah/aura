from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import logger
from app.api.routes import health, tools, models, tasks, files, settings as settings_routes

# Ensure tools and models are loaded
import app.tools
import app.models

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing AURA Platform backend services...")
    settings.init_directories()
    yield
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
    allow_origins=["*"], # Allow development UI access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(health.router) # /health
app.include_router(health.router, prefix=settings.API_V1_STR) # /api/health
app.include_router(tools.router, prefix=settings.API_V1_STR) # /api/tools
app.include_router(models.router, prefix=settings.API_V1_STR) # /api/models
app.include_router(tasks.router, prefix=settings.API_V1_STR) # /api/tasks
app.include_router(files.router, prefix=settings.API_V1_STR) # /api/files
app.include_router(settings_routes.router, prefix=settings.API_V1_STR) # /api/settings

# Mount static directories for charts and reports
app.mount("/data/charts", StaticFiles(directory=str(settings.CHARTS_DIR)), name="charts")
app.mount("/data/reports", StaticFiles(directory=str(settings.REPORTS_DIR)), name="reports")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
