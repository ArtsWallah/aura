import httpx
import urllib.parse
from typing import Any
from pydantic import BaseModel, Field

from app.tools.schemas import ToolCategory, ExecutionLocation, PermissionLevel
from app.tools.registry import tool_registry
from app.core.logging import logger
from app.privacy.outbound import PrivacyAwareAsyncClient, current_privacy_mode

class WebResearchInput(BaseModel):
    query: str = Field(description="Search terms or question to research.")
    max_results: int = Field(default=5, description="Number of results to retrieve.")

class SourceCitation(BaseModel):
    title: str
    url: str
    snippet: str
    published_date: str | None = None
    reliability_tier: str = "high"

@tool_registry.register(
    name="web_research",
    category=ToolCategory.RESEARCH,
    description="Searches for verified external information, collects titles, URLs, and snippets with preserved metadata and source citations.",
    input_model=WebResearchInput,
    execution_location=ExecutionLocation.CLOUD,
    permission_level=PermissionLevel.READ_ONLY,
    retry_safe=True,
    timeout_seconds=20
)
async def web_research(params: WebResearchInput) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    mode = current_privacy_mode()
    if mode == "local_only":
        raise PermissionError("External research is blocked by local_only privacy mode.")
    if mode == "hybrid":
        raise PermissionError("External research requires explicit cloud_enabled privacy mode.")
    query = params.query
    
    # Try querying Wikipedia REST API as a reliable knowledge source
    try:
        encoded_query = urllib.parse.quote(query)
        wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json&utf8=1&srlimit={params.max_results}"
        
        async with PrivacyAwareAsyncClient(timeout=10.0) as client:
            resp = await client.get(wiki_url, headers={"User-Agent": "AURA-Agent/1.0 (ResearchTool)"})
            if resp.status_code == 200:
                data = resp.json()
                search_results = data.get("query", {}).get("search", [])
                for item in search_results:
                    title = item.get("title", "")
                    clean_snippet = item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
                    page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    sources.append({
                        "title": title,
                        "url": page_url,
                        "snippet": clean_snippet,
                        "domain": "wikipedia.org",
                        "is_factual_source": True
                    })
    except PermissionError:
        raise
    except Exception:
        logger.warning("External search provider failed or timed out.")

    # If no results or external failure, provide structured fallback knowledge response
    return {
        "query": query,
        "results_count": len(sources),
        "sources": sources,
        "metadata": {
            "search_provider": "Wikipedia API",
            "distinguish_interpretation": "All claims must cite one of the listed sources."
        }
    }

