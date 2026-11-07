from app.tools.registry import tool_registry
from app.tools.schemas import (
    ToolDefinition,
    ToolResult,
    ToolCategory,
    ExecutionLocation,
    PermissionLevel,
    StrictContract,
    StrictJsonValue,
)

# Ensure all tools are registered and import output contracts
from app.tools.dataset_tools import InspectDatasetOutput
from app.tools.analytics_tools import (
    DescriptiveStatisticsOutput,
    ColumnDescriptiveStatistics,
    GroupComparisonOutput,
    CorrelationAnalysisOutput,
    CorrelationPair,
    DetectOutliersOutput,
)
from app.tools.visualization_tools import GenerateChartOutput
from app.tools.research_tools import (
    WebResearchOutput,
    WebResearchSource,
    WebResearchMetadata,
)
from app.tools.document_tools import (
    ReadDocumentOutput,
    DocumentChunk,
    SearchDocumentChunksOutput,
    ScoredDocumentChunk,
)
from app.tools.report_tools import GenerateReportOutput
from app.tools.system_tools import SystemHealthOutput, SystemStorageInfo
from app.tools.transform_schemas import TransformDatasetOutput
import app.tools.transforms.tool  # Ensure transform_dataset is registered

__all__ = [
    "tool_registry",
    "ToolDefinition",
    "ToolResult",
    "ToolCategory",
    "ExecutionLocation",
    "PermissionLevel",
    "StrictContract",
    "StrictJsonValue",
    "InspectDatasetOutput",
    "DescriptiveStatisticsOutput",
    "ColumnDescriptiveStatistics",
    "GroupComparisonOutput",
    "CorrelationAnalysisOutput",
    "CorrelationPair",
    "DetectOutliersOutput",
    "GenerateChartOutput",
    "WebResearchOutput",
    "WebResearchSource",
    "WebResearchMetadata",
    "ReadDocumentOutput",
    "DocumentChunk",
    "SearchDocumentChunksOutput",
    "ScoredDocumentChunk",
    "GenerateReportOutput",
    "SystemHealthOutput",
    "SystemStorageInfo",
    "TransformDatasetOutput",
]

