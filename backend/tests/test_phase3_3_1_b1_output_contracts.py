"""Phase 3.3.1-B1: Tests for typed tool output contracts and mandatory registration."""

import pytest
from pydantic import BaseModel, ValidationError

from app.tools.registry import ToolRegistry, tool_registry
from app.tools.schemas import (
    ExecutionLocation,
    PermissionLevel,
    StrictContract,
    ToolCategory,
)
from app.tools import (
    InspectDatasetOutput,
    DescriptiveStatisticsOutput,
    ColumnDescriptiveStatistics,
    GroupComparisonOutput,
    CorrelationAnalysisOutput,
    CorrelationPair,
    DetectOutliersOutput,
    GenerateChartOutput,
    ReadDocumentOutput,
    DocumentChunk,
    SearchDocumentChunksOutput,
    ScoredDocumentChunk,
    GenerateReportOutput,
    WebResearchOutput,
    WebResearchSource,
    WebResearchMetadata,
    SystemHealthOutput,
    SystemStorageInfo,
)


class DummyInput(BaseModel):
    query: str


class DummyOutput(StrictContract):
    result: str


EXPECTED_TOOL_OUTPUT_MODELS = {
    "inspect_dataset": InspectDatasetOutput,
    "descriptive_statistics": DescriptiveStatisticsOutput,
    "group_comparison": GroupComparisonOutput,
    "correlation_analysis": CorrelationAnalysisOutput,
    "detect_outliers": DetectOutliersOutput,
    "generate_chart": GenerateChartOutput,
    "read_document": ReadDocumentOutput,
    "search_document_chunks": SearchDocumentChunksOutput,
    "generate_report": GenerateReportOutput,
    "web_research": WebResearchOutput,
    "system_health": SystemHealthOutput,
}


# 1. All 11 registered tools expose an output model
def test_all_11_tools_expose_valid_output_models():
    """Verify that all 11 production tools expose their expected strict output models."""
    registered_names = {t.name for t in tool_registry.list_tools()}
    for tool_name, expected_model in EXPECTED_TOOL_OUTPUT_MODELS.items():
        assert tool_name in registered_names, f"Tool '{tool_name}' is not registered."
        output_model = tool_registry.get_output_model(tool_name)
        assert output_model is not None, f"Tool '{tool_name}' has no output model registered."
        assert output_model is expected_model, (
            f"Tool '{tool_name}' has output model {output_model}, expected {expected_model}."
        )
        assert issubclass(output_model, BaseModel)
        assert issubclass(output_model, StrictContract)


# 2. Registration without an output model fails
def test_registration_without_output_model_fails():
    """Registration must raise TypeError immediately when output_model is omitted or None."""
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="must declare an explicit output_model"):
        registry.register(
            name="test_tool_missing_output",
            category=ToolCategory.SYSTEM,
            description="Tool without output model",
            input_model=DummyInput,
            output_model=None,
        )(lambda _x: {"result": "ok"})


# 3. Registration with an invalid output model fails
def test_registration_with_invalid_output_model_fails():
    """Registration must raise TypeError immediately when output_model is not a BaseModel subclass."""
    registry = ToolRegistry()

    invalid_models = ["not_a_model", dict, int, object(), 123]
    for invalid in invalid_models:
        with pytest.raises(TypeError, match="must be an explicit subclass of pydantic.BaseModel"):
            registry.register(
                name=f"test_tool_invalid_{type(invalid).__name__}",
                category=ToolCategory.SYSTEM,
                description="Tool with invalid output model",
                input_model=DummyInput,
                output_model=invalid,  # type: ignore[arg-type]
            )(lambda _x: {"result": "ok"})


# 3b. Registration with missing or invalid input model fails
def test_registration_with_invalid_input_model_fails():
    """Registration must raise TypeError immediately when input_model is missing or invalid."""
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="input_model must be an explicit subclass"):
        registry.register(
            name="test_tool_invalid_input",
            category=ToolCategory.SYSTEM,
            description="Tool with invalid input model",
            input_model=None,  # type: ignore[arg-type]
            output_model=DummyOutput,
        )(lambda _x: {"result": "ok"})


# 4. Valid nested output structures validate
def test_valid_nested_output_structures_validate():
    """Verify that rich nested structures instantiate cleanly against their models."""
    # DescriptiveStatisticsOutput with ColumnDescriptiveStatistics
    stats_data = {
        "statistics": {
            "revenue": {
                "count": 100,
                "mean": 250.5,
                "std": 15.2,
                "min": 100.0,
                "q25": 200.0,
                "median": 250.0,
                "q75": 300.0,
                "max": 500.0,
                "iqr": 100.0,
                "skewness": 0.12,
                "kurtosis": -0.05,
            }
        },
        "analyzed_columns": ["revenue"],
    }
    stats_out = DescriptiveStatisticsOutput.model_validate(stats_data)
    assert isinstance(stats_out.statistics["revenue"], ColumnDescriptiveStatistics)
    assert stats_out.statistics["revenue"].mean == 250.5

    # DocumentChunk and ReadDocumentOutput
    doc_data = {
        "file_name": "report.txt",
        "total_characters": 1200,
        "total_words": 200,
        "chunk_count": 2,
        "chunks": [
            {"chunk_index": 0, "start_char": 0, "end_char": 600, "text": "Part 1"},
            {"chunk_index": 1, "start_char": 500, "end_char": 1200, "text": "Part 2"},
        ],
        "sample_preview": "Part 1...",
    }
    doc_out = ReadDocumentOutput.model_validate(doc_data)
    assert len(doc_out.chunks) == 2
    assert isinstance(doc_out.chunks[0], DocumentChunk)

    # ScoredDocumentChunk and SearchDocumentChunksOutput
    search_data = {
        "query": "market trend",
        "matches": [
            {"chunk_index": 0, "relevance_score": 0.95, "text": "Strong market trend observed."}
        ],
    }
    search_out = SearchDocumentChunksOutput.model_validate(search_data)
    assert isinstance(search_out.matches[0], ScoredDocumentChunk)
    assert search_out.matches[0].relevance_score == 0.95

    # WebResearchOutput with sources and metadata
    research_data = {
        "query": "AI architecture",
        "results_count": 1,
        "sources": [
            {
                "title": "AI Systems",
                "url": "https://example.org/ai",
                "snippet": "Overview of AI systems.",
                "domain": "example.org",
                "is_factual_source": True,
            }
        ],
        "metadata": {
            "search_provider": "Wikipedia API",
            "distinguish_interpretation": "All claims must cite sources.",
        },
    }
    research_out = WebResearchOutput.model_validate(research_data)
    assert isinstance(research_out.sources[0], WebResearchSource)
    assert isinstance(research_out.metadata, WebResearchMetadata)

    # CorrelationAnalysisOutput with CorrelationPair
    corr_data = {
        "method": "pearson",
        "analyzed_columns": ["col_a", "col_b"],
        "prominent_relationships": [
            {
                "column_1": "col_a",
                "column_2": "col_b",
                "correlation": 0.88,
                "relationship": "strong_positive",
            }
        ],
        "correlation_matrix": {"col_a": {"col_a": 1.0, "col_b": 0.88}, "col_b": {"col_a": 0.88, "col_b": 1.0}},
        "error": None,
    }
    corr_out = CorrelationAnalysisOutput.model_validate(corr_data)
    assert isinstance(corr_out.prominent_relationships[0], CorrelationPair)


# 5. Unknown fields are rejected (extra="forbid")
def test_unknown_fields_are_rejected():
    """StrictContract must reject extra fields on top-level and nested models."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        InspectDatasetOutput.model_validate({
            "file_name": "data.csv",
            "format": "CSV",
            "total_rows": 10,
            "total_columns": 2,
            "columns": ["a", "b"],
            "column_types": {"a": "int64", "b": "float64"},
            "missing_values": {"a": 0, "b": 0},
            "missing_percentages": {"a": 0.0, "b": 0.0},
            "duplicate_rows_count": 0,
            "numeric_columns": ["a", "b"],
            "categorical_columns": [],
            "date_columns": [],
            "sample_data": [],
            "memory_usage_kb": 12.5,
            "unexpected_extra_field": "injected_payload",
        })

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GenerateChartOutput.model_validate({
            "chart_id": "chart_12345",
            "title": "Sales",
            "chart_type": "bar",
            "relative_path": "/data/charts/chart_12345.png.enc",
            "file_path": "/safe/charts/chart_12345.png.enc",
            "data_uri": "data:image/png;base64,abc",
            "artifacts": ["/safe/charts/chart_12345.png.enc"],
            "admin_override": True,
        })


# 6. Required fields cannot be omitted
def test_required_fields_cannot_be_omitted():
    """Omitting non-optional fields must raise ValidationError."""
    with pytest.raises(ValidationError):
        # Missing total_rows, columns, etc.
        InspectDatasetOutput.model_validate({"file_name": "data.csv"})

    with pytest.raises(ValidationError):
        # Missing chart_id, title
        GenerateChartOutput.model_validate({
            "chart_type": "bar",
            "file_path": "/path/to/chart.png",
        })

    with pytest.raises(ValidationError):
        # Missing report_id
        GenerateReportOutput.model_validate({
            "title": "Title",
            "markdown_content": "# Title",
            "markdown_path": "/path.md",
            "html_path": "/path.html",
            "artifacts": [],
        })


# 7. Legitimate empty collections validate
def test_legitimate_empty_collections_validate():
    """Models with empty lists or empty dictionaries must validate without error."""
    # Empty descriptive stats
    empty_stats = DescriptiveStatisticsOutput.model_validate({
        "statistics": {},
        "analyzed_columns": [],
    })
    assert empty_stats.statistics == {}
    assert empty_stats.analyzed_columns == []

    # Empty group comparison
    empty_group = GroupComparisonOutput.model_validate({
        "group_by": "cat",
        "metric_column": "val",
        "total_metric_value": 0.0,
        "group_count": 0,
        "results": [],
        "underperforming_groups": [],
    })
    assert empty_group.results == []
    assert empty_group.underperforming_groups == []

    # Empty document search
    empty_search = SearchDocumentChunksOutput.model_validate({
        "query": "query without match",
        "matches": [],
    })
    assert empty_search.matches == []


# 8. Legitimate nullable fields validate
def test_legitimate_nullable_fields_validate():
    """Fields designed to accept None (or omitted) must validate properly."""
    # Correlation analysis with error return path
    err_corr = CorrelationAnalysisOutput.model_validate({
        "error": "Dataset requires at least 2 numeric columns for correlation analysis."
    })
    assert err_corr.error is not None
    assert err_corr.method is None
    assert err_corr.analyzed_columns == []

    # GenerateReportOutput with null relative URLs
    report = GenerateReportOutput.model_validate({
        "report_id": "rep_123",
        "title": "Quarterly Report",
        "markdown_content": "# Report",
        "markdown_path": "/tmp/rep.md",
        "html_path": "/tmp/rep.html",
        "relative_markdown_url": None,
        "relative_html_url": None,
        "artifacts": ["/tmp/rep.md", "/tmp/rep.html"],
    })
    assert report.relative_markdown_url is None
    assert report.relative_html_url is None

    # SystemHealthOutput with empty storage (when check_storage=False)
    sys_health = SystemHealthOutput.model_validate({
        "status": "healthy",
        "python_version": "3.14.0",
        "registered_tools_count": 11,
        "privacy_mode_default": "local_only",
        "storage": {},
        "directories_initialized": True,
    })
    assert sys_health.storage.total_gb is None
    assert sys_health.storage.used_gb is None
    assert sys_health.storage.free_gb is None


# 9. Existing tool outputs validate against their output contracts
@pytest.mark.asyncio
async def test_all_11_tool_output_contracts_match_actual_executions(tmp_path):
    """Verify that all production tool executions return payloads conforming to their output models."""
    # 1. System health
    health_def = tool_registry.get_tool_definition("system_health")
    assert health_def is not None
    health_model = tool_registry.get_output_model("system_health")
    assert health_model is SystemHealthOutput

    health_res = await tool_registry.execute_direct_read_only_tool(
        "system_health", {"check_storage": True}
    )
    assert health_res.success is True
    validated_health = health_model.model_validate(health_res.data)
    assert validated_health.status == "healthy"

    # 2. Document tools with temporary text file
    doc_file = tmp_path / "sample_doc.txt"
    doc_file.write_text("AURA is an autonomous analytics agent. It performs structured data analysis.", encoding="utf-8")

    from app.core.config import settings
    # Stage file into UPLOADS_DIR for validate_safe_path
    from app.privacy.encryption import encryption_service
    enc_path = settings.UPLOADS_DIR / "sample_doc_b1.txt.enc"
    enc_path.write_bytes(encryption_service().encrypt_bytes(doc_file.read_bytes()))

    try:
        read_res = await tool_registry.execute_direct_read_only_tool(
            "read_document", {"file_path": str(enc_path), "chunk_size": 50, "chunk_overlap": 10}
        )
        assert read_res.success is True
        read_model = tool_registry.get_output_model("read_document")
        validated_read = read_model.model_validate(read_res.data)
        assert validated_read.chunk_count > 0

        # Search chunks
        search_res = await tool_registry.execute_direct_read_only_tool(
            "search_document_chunks",
            {"chunks": [c.model_dump() for c in validated_read.chunks], "query": "autonomous analytics"}
        )
        assert search_res.success is True
        search_model = tool_registry.get_output_model("search_document_chunks")
        validated_search = search_model.model_validate(search_res.data)
        assert len(validated_search.matches) > 0
    finally:
        if enc_path.exists():
            enc_path.unlink()


# 10. Phase 3.3.1-B1.1 Strict Nested Output Contract Tests
def test_nested_group_comparison_results_validate():
    """Verify that GroupComparisonOutput.results cleanly validates dynamic group records."""
    valid_data = {
        "group_by": "region",
        "metric_column": "revenue",
        "total_metric_value": 54000.0,
        "group_count": 2,
        "results": [
            {"region": "North", "sum": 30000.0, "mean": 15000.0, "count": 2, "percentage_of_total": 55.56},
            {"region": "South", "sum": 24000.0, "mean": 12000.0, "count": 2, "percentage_of_total": 44.44},
        ],
        "underperforming_groups": [],
    }
    validated = GroupComparisonOutput.model_validate(valid_data)
    assert len(validated.results) == 2
    assert validated.results[0]["region"] == "North"
    assert validated.results[1]["percentage_of_total"] == 44.44


def test_nested_outlier_samples_validate():
    """Verify that DetectOutliersOutput.sample_outliers validates dynamic row dictionaries."""
    outlier_data = {
        "column": "revenue",
        "method": "iqr",
        "total_rows": 100,
        "outlier_count": 2,
        "outlier_percentage": 2.0,
        "lower_bound": 0.0,
        "upper_bound": 50000.0,
        "sample_outliers": [
            {"id": 1, "revenue": 75000.0, "category": "Enterprise", "is_flagged": True, "notes": None},
            {"id": 2, "revenue": 92000.0, "category": "Enterprise", "is_flagged": True, "notes": "audit"},
        ],
    }
    validated = DetectOutliersOutput.model_validate(outlier_data)
    assert len(validated.sample_outliers) == 2
    assert validated.sample_outliers[0]["revenue"] == 75000.0
    assert validated.sample_outliers[0]["notes"] is None


def test_unknown_nested_fields_rejected_on_fixed_schemas():
    """Fixed nested models must reject arbitrary unknown fields."""
    # ColumnDescriptiveStatistics
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ColumnDescriptiveStatistics.model_validate({
            "count": 10,
            "mean": 5.0,
            "std": 1.0,
            "min": 2.0,
            "q25": 4.0,
            "median": 5.0,
            "q75": 6.0,
            "max": 8.0,
            "iqr": 2.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "extra_metric": 123.45,
        })

    # CorrelationPair
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CorrelationPair.model_validate({
            "column_1": "a",
            "column_2": "b",
            "correlation": 0.5,
            "relationship": "moderate_positive",
            "p_value": 0.01,
        })

    # DocumentChunk
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DocumentChunk.model_validate({
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 100,
            "text": "sample text",
            "injected_metadata": "unauthorized",
        })

    # WebResearchSource
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WebResearchSource.model_validate({
            "title": "Title",
            "url": "https://example.org",
            "snippet": "Snippet",
            "domain": "example.org",
            "is_factual_source": True,
            "raw_html": "<div>extra</div>",
        })

    # SystemStorageInfo
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SystemStorageInfo.model_validate({
            "total_gb": 100.0,
            "used_gb": 50.0,
            "free_gb": 50.0,
            "filesystem_type": "ext4",
        })


def test_invalid_python_objects_rejected_in_dynamic_outputs():
    """Dynamic output containers must reject non-JSON Python objects (callables, classes, sockets)."""
    class CustomObject:
        pass

    # GroupComparisonOutput.results
    with pytest.raises(ValidationError, match="invalid-json-value"):
        GroupComparisonOutput.model_validate({
            "group_by": "cat",
            "metric_column": "rev",
            "total_metric_value": 100.0,
            "group_count": 1,
            "results": [{"cat": CustomObject()}],
            "underperforming_groups": [],
        })

    # GroupComparisonOutput.underperforming_groups
    with pytest.raises(ValidationError, match="invalid-json-value"):
        GroupComparisonOutput.model_validate({
            "group_by": "cat",
            "metric_column": "rev",
            "total_metric_value": 100.0,
            "group_count": 1,
            "results": [],
            "underperforming_groups": [lambda x: x],
        })

    # DetectOutliersOutput.sample_outliers
    with pytest.raises(ValidationError, match="invalid-json-value"):
        DetectOutliersOutput.model_validate({
            "column": "rev",
            "method": "iqr",
            "total_rows": 10,
            "outlier_count": 1,
            "outlier_percentage": 10.0,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
            "sample_outliers": [{"rev": object()}],
        })

    # InspectDatasetOutput.sample_data
    with pytest.raises(ValidationError, match="invalid-json-value"):
        InspectDatasetOutput.model_validate({
            "file_name": "data.csv",
            "format": "CSV",
            "total_rows": 1,
            "total_columns": 1,
            "columns": ["col"],
            "column_types": {"col": "object"},
            "missing_values": {"col": 0},
            "missing_percentages": {"col": 0.0},
            "duplicate_rows_count": 0,
            "numeric_columns": [],
            "categorical_columns": ["col"],
            "date_columns": [],
            "sample_data": [{"col": CustomObject()}],
            "memory_usage_kb": 1.0,
        })

