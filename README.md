# AURA: Autonomous Understanding, Reasoning & Analytics

**AURA** is an enterprise-grade, privacy-aware autonomous AI agent platform designed around the foundational architectural principle:
> **"Use an LLM as the reasoning brain, specialized small AI models for narrow ML tasks, and deterministic tools for actual execution."**

AURA decomposes complex, high-level natural language requests into structured execution plans (DAGs), selects tools and models based on capability and privacy boundaries, executes deterministic calculations on-premise, validates outputs through an evaluator, and synthesizes publication-grade deliverables.

---

## 🏛️ System Architecture

```
                    ┌────────────────────────────┐
                    │  User Prompt / Dataset     │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │   Privacy Mode Validator   │
                    │ (Local / Hybrid / Cloud)   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │      Task Planner          │
                    │  (Gemini / Local Reasoner) │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │     Dependency DAG Loop    │
                    │   (Plan & Execute / ReAct) │
                    └─────────────┬──────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐     ┌──────────────────┐
│  Dataset Tools   │    │ Analytics Engine │     │  Visualizations  │
│  CSV, XLSX, XLS  │    │ Pandas / SciPy   │     │  Matplotlib PNGs │
└──────────────────┘    └──────────────────┘     └──────────────────┘
         ▲                        ▲                        ▲
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │  Step & Task Evaluator     │
                    │  (Schema, Utility, Retry)  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │    Report Synthesis        │
                    │   Markdown / HTML / Charts │
                    └────────────────────────────┘
```

---

## 🛡️ Three Privacy Execution Modes

1. **Mode 1 — Local Only (`local_only`)**:
   - Zero external cloud calls.
   - Private files, data frames, and calculations remain 100% on the local host.
   - LLM reasoning falls back to the local deterministic planner.
   
2. **Mode 2 — Hybrid (`hybrid`)**:
   - Raw datasets, records, and proprietary rows are strictly retained locally.
   - Only minimized, sanitized schemas and anonymized statistical summaries are transmitted for cloud reasoning.
   - PII redactor scrubs emails, phone numbers, and IP addresses.

3. **Mode 3 — Cloud Enabled (`cloud_enabled`)**:
   - Enables cloud reasoning via Google Gemini API.
   - Governed by configurable timeouts and audit logging.

---

## 🛠️ Registered Deterministic Tools

| Tool Name | Category | Description | Execution Location |
| :--- | :--- | :--- | :--- |
| `inspect_dataset` | Dataset | Analyzes schema, dtypes, nulls, duplicates, and memory usage of CSV/Excel files. | Local |
| `descriptive_statistics` | Analytics | Calculates mean, median, IQR, skewness, kurtosis, quartiles for numeric fields. | Local |
| `group_comparison` | Analytics | Aggregates metrics (sum, mean, count) across categorical dimensions to find underperforming areas. | Local |
| `correlation_analysis` | Analytics | Pearson/Spearman matrix and prominent relationship ranking. | Local |
| `detect_outliers` | Analytics | Statistical anomaly detection via IQR and Z-score methods. | Local |
| `generate_chart` | Visualization | Renders high-res Matplotlib bar, line, scatter, box, and heatmap visualizations. | Local |
| `web_research` | Research | Retrieves cited facts and sources with attribution. | Cloud / Fallback |
| `read_document` | Document | Chunks and previews text documents for analysis. | Local |
| `generate_report` | Report | Synthesizes findings, embedded charts, and recommendations into Markdown and HTML. | Local |
| `system_health` | System | Reports operational status, storage, and registered capability metrics. | Local |

---

## ⚡ Quickstart

### Backend Setup
```bash
# 1. Activate Python virtual environment
.\venv\Scripts\activate

# 2. Start the FastAPI backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Docs available at: `http://127.0.0.1:8000/docs`

### Frontend Setup
```bash
cd frontend
npm run dev
```
Dashboard available at: `http://localhost:3000`

---

## 🧪 Running Automated Tests
```bash
.\venv\Scripts\pytest -v backend/tests/
```
