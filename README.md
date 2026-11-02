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

## Phase 1 Security Configuration

The backend supports a local single-user deployment. It binds to loopback only; set `HOST` to `127.0.0.1` or `::1`. The frontend origins are controlled by `CORS_ORIGINS` (JSON list syntax), and sensitive settings and direct tool endpoints require an allowed `Origin`.

Set a persistent, private Fernet key in `ENCRYPTION_KEY` before using uploads, reports, charts, or task persistence. Fernet here provides AES-128-CBC encryption with HMAC-SHA256 authentication. Generate a URL-safe key with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Keep the key outside source control and preserve it: encrypted files and checkpoints cannot be read after the key is lost or changed. Upload staging files contain ciphertext only and are removed after atomic placement; plaintext upload bytes and report/chart data are processed in memory. Existing legacy plaintext artifacts are preserved until an operator explicitly calls `POST /api/artifacts/migrate-legacy` from a configured frontend origin. Migration encrypts files in place and leaves a file untouched if conversion fails.

`local_only` blocks outbound requests through AURA's shared HTTPX client boundary. Gemini and Wikipedia research use that client. This is an application-level control: unrelated code or a future integration that creates a raw HTTPX/requests client or uses a network SDK can bypass it; use OS or network egress controls when a process-wide guarantee is required. `hybrid` redacts recognizable PII from inspected outbound text and omits raw row fields; opaque binary and multipart uploads are denied. Cloud egress is available only for tasks explicitly run in `hybrid` or `cloud_enabled` mode.

The current deployment has one local OS user and no multi-user identity or ownership model. Do not bind it to a network interface or expose it through a reverse proxy as a multi-user service.

File validation resolves paths before checking the upload root, which is intended to reject symlinks that point outside it. The symlink escape test is skipped on Windows when the process cannot create symlinks; that behavior still needs verification on a host with symlink privileges.

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

