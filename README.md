# SwagBot - LangGraph Multi-Agent E-Commerce Chatbot

SwagBot is a multi-agent e-commerce chatbot built with LangGraph and instrumented with Datadog APM + LLM Observability. Requests are analyzed, routed to specialized agents in parallel, and synthesized into a single HTML response.

## Table of Contents

- [Architecture](#architecture)
- [Implemented Functionality](#implemented-functionality)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [LLM Platform Support](#llm-platform-support)
- [Open Prompt Manager (OPM)](#open-prompt-manager-opm)
- [Error and Latency Simulation](#error-and-latency-simulation)
- [Interaction Traffic Generator](#interaction-traffic-generator)
- [Experiments](#experiments)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Architecture

SwagBot uses this workflow pattern:

1. Planning Agent: classifies request and chooses specialists.
2. Orchestrator Agent: creates focused tasks and loads selective knowledge sources.
3. Specialist Agents (parallel):
   - Customer Service
   - Product Specialist
   - Promotion Specialist
   - Feedback Handler
4. Response Synthesizer: combines specialist responses into final HTML.

Each specialist runs retrieval + generation for its own scope. This keeps responses focused and improves trace clarity.

## Implemented Functionality

- Multi-platform LLM runtime selected by `LLM_PLATFORM` (`bedrock`, `openai`, `vertex`, `azure`).
- Per-role model selection (`PLANNING_MODEL`, `SPECIALIST_MODEL`, `SYNTHESIZER_MODEL`).
- Selective RAG loading based on routed agents (instead of loading all documents every time).
- Datadog instrumentation for workflow, agents, retrieval, and LLM calls.
- Frontend-to-backend trace correlation via Datadog headers and `span_context` payload.
- User feedback ingestion endpoint (`/api/evaluate`) that submits score metrics to Datadog LLM Obs intake.
- Prompt tracking and optional Open Prompt Management (OPM) integration via `OPM_BASE_URL`.
- Runtime UI configuration endpoint (`/config`) exposing active models, logos, and display metadata.
- Experiment suite for dataset creation and model comparison (`swagbot_utils_experiments.py`).

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker Desktop | Includes Docker Compose v2 |
| Datadog account | Required for APM, LLM Obs, and evaluation API |
| AWS account | Required only for `LLM_PLATFORM=bedrock` |
| Provider credentials | Required for non-bedrock platforms (OpenAI, Vertex, Azure) |
| Open Prompt Manager (OPM) | Optional — enables remote prompt management; falls back to local files when unavailable |

## Environment Variables

Create a `.env` file at repository root.

```dotenv
# Flask
FLASK_HOST=0.0.0.0
FLASK_PORT=3000

# LLM platform and models
LLM_PLATFORM=bedrock                 # bedrock | openai | vertex | azure
PLANNING_MODEL=anthropic.claude-3-haiku-20240307-v1:0
SPECIALIST_MODEL=anthropic.claude-3-haiku-20240307-v1:0
SYNTHESIZER_MODEL=anthropic.claude-3-haiku-20240307-v1:0

# Bedrock
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1

# OpenAI
OPENAI_API_KEY=

# Vertex AI
GOOGLE_PROJECT_ID=
GOOGLE_LOCATION=us-central1

# Azure OpenAI
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=2024-02-01

# Datadog
DD_API_KEY=
DD_APP_KEY=                           # passed into container as SWAGBOT_DD_APP_KEY
DD_SITE=datadoghq.com
DD_ENV=dev
DD_SERVICE=swagbot
DD_VERSION=2.0.0
DD_RUM_CLIENT_TOKEN=
DD_RUM_APPLICATION_ID=

# Optional UI/RUM helper
SWAGBOT_URL=

# LLM Observability toggles (set in docker-compose by default)
DD_LLMOBS_ENABLED=1
DD_LLMOBS_ML_APP=swagbot
DD_LLMOBS_AGENTLESS_ENABLED=0

# Prompt management
OPM_BASE_URL=http://host.docker.internal

# Error/latency simulation
ERROR_SIMULATION=true
ERROR_SIMULATION_RATE=0.15
LATENCY_SIMULATION=true
MAX_LATENCY_MS=1000
```

Notes:

- App-level defaults in code are not always the same as docker-compose overrides.
- For Vertex, configuration keys are `GOOGLE_PROJECT_ID` and `GOOGLE_LOCATION`.

## Quick Start

```bash
# Start all services
docker compose up -d

# Or use helper scripts
./restart-containers.sh
./restart-bot.sh
```

Services:

- Chat UI: http://localhost:3000
- Health: http://localhost:3000/health
- Runtime status: http://localhost:3000/status
- Active model/config view: http://localhost:3000/config
- Categories: http://localhost:3000/categories

## API Endpoints

### POST /data

Main chat endpoint.

Request:

```bash
curl -X POST http://localhost:3000/data \
  -H "Content-Type: application/json" \
  -d '{"data": "Do you have any promotions on headphones?"}'
```

Response is JSON (not a raw HTML string). Important fields include:

- `response`: final rendered HTML message.
- `output`: workflow output text.
- `category`: planner category.
- `agents_needed`: selected specialists.
- `documents` and `retrieved_count`: RAG evidence summary.
- `agent_outputs`: per-agent outputs.
- `workflow_path`: execution path.
- `span_context`: trace metadata for feedback correlation.

### GET /api/sample-requests

Returns sample requests loaded from `app/resources/sample-requests.txt`.

### POST /api/evaluate

Submits user feedback to Datadog LLM Obs evaluation intake.

Request requirements:

- `evaluation_type`: must be `thumbs_up` or `thumbs_down`.
- `span_context`: required (`trace_id` and `span_id` used to join metric to span).

Example:

```bash
curl -X POST http://localhost:3000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "span_context": {"trace_id": "...", "span_id": "..."},
    "evaluation_type": "thumbs_up",
    "user_request": "Do you have any promotions on headphones?",
    "response_text": "..."
  }'
```

### GET /

Renders the web UI from `app/templates/index.html` using dynamic platform and model metadata.

### GET /health

Basic health payload with service and workflow markers.

### GET /status

Returns service-level status, Datadog status, and configured categories/provider summary.

### GET /config

Returns current UI/runtime model configuration, platform branding fields, and workflow type.

### GET /categories

Returns category list from runtime configuration.

## LLM Platform Support

| Platform | `LLM_PLATFORM` | Required variables |
|---|---|---|
| AWS Bedrock | `bedrock` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Google Vertex AI | `vertex` | `GOOGLE_PROJECT_ID`, `GOOGLE_LOCATION` |
| Azure OpenAI | `azure` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` |

## Open Prompt Manager (OPM)

SwagBot integrates with [Open Prompt Manager](https://github.com/ale-sanchez-g/open-prompt-manager) to load and version agent prompts at runtime. When OPM is reachable, every agent fetches its prompt from OPM instead of the local `resources/prompt-*.txt` files. If OPM is unreachable or the prompt is not found, SwagBot falls back to the local files automatically — the service keeps running either way.

### What OPM provides

- Centralised, versioned storage for all agent prompts.
- REST API for prompt retrieval (`GET /api/prompts/`) and execution telemetry (`POST /api/prompts/{id}/executions`).
- A web UI for editing, tagging, and tracking prompt quality metrics.

### Running OPM

OPM is a separate Docker service. Clone and start it alongside SwagBot:

```bash
git clone https://github.com/ale-sanchez-g/open-prompt-manager
cd open-prompt-manager
docker-compose up -d
```

Services started by OPM:

- Backend API: http://localhost:8000/api
- API docs: http://localhost:8000/api/docs
- Web UI: http://localhost:80

### Connecting SwagBot to OPM

Set `OPM_BASE_URL` in your `.env` to point to the running OPM backend:

```dotenv
# Local development (OPM running on the same machine)
OPM_BASE_URL=http://localhost

# Docker Compose (OPM running on the host, SwagBot in a container)
OPM_BASE_URL=http://host.docker.internal
```

The `docker-compose.yml` in this project already sets `OPM_BASE_URL=http://host.docker.internal` so the SwagBot container can reach OPM running on the host without any extra configuration.

### Prompt name mapping

SwagBot looks up each agent prompt in OPM by name:

| Agent | OPM prompt name |
|---|---|
| Planning | `planning` |
| Orchestrator | `orchestrator` |
| Customer Service | `customer-service` |
| Product Specialist | `product-specialist` |
| Promotion Specialist | `promotion-specialist` |
| Feedback Handler | `feedback-handler` |
| Response Synthesizer | `synthesizer` |

Create prompts with these exact names in OPM so that SwagBot picks them up automatically.

---

## Error and Latency Simulation

The workflow supports simulated failures and delay injection.

| Variable | Effect |
|---|---|
| `ERROR_SIMULATION` | Enables simulated errors in workflow paths |
| `ERROR_SIMULATION_RATE` | Error probability (0.0 to 1.0) |
| `LATENCY_SIMULATION` | Enables injected latency |
| `MAX_LATENCY_MS` | Maximum artificial delay |

In `docker-compose.yml`, these are enabled by default for demo/observability behavior.

## Interaction Traffic Generator

`app/scripts/interactions-script.sh` runs continuously in a loop and sends diverse requests (including adversarial/sensitive patterns) to generate telemetry.

Run manually:

```bash
bash app/scripts/interactions-script.sh
```

Container behavior:

- The `swagbot` service entrypoint starts Flask, waits briefly, then launches this script.
- The script keeps running (it is not a one-time startup burst).

## Experiments

`app/swagbot_utils_experiments.py` provides model and dataset tooling:

- List models: `--list-models`
- List datasets: `--list-datasets`
- Create all datasets: `--create-datasets`
- Create one dataset: `--create-dataset {customer_service|product_specialist|promotion_specialist|comprehensive}`
- Compare one model: `--model-comparison <model_key>`
- Compare selected models: `--compare-models <k1> <k2> ...`
- Compare all models: `--compare-all-models`
- Use direct-agent mode: `--direct-agent`

Examples:

```bash
docker compose exec swagbot python /app/swagbot_utils_experiments.py --list-models

docker compose exec swagbot python /app/swagbot_utils_experiments.py \
  --create-dataset product_specialist

docker compose exec swagbot python /app/swagbot_utils_experiments.py \
  --compare-all-models \
  --dataset product_specialist \
  --direct-agent
```

## Project Structure

```
langgraph-demo/
├── docker-compose.yml
├── restart-bot.sh
├── restart-containers.sh
└── app/
    ├── Dockerfile
    ├── requirements.txt
    ├── swagbot_app.py
    ├── swagbot_langgraph_config.py
    ├── swagbot_langgraph_workflow.py
    ├── swagbot_utils.py
    ├── swagbot_utils_experiments.py
    ├── resources/
    │   ├── products.json
    │   ├── promotions.json
    │   ├── faqs.json
    │   ├── cs_info.json
    │   ├── sample-requests.txt
    │   ├── prompt-planning.txt
    │   ├── prompt-orchestrator.txt
    │   ├── prompt-customer-service.txt
    │   ├── prompt-product-specialist.txt
    │   ├── prompt-promotion-specialist.txt
    │   ├── prompt-feedback-handler.txt
    │   ├── prompt-synthesizer.txt
    │   └── prompt-metadata.json
    ├── scripts/
    │   └── interactions-script.sh
    ├── static/
    │   └── images/
    └── templates/
        └── index.html
```

## Troubleshooting

### Containers fail to start

```bash
docker compose config
docker compose ps
docker compose logs swagbot --tail=200
```

### Datadog traces or eval submissions missing

1. Confirm `DD_API_KEY` is set.
2. Confirm agent is healthy: `docker compose exec agent agent status`.
3. Confirm `DD_SITE` matches your Datadog region.
4. For `/api/evaluate`, confirm payload includes valid `span_context` and `evaluation_type`.

### Port 3000 conflict

```bash
lsof -i :3000
kill -9 <PID>
docker compose up -d
```