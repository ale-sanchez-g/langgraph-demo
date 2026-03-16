# SwagBot — LangGraph Multi-Agent E-Commerce Chatbot

SwagBot is a production-grade multi-agent e-commerce chatbot built with [LangGraph](https://github.com/langchain-ai/langgraph) and instrumented with [Datadog](https://www.datadoghq.com/) for full LLM observability. It intelligently routes customer requests to specialised AI agents that work in parallel to produce comprehensive, context-aware responses.

---

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Quick Start](#quick-start)
  - [1. Placeholder Product Images](#1-placeholder-product-images)
  - [2. Start the Services](#2-start-the-services)
- [Available Endpoints](#available-endpoints)
- [Project Structure](#project-structure)
- [LLM Platform Support](#llm-platform-support)
- [Error & Latency Simulation](#error--latency-simulation)
- [Running the Interaction Script](#running-the-interaction-script)
- [Datadog Observability](#datadog-observability)
- [Troubleshooting](#troubleshooting)

---

## Architecture

User requests flow through a four-stage LangGraph pipeline:

```
User Request
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  1. PLANNING AGENT                                  │
│     Analyses the request and decides which          │
│     specialist agents are needed.                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  2. ORCHESTRATOR AGENT                              │
│     Splits the request into focused sub-tasks,      │
│     one per specialist, and pre-loads the           │
│     relevant knowledge-base documents.              │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────┬──────────────┬──────────────┬────────┐
│  3. PARALLEL SPECIALIST AGENTS                      │
├──────────────┼──────────────┼──────────────┼────────┤
│ Customer     │ Product      │ Promotion    │Feedback│
│ Service      │ Specialist   │ Specialist   │Handler │
└──────────────┴──────────────┴──────────────┴────────┘
    │ (fan-in)
    ▼
┌─────────────────────────────────────────────────────┐
│  4. RESPONSE SYNTHESIZER                            │
│     Combines all specialist outputs into a          │
│     single, well-formatted HTML response.           │
└─────────────────────────────────────────────────────┘
    │
    ▼
 Final HTML response returned to user
```

Each specialist agent uses RAG (Retrieval-Augmented Generation) — it only sees the subset of the knowledge base relevant to its domain.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Docker ≥ 24 | |
| Docker Compose v2 | Included with Docker Desktop |
| AWS account | Required when `LLM_PLATFORM=bedrock` (default) |
| Datadog account | Required for APM / LLM Observability |
| Python 3.11+ | Only for the image-generation helper script |

> If you want to use a different LLM provider (OpenAI, Google Vertex AI, Azure OpenAI) see [LLM Platform Support](#llm-platform-support).

---

## Environment Variables

Copy the block below into a `.env` file in the project root and fill in your values. All variables marked **required** must be set before starting the stack.

```dotenv
# ─── AWS Bedrock ──────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID=          # required (bedrock)
AWS_SECRET_ACCESS_KEY=      # required (bedrock)
AWS_REGION=us-east-1

# ─── LLM Model Selection ──────────────────────────────────────────────────────
LLM_PLATFORM=bedrock        # bedrock | openai | vertex | azure
PLANNING_MODEL=anthropic.claude-3-haiku-20240307-v1:0
SPECIALIST_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
SYNTHESIZER_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

# ─── Flask ────────────────────────────────────────────────────────────────────
FLASK_HOST=0.0.0.0
FLASK_PORT=3000

# ─── Datadog ──────────────────────────────────────────────────────────────────
DD_SITE=datadoghq.com       # required — match your account site:
                            #   US1: datadoghq.com  |  US3: us3.datadoghq.com
                            #   EU:  datadoghq.eu   |  AP1: ap1.datadoghq.com
DD_API_KEY=                 # required — Org Settings → API Keys
DD_APP_KEY=                 # required — Org Settings → Application Keys (40-char hex)
DD_RUM_CLIENT_TOKEN=        # required for browser RUM
DD_RUM_APPLICATION_ID=      # required for browser RUM
DD_ENV=dev
DD_SERVICE=swagbot
DD_VERSION=2.0.0

# ─── Error / Latency Simulation (optional) ────────────────────────────────────
ERROR_SIMULATION=true
ERROR_SIMULATION_RATE=0.15
LATENCY_SIMULATION=true
MAX_LATENCY_MS=1000
```

> **Security note:** Never commit your `.env` file. It is listed in `.gitignore` by default.

---

## Quick Start

### 1. Placeholder Product Images

The application references product images under `app/static/images/`. The folder ships empty. Before starting the stack you must either:

**Option A — generate placeholder images (recommended for local development):**

```bash
# With Pillow installed (produces coloured labelled images)
pip install Pillow
python scripts/setup-images.py

# Without Pillow (produces minimal valid JPEG/PNG stubs)
python scripts/setup-images.py
```

The following images will be created:

| File | Product |
|------|---------|
| `headphones.jpg` | Dog Headphones |
| `steel-bottle.jpg` | Dog Steel Bottle |
| `t-shirt.jpg` | Dog T-Shirt |
| `t-shirt-i-love-u.jpg` | Dog T-Shirt I Love You |
| `sweatshirt-gray.jpg` | Dog Sweatshirt Gray |
| `sweatshirt-black.jpg` | Dog Sweatshirt Black |
| `notebook.jpg` | Dog Notebook |
| `sko-notebook.jpg` | Dog SKO Notebook |
| `mug.jpg` | Dog Mug |
| `hoodie.png` | Dog Hoodie |
| `plastic-bottle.png` | Dog Plastic Bottle |
| `sticker-pack.png` | Dog Sticker Pack |
| `beanie.png` | Dog Beanie |

**Option B — supply your own images:**

Drop real image files into `app/static/images/` using the filenames listed above.

---

### 2. Start the Services

```bash
# Full start (swagbot + Datadog Agent)
docker compose up -d

# Or use the helper scripts:
./restart-containers.sh   # Full restart of all services
./restart-bot.sh          # Restart only the swagbot container
```

Once running:

| URL | Description |
|-----|-------------|
| `http://localhost:3000` | Chat UI |
| `http://localhost:3000/health` | Health check |
| `http://localhost:3000/status` | Service status & configuration |
| `http://localhost:3000/config` | Active LLM platform & model names |
| `http://localhost:3000/categories` | Available request categories |

---

## Available Endpoints

### `POST /data`

Main chat endpoint.

```bash
curl -X POST http://localhost:3000/data \
  -H "Content-Type: application/json" \
  -d '{"data": "Do you have any promotions on headphones?"}'
```

Response: HTML string with the bot's answer.

---

### `POST /api/evaluate`

Submit thumbs-up / thumbs-down feedback linked to a previous response.

```bash
curl -X POST http://localhost:3000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "span_context": { ... },
    "evaluation_type": "thumbs_up",
    "user_request": "Do you have any promotions on headphones?",
    "response_text": "..."
  }'
```

---

### `GET /api/sample-requests`

Returns a list of pre-written sample requests from `app/resources/sample-requests.txt`.

---

## Project Structure

```
langgraph-demo/
├── docker-compose.yml              # Docker services (swagbot + Datadog agent)
├── restart-bot.sh                  # Quick-restart swagbot container
├── restart-containers.sh           # Full restart of all containers
├── scripts/
│   └── setup-images.py             # Generate placeholder product images ← run first
└── app/
    ├── Dockerfile                  # Python 3.11 image
    ├── requirements.txt            # Python dependencies
    ├── swagbot_app.py              # Flask app & REST API endpoints
    ├── swagbot_langgraph_workflow.py  # LangGraph agents & workflow graph
    ├── swagbot_langgraph_config.py    # Multi-platform LLM configuration
    ├── swagbot_utils.py            # HTML formatting & utility helpers
    ├── swagbot_utils_experiments.py   # Experimental utilities
    ├── resources/
    │   ├── products.json           # Product catalogue (name, price, image path)
    │   ├── faqs.json               # Frequently asked questions
    │   ├── promotions.json         # Active promotions & coupon codes
    │   ├── cs_info.json            # Customer service policies
    │   ├── sample-requests.txt     # Example user queries for the UI
    │   ├── prompt-planning.txt     # Planning agent system prompt
    │   ├── prompt-orchestrator.txt # Orchestrator agent system prompt
    │   ├── prompt-customer-service.txt
    │   ├── prompt-product-specialist.txt
    │   ├── prompt-promotion-specialist.txt
    │   ├── prompt-feedback-handler.txt
    │   ├── prompt-synthesizer.txt  # Final response synthesizer prompt
    │   └── prompt-metadata.json    # Prompt versioning metadata
    ├── scripts/
    │   └── interactions-script.sh  # Automated curl test requests
    └── static/
        └── images/                 # Product images (empty — see Quick Start)
```

---

## LLM Platform Support

Set `LLM_PLATFORM` in your `.env` to switch providers. Adjust the model names accordingly.

| Platform | `LLM_PLATFORM` value | Additional env vars needed |
|----------|----------------------|----------------------------|
| AWS Bedrock (default) | `bedrock` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Google Vertex AI | `vertex` | `GOOGLE_APPLICATION_CREDENTIALS`, `GCP_PROJECT`, `GCP_REGION` |
| Azure OpenAI | `azure` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` |

---

## Error & Latency Simulation

SwagBot includes a built-in simulation layer for resilience testing and LLM observability demos.

| Variable | Default | Description |
|----------|---------|-------------|
| `ERROR_SIMULATION` | `true` | Enable random errors |
| `ERROR_SIMULATION_RATE` | `0.15` | Fraction of requests that trigger an error (0–1) |
| `LATENCY_SIMULATION` | `true` | Add artificial latency |
| `MAX_LATENCY_MS` | `1000` | Maximum added latency in milliseconds |

Set both to `false` / `0` when you want clean production-like behaviour:

```dotenv
ERROR_SIMULATION=false
LATENCY_SIMULATION=false
```

---

## Running the Interaction Script

A ready-made interaction script fires a variety of test requests covering all agent types:

```bash
bash app/scripts/interactions-script.sh
```

This script is also executed automatically by the container on startup (after the Flask server starts) so Datadog receives an initial burst of traces.

---

## Datadog Observability

SwagBot is fully instrumented with:

| Feature | Details |
|---------|---------|
| **APM Tracing** | Every request is traced end-to-end via `ddtrace` |
| **LLM Observability** | Each LLM call is captured with input/output/tokens/latency |
| **RUM** | Browser session recording and page-load correlation |
| **Logs** | Structured JSON logs with trace-id injection |
| **Metrics** | Runtime metrics and custom business metrics |
| **User Feedback** | Thumbs-up/down evaluations linked to LLM spans |

The Datadog Agent sidecar (`agent` service in `docker-compose.yml`) collects APM traces on port `8126` and ships everything to Datadog.

To verify the agent is healthy:

```bash
docker compose exec agent agent status
```

---

## Troubleshooting

**Images not showing in responses**

Run the setup script to create the placeholder images, then restart the container so Flask can serve them:

```bash
python scripts/setup-images.py
./restart-bot.sh
```

**Container fails to start**

Check that all required environment variables are set:

```bash
docker compose config   # shows resolved config with all substitutions
```

**AWS credentials error**

Ensure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are exported in your shell or present in your `.env` file, and that the IAM user has `bedrock:InvokeModel` permission for the configured models.

**Datadog traces not appearing**

1. Confirm `DD_API_KEY` is valid.
2. Check the agent is running: `docker compose ps`.
3. Verify `DD_AGENT_HOST=agent` is set in the swagbot service — it must resolve to the Datadog Agent container.

**Port 3000 already in use**

```bash
lsof -i :3000        # find the process
kill -9 <PID>
docker compose up -d
```


## Experiments

The `swagbot_utils_experiments.py` script contains utilities for dataset creation, evaluation, and other experiments. For example, to create a dataset of 100 requests related to the "product specialist" category:

```bash
docker compose exec swagbot python /app/swagbot_utils_experiments.py --create-dataset product_specialist
```

To compare the performance of all three agents (customer service, product specialist, promotion specialist) on the "product specialist" dataset using direct agent prompting:

```bash
docker compose exec swagbot python /app/swagbot_utils_experiments.py \
  --compare-all-models \
  --dataset product_specialist \
  --direct-agent
  ```