"""
SwagBot LangGraph Flask Application
Clean implementation using LangGraph for better observability and RAG annotation
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
from typing import Dict, Any, Optional

# Datadog imports - Basic APM and logging injection only
import ddtrace
from ddtrace import tracer

# Import our LangGraph modules
from swagbot_langgraph_config import config
from swagbot_utils import UIConfigUtils

# ===============================================================================
# FLASK APPLICATION SETUP
# ===============================================================================

# Configure logging with Datadog correlation fields
FORMAT = ('%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] '
          '[dd.service=%(dd.service)s dd.env=%(dd.env)s dd.version=%(dd.version)s dd.trace_id=%(dd.trace_id)s dd.span_id=%(dd.span_id)s] '
          '- %(message)s')
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configure CORS for RUM and APM correlation compatibility
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Content-Type", "Authorization", "X-Requested-With",
            # Datadog tracing headers for RUM-APM correlation
            "x-datadog-trace-id", "x-datadog-parent-id", "x-datadog-origin", 
            "x-datadog-sampling-priority", "traceparent", "tracestate"
        ],
        "expose_headers": [
            "Content-Range", "X-Content-Range",
            # Expose Datadog trace headers for RUM-APM correlation
            "x-datadog-trace-id", "x-datadog-parent-id", "x-datadog-origin",
            "x-datadog-sampling-priority", "traceparent", "tracestate"
        ]
    }
})

# Add global headers to help with CORB issues
@app.after_request
def after_request(response):
    """Add headers to all responses to help with CORB and security"""
    # Add CORB-friendly headers as recommended by Chromium documentation
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    
    # Add Datadog trace headers for RUM-APM correlation
    # This allows the frontend RUM to link browser sessions with backend traces
    if hasattr(ddtrace.tracer, 'current_span'):
        current_span = ddtrace.tracer.current_span()
        if current_span:
            response.headers['x-datadog-trace-id'] = str(current_span.trace_id)
            response.headers['x-datadog-parent-id'] = str(current_span.span_id)
            if current_span.context.sampling_priority is not None:
                response.headers['x-datadog-sampling-priority'] = str(current_span.context.sampling_priority)
    
    return response

# Import the consolidated workflow
from swagbot_langgraph_workflow import swagbot_workflow as active_workflow
logger.info("🎯 Using Optimized Multi-Agent Workflow (LangGraph Best Practices)")

# ===============================================================================
# HELPER FUNCTIONS
# ===============================================================================

def _annotate_error_span(e: Exception, user_request: str) -> tuple:
    """
    Annotate error span with detailed information.
    Returns tuple of (error_type, error_code, error_message) for response.
    """
    error_type = e.__class__.__name__
    error_code = getattr(e, 'response', {}).get('Error', {}).get('Code', 'UnknownError') if hasattr(e, 'response') else 'UnknownError'
    error_message = str(e)
    
    if not hasattr(ddtrace.tracer, 'current_span'):
        return (error_type, error_code, error_message)
    
    try:
        flask_span = ddtrace.tracer.current_span()
        if not flask_span:
            return (error_type, error_code, error_message)
        
        # Basic error tags
        flask_span.error = 1
        flask_span.set_tag("input.raw", user_request)
        flask_span.set_tag("output.raw", "I'm sorry, I encountered an error processing your request.")
        flask_span.set_tag("endpoint", "/data")
        
        # Enhanced error details
        flask_span.set_tag("error.message", error_message)
        flask_span.set_tag("error.type", error_type)
        flask_span.set_tag("error.code", error_code)
        
        # Service context
        flask_span.set_tag("service", "swagbot-langgraph")
        flask_span.set_tag("workflow_type", "langgraph")
        flask_span.set_metric("request_failed", 1)
        
        # Token-related error handling (combined conditions to reduce nesting)
        is_token_error = "token" in error_message.lower() or "length" in error_message.lower()
        if is_token_error:
            flask_span.set_tag("error.category", "token_limit")
            if hasattr(e, 'token_count'):
                flask_span.set_metric("token_count", e.token_count)
        
        logger.debug("📊 Annotated error span with enhanced details")
    except Exception as annotation_error:
        logger.warning(f"Could not annotate error span: {annotation_error}")
    
    return (error_type, error_code, error_message)

# ===============================================================================
# CORE APPLICATION ENDPOINTS
# ===============================================================================

@app.route('/data', methods=['POST'])
def get_data():
    """
    Main data endpoint - processes requests using LangGraph workflow
    """
    try:
        # Parse incoming request
        data = request.get_json()
        user_request = data.get('data', '').strip()
        
        if not user_request:
            return jsonify({"error": "Data is required"}), 400
        
        logger.info(f"Processing request: {user_request[:50]}...")
        
        # Process request using selected workflow
        result = active_workflow.process_request(user_request)
        
        # The response is already in HTML format, no parsing needed
        response_content = result.get("output", "")
        
        # Clean up any remaining artifacts or ensure HTML is properly formatted  
        if response_content:
            # Remove any accidental "Final" prefixes if they slip through
            response_content = response_content.strip()
            if response_content.startswith('"Final":'):
                # Just clean the prefix since we no longer use "Final" format
                response_content = response_content.replace('"Final":', '').strip()
        
        # Update the result with the cleaned response
        result["response"] = response_content
        
        # Check if there was an error in the workflow result
        if result.get("error"):
            # Mark the current span as error
            if hasattr(ddtrace.tracer, 'current_span'):
                flask_span = ddtrace.tracer.current_span()
                if flask_span:
                    error_message = result["error"]
                    error = Exception(error_message)  # Create an exception object
                    flask_span.error = 1
                    flask_span.set_exc_info(error, error, None)  # This sets both error message and stack trace
                    flask_span.set_tag("error.type", "WorkflowError")
                    flask_span.set_tag("error.code", "500")
            # Add span context for RUM correlation
            if hasattr(ddtrace.tracer, 'current_span'):
                flask_span = ddtrace.tracer.current_span()
                if flask_span:
                    result["span_context"] = {
                        "trace_id": str(flask_span.trace_id),
                        "span_id": str(flask_span.span_id)
                    }
            
            # Return 500 Internal Server Error for workflow errors
            return jsonify(result), 500
        
        # Annotate the Flask endpoint span with input/output
        if hasattr(ddtrace.tracer, 'current_span'):
            try:
                flask_span = ddtrace.tracer.current_span()
                if flask_span:
                    flask_span.set_tag("input.raw", user_request)
                    flask_span.set_tag("output.raw", result["response"])
                    flask_span.set_tag("endpoint", "/data")
                    flask_span.set_tag("service", "swagbot-langgraph")
                    flask_span.set_tag("workflow_type", "langgraph")
                    flask_span.set_metric("request_processed", 1)
                    logger.debug("📊 Annotated Flask endpoint span")
                    
                    # Add span context to result
                    result["span_context"] = {
                        "trace_id": str(flask_span.trace_id),
                        "span_id": str(flask_span.span_id)
                    }
            except Exception as e:
                logger.warning(f"Could not annotate Flask endpoint span: {e}")
        
        logger.info("✅ Request processed successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error processing request: {e}", exc_info=True)
        
        # Annotate error span and get error details
        error_type, error_code, error_message = _annotate_error_span(
            e, 
            user_request if 'user_request' in locals() else "Invalid request"
        )
        
        # Return detailed error response
        return jsonify({
            "error": error_type,
            "code": error_code,
            "message": error_message,
            "details": {
                "type": error_type,
                "code": error_code,
                "message": error_message,
                "workflow_type": "langgraph"
            }
        }), 500


@app.route('/api/sample-requests', methods=['GET'])
def get_sample_requests():
    """
    Get sample requests endpoint - returns sample requests as JSON
    """
    try:
        sample_requests = load_sample_requests()
        return jsonify(sample_requests)
    except Exception as e:
        logger.error(f"Error getting sample requests: {e}")
        return jsonify(["How can I help you today?"]), 500


@app.route('/api/evaluate', methods=['POST'])
def submit_evaluation():
    """
    Submit user evaluation (thumbs up/down) to Datadog LLM Observability
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        span_context = data.get('span_context')  # Complete span context from workflow
        evaluation_type = data.get('evaluation_type')  # 'thumbs_up' or 'thumbs_down'
        user_request = data.get('user_request', '')
        response_text = data.get('response_text', '')
        
        # Validate evaluation_type first
        if not evaluation_type:
            return jsonify({"error": "evaluation_type is required"}), 400
        
        if evaluation_type not in ['thumbs_up', 'thumbs_down']:
            return jsonify({"error": "evaluation_type must be 'thumbs_up' or 'thumbs_down'"}), 400
        
        # Check if we have span context
        if not span_context:
            logger.error(f"❌ Cannot submit evaluation: span_context not available from LangGraph workflow")
            raise ValueError("Evaluation submission requires span_context from LangGraph workflow")
        
        # Convert to Datadog evaluation format
        evaluation_score = 1.0 if evaluation_type == 'thumbs_up' else 0.0
        evaluation_label = "user_satisfaction"
        
        # Submit evaluation using Datadog API
        import requests
        import uuid
        import time
        import os

        # Prepare evaluation payload for Datadog API
        evaluation_payload = {
            "data": {
                "type": "evaluation_metric",
                "attributes": {
                    "metrics": [{
                        "join_on": {
                            "span": {
                                "span_id": str(span_context.get("span_id")),
                                "trace_id": str(span_context.get("trace_id"))
                            }
                        },
                        "ml_app": "swagbot",
                        "timestamp_ms": int(time.time() * 1000),
                        "metric_type": "score",
                        "label": evaluation_label,
                        "score_value": evaluation_score,
                        "source": "Swagbot UI"
                    }]
                }
            }
        }

        # Get API key and site
        api_key = os.getenv('DD_API_KEY')
        dd_site = os.getenv('DD_SITE', 'datadoghq.com')

        if not api_key:
            raise ValueError("DD_API_KEY required for evaluation submission")

        # Submit to Datadog API
        headers = {
            'Content-Type': 'application/json',
            'DD-API-KEY': api_key
        }

        api_url = f"https://api.{dd_site}/api/intake/llm-obs/v2/eval-metric"

        response = requests.post(
            api_url,
            json=evaluation_payload,
            headers=headers,
            timeout=10
        )

        if response.status_code in [200, 201, 202]:
            logger.info(f"✅ Evaluation submitted successfully via API")
        else:
            logger.error(f"API submission failed: {response.status_code} - {response.text}")
            raise ValueError(f"Evaluation submission failed: {response.status_code}")

        logger.info(f"📝 User feedback received: {evaluation_type} (score: {evaluation_score})")
        logger.info(f"✅ Feedback logged successfully: {evaluation_type}")
        
        return jsonify({
            "success": True,
            "message": f"Feedback '{evaluation_type}' submitted successfully to Datadog",
            "span_context": span_context,
            "evaluation_score": evaluation_score
        })
    
    except Exception as e:
        logger.error(f"❌ Error processing evaluation submission: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


@app.route('/')
def index():
    """Render the main SwagBot interface with dynamic configuration"""
    ui_config = get_ui_config()
    response = make_response(render_template('index.html', **ui_config))
    
    # Add headers to help with RUM CORB issues
    response.headers['Cross-Origin-Embedder-Policy'] = 'unsafe-none'
    response.headers['Cross-Origin-Opener-Policy'] = 'unsafe-none'
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    
    # Add proper Content-Type and nosniff headers as recommended by Chromium CORB docs
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    return response


# ===============================================================================
# ESSENTIAL MONITORING & CONFIGURATION ENDPOINTS
# ===============================================================================

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "swagbot-langgraph",
        "datadog_enabled": get_datadog_status().get("enabled", False),
        "workflow_type": "langgraph"
    })


@app.route('/status')
def status():
    """Status endpoint showing configuration"""
    return jsonify({
        "service": "swagbot-langgraph",
        "workflow_type": "langgraph",
        "datadog": get_datadog_status(),
        "config": {
            "available_categories": ["Customer-Service", "Product-Information", "Promotions", "Feedback", "Other"],
            "llm_provider": config.llm_platform.upper()
        }
    })


@app.route('/config')
def get_config():
    """Get current configuration for debugging"""
    try:
        ui_config = get_ui_config()
        return jsonify({
            "platform": ui_config["platform"]["name"],
            "platform_logo": ui_config["platform"]["logo"],
            "cloud_provider_logo": ui_config["platform"]["cloud_provider_logo"],
            "primary_logo": ui_config["primary_logo"],
            "bot_name": ui_config["bot_name"],
            "welcome_message": ui_config["welcome_message"],
            "planning": {
                "model": ui_config["agents"]["planning"]["model"],
                "description": ui_config["agents"]["planning"]["description"],
                "model_id": ui_config["agents"]["planning"]["model_id"],
                "friendly_name": ui_config["agents"]["planning"]["friendly_name"]
            },
            "specialists": {
                "model": ui_config["agents"]["specialists"]["model"], 
                "description": ui_config["agents"]["specialists"]["description"],
                "model_id": ui_config["agents"]["specialists"]["model_id"],
                "friendly_name": ui_config["agents"]["specialists"]["friendly_name"]
            },
            "dual_model": ui_config["show_dual_model"],
            "theme_color": ui_config["theme_color"],
            "logo_path": ui_config["logo_path"],
            "workflow_type": "langgraph"
        })
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({"error": "Failed to get configuration"}), 500


@app.route('/categories', methods=['GET'])
def get_categories():
    """Get available categories"""
    try:
        categories = getattr(config, 'CATEGORIES', [])
        return jsonify({"categories": categories})
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return jsonify({"error": "Failed to get categories"}), 500


# ===============================================================================
# UTILITY ENDPOINTS
# ===============================================================================

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return app.send_static_file(filename)


# ===============================================================================
# CONFIGURATION & SUPPORT FUNCTIONS
# ===============================================================================

def get_rum_config():
    """
    Get RUM configuration from environment variables
    Returns dict with RUM settings for frontend template
    """
    application_id = os.getenv('DD_RUM_APPLICATION_ID', '').strip()
    client_token = os.getenv('DD_RUM_CLIENT_TOKEN', '').strip()
    swagbot_url = os.getenv('SWAGBOT_URL', '').strip()
    
    # RUM is enabled only if both required credentials are provided
    enabled = bool(application_id and client_token)
    
    # Debug logging
    logger.info(f"🔍 RUM Config Debug:")
    logger.info(f"  - Application ID: {'***' + application_id[-8:] if application_id else 'NOT SET'}")
    logger.info(f"  - Client Token: {'***' + client_token[-8:] if client_token else 'NOT SET'}")
    logger.info(f"  - SwagBot URL: {swagbot_url}")
    logger.info(f"  - RUM Enabled: {enabled}")
    
    return {
        'enabled': enabled,
        'application_id': application_id,
        'client_token': client_token,
        'swagbot_url': swagbot_url
    }


def load_sample_requests() -> list:
    """Load sample requests from resources/sample-requests.txt file"""
    try:
        sample_requests_path = os.path.join(os.path.dirname(__file__), 'resources', 'sample-requests.txt')
        with open(sample_requests_path, 'r', encoding='utf-8') as file:
            # Read all lines, strip whitespace, and filter out empty lines
            requests = [line.strip() for line in file.readlines() if line.strip()]
        logger.info(f"✅ Loaded {len(requests)} sample requests from file")
        return requests
    except FileNotFoundError:
        logger.warning("⚠️  Sample requests file not found, using fallback requests")
        # Fallback sample requests if file is missing
        return [
            "What's the price of the dog steel bottle and are there any promotions running at the moment?",
            "I'd like to buy headphones but want to see customer reviews and ratings first",
            "I bought a mug but it arrived broken, can I get a replacement and see similar products?",
            "I want to return an item I bought on sale, what's your return policy for discounted items?",
            "Show me your best selling t-shirts, any current deals available, and what customers are saying about them"
        ]
    except Exception as e:
        logger.error(f"❌ Error loading sample requests: {e}")
        return ["How can I help you today?"]  # Minimal fallback


def get_ui_config() -> Dict[str, Any]:
    """Get UI configuration based on current setup"""
    # Get configurations for each agent type
    planning_config = config.get_planning_config()
    specialist_config = config.get_specialist_config()
    synthesizer_config = config.get_synthesizer_config()
    
    # Get model IDs from configuration
    planning_model = planning_config.get("model_id")
    specialist_model = specialist_config.get("model_id")
    synthesizer_model = synthesizer_config.get("model_id")
    

    
    # Get display names for all models using utilities
    planning_display = UIConfigUtils.get_friendly_model_name(planning_model)
    specialist_display = UIConfigUtils.get_friendly_model_name(specialist_model)
    synthesizer_display = UIConfigUtils.get_friendly_model_name(synthesizer_model)
    
    # Dynamic platform detection
    platform_map = {
        "bedrock": "AWS Bedrock",
        "openai": "OpenAI API", 
        "vertex": "Google Vertex AI",
        "azure": "Azure OpenAI"
    }
    platform_name = platform_map.get(config.llm_platform, "AWS Bedrock")
    
    # Create welcome message (matching original LangChain format)
        # Determine primary provider based on specialist model
    primary_provider = "Anthropic"  # Default for Bedrock Claude models
    if 'gemini' in specialist_model.lower() or 'google' in specialist_model.lower():
        primary_provider = "Google"
    elif 'openai' in specialist_model.lower() or 'gpt' in specialist_model.lower():
        primary_provider = "OpenAI"
    elif 'anthropic' in specialist_model.lower() or 'claude' in specialist_model.lower():
        primary_provider = "Anthropic"
    
    welcome_message = f"Hello! I'm SwagBot, powered by {platform_name} with {primary_provider} models. I can help you with product information, customer service questions, current promotions, and more. What can I help you find today?"
    
    # Platform-specific UI configuration
    platform_ui_config = {
        "bedrock": {
            "name": "Bedrock",
            "logo": "/static/images/bedrock.png",
            "cloud_provider_logo": "/static/images/aws.png",
            "theme_color": "#FF9900"  # AWS Orange
        },
        "openai": {
            "name": "OpenAI",
            "logo": "/static/images/openai-logo.png",
            "cloud_provider_logo": "",  # No separate cloud provider logo - already shown in platform badge
            "theme_color": "#00A67E"  # OpenAI Green
        },
        "vertex": {
            "name": "Vertex AI",
            "logo": "/static/images/Vertex_AI_Logo.png", 
            "cloud_provider_logo": "/static/images/gcp.svg",
            "theme_color": "#4285F4"  # Google Blue
        },
        "azure": {
            "name": "Azure OpenAI",
            "logo": "/static/images/azure-openai-logo.png",
            "cloud_provider_logo": "/static/images/azure.png",
            "theme_color": "#0078D4"  # Azure Blue
        }
    }
    
    current_platform_ui = platform_ui_config.get(config.llm_platform, platform_ui_config["bedrock"])
    
    # LangGraph Multi-Agent Workflow Configuration
    current_workflow = {
        "name": "LangGraph Multi-Agent Workflow",
        "description": "Orchestrator-driven workflow with parallel execution of specialized agents",
        "agent_count": 5,
        "agents": ["Planning Agent", "Customer Service", "Product Specialist", "Promotion Specialist", "Feedback Handler", "Response Synthesizer"],
        "workflow_steps": [
            {
                "step": 1,
                "name": "Planning",
                "description": "Request analysis & agent selection",
                "agent": "Planning Agent",
                "model": planning_display
            },
            {
                "step": 2,
                "name": "Orchestration",
                "description": "Parallel execution of specialized agents",
                "agent": "Specialized Agents",
                "model": specialist_display
            },
            {
                "step": 3,
                "name": "Synthesis",
                "description": "Response coordination & formatting",
                "agent": "Response Synthesizer",
                "model": synthesizer_display
            }
        ],
        "features": [
            "Parallel agent execution",
            "Orchestrator-driven routing",
            "Context-aware responses",
            "Multi-domain expertise"
        ]
    }
    
    return {
        "platform": {
            "name": current_platform_ui["name"],
            "logo": current_platform_ui["logo"],
            "cloud_provider_logo": current_platform_ui["cloud_provider_logo"]
        },
        "primary_logo": "/static/images/default-chatbot-logo.jpg",
        "bot_name": f"SwagBot - {current_platform_ui['name']} ({current_workflow['name']})",
        "welcome_message": welcome_message,
        "workflow": {
            "type": "langgraph",
            "name": current_workflow["name"],
            "description": current_workflow["description"],
            "agent_count": current_workflow["agent_count"],
            "agents": current_workflow["agents"],
            "workflow_steps": current_workflow["workflow_steps"],
            "features": current_workflow["features"]
        },
        "workflow_type": "langgraph",
        "agents": {
            "planning": {
                "name": "Planning Agent",
                "model": planning_display,
                "description": "Analyzes requests and determines needed specialist agents",
                "model_id": planning_model,
                "friendly_name": planning_display,
                "logo": UIConfigUtils.get_model_logo(planning_model),
                "role": "Request analysis & agent routing"
            },
            "specialists": {
                "name": "Specialized Agents",
                "model": specialist_display,
                "description": "Customer Service, Product, Promotion & Feedback specialists",
                "model_id": specialist_model,
                "friendly_name": specialist_display,
                "logo": UIConfigUtils.get_model_logo(specialist_model),
                "role": "Domain expertise & knowledge retrieval",
                "agents": ["Customer Service", "Product Specialist", "Promotion Specialist", "Feedback Handler"]
            },
            "synthesizer": {
                "name": "Response Synthesizer",
                "model": synthesizer_display,
                "description": "Coordinates and synthesizes specialist responses",
                "model_id": synthesizer_model,
                "friendly_name": synthesizer_display,
                "logo": UIConfigUtils.get_model_logo(synthesizer_model),
                "role": "Multi-agent coordination & response synthesis"
            }
        },
        "show_multi_agent": True,
        "show_dual_model": planning_model != specialist_model,
        "theme_color": current_platform_ui["theme_color"],
        "logo_path": "/static/images/default-chatbot-logo.jpg",
        "rum_config": get_rum_config()
    }


def get_datadog_status() -> dict:
    """Get current Datadog configuration status"""
    try:
        # Basic APM status check for lab #2
        current_span = tracer.current_span()
        return {
            "apm_enabled": current_span is not None,
            "service": "swagbot",
            "env": "dev", 
            "version": "1.0",
            "trace_active": current_span is not None
        }
    except Exception as e:
        return {"apm_enabled": False, "error": f"Status check failed: {str(e)}"}


# ===============================================================================
# APPLICATION INITIALIZATION
# ===============================================================================

def initialize_app():
    """Initialize the Flask application with all necessary components"""
    logger.info("🚀 Starting SwagBot LangGraph Application...")
    
    # Datadog already initialized at module level
    status = get_datadog_status()
    logger.info(f"📊 Datadog Status: {status}")
    
    logger.info("✅ SwagBot LangGraph Application initialized successfully")


# ===============================================================================
# APPLICATION STARTUP
# ===============================================================================

if __name__ == '__main__':
    initialize_app()
    # Bind to 0.0.0.0 for Docker - security is handled by container isolation and port mapping
    app.run(debug=True, host='0.0.0.0', port=config.FLASK_PORT) 