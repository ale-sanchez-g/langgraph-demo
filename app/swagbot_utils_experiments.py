#!/usr/bin/env python3

# Suppress Google Cloud warnings at the very beginning
import os
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

"""
SwagBot LangGraph Model Comparison Experiments

This module provides comprehensive model comparison functionality for the SwagBot LangGraph workflow,
enabling systematic testing and evaluation of different LLM models using Datadog LLM Observability.

Platform-Aware: Automatically detects LLM_PLATFORM environment variable and tests appropriate models.

Supported Platforms:
- AWS Bedrock: Claude variants (Haiku, Sonnet, 3.5-Sonnet, etc.)
- Google Vertex AI: Gemini models (Flash, Pro, Ultra)
- OpenAI: GPT models (4o-mini, 4o, 4-turbo)
- Azure OpenAI: GPT models (4o-mini, 4o, 4)

Usage:
    # Bedrock (LLM_PLATFORM=bedrock)
    python swagbot_utils_experiments.py --model-comparison claude-3-haiku
    python swagbot_utils_experiments.py --model-comparison claude-3-sonnet
    
    # Vertex AI (LLM_PLATFORM=vertex)
    python swagbot_utils_experiments.py --model-comparison gemini-flash
    python swagbot_utils_experiments.py --model-comparison gemini-pro
    
    # OpenAI (LLM_PLATFORM=openai)
    python swagbot_utils_experiments.py --model-comparison gpt-4o-mini
    python swagbot_utils_experiments.py --model-comparison gpt-4o
    
    # Universal commands (work with any platform)
    python swagbot_utils_experiments.py --compare-all-models
    python swagbot_utils_experiments.py --list-models
"""

import json
import logging
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime

# Suppress Google Cloud warnings BEFORE importing any Google libraries
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_LOGS'] = '1'

# Suppress absl logging at the C++ level
import sys
if hasattr(sys, 'stderr'):
    # Redirect stderr temporarily to suppress C++ warnings
    import io
    original_stderr = sys.stderr
    sys.stderr = io.StringIO()

# Datadog LLM Observability for Experiments
from ddtrace.llmobs import LLMObs

# Import your existing workflow
from swagbot_langgraph_workflow import process_swagbot_request
from swagbot_langgraph_config import LangChainConfig

# Restore stderr after imports
if 'original_stderr' in locals():
    sys.stderr = original_stderr

# Set up minimal logging
logging.basicConfig(level=logging.ERROR)  # Only show errors by default
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep info level just for this module

# Reduce verbosity of other modules
logging.getLogger('swagbot_langgraph_config').setLevel(logging.ERROR)
logging.getLogger('swagbot_langgraph_workflow').setLevel(logging.ERROR)
logging.getLogger('botocore').setLevel(logging.ERROR)
logging.getLogger('langchain_google_vertexai').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.ERROR)

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message="All log messages before absl::InitializeLog()")
warnings.filterwarnings("ignore", message="ALTS creds ignored")
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress Google Cloud and ALTS warnings at environment level
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# Suppress absl logging warnings
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
    # Disable absl logging completely
    absl.logging.use_python_logging()
except ImportError:
    # absl not installed, no action needed
    pass

# Additional suppression for Google Cloud libraries
try:
    import google.cloud.logging
    google.cloud.logging.Client().setup_logging(log_level=logging.ERROR)
except ImportError:
    # Google Cloud logging not available, no action needed
    pass

# Suppress gRPC warnings at the Python level
try:
    import grpc
    # Monkey patch gRPC logging
    original_grpc_log = grpc._common._LOGGER
    grpc._common._LOGGER = logging.getLogger('grpc').setLevel(logging.ERROR)
except (ImportError, AttributeError):
    # gRPC not available or internal structure changed, no action needed
    pass

# Manual LangGraph instrumentation patch (required for ddtrace 3.12+)
try:
    from ddtrace import patch
    patch(langgraph=True)
    logger.info("✅ Manual LangGraph instrumentation patch applied for experiments")
except Exception as e:
    logger.warning(f"⚠️ Failed to apply LangGraph patch in experiments: {e}")

class SuppressStderr:
    """Context manager to suppress stderr output during Google Cloud operations."""
    
    def __init__(self):
        """Initialize the context manager."""
        self.original_stderr = None
        self.devnull = None
    
    def __enter__(self):
        import sys
        import os
        self.original_stderr = sys.stderr
        self.devnull = open(os.devnull, 'w')
        sys.stderr = self.devnull
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        if self.devnull:
            self.devnull.close()
        sys.stderr = self.original_stderr

class SwagBotModelComparisonSuite:
    """
    Platform-aware model comparison experiment suite for SwagBot LangGraph workflow.
    
    Automatically adapts to the configured LLM platform (Bedrock, Vertex AI, OpenAI, Azure).
    
    Features:
    - Multi-platform support (Bedrock, Vertex AI, OpenAI, Azure OpenAI)
    - Platform-specific model selection and configuration
    - Model-specific configuration overrides
    - Comparative evaluation across identical datasets
    - Performance and cost analysis
    - Datadog LLM Observability integration with consistent span naming
    """
    
    # Experiment execution configuration
    EXPERIMENT_PARALLEL_JOBS = 3  # Number of parallel jobs for experiment execution
    
    def __init__(self):
        self.config = LangChainConfig()
        self.platform = self.config.llm_platform
        logger.info(f"🚀 Initializing experiments for platform: {self.platform.upper()}")
        self.available_models = self._get_available_models_for_platform()
        logger.info(f"📋 Loaded {len(self.available_models)} models for {self.platform.upper()} platform")
        self._ensure_llmobs_enabled()
    
    def _get_available_models_for_platform(self) -> Dict[str, Dict[str, Any]]:
        """Get available models based on the current platform."""
        if self.platform == "bedrock":
            return self._get_available_bedrock_models()
        elif self.platform == "vertex":
            return self._get_available_vertex_models()
        elif self.platform == "openai":
            return self._get_available_openai_models()
        elif self.platform == "azure":
            return self._get_available_azure_models()
        else:
            raise ValueError(f"Unsupported platform for experiments: {self.platform}")
    
    def _get_available_bedrock_models(self) -> Dict[str, Dict[str, Any]]:
        """Define the core models for experimentation."""
        return {
            # Claude Models
            "claude-3-haiku": {
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "display_name": "Claude 3 Haiku",
                "description": "Fast, lightweight model (baseline)",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "low",
                "provider": "anthropic"
            },
            "claude-3-sonnet": {
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "display_name": "Claude 3 Sonnet",
                "description": "Balanced model for general tasks",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high",
                "provider": "anthropic"
            },
            "claude-3.5-sonnet": {
                "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "display_name": "Claude 3.5 Sonnet",
                "description": "Latest balanced model with improved capabilities",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high",
                "provider": "anthropic"
            },
            # Claude 3.7 Sonnet Model
            "claude-3-7-sonnet": {
                "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "display_name": "Claude 3.7 Sonnet",
                "description": "Latest Claude Sonnet model with improved capabilities",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high",
                "provider": "anthropic"
            },
            # Claude Sonnet 4 Model
            "claude-sonnet-4": {
                "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "provider": "anthropic",
                "display_name": "Claude Sonnet 4",
                "description": "Latest Claude Sonnet model with improved capabilities",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high"
            }
        }
    
    def _get_available_vertex_models(self) -> Dict[str, Dict[str, Any]]:
        """Define Vertex AI models for experimentation - only available models."""
        return {
            "gemini-2.0-flash-lite": {
                "model_id": "gemini-2.0-flash-lite",
                "display_name": "Gemini 2.0 Flash Lite",
                "description": "Lightweight 2.0 model for fast responses",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "low",
                "provider": "google"
            },
            "gemini-2.5-flash": {
                "model_id": "gemini-2.5-flash",
                "display_name": "Gemini 2.5 Flash",
                "description": "Fast, efficient model with improved capabilities",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "medium",
                "provider": "google"
            },
            "gemini-2.5-pro": {
                "model_id": "gemini-2.5-pro",
                "display_name": "Gemini 2.5 Pro",
                "description": "Advanced reasoning model with enhanced capabilities",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high",
                "provider": "google"
            },
            "gemini-2.5-flash-lite": {
                "model_id": "gemini-2.5-flash-lite",
                "display_name": "Gemini 2.5 Flash Lite",
                "description": "Ultra-fast, lightweight model for quick planning tasks",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "low",
                "provider": "google"
            },
            "gemini-2.0-flash": {
                "model_id": "gemini-2.0-flash-001",
                "display_name": "Gemini 2.0 Flash",
                "description": "Workhorse model for all daily tasks with strong performance and low latency",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "medium",
                "provider": "google"
            }
        }
    
    def _get_available_openai_models(self) -> Dict[str, Dict[str, Any]]:
        """Define OpenAI models for experimentation."""
        return {
            "gpt-4o-mini": {
                "model_id": "gpt-4o-mini",
                "display_name": "GPT-4o Mini",
                "description": "Fast, cost-effective model for quick responses",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "low",
                "provider": "openai"
            },
            "gpt-4o": {
                "model_id": "gpt-4o",
                "display_name": "GPT-4o",
                "description": "Advanced model for complex reasoning tasks",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "medium",
                "provider": "openai"
            },
            "gpt-4-turbo": {
                "model_id": "gpt-4-turbo",
                "display_name": "GPT-4 Turbo",
                "description": "High-performance model for demanding tasks",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high",
                "provider": "openai"
            }
        }
    
    def _get_available_azure_models(self) -> Dict[str, Dict[str, Any]]:
        """Define Azure OpenAI models for experimentation."""
        return {
            "gpt-4o-mini": {
                "model_id": "gpt-4o-mini",
                "display_name": "GPT-4o Mini (Azure)",
                "description": "Fast, cost-effective model for quick responses",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "low",
                "provider": "azure-openai"
            },
            "gpt-4o": {
                "model_id": "gpt-4o",
                "display_name": "GPT-4o (Azure)",
                "description": "Advanced model for complex reasoning tasks",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "medium",
                "provider": "azure-openai"
            },
            "gpt-4": {
                "model_id": "gpt-4",
                "display_name": "GPT-4 (Azure)",
                "description": "High-performance model for demanding tasks",
                "temperature": 0.3,
                "max_tokens": 1000,
                "cost_tier": "high",
                "provider": "azure-openai"
            }
        }
    
    def _ensure_llmobs_enabled(self):
        """Enable Datadog LLM Observability (prerequisite)."""
        # Check for experiment-specific app key
        experiments_app_key = os.getenv('SWAGBOT_DD_APP_KEY')
        api_key = os.getenv('DD_API_KEY')
        
        if experiments_app_key:
            logger.info(f"🔑 Using dedicated experiments app key: SWAGBOT_DD_APP_KEY")
        else:
            logger.warning("⚠️ No SWAGBOT_DD_APP_KEY found - experiments may have limited functionality")
        
        if not api_key:
            logger.warning("⚠️ No DD_API_KEY found - experiments may not work properly")
        
        # Enable LLMObs with consistent naming [[memory:3279953]]
        LLMObs.enable(
            site=os.getenv('DD_SITE', 'datadoghq.com'),
            api_key=api_key,
            app_key=experiments_app_key,  # Use dedicated app key for experiments
            project_name="swagbot_langgraph_workflow"  # Project name for experiments
        )
        
        logger.info("✅ Datadog LLM Observability enabled for experiments")
    


    def create_customer_service_dataset(self) -> Any:
        """Create or update dataset for Customer Service agent evaluation."""
        records = [
            {
                "input_data": "I bought a mug but it arrived broken, can I get a replacement?",
                "expected_output": "We apologize for the damaged mug. We'll process a replacement immediately. Our return policy covers damaged items with free replacement shipping."
            },
            {
                "input_data": "What's your return policy for items bought on sale?",
                "expected_output": "Sale items can be returned within 30 days for store credit or exchange. Original packaging required. Final sale items (marked as such) cannot be returned."
            },
            {
                "input_data": "I can't track my order, it's been 5 days since I placed it",
                "expected_output": "I'll help you track your order. Orders typically ship within 2-3 business days. Please provide your order number so I can check the current status and provide tracking information."
            },
            {
                "input_data": "The headphones I received don't match the description online",
                "expected_output": "I'm sorry the headphones don't match expectations. We'll arrange a return and full refund. Please keep the original packaging and we'll email you a prepaid return label."
            },
            {
                "input_data": "I was charged twice for the same order, can you help?",
                "expected_output": "Double charges are usually pre-authorizations that will be released. I'll check your account and if there's an actual duplicate charge, we'll process an immediate refund within 3-5 business days."
            }
        ]
        
        return LLMObs.create_dataset(
            dataset_name="swagbot_customer_service_eval",
            description="Customer service scenarios for SwagBot evaluation",
            records=records
        )
    
    def create_product_specialist_dataset(self) -> Any:
        """Create or update dataset for Product Specialist agent evaluation."""
        records = [
            {
                "input_data": "Tell me about the Dog Steel Bottle",
                "expected_output": "The Dog Steel Bottle is a robust stainless steel vacuum flask with sports lid, priced at $29.99. It's perfect for keeping drinks hot or cold, made from durable materials."
            },
            {
                "input_data": "How much do the Dog Headphones cost?",
                "expected_output": "The Dog Headphones are priced at $229.99. They feature soft, plush cushions for excellent comfort and sound isolation."
            },
            {
                "input_data": "Do you have any eco-friendly products?",
                "expected_output": "Our stainless steel products like the Dog Steel Bottle are eco-friendly - reusable, durable, and made from recyclable materials that reduce single-use plastic waste."
            },
            {
                "input_data": "What categories of products do you sell?",
                "expected_output": "We offer products in several categories including accessories (headphones, bottles), apparel (t-shirts), drinkware (mugs), and lifestyle items."
            }
        ]
        
        return LLMObs.create_dataset(
            dataset_name="swagbot_product_specialist_eval",
            description="Product information scenarios for SwagBot evaluation",
            records=records
        )
    
    def create_promotion_specialist_dataset(self) -> Any:
        """Create dataset for Promotion Specialist agent evaluation."""
        records = [
            {
                "input_data": "Are there any current discounts for the steel bottle?",
                "expected_output": "Yes! Use code STEELBOTTLE10 for 10% off the Dog Steel Bottle. This promotion is valid until December 31st, 2024."
            },
            {
                "input_data": "Do you have any deals on t-shirts this week?",
                "expected_output": "Currently we have TSHIRT15 for 15% off all t-shirts, and BUNDLE20 for 20% off when you buy 2 or more t-shirts."
            },
            {
                "input_data": "What promotions are running right now?",
                "expected_output": "Active promotions include: STEELBOTTLE10 (10% off steel bottles), TSHIRT15 (15% off t-shirts), BUNDLE20 (20% off 2+ t-shirts), and FREESHIP50 (free shipping on orders over $50)."
            },
            {
                "input_data": "Is there a discount code for first-time buyers?",
                "expected_output": "Welcome! New customers can use WELCOME20 for 20% off their first order over $30. This offer is valid for 30 days from signup."
            },
            {
                "input_data": "Any coupon codes for the holidays?",
                "expected_output": "Holiday specials include HOLIDAY25 for 25% off orders over $75, and GIFT15 for 15% off gift items. Both valid through December 31st."
            }
        ]
        
        return LLMObs.create_dataset(
            dataset_name="swagbot_promotion_specialist_eval",
            description="Promotion and discount scenarios for SwagBot evaluation",
            records=records
        )
    
    def create_comprehensive_dataset(self) -> Any:
        """Create or update comprehensive dataset covering multiple agent scenarios."""
        records = [
            {
                "input_data": "Tell me about the Dog Steel Bottle and any current promotions",
                "expected_output": "The Dog Steel Bottle ($24.99) is a robust stainless steel vacuum flask with sports lid. Currently, you can save 10% with code STEELBOTTLE10, making it just $22.49!"
            },
            {
                "input_data": "I want to buy headphones but want to see customer reviews first",
                "expected_output": "Our Dog Headphones ($229.99) have excellent reviews for comfort and sound quality. Customers love the soft, plush cushions. I can help you find detailed reviews and ratings."
            },
            {
                "input_data": "What's your return policy and current deals on bottles?",
                "expected_output": "Returns are accepted within 30 days with original packaging. For bottles, we currently have STEELBOTTLE10 for 10% off the Dog Steel Bottle, making it $22.49 instead of $24.99."
            },
            {
                "input_data": "Show me your best selling products and any deals available",
                "expected_output": "Our bestsellers include Dog Headphones ($229.99) and Dog Steel Bottle ($24.99). Current deals: STEELBOTTLE10 (10% off bottles), TSHIRT15 (15% off shirts), and FREESHIP50 (free shipping over $50)."
            },
            {
                "input_data": "I need help with a damaged product and want to know about your eco-friendly options",
                "expected_output": "For damaged products, we offer free replacements with prepaid return shipping. Our eco-friendly options include the reusable Dog Steel Bottle made from recyclable stainless steel."
            }
        ]
        
        return LLMObs.create_dataset(
            dataset_name="swagbot_comprehensive_eval",
            description="Multi-agent scenarios testing complete SwagBot workflow",
            records=records
        )
    
    # ===============================================================================
    # EVALUATOR FUNCTIONS
    # ===============================================================================
    
    def exact_match_evaluator(self, input_data: str, output_data: str, expected_output: str) -> bool:
        """Simple exact match evaluator."""
        return output_data.strip().lower() == expected_output.strip().lower()
    
    def contains_key_info_evaluator(self, input_data: str, output_data: str, expected_output: str) -> bool:
        """Boolean evaluator that checks if output contains key information from expected output."""
        import re
        
        # Handle cases where expected output indicates unavailable information
        unavailable_phrases = ["price not available", "not available", "price information not available"]
        expected_lower = expected_output.lower()
        output_lower = output_data.lower()
        
        if any(phrase in expected_lower for phrase in unavailable_phrases):
            # Check if the model appropriately indicates unavailability
            unavailable_indicators = [
                "not available", "no price", "price not", "don't have price", 
                "cannot provide price", "price information is not", "pricing is not available",
                "price is not listed", "no pricing information", "unavailable"
            ]
            
            return any(indicator in output_lower for indicator in unavailable_indicators)
        
        # For other cases, check for key terms presence
        # Extract key terms from expected output
        key_terms = []
        
        # Price patterns
        prices = re.findall(r'\$[\d,]+\.?\d*', expected_output)
        key_terms.extend(prices)
        
        # Product names and important terms
        products = ['Dog Steel Bottle', 'Dog Headphones', 'headphones', 'bottle', 'steel bottle']
        for product in products:
            if product.lower() in expected_lower:
                key_terms.append(product)
        
        # Promotion codes
        promo_codes = re.findall(r'[A-Z]+\d+', expected_output)
        key_terms.extend(promo_codes)
        
        # Check if at least 90% of key terms are found (as requested by user)
        if not key_terms:
            return True  # No specific terms to check
        
        found_terms = sum(1 for term in key_terms if term.lower() in output_lower)
        match_percentage = found_terms / len(key_terms)
        
        return match_percentage >= 0.9  # 90% threshold as requested
    
    def response_quality_evaluator(self, input_data: str, output_data: str, expected_output: str) -> float:
        """Score-based comprehensive response quality evaluator (0.0-1.0)."""
        import re
        from difflib import SequenceMatcher
        
        output_lower = output_data.lower()
        input_lower = input_data.lower()
        score_components = []
        
        # 1. Length Quality (0.2 weight)
        # Good responses should be substantial but not overly verbose
        response_length = len(output_data.strip())
        if response_length == 0:
            length_score = 0.0
        elif response_length < 20:
            length_score = 0.3  # Too short
        elif response_length < 100:
            length_score = 0.7  # Acceptable
        elif response_length < 500:
            length_score = 1.0  # Good length
        elif response_length < 1000:
            length_score = 0.8  # A bit long but okay
        else:
            length_score = 0.5  # Too verbose
        score_components.append((length_score, 0.2))
        
        # 2. Key Information Match (0.4 weight) - Uses contains_key_info logic
        key_info_score = self.contains_key_info_evaluator(input_data, output_data, expected_output)
        score_components.append((key_info_score, 0.4))
        
        # 3. Response Appropriateness (0.2 weight)
        appropriateness_score = 1.0
        
        # Negative indicators
        negative_phrases = ['error', 'sorry, i cannot', 'i don\'t know', 'i am unable', 'i cannot help']
        if any(phrase in output_lower for phrase in negative_phrases):
            appropriateness_score -= 0.5
        
        # Check for proper punctuation and structure
        if not any(punct in output_data for punct in ['.', '!', '?']):
            appropriateness_score -= 0.3
        
        # Check if response is empty or too generic
        if output_data.strip() == '' or len(output_data.strip().split()) < 5:
            appropriateness_score = 0.0
        
        score_components.append((max(0.0, appropriateness_score), 0.2))
        
        # 4. Context Relevance (0.2 weight)
        # Check if response addresses the specific question type
        relevance_score = 0.7  # Base score
        
        # Price-related questions
        if any(word in input_lower for word in ['price', 'cost', 'how much']):
            if any(indicator in output_lower for indicator in ['$', 'price', 'cost', 'not available']):
                relevance_score = 1.0
            else:
                relevance_score = 0.3  # Didn't address price question
        
        # Product information questions
        if any(word in input_lower for word in ['about', 'tell me', 'information', 'describe']):
            if any(word in output_lower for word in ['product', 'description', 'features', 'specifications']):
                relevance_score = 1.0
        
        # Promotion/discount questions
        if any(word in input_lower for word in ['discount', 'deal', 'promotion', 'coupon']):
            if any(word in output_lower for word in ['discount', 'code', '%', 'promotion', 'deal']):
                relevance_score = 1.0
            else:
                relevance_score = 0.3
        
        score_components.append((relevance_score, 0.2))
        
        # Calculate weighted final score
        final_score = sum(score * weight for score, weight in score_components)
        
        # Ensure score is between 0.0 and 1.0
        return round(max(0.0, min(1.0, final_score)), 2)
    
    def html_format_evaluator(self, input_data: str, output_data: str, expected_output: str) -> bool:
        """Evaluator to check if output contains proper HTML formatting."""
        # Check for HTML tags that should be present in SwagBot responses
        html_indicators = [
            '<p>' in output_data,  # Paragraphs
            '</p>' in output_data,  # Closing paragraphs
        ]
        
        # Additional HTML checks
        if any(word in input_data.lower() for word in ['price', 'product', 'information']):
            html_indicators.append('<strong>' in output_data)  # Should emphasize important info
        
        if any(word in input_data.lower() for word in ['list', 'products', 'categories', 'options']):
            html_indicators.append('<ul>' in output_data or '<li>' in output_data)  # Should use lists
        
        return sum(html_indicators) >= 2  # At least basic HTML structure
    
    # ===============================================================================
    # SUMMARY EVALUATORS
    # ===============================================================================
    
    def avg_response_quality_summary(self, inputs, outputs, expected_outputs, evaluators_results):
        """Summary evaluator to calculate average response quality across all experiments."""
        quality_scores = evaluators_results.get("response_quality_evaluator", [])
        if not quality_scores:
            return None
        # Filter out None values
        valid_scores = [score for score in quality_scores if score is not None]
        if not valid_scores:
            return None
        return round(sum(valid_scores) / len(valid_scores), 3)
    
    def key_info_match_rate_summary(self, inputs, outputs, expected_outputs, evaluators_results):
        """Summary evaluator to calculate percentage of outputs containing key information."""
        matches = evaluators_results.get("contains_key_info_evaluator", [])
        if not matches:
            return None
        return round(matches.count(True) / len(matches) * 100, 1)
    
    def html_format_compliance_summary(self, inputs, outputs, expected_outputs, evaluators_results):
        """Summary evaluator to calculate percentage of outputs with proper HTML formatting."""
        html_checks = evaluators_results.get("html_format_evaluator", [])
        if not html_checks:
            return None
        return round(html_checks.count(True) / len(html_checks) * 100, 1)
    
    # ===============================================================================
    # MODEL-SPECIFIC TASK WRAPPER
    # ===============================================================================
    
    def create_direct_agent_task(self, model_key: str, agent_type: str = "product_specialist"):
        """Create a task function that tests a specific agent directly without the full workflow."""

        def direct_agent_task(input_data: str, config: Dict[str, Any]) -> str:
            """
            Direct agent testing that bypasses the full workflow.
            Tests only the specified agent with the given model, including proper knowledge base and prompts.
            """
            try:
                model_config = self.available_models[model_key]

                # Import the workflow to access the LLMCaller and knowledge loading
                from swagbot_langgraph_workflow import LLMCaller, LangChainConfig, SwagBotWorkflow
                import json
                import os

                # Create a temporary config with the test model
                temp_config = LangChainConfig()
                temp_config.specialist_model = model_config.get('inference_profile', model_config['model_id'])

                # Create LLM caller
                llm_caller = LLMCaller(temp_config)
                
                # Create a temporary workflow instance to access production methods
                workflow = SwagBotWorkflow()
                
                # Load agent prompt using EXACT production method
                agent_prompt = workflow._load_agent_prompt("product-specialist")
                
                # Extract user request from input data
                if isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], dict):
                    # Handle list of messages format
                    user_request = input_data[0].get('content', '')
                elif isinstance(input_data, dict):
                    # Handle single message format
                    user_request = input_data.get('content', '')
                else:
                    # Handle plain string
                    user_request = input_data
                
                # Retrieve documents using EXACT production method
                documents = workflow._retrieve_documents("product_specialist", user_request, None)
                
                # Build context using EXACT production method
                context = workflow.llm_caller._build_context_from_documents(documents, include_header=True)
                
                # Build prompt using EXACT production format
                full_prompt = f"""{agent_prompt}

User Request: {user_request}

Context:
{context}

Response:"""

                # Minimal logging for experiment progress
                if logger.getEffectiveLevel() <= logging.DEBUG:
                    logger.debug(f"Processing record with {model_key}: {user_request[:50]}...")

                # Call the LLM directly using existing agent method with proper context
                # This call should be automatically instrumented by ddtrace [[memory:4152481]]
                # Suppress Google Cloud warnings during LLM call
                with SuppressStderr():
                    response = llm_caller.call_agent_llm(
                        prompt=full_prompt,
                        agent_name=agent_type,
                        documents=[],  # Knowledge already included in prompt
                        user_request=input_data
                    )

                return response.strip()

            except Exception as e:
                logger.error(f"Error in direct agent task for {model_key}: {e}")
                return f"Error testing {model_config['display_name']} on {agent_type}: {str(e)}"

        return direct_agent_task
    
    def create_model_specific_task(self, model_key: str):
        """Create a task function that uses a specific model configuration for full workflow."""
        
        def model_task(input_data: str, config: Dict[str, Any]) -> str:
            """
            Model-specific wrapper that overrides the default model configuration.
            
            This function:
            1. Temporarily overrides the model configuration
            2. Uses the existing process_swagbot_request function with new model
            3. Extracts and returns the output
            4. Uses consistent span naming for Datadog [[memory:4152491]]
            """
            try:
                model_config = self.available_models[model_key]
                original_config = {}
                
                # Temporarily override environment variables for this model
                env_overrides = {
                    'SPECIALIST_MODEL': model_config.get('inference_profile', model_config['model_id']),
                    'PLANNING_MODEL': model_config.get('inference_profile', model_config['model_id']),
                    'SYNTHESIZER_MODEL': model_config.get('inference_profile', model_config['model_id'])
                }
                
                # Store original values
                for key, value in env_overrides.items():
                    original_config[key] = os.getenv(key)
                    os.environ[key] = value
                
                try:
                    # Call workflow with new model configuration
                    result = process_swagbot_request(input_data)
                    
                    # Extract the output
                    if isinstance(result, dict):
                        output = result.get('output', str(result))
                    else:
                        output = str(result)
                    
                    # Ensure we return a string
                    return output if isinstance(output, str) else str(output)
                    
                finally:
                    # Restore original environment
                    for key, original_value in original_config.items():
                        if original_value is not None:
                            os.environ[key] = original_value
                        elif key in os.environ:
                            del os.environ[key]
                            
            except Exception as e:
                logger.error(f"Error in model_task for {model_key}: {e}")
                return f"Error processing request with {model_config['display_name']}: {str(e)}"
        
        return model_task
    
    # ===============================================================================
    # MODEL COMPARISON EXPERIMENTS
    # ===============================================================================
    
    def run_single_model_experiment(self, model_key: str, dataset_type: str = "comprehensive", direct_agent: bool = False) -> tuple:
        """Run experiment with a specific model using existing dataset."""
        if model_key not in self.available_models:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.available_models.keys())}")
        
        start_time = datetime.now()
        model_config = self.available_models[model_key]
        test_mode = "direct agent" if direct_agent else "full workflow"
        
        # Get dataset name
        dataset_names = {
            "customer_service": "swagbot_customer_service_eval",
            "product_specialist": "swagbot_product_specialist_eval",
            "promotion_specialist": "swagbot_promotion_specialist_eval",
            "comprehensive": "swagbot_comprehensive_eval"
        }
        
        dataset_name = dataset_names.get(dataset_type)
        if not dataset_name:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        dataset = LLMObs.pull_dataset(dataset_name)
        if dataset is None:
            raise ValueError(f"Dataset '{dataset_name}' not found. Please create it first using --create-dataset {dataset_type}")
        
        # Extract and log dataset version for reproducibility
        dataset_version = getattr(dataset, 'current_version', None) or getattr(dataset, 'version', 'unknown')
        logger.info(f"📊 Using dataset '{dataset_name}' version: {dataset_version}")
            
        # Log experiment start
        # Get number of records from the dataset
        try:
            if hasattr(dataset, 'records'):
                num_records = len(dataset.records)
            elif isinstance(dataset, dict) and 'records' in dataset:
                num_records = len(dataset['records'])
            else:
                num_records = len(dataset)  # Assume dataset is directly a list of records
            logger.info(f"🔬 Testing {model_config['display_name']} on {dataset_type} dataset ({num_records} records)")
        except Exception as e:
            logger.warning(f"Could not determine number of records in dataset: {e}")
            logger.info(f"🔬 Testing {model_config['display_name']} on {dataset_type} dataset")
        
        # Create task function based on test mode
        if direct_agent:
            # Direct agent testing - no HTML evaluation needed
            model_task = self.create_direct_agent_task(model_key, dataset_type)
            evaluators = [
                self.contains_key_info_evaluator,
                self.response_quality_evaluator
                # No HTML evaluator for direct agent testing
            ]
            summary_evaluators = [
                self.avg_response_quality_summary,
                self.key_info_match_rate_summary
            ]
            experiment_name = f"swagbot_direct_agent_{model_key}_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            description = f"Direct {dataset_type} agent testing with {model_config['display_name']}"
        else:
            # Full workflow testing
            model_task = self.create_model_specific_task(model_key)
            evaluators = [
                self.contains_key_info_evaluator,
                self.response_quality_evaluator,
                self.html_format_evaluator
            ]
            summary_evaluators = [
                self.avg_response_quality_summary,
                self.key_info_match_rate_summary,
                self.html_format_compliance_summary
            ]
            experiment_name = f"swagbot_model_comparison_{model_key}_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            description = f"Full workflow testing of {model_config['display_name']} on {dataset_type} scenarios"
        
        experiment = LLMObs.experiment(
            name=experiment_name,
            task=model_task,
            dataset=dataset,
            evaluators=evaluators,
            summary_evaluators=summary_evaluators,
            description=description,
            config={
                "model_name": model_key,
                "dataset_version": dataset_version,
                "dataset_type": dataset_type
            }
        )
        
        # Run experiment with parallel processing for faster execution
        result = experiment.run(jobs=self.EXPERIMENT_PARALLEL_JOBS)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log experiment completion and URL
        logger.info(f"✅ {model_config['display_name']} completed in {duration:.1f}s")
        if hasattr(experiment, 'url'):
            print(f"\n🔗 View experiment results: {experiment.url}\n")
        
        return (result, experiment)
    
    def run_model_comparison_suite(self, models_to_test: List[str] = None, dataset_type: str = "comprehensive", direct_agent: bool = False) -> Dict[str, Any]:
        """Run comparison across multiple models using the same dataset."""
        if models_to_test is None:
            # Test all 3 core models
            models_to_test = ["claude-haiku", "claude-instant", "claude-sonnet"]
        
        start_time = datetime.now()
        logger.info(f"🚀 Comparing models on {dataset_type} dataset: {', '.join(models_to_test)}")
        results = {}
        completed = 0
        experiment_urls = []
        
        for model_key in models_to_test:
            if model_key not in self.available_models:
                logger.warning(f"⚠️ Skipping unknown model: {model_key}")
                continue
                
            try:
                result, experiment = self.run_single_model_experiment(model_key, dataset_type, direct_agent)
                results[model_key] = {
                    "experiment_result": result,
                    "model_config": self.available_models[model_key],
                    "status": "completed"
                }
                if hasattr(experiment, 'url'):
                    experiment_urls.append((self.available_models[model_key]['display_name'], experiment.url))
                completed += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to test {model_key}: {e}")
                results[model_key] = {
                    "error": str(e),
                    "model_config": self.available_models[model_key],
                    "status": "failed"
                }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"✅ Comparison completed in {duration:.1f}s - {completed}/{len(models_to_test)} models tested")
        
        # Print summary of all experiment URLs
        if experiment_urls:
            print("\n" + "="*80)
            print("📊 EXPERIMENT RESULTS SUMMARY")
            print("="*80)
            for model_name, url in experiment_urls:
                print(f"  🔗 {model_name}: {url}")
            print("="*80 + "\n")
        
        return results
    
    def run_all_model_comparison(self, dataset_type: str = "comprehensive", direct_agent: bool = False) -> Dict[str, Any]:
        """Test all available models."""
        return self.run_model_comparison_suite(list(self.available_models.keys()), dataset_type, direct_agent)
    
    def list_available_models(self) -> List[str]:
        """List all available models for testing."""
        return list(self.available_models.keys())
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset types."""
        return ["customer_service", "product_specialist", "promotion_specialist", "comprehensive"]
    

    
    # ===============================================================================
    # DATASET MANAGEMENT METHODS
    # ===============================================================================
    
    def create_all_datasets(self) -> Dict[str, str]:
        """Create all datasets in Datadog. Datadog will auto-version when records change."""
        logger.info("🗂️ Creating all datasets in Datadog...")
        
        results = {}
        dataset_creators = {
            "customer_service": self.create_customer_service_dataset,
            "product_specialist": self.create_product_specialist_dataset,
            "promotion_specialist": self.create_promotion_specialist_dataset,
            "comprehensive": self.create_comprehensive_dataset
        }
        
        for dataset_name, creator_func in dataset_creators.items():
            try:
                dataset = creator_func()
                dataset_id = getattr(dataset, 'id', 'unknown')
                results[dataset_name] = dataset_id
                logger.info(f"✅ Created dataset '{dataset_name}'")
            except Exception as e:
                logger.error(f"❌ Failed to create dataset '{dataset_name}': {e}")
                results[dataset_name] = f"ERROR: {e}"
        
        logger.info("🎯 Dataset creation completed!")
        logger.info("💡 Datadog will automatically version datasets when records are updated")
        return results
    
    def get_dataset_by_name(self, dataset_type: str) -> Any:
        """Get an existing dataset by name instead of creating a new one."""
        dataset_names = {
            "customer_service": "swagbot_customer_service_eval",
            "product_specialist": "swagbot_product_specialist_eval", 
            "promotion_specialist": "swagbot_promotion_specialist_eval",
            "comprehensive": "swagbot_comprehensive_eval"
        }
        
        dataset_name = dataset_names.get(dataset_type)
        if not dataset_name:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Note: For now, we'll return a dataset name that LLMObs.experiment can use
        # In practice, you might need to query existing datasets via API
        logger.info(f"📋 Using existing dataset: {dataset_name}")
        return dataset_name
    
    def get_model_info(self, model_key: str = None) -> Dict[str, Any]:
        """Get detailed information about available models."""
        if model_key:
            if model_key not in self.available_models:
                raise ValueError(f"Unknown model: {model_key}")
            return self.available_models[model_key]
        
        return self.available_models
    
    def run_model_comparison(self, model_key: str, dataset_type: str = "comprehensive", direct_agent: bool = False) -> Any:
        """Run a model comparison experiment."""
        test_mode = "direct agent" if direct_agent else "full workflow"
        logger.info(f"🚀 Starting model comparison: {model_key} on {dataset_type} dataset ({test_mode})...")
        result, experiment = self.run_single_model_experiment(model_key, dataset_type, direct_agent)
        
        model_config = self.available_models[model_key]
        logger.info(f"✅ {model_config['display_name']} experiment completed!")
        
        # Print experiment URL
        if hasattr(experiment, 'url'):
            print(f"\n🔗 View results: {experiment.url}\n")
        
        return result

def main():
    """Main CLI interface for running model comparison experiments."""
    parser = argparse.ArgumentParser(description="SwagBot LangGraph Model Comparison Experiments")
    
    # Model comparison arguments
    parser.add_argument("--model-comparison", "-m", 
                       help="Run experiment with specific model (e.g., claude-haiku, claude-sonnet)")
    parser.add_argument("--dataset", "-d", 
                       choices=["customer_service", "product_specialist", "promotion_specialist", "comprehensive"],
                       default="comprehensive",
                       help="Dataset type to use for testing (default: comprehensive)")
    parser.add_argument("--compare-models", "-c", nargs="+",
                       help="Compare multiple models (e.g., --compare-models claude-haiku claude-sonnet)")
    parser.add_argument("--compare-all-models", "-a", action="store_true",
                       help="Compare all available models")
    parser.add_argument("--direct-agent", action="store_true",
                       help="Test only the specific agent directly (no full workflow, no HTML)")
    
    # Information arguments
    parser.add_argument("--list-models", "-l", action="store_true", 
                       help="List available models")
    parser.add_argument("--list-datasets", action="store_true", 
                       help="List available datasets")
    parser.add_argument("--model-info", 
                       help="Get detailed info about a specific model")
    
    # Dataset management arguments
    parser.add_argument("--create-datasets", action="store_true",
                       help="Create all datasets in Datadog (run this once)")
    parser.add_argument("--create-dataset", 
                       choices=["customer_service", "product_specialist", "promotion_specialist", "comprehensive"],
                       help="Create a specific dataset in Datadog")
    
    args = parser.parse_args()
    
    suite = SwagBotModelComparisonSuite()
    
    if args.list_models:
        models = suite.list_available_models()
        print(f"🤖 Available {suite.platform.upper()} models for testing:")
        for model_key in models:
            model_config = suite.available_models[model_key]
            print(f"  • {model_key}: {model_config['display_name']}")
            print(f"    ID: {model_config['model_id']}")
            print(f"    Description: {model_config['description']}")
            print(f"    Cost Tier: {model_config['cost_tier']}")
            print()
        return
    
    if args.list_datasets:
        datasets = suite.list_available_datasets()
        print("📋 Available datasets:")
        for dataset in datasets:
            print(f"  • {dataset}")
        return
    
    if args.create_datasets:
        print("🗂️ Creating all datasets in Datadog...")
        results = suite.create_all_datasets()
        print("\n📊 Dataset Creation Results:")
        for dataset_name, result in results.items():
            if result.startswith("ERROR"):
                print(f"  ❌ {dataset_name}: {result}")
            else:
                print(f"  ✅ {dataset_name}: {result}")
        print("\n💡 Datasets created! Datadog will auto-version when records change.")
        return
    
    if args.create_dataset:
        print(f"🗂️ Creating dataset: {args.create_dataset}")
        dataset_creators = {
            "customer_service": suite.create_customer_service_dataset,
            "product_specialist": suite.create_product_specialist_dataset,
            "promotion_specialist": suite.create_promotion_specialist_dataset,
            "comprehensive": suite.create_comprehensive_dataset
        }
        try:
            dataset = dataset_creators[args.create_dataset]()
            dataset_id = getattr(dataset, 'id', 'unknown')
            print(f"✅ Created dataset '{args.create_dataset}'")
        except Exception as e:
            print(f"❌ Failed to create dataset '{args.create_dataset}': {e}")
        return
    
    if args.model_info:
        try:
            info = suite.get_model_info(args.model_info)
            print(f"📊 Model Information: {args.model_info}")
            print(f"  Display Name: {info['display_name']}")
            print(f"  Model ID: {info['model_id']}")
            print(f"  Description: {info['description']}")
            print(f"  Cost Tier: {info['cost_tier']}")
            print(f"  Temperature: {info['temperature']}")
            print(f"  Max Tokens: {info['max_tokens']}")
        except ValueError as e:
            print(f"❌ {e}")
        return
    
    if args.compare_all_models:
        test_mode = "direct agent" if args.direct_agent else "full workflow"
        print(f"🚀 Comparing all models on {args.dataset} dataset ({test_mode})...")
        suite.run_all_model_comparison(args.dataset, args.direct_agent)
        return
    
    if args.compare_models:
        test_mode = "direct agent" if args.direct_agent else "full workflow"
        print(f"🚀 Comparing models: {', '.join(args.compare_models)} on {args.dataset} dataset ({test_mode})...")
        suite.run_model_comparison_suite(args.compare_models, args.dataset, args.direct_agent)
        return
    
    if args.model_comparison:
        suite.run_model_comparison(args.model_comparison, args.dataset, args.direct_agent)
        return
    
    # No arguments provided - show help
    print("🤖 SwagBot Model Comparison Suite")
    print("=" * 50)
    print("Please provide command-line arguments to run experiments.")
    print("\n📋 Examples:")
    print("  # STEP 1: Create datasets in Datadog (run once):")
    print("  python swagbot_experiments.py --create-datasets")
    print()
    print("  # STEP 2: Run experiments:")
    print("  python swagbot_experiments.py --compare-all-models --dataset product_specialist --direct-agent")
    print()
    print("  # Dataset management:")
    print("  python swagbot_experiments.py --create-dataset product_specialist  # Create/update single dataset")
    print("  python swagbot_experiments.py --list-datasets                        # List available datasets")
    print()
    print("  # Compare specific models:")
    print("  python swagbot_experiments.py --compare-models claude-haiku claude-sonnet --dataset product_specialist --direct-agent")
    print()
    print("  # Test single model:")
    print("  python swagbot_experiments.py --model-comparison claude-haiku --dataset product_specialist --direct-agent")
    print()
    print("  # Model information:")
    print("  python swagbot_experiments.py --list-models")
    print()
    print("💡 Note: Datadog automatically versions datasets when records are updated")
    print("📘 For full help: python swagbot_experiments.py --help")
    return

if __name__ == "__main__":
    main()
