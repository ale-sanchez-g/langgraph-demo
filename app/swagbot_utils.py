"""
SwagBot Utilities - Pure utility functions (NO Datadog dependencies)

This file contains only pure utility functions that don't require instrumentation.

Key principle: Clean separation of concerns
- Utilities: Pure functions for data processing, HTML handling, document operations
- Instrumentation: All in the workflow file for proper organization
"""

import os
import json
import re
import logging
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Module-level OPM prompt ID cache - populated by load_agent_prompt_from_opm at load time
_opm_prompt_id_cache: Dict[str, int] = {}


class LangGraphUtils:
    """LangGraph state management utilities - pure functions for state merging"""
    
    @staticmethod
    def merge_agent_responses(left: Dict[str, str], right: Dict[str, str]) -> Dict[str, str]:
        """Merge agent responses from parallel execution."""
        if not left:
            left = {}
        if not right:
            return left
        return {**left, **right}

    @staticmethod
    def merge_agent_contexts(left: Dict[str, List[Dict[str, Any]]], right: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Merge agent contexts from parallel execution."""
        if not left:
            left = {}
        if not right:
            return left
        return {**left, **right}

    @staticmethod
    def merge_errors(left: List[str], right: List[str]) -> List[str]:
        """Merge error messages from parallel execution."""
        if not left:
            left = []
        if not right:
            return left
        return left + right


class HTMLUtils:
    """HTML processing utilities - pure functions with no Datadog dependencies"""
    
    @staticmethod
    def strip_html_tags(text: str) -> str:
        """Strip HTML tags from text to create clean version"""
        if not text:
            return text
        
        # Remove HTML tags using regex
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        # Replace common HTML entities
        clean_text = clean_text.replace('&lt;', '<')
        clean_text = clean_text.replace('&gt;', '>')
        clean_text = clean_text.replace('&amp;', '&')
        clean_text = clean_text.replace('&quot;', '"')
        clean_text = clean_text.replace('&apos;', "'")
        clean_text = clean_text.replace('&nbsp;', ' ')
        
        # Clean up extra whitespace
        clean_text = ' '.join(clean_text.split())
        
        return clean_text


class ResponseUtils:
    """Response formatting utilities - pure functions for clean data handling"""
    
    @staticmethod
    def format_workflow_response(final_state: Dict[str, Any], platform: str = "bedrock") -> Dict[str, Any]:
        """Format workflow response for API compatibility"""
        return {
            "response": final_state.get("response", "I'm sorry, I couldn't process your request."),
            "category": final_state.get("category", "Unknown"),
            "reason": "Processed successfully" if not final_state.get("error") else f"Error: {final_state.get('error')}",
            "metrics": {
                "retrieved_documents": final_state.get("retrieved_count", 0),
                "category_confidence": "high" if not final_state.get("error") else "low",
                "processing_time": "N/A"
            },
            "platform": platform.upper(),
            "model_id": final_state.get("model_id", "unknown"),
            "agents_used": {
                "planning": "langgraph_planning_agent",
                "specialists": "langgraph_specialist_agents"
            },
            "workflow_type": "langgraph"
        }
    
    @staticmethod
    def format_enhanced_workflow_response(final_state: Dict[str, Any], platform: str = "bedrock") -> Dict[str, Any]:
        """Format enhanced workflow response for API compatibility"""
        agent_outputs = final_state.get("agent_outputs", {})
        workflow_path = final_state.get("workflow_path", [])
        
        return {
            "response": final_state.get("response", "I'm sorry, I couldn't process your request."),
            "category": final_state.get("category", "Unknown"),
            "reason": "Processed successfully" if not final_state.get("error") else f"Error: {final_state.get('error')}",
            "metrics": {
                "retrieved_documents": final_state.get("retrieved_count", 0),
                "category_confidence": final_state.get("confidence", 0.5),
                "processing_time": "N/A",
                "agents_involved": len(agent_outputs)
            },
            "platform": platform.upper(),
            "model_id": final_state.get("model_id", "unknown"),
            "agents_used": {
                "planning": "enhanced_planning_agent",
                "specialists": list(agent_outputs.keys()),
                "coordinator": "enhanced_coordinator_agent"
            },
            "workflow_type": "enhanced_langgraph",
            "workflow_path": workflow_path,
            "planning_analysis": final_state.get("planning_analysis", {}),
            "agent_count": len(agent_outputs)
        }
    
    @staticmethod
    def format_error_response(error: Exception, platform: str = "bedrock") -> Dict[str, Any]:
        """Format error response for API compatibility"""
        return {
            "response": "I'm sorry, I encountered an unexpected error. Please try again later.",
            "category": "Error",
            "reason": f"Workflow error: {str(error)}",
            "metrics": {
                "retrieved_documents": 0,
                "category_confidence": "low",
                "processing_time": "N/A"
            },
            "platform": platform.upper(),
            "model_id": "unknown",
            "agents_used": {
                "planning": "error",
                "specialists": "error"
            },
            "workflow_type": "langgraph",
            "error": str(error)
        }


class DocumentUtils:
    """Document processing utilities - pure functions with no external dependencies"""
    
    @staticmethod
    def score_document_match(query_words: List[str], text: str, boost: float = 1.0) -> float:
        """Score how well a document matches the query"""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        score = 0.0
        
        for word in query_words:
            if word in text_lower:
                score += boost
                
        return score
    
    @staticmethod
    def rank_and_filter_documents(documents: List[Dict[str, Any]], max_results: int = 10) -> List[Dict[str, Any]]:
        """Sort documents by score and return top results"""
        if not documents:
            return []
            
        # Sort by score (descending)
        sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
        
        # Return top results
        return sorted_docs[:max_results]


class DocumentHandlers:
    """Document retrieval handlers - pure functions for different document types"""
    
    @staticmethod
    def search_faqs(query_words: List[str], faqs: List[Dict[str, Any]], category: str = None) -> List[Dict[str, Any]]:
        """Search FAQ documents for relevant matches"""
        results = []
        
        for faq in faqs:
            question = faq.get("question", "").lower()
            answer = faq.get("answer", "").lower()
            
            # Calculate score
            score = 0.0
            score += DocumentUtils.score_document_match(query_words, question, boost=5.0)
            score += DocumentUtils.score_document_match(query_words, answer, boost=2.0)
            
            # Bonus for customer service category
            if category == "Customer-Service":
                score += 3.0
                
            if score > 0:
                results.append({
                    "id": faq.get("id", "unknown"),
                    "type": "faq",
                    "content": f"Q: {faq.get('question', '')}\nA: {faq.get('answer', '')}",
                    "score": score,
                    "name": f"FAQ: {faq.get('question', 'Unknown question')}"
                })
        
        return results
    
    @staticmethod
    def search_products(query_words: List[str], products: List[Dict[str, Any]], category: str = None) -> List[Dict[str, Any]]:
        """Search product documents for relevant matches"""
        results = []
        
        for product in products:
            name = product.get("name", "").lower()
            description = product.get("description", "").lower()
            categories = " ".join(product.get("categories", [])).lower()
            
            # Calculate score
            score = 0.0
            score += DocumentUtils.score_document_match(query_words, name, boost=8.0)
            score += DocumentUtils.score_document_match(query_words, description, boost=3.0)
            score += DocumentUtils.score_document_match(query_words, categories, boost=2.0)
            
            # Bonus for product information category
            if category == "Product-Information":
                score += 5.0
                
            if score > 0:
                # Format price correctly (nanos = billionths of a dollar)
                price_info = product.get("priceUsd", {})
                if isinstance(price_info, dict) and "units" in price_info:
                    units = price_info.get("units", 0)
                    nanos = price_info.get("nanos", 0)
                    # Convert nanos (billionths) to decimal: nanos/1000000000
                    total_price = units + (nanos / 1000000000)
                    price = f"${total_price:.2f}"
                else:
                    price = "Price not available"
                    
                results.append({
                    "id": product.get("id", "unknown"),
                    "type": "product",
                    "content": f"Product: {product.get('name', '')}\nPrice: {price}\nDescription: {product.get('description', '')}\nCategories: {', '.join(product.get('categories', []))}\nImage: {product.get('picture', '')}",
                    "score": score,
                    "name": f"Product: {product.get('name', '')}"
                })
        
        return results
    
    @staticmethod
    def search_promotions(query_words: List[str], promotions: List[Dict[str, Any]], category: str = None) -> List[Dict[str, Any]]:
        """Search promotion documents for relevant matches"""
        results = []
        
        for promo in promotions:
            code = promo.get("code", "").lower()
            description = promo.get("description", "").lower()
            product = promo.get("applicable_product", "").lower()
            
            # Calculate score
            score = 0.0
            score += DocumentUtils.score_document_match(query_words, code, boost=3.0)
            score += DocumentUtils.score_document_match(query_words, description, boost=2.0)
            score += DocumentUtils.score_document_match(query_words, product, boost=2.0)
            
            # Bonus for promotion-related queries
            if category == "Promotions" or any(word in " ".join(query_words) for word in ["promotion", "discount", "deal", "sale", "code"]):
                score += 2.0
                
            if score > 0:
                results.append({
                    "id": promo.get("id", "unknown"),
                    "type": "promotion",
                    "content": f"Promotion: {promo.get('description', '')}\nCode: {promo.get('code', '')}\nDiscount: {promo.get('discount_percentage', 0)}%\nValid until: {promo.get('end_date', '')}\nMinimum purchase: ${promo.get('minimum_purchase', 0)}\nApplicable to: {promo.get('applicable_product', '')}",
                    "score": score,
                    "name": f"Promotion: {promo.get('code', '')}"
                })
        
        return results
    
    @staticmethod
    def search_customer_service(query_words: List[str], cs_info: Dict[str, Any], category: str = None) -> List[Dict[str, Any]]:
        """Search customer service information for relevant matches"""
        if not cs_info:
            return []
            
        results = []
        cs_terms = ["contact", "phone", "email", "hours", "customer", "service", "support", "help", "call", "reach"]
        
        # Calculate score
        score = 0.0
        for word in query_words:
            if word in cs_terms:
                score += 3.0
        
        # Bonus for customer service category
        if category == "Customer-Service":
            score += 5.0
            
        # Check for specific requests
        query_text = " ".join(query_words)
        if any(term in query_text for term in ["phone", "number", "call"]):
            score += 5.0
        if any(term in query_text for term in ["email", "contact"]):
            score += 5.0  
        if any(term in query_text for term in ["hours", "open", "close", "time"]):
            score += 5.0
            
        if score > 0:
            # Format hours information
            hours_info = cs_info.get("hours", {})
            hours_text = ""
            if hours_info:
                hours_list = []
                for day, times in hours_info.items():
                    open_time = times.get("open", "")
                    close_time = times.get("close", "")
                    hours_list.append(f"{day.capitalize()}: {open_time} - {close_time}")
                hours_text = "\n".join(hours_list)
            
            cs_content = f"Customer Service Information:\n"
            cs_content += f"Phone: {cs_info.get('phone', 'Not available')}\n"
            cs_content += f"Email: {cs_info.get('email', 'Not available')}\n"
            if hours_text:
                cs_content += f"Hours:\n{hours_text}"
            
            results.append({
                "id": "customer_service_info",
                "type": "customer_service",
                "content": cs_content,
                "score": score,
                "name": "Customer Service Contact Information"
            })
        
        return results


class CostCalculationUtils:
    """LLM cost calculation utilities - pure functions for token-based cost estimation"""
    
    @staticmethod
    def calculate_vertex_ai_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
        """
        Calculate simple total cost estimate for Vertex AI models based on token usage.
        
        Returns total cost in USD as a float.
        """
        # Validate inputs
        validated_input_tokens = max(0, int(input_tokens or 0))
        validated_output_tokens = max(0, int(output_tokens or 0))
        validated_model_id = str(model_id or "unknown").strip()
        
        # Vertex AI pricing (approximate, as of 2024 - prices may vary)
        # Prices per 1000 tokens in USD
        pricing_map = {
            # Gemini 2.0 models (newer)
            "gemini-2.0-flash-lite": {"input": 0.000075, "output": 0.0003},
            "gemini-2.0-flash": {"input": 0.000075, "output": 0.0003},
            
            # Gemini 1.5 models
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.00375},
            "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
            
            # Gemini 1.0 models
            "gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
            
            # Default pricing for unknown models
            "default": {"input": 0.001, "output": 0.003}
        }
        
        # Normalize model ID to match pricing keys
        model_key = validated_model_id.lower()
        if "gemini-2.0-flash-lite" in model_key:
            rates = pricing_map["gemini-2.0-flash-lite"]
        elif "gemini-2.0-flash" in model_key:
            rates = pricing_map["gemini-2.0-flash"]
        elif "gemini-1.5-pro" in model_key:
            rates = pricing_map["gemini-1.5-pro"]
        elif "gemini-1.5-flash" in model_key:
            rates = pricing_map["gemini-1.5-flash"]
        elif "gemini-1.0-pro" in model_key:
            rates = pricing_map["gemini-1.0-pro"]
        else:
            rates = pricing_map["default"]
        
        # Calculate total cost
        try:
            input_cost = (validated_input_tokens / 1000.0) * rates["input"]
            output_cost = (validated_output_tokens / 1000.0) * rates["output"]
            total_cost = input_cost + output_cost
            return max(0.0, float(total_cost))
        except Exception:
            logger.warning(f"Vertex AI cost calculation error for model {validated_model_id}, returning 0.0")
            return 0.0

    @staticmethod
    def calculate_azure_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
        """
        Calculate simple total cost estimate for Azure OpenAI models based on token usage.
        
        Returns total cost in USD as a float.
        """
        # Validate inputs
        validated_input_tokens = max(0, int(input_tokens or 0))
        validated_output_tokens = max(0, int(output_tokens or 0))
        validated_model_id = str(model_id or "unknown").strip()
        
        # Azure OpenAI pricing (approximate, as of 2024 - prices may vary)
        # Prices per 1000 tokens in USD
        pricing_map = {
            # GPT-4o models
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            
            # GPT-4 Turbo models
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            
            # GPT-4 models
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            
            # GPT-3.5 Turbo models
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            
            # Default pricing for unknown models
            "default": {"input": 0.002, "output": 0.006}
        }
        
        # Normalize model ID to match pricing keys
        model_key = validated_model_id.lower()
        if "gpt-4o-mini" in model_key:
            rates = pricing_map["gpt-4o-mini"]
        elif "gpt-4o" in model_key:
            rates = pricing_map["gpt-4o"]
        elif "gpt-4-turbo" in model_key:
            rates = pricing_map["gpt-4-turbo"]
        elif "gpt-4-32k" in model_key:
            rates = pricing_map["gpt-4-32k"]
        elif "gpt-4" in model_key:
            rates = pricing_map["gpt-4"]
        elif "gpt-3.5-turbo-16k" in model_key:
            rates = pricing_map["gpt-3.5-turbo-16k"]
        elif "gpt-3.5-turbo" in model_key:
            rates = pricing_map["gpt-3.5-turbo"]
        else:
            rates = pricing_map["default"]
        
        # Calculate total cost
        try:
            input_cost = (validated_input_tokens / 1000.0) * rates["input"]
            output_cost = (validated_output_tokens / 1000.0) * rates["output"]
            total_cost = input_cost + output_cost
            return max(0.0, float(total_cost))
        except Exception:
            logger.warning(f"Azure OpenAI cost calculation error for model {validated_model_id}, returning 0.0")
            return 0.0

    @staticmethod
    def calculate_openai_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
        """Calculate simple cost estimate for OpenAI models (unused - platform provides estimates)."""
        # Use same pricing as Azure (OpenAI pricing is similar)
        return CostCalculationUtils.calculate_azure_cost(input_tokens, output_tokens, model_id)

    @staticmethod
    def calculate_bedrock_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
        """
        Calculate simple total cost estimate for AWS Bedrock models (unused - platform provides estimates).
        
        Returns total cost in USD as a float.
        """
        # Validate inputs
        validated_input_tokens = max(0, int(input_tokens or 0))
        validated_output_tokens = max(0, int(output_tokens or 0))
        validated_model_id = str(model_id or "unknown").strip()
        
        # AWS Bedrock pricing (approximate, as of 2024 - prices may vary)
        # Prices per 1000 tokens in USD
        pricing_map = {
            # Claude models
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-instant": {"input": 0.00163, "output": 0.00551},
            
            # Mistral models
            "mistral-7b": {"input": 0.00015, "output": 0.0002},
            "mistral-8x7b": {"input": 0.00045, "output": 0.0007},
            "mistral-large": {"input": 0.008, "output": 0.024},
            
            # Default pricing for unknown models
            "default": {"input": 0.002, "output": 0.006}
        }
        
        # Normalize model ID to match pricing keys
        model_key = validated_model_id.lower()
        if "claude-3-5-sonnet" in model_key:
            rates = pricing_map["claude-3-5-sonnet"]
        elif "claude-3-sonnet" in model_key:
            rates = pricing_map["claude-3-sonnet"]
        elif "claude-3-haiku" in model_key:
            rates = pricing_map["claude-3-haiku"]
        elif "claude-instant" in model_key:
            rates = pricing_map["claude-instant"]
        elif "mistral-large" in model_key:
            rates = pricing_map["mistral-large"]
        elif "mistral-8x7b" in model_key:
            rates = pricing_map["mistral-8x7b"]
        elif "mistral-7b" in model_key:
            rates = pricing_map["mistral-7b"]
        else:
            rates = pricing_map["default"]
        
        # Calculate total cost
        try:
            input_cost = (validated_input_tokens / 1000.0) * rates["input"]
            output_cost = (validated_output_tokens / 1000.0) * rates["output"]
            total_cost = input_cost + output_cost
            return max(0.0, float(total_cost))
        except Exception:
            logger.warning(f"Bedrock cost calculation error for model {validated_model_id}, returning 0.0")
            return 0.0


class VertexInstrumentationUtils:
    """Utilities for Vertex AI LLM instrumentation - reduces code duplication"""
    
    @staticmethod
    def build_annotation_params(prompt: str, context: str, user_request: str, agent_name: str,
                                model_config: Dict[str, Any], llm_type: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build annotation parameters for Vertex AI LLM calls.
        
        Returns dict with input_data, metadata, tags, and prompt for LLMObs.annotate()
        """
        
        return {
            "input_data": prompt,
            "metadata": {
                "agent_name": agent_name,
                "llm_type": llm_type,
                "model_name": model_config.get("model_id", "unknown"),
                "model_provider": "google",
                "platform": "vertex",
                "context_provided": bool(documents),
                "context_length": len(context) if context else 0,
                "user_request": user_request,
                "prompt_length": len(prompt)
            },
            "tags": {
                "llm.model_name": model_config.get("model_id", "unknown"),
                "llm.model_provider": "google",
                "llm.platform": "vertex",
                "llm.agent": agent_name,
                "llm.llm_type": llm_type,
                "llm.context_provided": str(bool(documents)).lower()
            },
            "prompt": {
                "variables": {
                    "prompt": prompt,
                    "context": context or "",
                    "user_request": user_request or "",
                    "agent_name": agent_name
                },
                "rag_query_variables": ["user_request"],
                "rag_context_variables": ["context"]
            }
        }
    
    @staticmethod
    def extract_token_metrics(response, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract token metrics from Vertex AI response.
        
        Returns dict with input_tokens, output_tokens, total_tokens, and total_cost_usd
        """
        metrics = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0
        }
        
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                
                # Handle both dict and object structures
                if isinstance(usage, dict):
                    metrics["input_tokens"] = usage.get('input_tokens', 0)
                    metrics["output_tokens"] = usage.get('output_tokens', 0)
                    metrics["total_tokens"] = usage.get('total_tokens', 0)
                else:
                    # Object structure (original SDK)
                    metrics["input_tokens"] = getattr(usage, 'prompt_token_count', 0)
                    metrics["output_tokens"] = getattr(usage, 'candidates_token_count', 0)
                    metrics["total_tokens"] = getattr(usage, 'total_token_count', 0)
                
                # Calculate cost if we have token counts
                if metrics["input_tokens"] > 0 or metrics["output_tokens"] > 0:
                    metrics["total_cost_usd"] = CostCalculationUtils.calculate_vertex_ai_cost(
                        input_tokens=metrics["input_tokens"],
                        output_tokens=metrics["output_tokens"],
                        model_id=model_config.get("model_id", "unknown")
                    )
                    
                logger.info(f"📊 Vertex AI tokens - Input: {metrics['input_tokens']}, "
                          f"Output: {metrics['output_tokens']}, Total: {metrics['total_tokens']}")
                if metrics["total_cost_usd"] > 0:
                    logger.info(f"💰 Vertex AI cost estimate: ${metrics['total_cost_usd']:.6f} USD")
            else:
                logger.warning("⚠️  No usage_metadata found in Vertex AI response")
                
        except Exception as e:
            logger.warning(f"Could not extract token metrics: {e}")
            
        return metrics
    
    @staticmethod
    def build_output_metadata(response_content: str, agent_name: str, llm_type: str,
                             model_config: Dict[str, Any], documents: List[Dict[str, Any]],
                             token_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Build output metadata for Vertex AI annotation."""
        metadata = {
            "agent_name": agent_name,
            "llm_type": llm_type,
            "model_name": model_config.get("model_id", "unknown"),
            "model_provider": "google",
            "platform": "vertex",
            "response_length": len(response_content),
            "context_provided": bool(documents)
        }
        
        # Merge token metrics
        metadata.update(token_metrics)
        
        return metadata


class DocumentRetrievalUtils:
    """Utilities for document retrieval and RAG operations"""
    
    @staticmethod
    def normalize_query_words(words: List[str]) -> List[str]:
        """Normalize query words for better matching (lowercase, strip, dedupe)."""
        return list(set(word.lower().strip() for word in words if word.strip()))
    
    @staticmethod
    def analyze_request_keywords(user_request: str) -> List[str]:
        """Extract and normalize keywords from user request."""
        if not user_request:
            return []
            
        # Convert to lowercase for case-insensitive matching
        request_lower = user_request.lower()
        
        # Define keyword patterns
        keywords = []
        
        # Product-related keywords
        product_keywords = ['hoodie', 'mug', 'notebook', 'sticker', 'bottle', 't-shirt', 'tshirt',
                          'sweatshirt', 'headphones', 'beanie', 'product', 'item', 'merch']
        keywords.extend([kw for kw in product_keywords if kw in request_lower])
        
        # Promotion-related keywords  
        promo_keywords = ['discount', 'sale', 'promo', 'promotion', 'offer', 'deal', 'coupon',
                         'free shipping', 'percent off', '% off', 'black friday', 'cyber monday']
        keywords.extend([kw for kw in promo_keywords if kw in request_lower])
        
        # Customer service keywords
        cs_keywords = ['return', 'refund', 'exchange', 'policy', 'shipping', 'delivery',
                      'order', 'track', 'cancel', 'help', 'support', 'contact', 'faq']
        keywords.extend([kw for kw in cs_keywords if kw in request_lower])
        
        # Feedback keywords
        feedback_keywords = ['feedback', 'review', 'complaint', 'suggestion', 'compliment',
                            'issue', 'problem', 'love', 'hate', 'disappointed', 'happy']
        keywords.extend([kw for kw in feedback_keywords if kw in request_lower])
        
        return keywords
    
    @staticmethod
    def simple_relevance_score(doc_content: str, query_words: List[str]) -> float:
        """Calculate simple relevance score based on keyword matches."""
        if not query_words or not doc_content:
            return 0.0
            
        doc_lower = doc_content.lower()
        matches = sum(1 for word in query_words if word in doc_lower)
        return min(1.0, matches / max(1, len(query_words)))


class KnowledgeBaseUtils:
    """Utilities for loading and managing knowledge base resources"""
    
    @staticmethod
    def load_knowledge_source(source_type: str, resources_dir: str = None) -> Any:
        """Load a single knowledge source JSON file."""
        if resources_dir is None:
            resources_dir = os.path.join(os.path.dirname(__file__), "resources")
        
        file_mapping = {
            "faqs": "faqs.json",
            "products": "products.json", 
            "promotions": "promotions.json",
            "customer_service": "cs_info.json"
        }
        
        if source_type not in file_mapping:
            logger.warning(f"📂 Unknown knowledge source: {source_type}")
            return [] if source_type != "customer_service" else {}
            
        resource_file = file_mapping[source_type]
        file_path = os.path.join(resources_dir, resource_file)
        
        try:
            if os.path.exists(file_path):
                logger.info(f"📂 Loading {resource_file} from {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if source_type == "faqs":
                        result = data.get("FAQ", [])
                        logger.info(f"✅ Loaded {len(result)} FAQs")
                        return result
                    elif source_type == "products":
                        result = data.get("products", [])
                        logger.info(f"✅ Loaded {len(result)} products")
                        return result
                    elif source_type == "promotions":
                        result = data if isinstance(data, list) else data.get("promotions", [])
                        logger.info(f"✅ Loaded {len(result)} promotions")
                        return result
                    else:  # customer_service
                        logger.info(f"✅ Loaded customer service info")
                        return data
            else:
                logger.warning(f"⚠️ Knowledge source file not found: {file_path}")
                return [] if source_type != "customer_service" else {}
                
        except Exception as e:
            logger.error(f"📂 Error loading {source_type}: {e}")
            return [] if source_type != "customer_service" else {}
    
    @staticmethod
    def load_selective_knowledge_base(agents_needed: List[str], resources_dir: str = None) -> Dict[str, Any]:
        """Load only the knowledge sources needed for the selected agents."""
        # Mapping: agent → required knowledge sources
        knowledge_requirements = {
            "customer_service": ["faqs", "customer_service"],
            "product_specialist": ["products"], 
            "promotion_specialist": ["promotions"],
            "feedback_handler": ["faqs"]
        }
        
        # Determine which sources to load
        required_sources = set()
        for agent in agents_needed:
            if agent in knowledge_requirements:
                required_sources.update(knowledge_requirements[agent])
        
        # Load only required sources
        knowledge_base = {}
        for source in required_sources:
            knowledge_base[source] = KnowledgeBaseUtils.load_knowledge_source(source, resources_dir)
        
        # Initialize empty sources for agents that weren't needed
        all_sources = ["faqs", "products", "promotions", "customer_service"]
        for source in all_sources:
            if source not in knowledge_base:
                knowledge_base[source] = [] if source != "customer_service" else {}
        return knowledge_base
    
    @staticmethod
    def load_full_knowledge_base(resources_dir: str = None) -> Dict[str, Any]:
        """Load all knowledge base resources from JSON files."""
        if resources_dir is None:
            resources_dir = os.path.join(os.path.dirname(__file__), "resources")
        
        knowledge_base = {"faqs": [], "products": [], "promotions": [], "customer_service": {}}
        
        try:
            for resource_file, key in [
                ("faqs.json", "faqs"),
                ("products.json", "products"),
                ("promotions.json", "promotions"),
                ("cs_info.json", "customer_service")
            ]:
                file_path = os.path.join(resources_dir, resource_file)
                if os.path.exists(file_path):
                    logger.info(f"📂 Loading {resource_file} from {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if key == "faqs":
                            knowledge_base[key] = data.get("FAQ", [])
                            logger.info(f"✅ Loaded {len(knowledge_base[key])} FAQs")
                        elif key == "products":
                            knowledge_base[key] = data.get("products", [])
                            logger.info(f"✅ Loaded {len(knowledge_base[key])} products")
                        elif key == "promotions":
                            knowledge_base[key] = data if isinstance(data, list) else data.get("promotions", [])
                            logger.info(f"✅ Loaded {len(knowledge_base[key])} promotions")
                        else:
                            knowledge_base[key] = data
                            logger.info(f"✅ Loaded customer service info")
                else:
                    logger.warning(f"⚠️ File not found: {file_path}")
                    knowledge_base[key] = [] if key != "customer_service" else {}
                    
        except Exception as e:
            logger.error(f"📚 Knowledge base loading error: {e}")
        return knowledge_base


class PromptTrackingUtils:
    """Utilities for prompt tracking and versioning in LLM Observability"""
    
    @staticmethod
    def load_prompt_metadata(resources_dir: str = None) -> Dict[str, Dict[str, str]]:
        """Load prompt metadata from JSON file."""
        if resources_dir is None:
            resources_dir = os.path.join(os.path.dirname(__file__), "resources")
        
        try:
            metadata_path = os.path.join(resources_dir, "prompt-metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                logger.info(f"✅ Loaded prompt metadata for {len(metadata)} agent types")
                return metadata
        except FileNotFoundError:
            logger.warning("📂 Prompt metadata file not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading prompt metadata: {e}")
            return {}
    
    @staticmethod
    def get_prompt_metadata(agent_type: str, resources_dir: str = None) -> Dict[str, str]:
        """Get prompt ID and version for an agent type."""
        metadata = PromptTrackingUtils.load_prompt_metadata(resources_dir)
        
        if agent_type in metadata:
            return {
                "id": metadata[agent_type]["id"],
                "version": metadata[agent_type]["version"],
                "description": metadata[agent_type].get("description", "")
            }
        
        # Fallback metadata if not found
        logger.warning(f"No metadata found for {agent_type}, using default")
        return {
            "id": f"swagbot.{agent_type.replace('-', '_')}",
            "version": "1.0.0",
            "description": f"Agent prompt for {agent_type}"
        }
    
    @staticmethod
    def load_prompt_with_metadata(agent_type: str, resources_dir: str = None,
                                  opm_base_url: str = "http://localhost") -> Dict[str, Any]:
        """Load prompt content with tracking metadata, preferring OPM over local files."""
        if resources_dir is None:
            resources_dir = os.path.join(os.path.dirname(__file__), "resources")
        
        # Load the prompt template from OPM (falls back to file automatically)
        template = SynthesisUtils.load_agent_prompt_from_opm(agent_type, opm_base_url)
        
        # Get metadata
        metadata = PromptTrackingUtils.get_prompt_metadata(agent_type, resources_dir)
        
        return {
            "template": template,
            "id": metadata["id"],
            "version": metadata["version"],
            "description": metadata.get("description", "")
        }


class OPMTelemetryUtils:
    """Utilities for recording prompt execution telemetry to OPM."""

    @staticmethod
    def get_prompt_id(agent_type: str) -> Optional[int]:
        """Return the cached OPM prompt ID for an agent type, or None if not loaded from OPM."""
        return _opm_prompt_id_cache.get(agent_type)

    @staticmethod
    def record_execution(
        prompt_id: int,
        rendered_prompt: str,
        response: str,
        execution_time_ms: int,
        input_variables: Dict[str, Any] = None,
        token_count: int = None,
        cost: float = None,
        success: int = 1,
        rating: int = None,
        metadata: Dict[str, Any] = None,
        opm_base_url: str = "http://localhost",
    ) -> None:
        """Fire-and-forget: POST execution telemetry to /api/prompts/{id}/executions."""
        try:
            if token_count is None:
                token_count = (len(rendered_prompt) + len(response)) // 4

            payload: Dict[str, Any] = {
                "rendered_prompt": rendered_prompt,
                "response": response,
                "execution_time_ms": execution_time_ms,
                "token_count": token_count,
                "success": success,
            }
            if input_variables is not None:
                payload["input_variables"] = input_variables
            if cost is not None:
                payload["cost"] = cost
            if rating is not None:
                payload["rating"] = rating
            if metadata is not None:
                payload["metadata"] = metadata

            url = f"{opm_base_url}/api/prompts/{prompt_id}/executions"
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310
                logger.debug(f"📊 OPM execution recorded for prompt {prompt_id} (HTTP {resp.status})")
        except Exception as exc:
            logger.debug(f"📊 OPM telemetry skipped for prompt {prompt_id}: {exc}")


class SynthesisUtils:
    """Utilities for response synthesis and prompt management"""
    
    @staticmethod
    def load_agent_prompt(agent_type: str, resources_dir: str = None) -> str:
        """Load agent-specific prompt from file with fallback."""
        if resources_dir is None:
            resources_dir = os.path.join(os.path.dirname(__file__), "resources")
        
        try:
            prompt_path = os.path.join(resources_dir, f"prompt-{agent_type.replace('_', '-')}.txt")
            logger.info(f"📂 Loading {agent_type} prompt from {prompt_path}")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content_length = len(content)
                token_estimate = content_length // 4
                logger.info(f"✅ Successfully loaded {agent_type} prompt ({content_length} chars ≈ {token_estimate} tokens)")
                return content
        except FileNotFoundError:
            logger.warning(f"📂 {agent_type} prompt file not found, using fallback")
            
            fallbacks = {
                "planning": """Analyze user requests and route to appropriate agents.

JSON only: {"primary_category": "Customer-Service|Product-Information|Promotions|Feedback", "confidence": 0.9, "agents_needed": ["agent1"], "reasoning": "Brief explanation"}

Agents: customer_service, product_specialist, promotion_specialist, feedback_handler""",
                "orchestrator": """Extract focused subsets of the user request for each agent.

ORIGINAL REQUEST: "{user_request}"
CATEGORY: {category}
AGENTS NEEDED: {agents_needed}

Agent responsibilities:
- customer_service: Support, returns, complaints, account issues
- product_specialist: Product info, specifications, pricing
- promotion_specialist: Discounts, deals, coupon requests
- feedback_handler: Reviews, ratings, feedback submission

Return JSON object only:
{"agent_name": "subset of original request", ...}

Example: {"product_specialist": "What's the price?", "promotion_specialist": "Any discounts?"}""",
                "customer-service": "Handle customer support, returns, complaints. Professional tone, clear solutions.",
                "product-specialist": "Provide product info, specs, recommendations. Use product data, include prices/images.",
                "promotion-specialist": "Find deals, discounts, special offers. Clear savings info, coupon codes.",
                "feedback-handler": "Process feedback, reviews, suggestions. Professional, appreciative tone.",
                "synthesizer": "Combine specialist responses into beautifully formatted HTML for web display. No agent mentions."
            }
            return fallbacks.get(agent_type, f"You are a helpful {agent_type.replace('-', ' ')} assistant.")
    
    @staticmethod
    def load_agent_prompt_from_opm(agent_type: str, opm_base_url: str = "http://localhost") -> str:
        """Load agent-specific prompt from OPM API with fallback to file-based loading.

        Maps agent_type to the corresponding OPM prompt name, searches for it via the
        OPM REST API, and returns the prompt content string. Falls back to
        load_agent_prompt() if OPM is unreachable or the prompt is not found.
        """
        name_map = {
            "planning":            "SwagBot Planning Agent",
            "orchestrator":        "SwagBot Orchestrator",
            "customer-service":    "SwagBot Customer Service",
            "customer_service":    "SwagBot Customer Service",
            "product-specialist":  "SwagBot Product Specialist",
            "product_specialist":  "SwagBot Product Specialist",
            "promotion-specialist": "SwagBot Promotion Specialist",
            "promotion_specialist": "SwagBot Promotion Specialist",
            "feedback-handler":    "SwagBot Feedback Handler",
            "feedback_handler":    "SwagBot Feedback Handler",
            "synthesizer":         "SwagBot Response Synthesizer",
        }

        prompt_name = name_map.get(agent_type)
        if not prompt_name:
            logger.warning(f"🔍 No OPM name mapping for agent_type '{agent_type}', falling back to file")
            return SynthesisUtils.load_agent_prompt(agent_type)

        try:
            # Step 1: search for the prompt by name to get its ID
            search_params = urllib.parse.urlencode({"search": prompt_name, "limit": 10})
            search_url = f"{opm_base_url}/api/prompts/?{search_params}"
            logger.info(f"🌐 Searching OPM for prompt '{prompt_name}' at {search_url}")

            with urllib.request.urlopen(search_url, timeout=5) as resp:  # noqa: S310
                search_results = json.loads(resp.read().decode())

            # Find exact name match
            prompt_id = None
            for item in search_results:
                if item.get("name") == prompt_name:
                    prompt_id = item["id"]
                    break

            if prompt_id is None:
                logger.warning(f"🔍 Prompt '{prompt_name}' not found in OPM, falling back to file")
                return SynthesisUtils.load_agent_prompt(agent_type)

            # Cache the prompt ID so telemetry can reference it later
            _opm_prompt_id_cache[agent_type] = prompt_id

            # Step 2: fetch full prompt (includes content)
            detail_url = f"{opm_base_url}/api/prompts/{prompt_id}"
            logger.info(f"🌐 Fetching OPM prompt id={prompt_id} from {detail_url}")

            with urllib.request.urlopen(detail_url, timeout=5) as resp:  # noqa: S310
                prompt_data = json.loads(resp.read().decode())

            content = prompt_data.get("content", "")
            version = prompt_data.get("version", "unknown")
            content_length = len(content)
            token_estimate = content_length // 4
            logger.info(
                f"✅ Loaded '{prompt_name}' v{version} from OPM "
                f"({content_length} chars ≈ {token_estimate} tokens)"
            )
            return content

        except urllib.error.URLError as exc:
            logger.warning(f"⚠️  OPM unreachable ({exc}), falling back to file for '{agent_type}'")
            return SynthesisUtils.load_agent_prompt(agent_type)
        except Exception as exc:
            logger.warning(f"⚠️  OPM error ({exc}), falling back to file for '{agent_type}'")
            return SynthesisUtils.load_agent_prompt(agent_type)

    @staticmethod
    def enhance_single_response(response: str, user_request: str, synthesizer_prompt: str,
                               agent_contexts: Dict[str, List[Dict[str, Any]]], llm_caller,
                               opm_base_url: str = "http://localhost") -> str:
        """Enhance single agent response using original user request."""
        try:
            # Collect documents from the single agent for context
            all_documents = []
            if agent_contexts:
                for agent_name, documents in agent_contexts.items():
                    all_documents.extend(documents)
                logger.info(f"📝 Single-agent synthesizer using {len(all_documents)} context documents")
            
            # Build document context
            document_context = ""
            if all_documents:
                document_context = "\n\nRELEVANT SOURCE DOCUMENTS:\n"
                for i, doc in enumerate(all_documents[:10], 1):
                    doc_name = doc.get('name', f'Document {i}')
                    doc_content = doc.get('content', '')[:500]
                    document_context += f"\n{i}. {doc_name}:\n{doc_content}...\n"
                logger.info(f"📚 Including {len(all_documents[:10])} source documents in synthesizer context")
            
            # Load prompt metadata for tracking
            prompt_data = PromptTrackingUtils.load_prompt_with_metadata("synthesizer", opm_base_url=opm_base_url)
            
            # Build prompt variables
            prompt_variables = {
                "synthesizer_prompt": synthesizer_prompt,
                "user_request": user_request,
                "response": response,
                "document_context": document_context
            }
            
            enhancement_prompt = f"""{synthesizer_prompt}

Original User Request: {user_request}

Specialist Response: {response}{document_context}

Instructions: Address the user's complete original request using BOTH the specialist response AND the source documents as your information sources. Ensure accuracy by referencing the original documents when needed.

Enhanced Response:"""

            enhanced_response = llm_caller.call_synthesis_llm(
                prompt=enhancement_prompt,
                user_request=user_request,
                documents=all_documents,
                prompt_template=synthesizer_prompt,
                prompt_id=prompt_data["id"],
                prompt_version=prompt_data["version"],
                prompt_variables=prompt_variables
            )
            
            logger.info("✅ Synthesizer enhanced single response with full context")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Single response enhancement error: {e}")
            return f"{response}\n\n<p><em>Response processed by SwagBot autonomous agents</em></p>"
    
    @staticmethod
    def synthesize_multi_agent_responses(agent_responses: Dict[str, str], 
                                        agent_contexts: Dict[str, List[Dict[str, Any]]], 
                                        user_request: str, synthesizer_prompt: str, llm_caller,
                                        opm_base_url: str = "http://localhost") -> str:
        """Combine multiple agent responses to address original user request."""
        try:
            # Collect all documents from specialized agents
            all_documents = []
            for agent_name, documents in agent_contexts.items():
                all_documents.extend(documents)
            
            logger.info(f"📝 Synthesizer collected {len(all_documents)} total documents from {len(agent_contexts)} agents")
            
            # Build specialist responses summary
            specialist_summary = ""
            agent_display_names = {
                "customer_service": "Customer Service Specialist",
                "product_specialist": "Product Specialist", 
                "promotion_specialist": "Promotion Specialist",
                "feedback_handler": "Feedback Specialist"
            }
            
            for agent_name, response in agent_responses.items():
                display_name = agent_display_names.get(agent_name, agent_name.replace('_', ' ').title())
                # Clean HTML tags for the prompt
                clean_response = HTMLUtils.strip_html_tags(response) if hasattr(HTMLUtils, 'strip_html_tags') else response
                specialist_summary += f"\n{display_name}:\n{clean_response}\n"
            
            # Build document context
            document_context = ""
            if all_documents:
                document_context = "\n\nRELEVANT SOURCE DOCUMENTS:\n"
                for i, doc in enumerate(all_documents[:10], 1):
                    doc_name = doc.get('name', f'Document {i}')
                    doc_content = doc.get('content', '')[:500]
                    document_context += f"\n{i}. {doc_name}:\n{doc_content}...\n"
                logger.info(f"📚 Including {len(all_documents[:10])} source documents in synthesizer context")
            
            # Load prompt metadata for tracking
            prompt_data = PromptTrackingUtils.load_prompt_with_metadata("synthesizer", opm_base_url=opm_base_url)
            
            # Build prompt variables
            prompt_variables = {
                "synthesizer_prompt": synthesizer_prompt,
                "user_request": user_request,
                "specialist_summary": specialist_summary,
                "document_context": document_context
            }
            
            synthesis_prompt = f"""{synthesizer_prompt}

Original User Request: {user_request}

Specialist Responses: {specialist_summary}{document_context}

Instructions: Create a comprehensive response that fully addresses the original user request. Use BOTH the specialist responses AND the source documents as your information sources.

Synthesized Response:"""

            synthesized_response = llm_caller.call_synthesis_llm(
                prompt=synthesis_prompt,
                user_request=user_request,
                documents=all_documents,
                prompt_template=synthesizer_prompt,
                prompt_id=prompt_data["id"],
                prompt_version=prompt_data["version"],
                prompt_variables=prompt_variables
            )
            
            logger.info("✅ Synthesizer combined multiple agent responses with full context")
            return synthesized_response
            
        except Exception as e:
            logger.error(f"Multi-agent synthesis error: {e}")
            # Fallback
            fallback = f"<h3>Regarding your request: {user_request}</h3>\n<p>Here's what our specialists found:</p>\n"
            agent_display_names = {
                "customer_service": "Customer Service Specialist",
                "product_specialist": "Product Specialist", 
                "promotion_specialist": "Promotion Specialist",
                "feedback_handler": "Feedback Specialist"
            }
            for agent_name, response in agent_responses.items():
                display_name = agent_display_names.get(agent_name, agent_name.replace('_', ' ').title())
                fallback += f"<h4>{display_name}:</h4>\n{response}\n"
            return fallback


class UIConfigUtils:
    """Utilities for UI configuration and model display"""
    
    @staticmethod
    def get_friendly_model_name(model_id: str) -> str:
        """Get user-friendly model display name"""
        model_lower = model_id.lower()
        
        # Claude models
        if 'claude-instant' in model_lower:
            return "Claude Instant"
        elif 'claude-3-haiku' in model_lower:
            return "Claude 3 Haiku"
        elif 'claude-3-sonnet' in model_lower:
            return "Claude 3 Sonnet"
        elif 'claude-3-opus' in model_lower:
            return "Claude 3 Opus"
        elif 'claude-3-5-sonnet' in model_lower:
            return "Claude 3.5 Sonnet"
        elif 'claude-3-5-haiku' in model_lower:
            return "Claude 3.5 Haiku"
        # Fallback
        else:
            return model_id.split('/')[-1] if '/' in model_id else model_id
    
    @staticmethod
    def get_model_logo(model_id: str) -> str:
        """Get the logo path for a model based on its provider"""
        model_lower = model_id.lower()
        
        # Google Gemini models
        if 'gemini' in model_lower or 'google' in model_lower:
            return "/static/images/gemini-logo.png"
        # Anthropic Claude models  
        elif 'claude' in model_lower or 'anthropic' in model_lower:
            return "/static/images/anthropic-logo.png"
        # OpenAI models
        elif 'gpt' in model_lower or 'openai' in model_lower:
            return "/static/images/openai-logo.png"
        # Fallback to generic logo
        else:
            return "/static/images/default-chatbot-logo.jpg"
    
    @staticmethod
    def get_model_description(model_id: str) -> str:
        """Get description for model tooltips"""
        model_lower = model_id.lower()
        
        if 'claude-instant' in model_lower:
            return "Claude Instant - Ultra-fast, cost-effective model ideal for quick planning and simple tasks."
        elif 'claude-3-haiku' in model_lower:
            return "Claude 3 Haiku - Fast and efficient model with excellent reasoning capabilities."
        elif 'claude-3-sonnet' in model_lower:
            return "Claude 3 Sonnet - Powerful model with advanced reasoning and analysis capabilities."
        elif 'claude-3-opus' in model_lower:
            return "Claude 3 Opus - Anthropic's most capable model with exceptional reasoning abilities."
        elif 'claude-3-5-sonnet' in model_lower:
            return "Claude 3.5 Sonnet - Enhanced version with improved reasoning, coding, and analysis capabilities."
        elif 'claude-3-5-haiku' in model_lower:
            return "Claude 3.5 Haiku - Next-generation fast model with improved capabilities."
        else:
            return f"AI Model: {model_id} - Advanced language model for intelligent text processing and generation."


class ParsingUtils:
    """Utilities for parsing LLM responses"""
    
    @staticmethod
    def extract_json_from_response(response: str) -> str:
        """Extract JSON from LLM response, handling markdown fences."""
        if not response:
            return ""
            
        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL | re.IGNORECASE)
        if json_match:
            return json_match.group(1).strip()
        
        # Try to find JSON object or array directly
        # Look for opening brace/bracket
        json_start = -1
        for i, char in enumerate(response):
            if char in ['{', '[']:
                json_start = i
                break
        
        if json_start == -1:
            return response.strip()
        
        # Find matching closing brace/bracket
        if response[json_start] == '{':
            brace_count = 0
            for idx, char in enumerate(response[json_start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return response[json_start:json_start+idx+1]
        elif response[json_start] == '[':
            bracket_count = 0
            for idx, char in enumerate(response[json_start:]):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        return response[json_start:json_start+idx+1]
        
        return response.strip()
    
    @staticmethod
    def keyword_based_planning_fallback(user_request: str) -> Dict[str, Any]:
        """Generate fallback planning result based on keyword analysis."""
        keywords = DocumentRetrievalUtils.analyze_request_keywords(user_request)
        agents_needed = []
        
        # Product-related
        if any(kw in keywords for kw in ['hoodie', 'mug', 'notebook', 'sticker', 'bottle', 
                                         't-shirt', 'tshirt', 'sweatshirt', 'product', 'item']):
            agents_needed.append("product_specialist")
        
        # Promotion-related
        if any(kw in keywords for kw in ['discount', 'sale', 'promo', 'promotion', 'offer', 'deal']):
            agents_needed.append("promotion_specialist")
        
        # Customer service
        if any(kw in keywords for kw in ['return', 'refund', 'policy', 'shipping', 'help', 
                                         'support', 'order', 'track']):
            agents_needed.append("customer_service")
        
        # Feedback
        if any(kw in keywords for kw in ['feedback', 'review', 'complaint', 'suggestion']):
            agents_needed.append("feedback_handler")
        
        # Default to customer service if nothing detected
        if not agents_needed:
            agents_needed = ["customer_service"]
        
        return {
            "agents_needed": agents_needed,
            "reasoning": f"Keyword-based analysis detected: {', '.join(agents_needed)}",
            "agent_prompts": {}
        }


# ===============================================================================
# UTILITIES LOADED
# ===============================================================================

logger.info("📚 SWAGBOT UTILITIES LOADED")
logger.info("   ✅ Pure utility functions + instrumentation helpers")
logger.info("   ✅ LLM cost calculation utilities for all platforms")
logger.info("   ✅ Vertex AI instrumentation utilities")
logger.info("   ✅ Document retrieval and RAG utilities")
logger.info("   ✅ Parsing utilities for LLM responses")
logger.info("   ✅ Production-ready clean architecture")