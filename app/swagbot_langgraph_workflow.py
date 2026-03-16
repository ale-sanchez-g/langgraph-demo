"""
SwagBot LangGraph Optimized Workflow - Production Implementation

This implementation follows LangGraph best practices with the optimal workflow pattern:
- Planning → Orchestrator → Specialized Agents → Synthesizer
- Each specialized agent handles its own complete workflow (retrieval + LLM + response)
- Proper LangGraph flow control with parallel execution
- Clean separation of concerns and autonomous agent patterns

Architecture:
Planning → Orchestrator → [Specialized Agents in Parallel] → Synthesizer → END

This is the main workflow implementation for SwagBot
"""

import json
import os
import logging
import operator
import random
import re
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Sequence, Annotated
from botocore.exceptions import ClientError
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from swagbot_langgraph_config import LangChainConfig
from swagbot_utils import (DocumentUtils, HTMLUtils, DocumentHandlers, CostCalculationUtils, LangGraphUtils,
                           VertexInstrumentationUtils, DocumentRetrievalUtils, ParsingUtils, KnowledgeBaseUtils,
                           SynthesisUtils, PromptTrackingUtils)

# Datadog LLM Observability decorators
from ddtrace.llmobs.decorators import agent, workflow, retrieval, llm, task, tool
from ddtrace.llmobs import LLMObs
from ddtrace import tracer

# Note: Logging configuration is handled in swagbot_app.py for consistency
logger = logging.getLogger(__name__)

# Global config instance
config = LangChainConfig()



# ===============================================================================
# WORKFLOW STATE SCHEMA
# ===============================================================================

class WorkflowState(TypedDict):
    """
    State schema for optimized workflow following LangGraph best practices.
    Planning → Orchestrator → Specialized Agents → Synthesizer
    
    Uses Datadog LLM Observability standard field names for better tracing.
    """
    # Primary I/O - Datadog LLM Observability Standard
    input: str                  # User request input
    output: str                 # Final response output
    
    # Planning results
    planning_result: Dict[str, Any]
    agents_needed: List[str]
    
    # Agent-specific tasks for focused execution
    agent_tasks: Dict[str, str]  # Maps agent_name -> specific task for that agent
    
    # Selective knowledge base (loaded based on agents_needed)
    knowledge_base: Dict[str, Any]  # Pre-loaded knowledge sources for agents
    
    # Agent results (each agent produces complete output)
    agent_responses: Annotated[Dict[str, str], LangGraphUtils.merge_agent_responses]
    agent_contexts: Annotated[Dict[str, List[Dict[str, Any]]], LangGraphUtils.merge_agent_contexts]
    
    # Metadata
    workflow_path: Annotated[List[str], operator.add]
    error: Optional[str]

# =============================================================================== 
# MAIN WORKFLOW FUNCTION
# ===============================================================================

# Core workflow execution function
def execute_workflow_core(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    return swagbot_workflow.app.invoke(initial_state)

# Main workflow execution function
@workflow(name="swagbot_workflow")
def process_swagbot_request(input: str) -> Dict[str, Any]:
    try:
        initial_state = {
            "input": input,
            "output": "",
            "planning_result": {},
            "agents_needed": [],
            "agent_tasks": {},
            "agent_responses": {},
            "agent_contexts": {},
            "workflow_path": [],
            "error": None
        }
        
        result = execute_workflow_core(initial_state)
        
        # Check for error in result
        if result.get("error"):
            # Set error on workflow span
            workflow_span = tracer.current_span()
            if workflow_span:
                workflow_span.error = 1
                workflow_span.set_tag("error.message", result["error"])
                workflow_span.set_tag("error.type", "WorkflowError")
                workflow_span.set_tag("error.code", "ValidationException")
        
        # Paste custom annotation here
        output = result.get("output", "No response generated")
        agents_used = list(result.get("agent_responses", {}).keys())

        # Clean HTML tags for better UI display
        clean_response = HTMLUtils.strip_html_tags(output)

        LLMObs.annotate(
            input_data=input,
            output_data=clean_response,
            metadata={
                "workflow_type": "swagbot_langgraph",
                "request_length": len(input),
                "response_length": len(output),
                "agents_count": len(agents_used),
                "error": bool(result.get("error"))
            }
        )
        
        # Capture trace context using the proper export format
        # Extract the full span context dictionary that can be used directly for evaluation submission
        span_context = LLMObs.export_span(span=None)
        
        # Add the complete span context to result for evaluation submission
        result["span_context"] = span_context
        
        return swagbot_workflow._convert_to_response_format(result)
        
    except Exception as e:
        # Set error on workflow span for any unexpected errors
        workflow_span = tracer.current_span()
        if workflow_span:
            workflow_span.error = 1
            workflow_span.set_tag("error.message", str(e))
            workflow_span.set_tag("error.type", e.__class__.__name__)
            workflow_span.set_tag("error.code", "UnexpectedError")
        raise

# ===============================================================================
# MAIN WORKFLOW CLASS - Core Instrumented Functions
# ===============================================================================

class SwagBotWorkflow:
    """
    Production SwagBot workflow implementation following LangGraph best practices.
    Features:
    - Planning → Orchestrator → Specialized Agents → Synthesizer architecture
    - Each specialized agent is completely self-contained
    - Proper LangGraph flow control with parallel execution
    - Autonomous agent patterns for scalability
    """
    
    def __init__(self):
        self.llm_caller = LLMCaller(config)
        self.app = self._build_workflow()

        self.synthesizer_prompt = None
        
    def _build_workflow(self) -> StateGraph:
        """
        Build optimized workflow following LangGraph best practices.
        Planning → Orchestrator → Specialized Agents → Synthesizer
        """
        workflow = StateGraph(WorkflowState)
        
        # ============= PLANNING PHASE =============
        workflow.add_node("Planning", self._planning)
        
        # ============= ORCHESTRATOR PHASE =============
        workflow.add_node("Orchestrator", self._orchestrator)
        
        # ============= SPECIALIZED AGENTS =============
        workflow.add_node("Customer Service", self._customer_service_agent)
        workflow.add_node("Product Specialist", self._product_specialist_agent)
        workflow.add_node("Promotion Specialist", self._promotion_specialist_agent)
        workflow.add_node("Feedback Handler", self._feedback_handler_agent)
        
        # ============= SYNTHESIZER PHASE =============
        workflow.add_node("Response Synthesizer", self._synthesizer)
        workflow.add_node("error_handler", self._error_handler)
        
        # ============= AGENT-CENTRIC WORKFLOW FLOW =============
        workflow.add_edge(START, "Planning")
        
        # Planning routes to Orchestrator Agent or error
        workflow.add_conditional_edges(
            "Planning",
            self._route_after_planning,
            {
                "dispatch": "Orchestrator",
                "error": "error_handler"
            }
        )
        
        # Orchestrator Agent fans out to specialized agents in parallel
        workflow.add_conditional_edges(
            "Orchestrator",
            self._dispatch_specialized_agents,
            ["Customer Service", "Product Specialist", "Promotion Specialist", "Feedback Handler"]
        )
        
        # All specialized agents fan-in to Response Synthesizer Agent
        workflow.add_edge("Customer Service", "Response Synthesizer")
        workflow.add_edge("Product Specialist", "Response Synthesizer")
        workflow.add_edge("Promotion Specialist", "Response Synthesizer")
        workflow.add_edge("Feedback Handler", "Response Synthesizer")
        
        # Add conditional edge from Response Synthesizer to error_handler or END
        workflow.add_conditional_edges(
            "Response Synthesizer",
            self._route_after_synthesis,
            {
                "success": END,
                "error": "error_handler"
            }
        )
        workflow.add_edge("error_handler", END)
        
        compiled_workflow = workflow.compile()
        
        logger.info("🎯 AGENTIC AI SWAGBOT WORKFLOW:")
        logger.info("   🤖 Planning Agent → 🎯 Orchestrator Agent → 🚀 [Specialist Agents] → 📝 Synthesizer Agent")
                
        return compiled_workflow
    
    def process_request(self, input: str) -> Dict[str, Any]:
        """Instance method that delegates to module-level function for cleaner tracing."""
        return process_swagbot_request(input)
    
    # ===============================================================================
    # CORE WORKFLOW AGENTS - Primary Instrumented Functions
    # ===============================================================================
    # Architecture: Planning → Orchestrator → [Specialized Agents] → Synthesizer
    
    @agent(name="Planning Agent")
    def _planning(self, state: WorkflowState) -> WorkflowState:
        """
        Planning agent with comprehensive Datadog instrumentation.
        Analyzes user requests and determines appropriate specialist agents.
        """
        logger.info("📋 Planning agent analyzing request")
        
        try:
            # Load planning prompt with metadata
            prompt_data = PromptTrackingUtils.load_prompt_with_metadata("planning")
            planning_prompt = prompt_data["template"]
            
            # Build prompt variables
            prompt_variables = {
                "user_request": state['input']
            }
            
            # Build full prompt
            full_prompt = f"{planning_prompt}\n\nUser Request: {state['input']}\n\nProvide your analysis as JSON:"
            
            # Execute planning LLM call with instrumentation and prompt tracking
            response = self.llm_caller.call_planning_llm(
                prompt=full_prompt,
                user_request=state['input'],
                prompt_template=planning_prompt,
                prompt_id=prompt_data["id"],
                prompt_version=prompt_data["version"],
                prompt_variables=prompt_variables
            )
            
            # Parse JSON response with fallback
            planning_result = self._parse_planning_response(response, state['input'])
            
            return {
                "planning_result": planning_result,
                "agents_needed": planning_result.get("agents_needed", ["customer_service"]),
                "workflow_path": ["Planning"]
            }
            
        except Exception as e:
            logger.error(f"Planning agent error: {e}")
            return {"error": f"Planning failed: {str(e)}"}
    
    # ===============================================================================
    # ORCHESTRATOR - Coordination Function
    # ===============================================================================
    
    @agent(name="Orchestrator Agent")
    def _orchestrator(self, state: WorkflowState) -> WorkflowState:
        """
        Orchestrator creates focused request subsets for each specialized agent.
        Prevents scope overlap and ensures each agent handles specific concerns.
        """
        import time
        
        logger.info("🚀 Orchestrator creating agent-specific request subsets")
        
        # Small pause for better Datadog Flamegraph visibility
        time.sleep(0.1)
        
        agents_needed = state.get("agents_needed", [])
        user_request = state.get("input", "")
        planning_result = state.get("planning_result", {})
        
        logger.info(f"🎯 Orchestrator processing {len(agents_needed)} agents: {agents_needed}")
        
        # Load selective knowledge base based on agents needed
        # REVERT: Replace with: knowledge_base = self._load_full_knowledge_base_backup()
        knowledge_base = self._load_selective_knowledge_base(agents_needed)
        logger.info("📚 Selective knowledge base loaded and cached for agents")
        
        # Create agent-specific request subsets
        agent_tasks = self._create_agent_prompts_from_planning(
            user_request, 
            planning_result, 
            agents_needed
        )
        
        logger.info("📋 Agent-specific request subsets created:")
        for agent_name, subset in agent_tasks.items():
            logger.info(f"   🤖 {agent_name}: {subset[:100]}...")
        
        return {
            "workflow_path": ["Orchestrator"], 
            "agent_tasks": agent_tasks,
            "knowledge_base": knowledge_base
        }
    
    # ===============================================================================
    # SPECIALIZED AGENTS - Core Instrumented Functions
    # ===============================================================================

    @agent(name="Customer Service Agent")
    def _customer_service_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Customer Service Agent with LLM support.
        Handles customer support, returns, complaints, and general assistance.
        """
        logger.info("🎧 Customer Service Agent executing autonomous workflow")
        
        try:
            # Get agent-specific task for focused execution
            agent_tasks = state.get("agent_tasks", {})
            specific_task = agent_tasks.get("customer_service", state.get("input", ""))
            logger.info(f"🎯 Customer Service processing: {specific_task[:100]}...")
            
            # Load agent prompt and retrieve relevant documents
            agent_prompt = self._load_agent_prompt("customer-service")
            knowledge_base = state.get("knowledge_base", None)
            documents = self._retrieve_documents("customer_service", specific_task, knowledge_base)
            
            # Execute LLM call with RAG annotation
            response = self._call_agent_with_rag(
                agent_type="customer_service",
                agent_task=specific_task,
                original_request=state['input'],
                documents=documents,
                agent_prompt=agent_prompt
            )
            
            return {
                "agent_responses": {"customer_service": response},
                "agent_contexts": {"customer_service": documents},
                "workflow_path": ["customer_service_agent"]
            }
            
        except Exception as e:
            logger.error(f"Customer service agent error: {e}")
            return {"error": f"Customer service agent failed: {str(e)}"}
    
    @agent(name="Product Specialist Agent")
    def _product_specialist_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Product Specialist Agent with Datadog instrumentation.
        Handles product information, specifications, pricing, and recommendations.
        """
        logger.info("🛍️ Product Specialist Agent executing autonomous workflow")
        
        try:
            # Get agent-specific task for focused execution
            agent_tasks = state.get("agent_tasks", {})
            specific_task = agent_tasks.get("product_specialist", state.get("input", ""))
            original_request = state.get("input", "")
            
            logger.info(f"🎯 Product Specialist processing:")
            logger.info(f"   Original request: {original_request}")
            logger.info(f"   Focused subset: {specific_task}")
            
            # Load agent prompt and retrieve product documents
            agent_prompt = self._load_agent_prompt("product-specialist")
            knowledge_base = state.get("knowledge_base", None)
            documents = self._retrieve_documents("product_specialist", specific_task, knowledge_base)
            
            # Execute LLM call with RAG using focused subset
            response = self._call_agent_with_rag(
                agent_type="product_specialist",
                agent_task=specific_task,
                original_request=state['input'],
                documents=documents,
                agent_prompt=agent_prompt
            )
            
            return {
                "agent_responses": {"product_specialist": response},
                "agent_contexts": {"product_specialist": documents},
                "workflow_path": ["product_specialist_agent"]
            }
            
        except Exception as e:
            logger.error(f"Product specialist agent error: {e}")
            return {"error": f"Product specialist agent failed: {str(e)}"}
    
    @agent(name="Promotion Specialist Agent")
    def _promotion_specialist_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Promotion Specialist Agent with focused request processing.
        Handles discounts, deals, special offers, and promotional campaigns.
        """
        logger.info("🎁 Promotion Specialist Agent executing autonomous workflow")
        
        try:
            # Get agent-specific request subset for precise handling
            agent_tasks = state.get("agent_tasks", {})
            specific_task = agent_tasks.get("promotion_specialist", state.get("input", ""))
            original_request = state.get("input", "")
            
            logger.info(f"🎯 Promotion Specialist processing:")
            logger.info(f"   Original request: {original_request}")
            logger.info(f"   Focused subset: {specific_task}")
            
            # Load agent prompt and retrieve promotional documents
            agent_prompt = self._load_agent_prompt("promotion-specialist")
            knowledge_base = state.get("knowledge_base", None)
            documents = self._retrieve_documents("promotion_specialist", specific_task, knowledge_base)
            
            response = self._call_agent_with_rag(
                agent_type="promotion_specialist", 
                agent_task=specific_task,
                original_request=state['input'],
                documents=documents,
                agent_prompt=agent_prompt
            )
            
            return {
                "agent_responses": {"promotion_specialist": response},
                "agent_contexts": {"promotion_specialist": documents},
                "workflow_path": ["promotion_specialist_agent"]
            }
            
        except Exception as e:
            logger.error(f"Promotion specialist agent error: {e}")
            return {"error": f"Promotion specialist agent failed: {str(e)}"}
    
    @agent(name="Feedback Handler Agent")
    def _feedback_handler_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Feedback Handler Agent for customer reviews and suggestions.
        Processes feedback, reviews, ratings, and improvement suggestions.
        """
        logger.info("⭐ Feedback Handler Agent executing autonomous workflow")
        
        try:
            # Get agent-specific task
            agent_tasks = state.get("agent_tasks", {})
            specific_task = agent_tasks.get("feedback_handler", state.get("input", ""))
            logger.info(f"🎯 Feedback Handler processing: {specific_task[:100]}...")
            
            # Load agent prompt and retrieve relevant documents
            agent_prompt = self._load_agent_prompt("feedback-handler")
            knowledge_base = state.get("knowledge_base", None)
            documents = self._retrieve_documents("feedback_handler", specific_task, knowledge_base)
            
            response = self._call_agent_with_rag(
                agent_type="feedback_handler",
                agent_task=specific_task,
                original_request=state['input'],
                documents=documents,
                agent_prompt=agent_prompt
            )
            
            return {
                "agent_responses": {"feedback_handler": response},
                "agent_contexts": {"feedback_handler": documents},
                "workflow_path": ["feedback_handler_agent"]
            }
            
        except Exception as e:
            logger.error(f"Feedback handler agent error: {e}")
            return {"error": f"Feedback handler agent failed: {str(e)}"}
    
    # ===============================================================================
    # RESPONSE SYNTHESIZER - Final Instrumented Function
    # ===============================================================================
    
    @agent(name="Response Synthesizer Agent")
    def _synthesizer(self, state: WorkflowState) -> WorkflowState:
        """
        Response synthesizer that combines agent outputs into final response.
        Only component that sees the complete original user request.
        """
        agent_responses = state.get("agent_responses", {})
        agent_contexts = state.get("agent_contexts", {})
        user_request = state.get("input", "")
        
        logger.info(f"📝 Synthesizer processing {len(agent_responses)} specialist inputs")
        
        try:
            # LAB 5: Load prompt first so we see it in the trace
            self.synthesizer_prompt = self._load_agent_prompt("synthesizer")  # Store for reuse
            base_prompt_tokens = len(self.synthesizer_prompt) // 4  # Rough estimate: 4 chars ≈ 1 token
            logger.info(f"Current synthesizer prompt length: {len(self.synthesizer_prompt)} chars ≈ {base_prompt_tokens} tokens")

            # LAB 5: Check if error simulation is enabled and prompt is too large
            error_simulation = os.getenv("ERROR_SIMULATION", "false").lower() == "true"
            if error_simulation and base_prompt_tokens > 1000:
                # Set error on current span
                span = tracer.current_span()
                if span:
                    span.error = 1
                    span.set_tag("error.message", f"Input length exceeds recommended length (800 tokens). The synthesizer's prompt is too large. Current length: {base_prompt_tokens} tokens")
                    span.set_tag("error.type", "ValidationException")
                    span.set_tag("error.code", "ValidationException")
                    span.set_tag("token_count", base_prompt_tokens)
                    span.set_tag("prompt_file", os.path.join(os.path.dirname(__file__), "resources", "prompt-synthesizer.txt"))
                
                error = ClientError(
                    error_response={
                        "Error": {
                            "Message": "Input length exceeds model's context length. The prompt is too large for the model to process.",
                            "Code": "ValidationException",
                            "Type": "Client"
                        }
                    },
                    operation_name="InvokeModel"
                )
                raise error
            
            # Estimate tokens for agent responses (4 chars ≈ 1 token)
            response_tokens = sum(len(response) // 4 for response in agent_responses.values())
            
            # Estimate tokens for context documents
            context_tokens = sum(
                len(doc.get("content", "")) // 4 
                for docs in agent_contexts.values() 
                for doc in docs
            )
            
            # Estimate tokens for user request
            request_tokens = len(user_request) // 4
            
            total_tokens = base_prompt_tokens + response_tokens + context_tokens + request_tokens
            # LAB 5: Log token usage for monitoring
            logger.info(f"Token estimates - Base: {base_prompt_tokens} ({(base_prompt_tokens/total_tokens*100):.1f}% of total), Responses: {response_tokens} ({(response_tokens/total_tokens*100):.1f}%), Context: {context_tokens} ({(context_tokens/total_tokens*100):.1f}%), Request: {request_tokens} ({(request_tokens/total_tokens*100):.1f}%), Total: {total_tokens}")

            # LAB 5: Check if error simulation is enabled and prompt is too large
            error_simulation = os.getenv("ERROR_SIMULATION", "false").lower() == "true"
            if error_simulation and hasattr(self, 'synthesizer_prompt_tokens') and self.synthesizer_prompt_tokens > 800:
                # Set error on current span
                span = tracer.current_span()
                if span:
                    span.error = 1
                    span.set_tag("error.message", f"Input length exceeds recommended length (800 tokens). The synthesizer's prompt is too large. Current length: {self.synthesizer_prompt_tokens} tokens")
                    span.set_tag("error.type", "ValidationException")
                    span.set_tag("error.code", "ValidationException")
                    span.set_tag("token_count", self.synthesizer_prompt_tokens)
                    span.set_tag("prompt_file", getattr(self, 'synthesizer_prompt_path', 'unknown'))
                
                error = ClientError(
                    error_response={
                        "Error": {
                            "Message": "Input length exceeds model's context length. The prompt is too large for the model to process.",
                            "Code": "ValidationException",
                            "Type": "Client"
                        }
                    },
                    operation_name="InvokeModel"
                )
                raise error

            if len(agent_responses) == 1:
                # Single agent response - light enhancement
                single_response = list(agent_responses.values())[0]
                final_response = self._enhance_single_response(single_response, user_request, agent_contexts)
            else:
                # Multi-agent synthesis
                final_response = self._synthesize_multi_agent_responses(
                    agent_responses, agent_contexts, user_request
                )
            
            return {
                "output": final_response,
                "workflow_path": ["Response Synthesizer"]
            }
            
        except Exception as e:
            logger.error(f"Response Synthesizer error: {e}")
            # Set error on current span if not already set
            span = tracer.current_span()
            if span:
                span.error = 1
                span.set_tag("error.message", str(e))
                span.set_tag("error.type", e.__class__.__name__)
                if isinstance(e, ClientError):
                    error_code = e.response["Error"].get("Code", "UnknownError")
                    error_type = e.response["Error"].get("Type", "Unknown")
                    span.set_tag("error.code", error_code)
                    span.set_tag("error.type", error_type)
            return {
                "error": f"Response synthesis failed: {str(e)}",
                "workflow_path": ["Response Synthesizer", "error_handler"]
            }
    
    # ===============================================================================
    # WORKFLOW ROUTING AND HELPER METHODS
    # ===============================================================================
    
    def _route_after_planning(self, state: WorkflowState) -> str:
        """Route workflow after planning phase."""
        if state.get("error"):
            return "error"
        return "dispatch"
        
    def _route_after_synthesis(self, state: WorkflowState) -> str:
        """Route workflow after synthesis phase."""
        if state.get("error"):
            return "error"
        return "success"
    
    def _dispatch_specialized_agents(self, state: WorkflowState) -> Sequence[Send]:
        """Dispatch specialized agents in parallel using LangGraph Send API."""
        agents_needed = state.get("agents_needed", ["customer_service"])
        agent_tasks = state.get("agent_tasks", {})
        
        send_list = []
        agent_map = {
            "customer_service": "Customer Service",
            "product_specialist": "Product Specialist",
            "promotion_specialist": "Promotion Specialist",
            "feedback_handler": "Feedback Handler"
        }
        
        for agent_name in agents_needed:
            if agent_node := agent_map.get(agent_name):
                specific_task = agent_tasks.get(agent_name, state.get('user_request', ''))
                logger.info(f"📤 Dispatching {agent_node} with task: {specific_task[:80]}...")
                send_list.append(Send(agent_node, state))
        
        return send_list
    
    @task(name="create_agent_prompts")
    def _create_agent_prompts_from_planning(self, user_request: str, planning_result: Dict[str, Any], agents_needed: List[str]) -> Dict[str, str]:
        """
        Generate focused request subsets for each agent using LLM.
        Creates clean separation of concerns between agents.
        """
        logger.info("🔄 Generating focused request subsets for each agent")
        
        # Extract planning insights
        category = planning_result.get("primary_category", "General")
        reasoning = planning_result.get("reasoning", "")
        
        # Load orchestrator prompt with metadata
        prompt_data = PromptTrackingUtils.load_prompt_with_metadata("orchestrator")
        orchestrator_prompt = prompt_data["template"]
        
        # Build prompt variables
        prompt_variables = {
            "user_request": user_request,
            "category": category,
            "reasoning": reasoning,
            "agents_needed": ', '.join(agents_needed)
        }
        
        # Build subset generation request
        subset_generation_request = orchestrator_prompt.format(**prompt_variables)

        try:
            # Use planning model for request subset generation with prompt tracking
            response = self.llm_caller.call_planning_llm(
                prompt=subset_generation_request,
                user_request=user_request,
                prompt_template=orchestrator_prompt,
                prompt_id=prompt_data["id"],
                prompt_version=prompt_data["version"],
                prompt_variables=prompt_variables
            )
            
            # Parse response with fallback
            agent_subsets = self._parse_agent_prompts(response, user_request, agents_needed)
            
            logger.info("✅ Request subsets generated successfully")
            return agent_subsets
            
        except Exception as e:
            logger.warning(f"Subset generation failed: {e}, using fallback")
            return self._create_simple_agent_tasks(user_request, agents_needed)
    
    def _call_agent_with_rag(self, agent_type: str, agent_task: str, original_request: str,
                           documents: List[Dict[str, Any]], agent_prompt: str) -> str:
        """
        Execute agent LLM call with RAG using focused request subset.
        
        Uses clean approach where agent_task is a focused subset of the original request,
        providing better precision and preventing scope creep.
        """
        try:
            # Build context from documents using standardized method
            logger.info(f"🔍 Building LLM prompt context from {len(documents)} documents for {agent_type}:")
            context = self.llm_caller._build_context_from_documents(documents, include_header=True)
            
            if documents:
                for i, doc in enumerate(documents[:3], 1):
                    doc_name = doc.get('name', 'Document')
                    doc_content = doc.get('content', '')[:200]
                    logger.info(f"   📄 Document {i}: {doc_name} (content length: {len(doc_content)} chars)")
                    logger.debug(f"      Content preview: {doc_content[:100]}...")
                
                logger.info(f"🔍 LLM prompt context (total length: {len(context)} chars):")
                logger.debug(f"   Context content: {context[:300]}...")
            
            # Load prompt metadata for tracking
            agent_type_normalized = agent_type.replace("_", "-")
            prompt_data = PromptTrackingUtils.load_prompt_with_metadata(agent_type_normalized)
            
            # Build prompt variables
            prompt_variables = {
                "agent_prompt": agent_prompt,
                "agent_task": agent_task,
                "context": context
            }
            
            # Build prompt using agent's focused subset
            full_prompt = f"""{agent_prompt}

User Request: {agent_task}

Context:
{context}

Response:"""

            logger.info(f"🔍 {agent_type} executing LLM call:")
            logger.info(f"   User Request in prompt: {agent_task}")
            logger.info(f"   Original request (for annotation): {original_request}")
            
            # Execute LLM call with comprehensive instrumentation and prompt tracking
            response = self.llm_caller.call_agent_llm(
                prompt=full_prompt,
                agent_name=agent_type,
                documents=documents,
                user_request=original_request,  # Keep original for annotation
                category="General",
                prompt_template=agent_prompt,
                prompt_id=prompt_data["id"],
                prompt_version=prompt_data["version"],
                prompt_variables=prompt_variables
            )
            
            return response
            
        except Exception as e:
            logger.error(f"RAG LLM call error for {agent_type}: {e}")
            return f"Sorry, I couldn't process your {agent_type.replace('_', ' ')} request at this time."
    
    def _enhance_single_response(self, response: str, user_request: str, agent_contexts: Dict[str, List[Dict[str, Any]]] = None) -> str:
        """Enhance single agent response using original user request."""
        return SynthesisUtils.enhance_single_response(
            response=response,
            user_request=user_request,
            synthesizer_prompt=self.synthesizer_prompt,
            agent_contexts=agent_contexts,
            llm_caller=self.llm_caller
        )
    
    def _synthesize_multi_agent_responses(self, agent_responses: Dict[str, str], 
                                        agent_contexts: Dict[str, List[Dict[str, Any]]], 
                                        user_request: str) -> str:
        """Combine multiple agent responses to address original user request."""
        return SynthesisUtils.synthesize_multi_agent_responses(
            agent_responses=agent_responses,
            agent_contexts=agent_contexts,
            user_request=user_request,
            synthesizer_prompt=self.synthesizer_prompt,
            llm_caller=self.llm_caller
        )

    def _error_handler(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors gracefully."""
        error = state.get("error", "Unknown error")
        logger.error(f"🚨 Workflow error: {error}")
        
        return {
            "output": "I apologize, but I encountered an error processing your request. Please try again.",
            "workflow_path": ["error_handler"]
        }
    
    def _parse_agent_prompts(self, response: str, user_request: str, agents_needed: List[str]) -> Dict[str, str]:
        """Parse agent subset response with robust fallback handling."""
        try:
            # Extract JSON using utility
            json_str = ParsingUtils.extract_json_from_response(response)
            parsed = json.loads(json_str)
            
            # Handle both object and array formats
            if isinstance(parsed, list):
                logger.info(f"🔄 Converting array format to object")
                agent_subsets = {}
                for item in parsed:
                    if isinstance(item, dict):
                        agent_subsets.update(item)
            else:
                agent_subsets = parsed
            
            # Validate all needed agents have subsets
            for agent_name in agents_needed:
                if agent_name not in agent_subsets:
                    logger.warning(f"Missing subset for {agent_name}, using fallback")
                    return self._create_simple_agent_tasks(user_request, agents_needed)
            
            logger.info(f"🎯 Final agent subsets: {agent_subsets}")
            return agent_subsets
            
        except Exception as e:
            logger.warning(f"Failed to parse agent subsets: {e}, using fallback")
            return self._create_simple_agent_tasks(user_request, agents_needed)
    
    def _create_simple_agent_tasks(self, user_request: str, agents_needed: List[str]) -> Dict[str, str]:
        """Fallback method: use original request for all agents."""
        agent_tasks = {}
        for agent_name in agents_needed:
            agent_tasks[agent_name] = user_request
        return agent_tasks
    

    
    @tool(name="load_agent_prompt")
    def _load_agent_prompt(self, agent_type: str) -> str:
        """Load agent-specific prompt from file with fallback."""
        resources_dir = os.path.join(os.path.dirname(__file__), "resources")
        return SynthesisUtils.load_agent_prompt(agent_type, resources_dir)
    
    @task(name="parse_planning_response")
    def _parse_planning_response(self, response: str, user_request: str) -> Dict[str, Any]:
        """Parse planning agent response with robust error handling."""
        try:
            # Extract JSON using utility
            json_str = ParsingUtils.extract_json_from_response(response)
            analysis = json.loads(json_str)
            
            # Validate required fields
            if "agents_needed" in analysis and analysis["agents_needed"]:
                logger.info(f"✅ Planning agent produced valid JSON")
                return analysis
            
            # Fallback if validation fails
            logger.warning("Invalid JSON structure, using keyword fallback")
            return self._keyword_based_planning_fallback(user_request)
            
        except Exception as e:
            logger.error(f"Planning response parsing error: {e}, using fallback")
            return self._keyword_based_planning_fallback(user_request)

    def _keyword_based_planning_fallback(self, user_request: str) -> Dict[str, Any]:
        """Fallback planning based on request keywords."""
        result = ParsingUtils.keyword_based_planning_fallback(user_request)
        return {
            "primary_category": "General",
            "confidence": 0.7,
            "agents_needed": result["agents_needed"],
            "reasoning": result["reasoning"]
        }

    def _analyze_request_keywords(self, user_request: str) -> List[str]:
        """Analyze request to determine needed agents."""
        result = ParsingUtils.keyword_based_planning_fallback(user_request)
        return result["agents_needed"]
    
    @retrieval(name="retrieve_agent_documents")
    def _retrieve_documents(self, agent_type: str, user_request: str, knowledge_base: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Document retrieval with enhanced Datadog instrumentation and UI display."""
        retrieval_start_time = time.time()
        
        try:
            docs_handler = DocumentHandlers()
            
            # Use pre-loaded knowledge base if available, otherwise fallback to full loading
            if knowledge_base is None:
                # Silently fall back to full loading - this is expected in some cases
                knowledge_base = self._load_full_knowledge_base_backup()
            else:
                logger.info(f"📚 Using pre-loaded selective knowledge base for {agent_type}")
            
            # Extract user request from dictionary if needed
            request_text = user_request.get('content') if isinstance(user_request, dict) else user_request
            
            # Simple query processing with basic normalization
            query_words = self._normalize_query_words(request_text.lower().split())
            results = []
            
            # Agent-specific retrieval logic
            if agent_type == "customer_service":
                faq_results = docs_handler.search_faqs(query_words, knowledge_base["faqs"], None)
                cs_results = docs_handler.search_customer_service(query_words, knowledge_base["customer_service"], None)
                results.extend(faq_results)
                results.extend(cs_results)
                
            elif agent_type == "product_specialist":
                product_results = docs_handler.search_products(query_words, knowledge_base["products"], None)
                results.extend(product_results)
                
            elif agent_type == "promotion_specialist":
                promo_results = docs_handler.search_promotions(query_words, knowledge_base["promotions"], None)
                results.extend(promo_results)
                
            elif agent_type == "feedback_handler":
                faq_results = docs_handler.search_faqs(query_words, knowledge_base["faqs"], None)
                results.extend(faq_results)
            
            # Rank and filter results
            filtered_results = DocumentUtils.rank_and_filter_documents(results, max_results=8)
            retrieval_time = (time.time() - retrieval_start_time) * 1000
            
            # Manual Span Annotation - Show retrieved documents
            LLMObs.annotate(
                input_data=f"Search query: {user_request}",
                output_data=[{
                    "text": doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                    "name": doc.get('name', 'Unknown'),
                    "score": round(doc.get('score', 0), 2),
                    "id": doc.get('id', f'doc_{i}')
                } for i, doc in enumerate(filtered_results, 1)],
                metadata={
                    "agent_type": agent_type,
                    "documents_found": len(filtered_results),
                    "retrieval_time_ms": round(retrieval_time, 1)
                },
                tags={"lab": "03-complete-instrumentation"}
            )
            
            return filtered_results
            
        except Exception as e:
            retrieval_time = (time.time() - retrieval_start_time) * 1000
            logger.error(f"Document retrieval error for {agent_type}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            return []
    
    @tool(name="load_knowledge_base")
    def _load_full_knowledge_base_backup(self) -> Dict[str, Any]:
        """Load knowledge base resources from JSON files."""
        resources_dir = os.path.join(os.path.dirname(__file__), "resources")
        return KnowledgeBaseUtils.load_full_knowledge_base(resources_dir)
    
    def _load_knowledge_source(self, source_type: str) -> Any:
        """Load a single knowledge source JSON file."""
        resources_dir = os.path.join(os.path.dirname(__file__), "resources")
        return KnowledgeBaseUtils.load_knowledge_source(source_type, resources_dir)
    
    @tool(name="load_selective_knowledge_base")
    def _load_selective_knowledge_base(self, agents_needed: List[str]) -> Dict[str, Any]:
        """Load only the knowledge sources needed for the selected agents."""
        resources_dir = os.path.join(os.path.dirname(__file__), "resources")
        return KnowledgeBaseUtils.load_selective_knowledge_base(agents_needed, resources_dir)
    
    def _normalize_query_words(self, words: List[str]) -> List[str]:
        """
        Basic normalization for common plural/singular product terms.
        Handles the most common cases without over-complicating.
        """
        normalized = []
        
        # Simple plural/singular mappings for common product terms
        plural_to_singular = {
            "t-shirts": "t-shirt",
            "shirts": "shirt", 
            "bottles": "bottle",
            "mugs": "mug",
            "notebooks": "notebook",
            "headphones": "headphone",  # though this is often plural
            "stickers": "sticker",
            "hoodies": "hoodie",
            "sweatshirts": "sweatshirt"
        }
        
        for word in words:
            # Add the original word
            normalized.append(word)
            
            # Add singular version if it's a known plural
            if word in plural_to_singular:
                normalized.append(plural_to_singular[word])
                
        return normalized

    def _convert_to_response_format(self, state: WorkflowState) -> Dict[str, Any]:
        """Convert workflow state to expected response format."""
        planning_result = state.get("planning_result", {})
        
        # Flatten agent contexts for compatibility
        all_documents = []
        for documents in state.get("agent_contexts", {}).values():
            all_documents.extend(documents)
        
        # Build response with trace information for evaluation
        response = {
            "output": state.get("output", ""),
            "input": state.get("input", ""),
            "category": planning_result.get("primary_category", "General"),
            "agents_needed": state.get("agents_needed", []),
            "confidence": planning_result.get("confidence", 0.8),
            "documents": all_documents,
            "retrieved_count": len(all_documents),
            "model_id": getattr(config, "specialist_model", "unknown"),
            "error": state.get("error", ""),
            "agent_outputs": state.get("agent_responses", {}),
            "workflow_path": state.get("workflow_path", []),
            "workflow_type": "optimized_production_workflow",
        }
        
        # Add span context for evaluation functionality
        if state.get("span_context"):
            response["span_context"] = state.get("span_context")
        
        return response

# ===============================================================================
# LLM CALLING FUNCTIONS - Post-Workflow
# ===============================================================================

class LLMCaller:
    """
    Simplified LLM Caller with 2 clean instrumentation approaches:
    1. Vertex AI: Custom decorator with advanced instrumentation
    2. Others: Auto-instrumentation with LLMObs.annotation_context
    """
    
    def __init__(self, config):
        self.config = config
        self.platform = config.llm_platform
        self.llm_instances = {}
        # Add instance tracking for debugging
        import uuid
        self.instance_id = str(uuid.uuid4())[:8]
        logger.info(f"🏗️  Creating LLMCaller instance {self.instance_id} for platform: {self.platform}")
        
    def _clean_markdown_fences(self, content: str) -> str:
        """Remove markdown code fences that models sometimes add around HTML responses."""
        if not content:
            return content
            
        # Remove opening markdown fences (```html, ```HTML, ```)
        cleaned_content = re.sub(r'^```(?:html|HTML)?\s*\n?', '', content, flags=re.MULTILINE)
        
        # Remove closing markdown fences
        cleaned_content = re.sub(r'\n?```\s*$', '', cleaned_content, flags=re.MULTILINE)
        
        return cleaned_content.strip()
    
    def _get_llm_type_from_agent(self, agent_name: str) -> str:
        """Map agent name to appropriate LLM type for metadata consistency."""
        if agent_name == "planning":
            return "planning"
        elif agent_name == "synthesizer":
            return "synthesizer"
        elif agent_name in ["customer_service", "product_specialist", "promotion_specialist", "feedback_handler"]:
            return "specialist"
        else:
            return "specialist"  # Default fallback
    
    def _build_context_from_documents(self, documents: List[Dict[str, Any]], include_header: bool = False) -> str:
        """
        Build standardized context from documents for both LLM prompts and RAG annotation.
        
        Args:
            documents: List of document dictionaries with 'name' and 'content' keys
            include_header: Whether to include "Relevant Information:" header (for LLM prompts)
        
        Returns:
            Formatted context string optimized for hallucination detection
        """
        if not documents:
            return ""
        
        context_parts = []
        if include_header:
            context_parts.append("Relevant Information:")
        
        # Improved context format for better hallucination detection
        for i, doc in enumerate(documents[:10], 1): 
            doc_name = doc.get('name', 'Document')
            doc_content = doc.get('content', '')[:500]  # Increased to 500 chars for better context
            doc_type = doc.get('type', 'unknown')
            doc_id = doc.get('id', f'doc_{i}')
            
            # Enhanced format with metadata for better hallucination detection
            context_part = f"Source {i} [ID: {doc_id}, Type: {doc_type}]:\nTitle: {doc_name}\nContent: {doc_content}{'...' if len(doc.get('content', '')) > 500 else ''}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    

    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status for debugging."""
        return {
            "instance_id": self.instance_id,
            "platform": self.platform,
            "cached_llm_types": list(self.llm_instances.keys()),
            "cache_count": len(self.llm_instances)
        }
    
    def get_llm_instance(self, llm_type: str = "specialist"):
        """Get or create LLM instance based on type and platform."""
        if llm_type not in self.llm_instances:
            logger.info(f"🔧 Creating NEW {llm_type} LLM instance (not in cache)")
            
            # Get config based on LLM type
            if llm_type == "planning":
                model_config = self.config.get_planning_config()
                logger.info(f"🧠 Planning model config: {model_config.get('model_id', 'unknown')}")
            elif llm_type == "specialist":
                model_config = self.config.get_specialist_config()
                logger.info(f"🤖 Specialist model config: {model_config.get('model_id', 'unknown')}")
            elif llm_type == "synthesizer":
                model_config = self.config.get_synthesizer_config()
                logger.info(f"📝 Synthesizer model config: {model_config.get('model_id', 'unknown')}")
            else:
                model_config = self.config.get_specialist_config()
                logger.info(f"🤖 Default to specialist model config: {model_config.get('model_id', 'unknown')}")
            
            # Create LLM based on platform
            if self.platform == "bedrock":
                self.llm_instances[llm_type] = LLMFactory.create_bedrock_llm(model_config, llm_type)
            elif self.platform == "vertex":
                self.llm_instances[llm_type] = LLMFactory.create_vertex_llm(model_config, llm_type)
            elif self.platform == "openai":
                self.llm_instances[llm_type] = LLMFactory.create_openai_llm(model_config, llm_type)
            elif self.platform == "azure":
                self.llm_instances[llm_type] = LLMFactory.create_azure_llm(model_config, llm_type)
            else:
                raise ValueError(f"Unsupported platform: {self.platform}")
        else:
            logger.debug(f"♻️  Using CACHED {llm_type} LLM instance")
        
        return self.llm_instances[llm_type]
    
    # ===============================================================================
    # AUTO-INSTRUMENTED LLM CALLING METHODS
    # ===============================================================================
    # These methods are automatically traced by Datadog LLM Observability
    
    def call_planning_llm(self, prompt: str, user_request: str = "", prompt_template: str = None, 
                          prompt_id: str = None, prompt_version: str = None, prompt_variables: Dict[str, Any] = None) -> str:
        """Execute planning LLM call with appropriate instrumentation."""
        logger.info(f"🧠 Planning LLM call using {self.platform}")
        
        llm_instance = self.get_llm_instance("planning")
        
        if self.platform == "vertex":
            return self._call_vertex_llm(llm_instance, prompt, "planning", [], user_request, "planning",
                                        prompt_template, prompt_id, prompt_version, prompt_variables)
        
        return self._call_standard_llm(llm_instance, prompt, "planning", [], user_request, "planning",
                                      prompt_template, prompt_id, prompt_version, prompt_variables)
    
    def call_agent_llm(self, prompt: str, agent_name: str, documents: List[Dict[str, Any]], 
                       user_request: str = "", category: str = "", prompt_template: str = None,
                       prompt_id: str = None, prompt_version: str = None, prompt_variables: Dict[str, Any] = None) -> str:
        """Execute agent LLM call with appropriate instrumentation."""
        logger.info(f"🤖 Agent LLM call for {agent_name} using {self.platform}")
        
        # Get appropriate LLM instance based on agent type
        if agent_name == "planning":
            llm_instance = self.get_llm_instance("planning")
            logger.debug(f"🧠 Using PLANNING model for {agent_name}")
        elif agent_name == "synthesizer":
            llm_instance = self.get_llm_instance("synthesizer")
            logger.debug(f"📝 Using SYNTHESIZER model for {agent_name}")
        else:
            # Specialist agents (customer_service, product_specialist, etc.)
            llm_instance = self.get_llm_instance("specialist")
            logger.debug(f"🤖 Using SPECIALIST model for {agent_name}")
        
        if self.platform == "vertex":
            return self._call_vertex_llm(llm_instance, prompt, agent_name, documents, user_request, category,
                                        prompt_template, prompt_id, prompt_version, prompt_variables)
        
        return self._call_standard_llm(llm_instance, prompt, agent_name, documents, user_request, category,
                                      prompt_template, prompt_id, prompt_version, prompt_variables)
    
    def call_synthesis_llm(self, prompt: str, user_request: str = "", documents: List[Dict[str, Any]] = None,
                           prompt_template: str = None, prompt_id: str = None, prompt_version: str = None, 
                           prompt_variables: Dict[str, Any] = None) -> str:
        """Execute synthesis LLM call with appropriate instrumentation."""
        if documents is None:
            documents = []
        
        logger.info(f"📝 Synthesis LLM call using {self.platform}")
        logger.info(f"📝 Synthesizer has {len(documents)} context documents from specialized agents")
        
        llm_instance = self.get_llm_instance("synthesizer")
        
        if self.platform == "vertex":
            return self._call_vertex_llm(llm_instance, prompt, "Response Synthesizer", documents, user_request, "synthesis",
                                        prompt_template, prompt_id, prompt_version, prompt_variables)
        
        return self._call_standard_llm(llm_instance, prompt, "Response Synthesizer", documents, user_request, "synthesis",
                                      prompt_template, prompt_id, prompt_version, prompt_variables)
    
    def _call_standard_llm(self, llm_instance, prompt: str, agent_name: str, documents: List[Dict[str, Any]], 
                          user_request: str = "", category: str = "", prompt_template: str = None,
                          prompt_id: str = None, prompt_version: str = None, prompt_variables: Dict[str, Any] = None) -> str:
        context = ""
        if documents:
            context = self._build_context_from_documents(documents, include_header=True)
                          
        if context:
            logger.debug(f"   🔍 Context content: {context[:300]}...")
        
        # Build prompt configuration with tracking metadata
        prompt_config = {}
        
        # If prompt metadata is provided, use structured tracking
        if prompt_template and prompt_id:
            logger.info(f"📋 Using prompt tracking: id={prompt_id}, version={prompt_version}")
            
            # Build variables dictionary
            if prompt_variables is None:
                prompt_variables = {}
            
            # Ensure we have the key variables for RAG
            if user_request and "user_request" not in prompt_variables:
                prompt_variables["user_request"] = user_request
            if context and "context" not in prompt_variables:
                prompt_variables["context"] = context
            
            prompt_config = {
                "id": prompt_id,
                "version": prompt_version,
                "template": prompt_template,
                "variables": prompt_variables,
                "rag_query_variables": ["user_request"] if user_request else [],
                "rag_context_variables": ["context"] if context else []
            }
        else:
            # Fallback to basic variables if no template provided
            prompt_config = {
                "variables": {
                    "prompt": prompt, 
                    "context": context or "", 
                    "user_request": user_request or "",
                    "agent_name": agent_name
                },
                "rag_query_variables": ["user_request"] if user_request else [], 
                "rag_context_variables": ["context"] if context else []
            }
        
        # Paste custom annotation here
        with LLMObs.annotation_context(
            prompt=prompt_config,
            tags={
                "agent": agent_name,  # Tag the span with the agent name
                "agent_type": "synthesizer" if agent_name == "Response Synthesizer" else (
                    "planning" if agent_name == "Planning" else 
                    "specialist" if agent_name in ["Product Specialist", "Promotion Specialist", "Customer Service"] else 
                    "unknown"
                ),
                "prompt_id": prompt_id if prompt_id else "unknown",
                "prompt_version": prompt_version if prompt_version else "unknown"
            }
        ):
            try:
                response = llm_instance.invoke(prompt)
                response_content = response.content.strip() if hasattr(response, 'content') else str(response)
                return self._clean_markdown_fences(response_content)
                
            except Exception as e:
                logger.error(f"Standard LLM call error for {agent_name}: {e}")
                raise



    @llm(name="vertex_ai_llm_call", model_provider="Google")
    # ===============================================================================
    # MANUAL LLM CALLING METHODS
    # ===============================================================================
    # These methods require manual instrumentation for specific providers
    
    def _call_vertex_llm(self, llm_instance, prompt: str, agent_name: str, documents: List[Dict[str, Any]], 
                        user_request: str = "", category: str = "", prompt_template: str = None,
                        prompt_id: str = None, prompt_version: str = None, prompt_variables: Dict[str, Any] = None) -> str:
        """Vertex AI LLM call with manual instrumentation using LLMObs.annotate."""
        logger.info(f"🔧 Vertex AI LLM call for {agent_name}")
        
        try:
            # Build context for RAG annotation if documents exist
            context = ""
            if documents:
                context = self._build_context_from_documents(documents, include_header=True)
            
            # Get model config for proper annotation based on agent type
            if agent_name == "planning":
                model_config = self.config.get_planning_config()
            elif agent_name == "synthesizer":
                model_config = self.config.get_synthesizer_config()
            else:
                model_config = self.config.get_specialist_config()
            
            # Build prompt configuration with tracking metadata
            if prompt_template and prompt_id:
                logger.info(f"📋 Vertex AI using prompt tracking: id={prompt_id}, version={prompt_version}")
                
                # Build variables dictionary
                if prompt_variables is None:
                    prompt_variables = {}
                
                # Ensure we have the key variables for RAG
                if user_request and "user_request" not in prompt_variables:
                    prompt_variables["user_request"] = user_request
                if context and "context" not in prompt_variables:
                    prompt_variables["context"] = context
                
                # Create Prompt dictionary for tracking
                prompt_obj = {
                    "id": prompt_id,
                    "version": prompt_version,
                    "template": prompt_template,
                    "variables": prompt_variables,
                    "rag_query_variables": ["user_request"] if user_request else [],
                    "rag_context_variables": ["context"] if context else []
                }
            else:
                # Fallback to basic prompt annotation
                prompt_obj = None
            
            # Build and apply input annotation using utility
            llm_type = self._get_llm_type_from_agent(agent_name)
            annotation_params = VertexInstrumentationUtils.build_annotation_params(
                prompt=prompt,
                context=context,
                user_request=user_request,
                agent_name=agent_name,
                model_config=model_config,
                llm_type=llm_type,
                documents=documents
            )
            
            # Add prompt tracking to metadata if available
            if prompt_obj:
                if "metadata" not in annotation_params:
                    annotation_params["metadata"] = {}
                annotation_params["metadata"]["prompt_id"] = prompt_id
                annotation_params["metadata"]["prompt_version"] = prompt_version
            
            # Add tags for prompt tracking
            if "tags" not in annotation_params:
                annotation_params["tags"] = {}
            annotation_params["tags"]["prompt_id"] = prompt_id if prompt_id else "unknown"
            annotation_params["tags"]["prompt_version"] = prompt_version if prompt_version else "unknown"
            
            logger.debug(f"🔍 Added prompt annotation for Vertex AI")
            
            # Apply input annotation with prompt object if available
            if prompt_obj:
                LLMObs.annotate(prompt=prompt_obj, **{k: v for k, v in annotation_params.items() if k != 'prompt'})
            else:
                LLMObs.annotate(**annotation_params)
            
            # Execute LLM call
            response = llm_instance.invoke(prompt)
            response_content = response.content.strip() if hasattr(response, 'content') else str(response)
            response_content = self._clean_markdown_fences(response_content)
            
            # Extract token metrics using utility
            token_metrics = VertexInstrumentationUtils.extract_token_metrics(response, model_config)
            
            # Build output metadata using utility
            output_metadata = VertexInstrumentationUtils.build_output_metadata(
                response_content=response_content,
                agent_name=agent_name,
                llm_type=llm_type,
                model_config=model_config,
                documents=documents,
                token_metrics=token_metrics
            )
            
            # Apply output annotation
            LLMObs.annotate(
                output_data=response_content,
                metadata=output_metadata,
                metrics={
                    "input_tokens": output_metadata.get("input_tokens", 0),
                    "output_tokens": output_metadata.get("output_tokens", 0),
                    "total_tokens": output_metadata.get("total_tokens", 0),
                    "total_cost": output_metadata.get("total_cost_usd", 0.0),
                    "context_length": len(context) if context else 0,
                    "response_length": len(response_content)
                },
                tags={
                    "llm.model_name": model_config.get("model_id", "unknown"),
                    "llm.model_provider": "google",
                    "llm.platform": "vertex",
                    "llm.agent": agent_name,
                    "llm.llm_type": llm_type,
                    "llm.total_cost_usd": f"{output_metadata.get('total_cost_usd', 0.0):.6f}",
                    "llm.context_provided": str(bool(documents)).lower(),
                    "llm.rag_enabled": str(bool(documents and context)).lower()
                }
            )
            
            # Summary log
            token_summary = f"Input: {output_metadata.get('input_tokens', 0)}, Output: {output_metadata.get('output_tokens', 0)}"
            cost_summary = f"${output_metadata.get('total_cost_usd', 0.0):.6f}" if output_metadata.get('total_cost_usd', 0) > 0 else "N/A"
            logger.info(f"✅ Vertex AI call - Agent: {agent_name} ({llm_type.upper()}), Model: {model_config.get('model_id')}, Tokens: {token_summary}, Cost: {cost_summary}")
            
            return response_content
            
        except Exception as e:
            logger.error(f"Vertex AI LLM call error for {agent_name}: {e}")
            # Annotate the error with any available information
            error_metadata = {
                "agent_name": agent_name,
                "platform": "vertex",
                "model_name": model_config.get("model_id", "unknown"),
                "error": True,
                "error_message": str(e)
            }
            
            # Include any token/cost info that might have been extracted before error
            if 'output_metadata' in locals() and output_metadata:
                error_metadata.update({
                    "input_tokens": output_metadata.get("input_tokens", 0),
                    "output_tokens": output_metadata.get("output_tokens", 0),
                    "total_tokens": output_metadata.get("total_tokens", 0),
                    "total_cost_usd": output_metadata.get("total_cost_usd", 0.0)
                })
            
            # LLMObs.annotate(
            #     output_data=f"Error: {str(e)}",
            #     metadata=error_metadata,
            #     tags={
            #         "llm.model_name": model_config.get("model_id", "unknown"),
            #         "llm.model_provider": "google",
            #         "llm.platform": "vertex",
            #         "llm.agent": agent_name,
            #         "llm.llm_type": self._get_llm_type_from_agent(agent_name),
            #         "llm.error": "true"
            #     }
            # )
            raise
    

# ===============================================================================
# GLOBAL WORKFLOW INSTANCE - Required for module-level functions above
# ===============================================================================

# Create global instance for the application
swagbot_workflow = SwagBotWorkflow()

# ===============================================================================
# LLM FACTORY - Supporting Infrastructure 
# ===============================================================================

class LLMFactory:
    """
    Centralized LLM Factory for creating platform-specific LLM instances.
    Supports AWS Bedrock, Google Vertex AI, OpenAI, and Azure OpenAI platforms.
    """
    
    @staticmethod
    def create_bedrock_llm(model_config: Dict[str, Any], llm_type: str = "specialist"):
        """Create AWS Bedrock LLM instance with proper configuration."""
        try:
            import boto3
            from langchain_aws import ChatBedrock
            
            bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=model_config.get("region_name", "us-east-1")
            )
            
            llm = ChatBedrock(
                client=bedrock_client,
                model_id=model_config["model_id"],
                model_kwargs={
                    "temperature": model_config.get("temperature", 0.3),
                    "max_tokens": model_config.get("max_tokens", 1000)
                }
            )
            
            logger.info(f"✅ Created Bedrock LLM: {model_config['model_id']} (Type: {llm_type})")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create Bedrock LLM: {e}")
            raise
    
    @staticmethod 
    def create_vertex_llm(model_config: Dict[str, Any], llm_type: str = "specialist"):
        """Create Google Vertex AI LLM instance with proper configuration."""
        try:
            from langchain_google_vertexai import ChatVertexAI
            
            llm = ChatVertexAI(
                model_name=model_config["model_id"],
                project=model_config.get("project_id"),
                location=model_config.get("location", "us-central1"),
                temperature=model_config.get("temperature", 0.3),
                max_output_tokens=model_config.get("max_tokens", 1000)
            )
            
            logger.info(f"✅ Created Vertex AI LLM: {model_config['model_id']} (Type: {llm_type})")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create Vertex AI LLM: {e}")
            raise
    
    @staticmethod
    def create_openai_llm(model_config: Dict[str, Any], llm_type: str = "specialist"):
        """Create OpenAI LLM instance with proper configuration."""
        try:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                model=model_config["model_id"],
                api_key=model_config.get("api_key"),
                temperature=model_config.get("temperature", 0.3),
                max_tokens=model_config.get("max_tokens", 1000)
            )
            
            logger.info(f"✅ Created OpenAI LLM: {model_config['model_id']} (Type: {llm_type})")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create OpenAI LLM: {e}")
            raise
    
    @staticmethod
    def create_azure_llm(model_config: Dict[str, Any], llm_type: str = "specialist"):
        """Create Azure OpenAI LLM instance with proper configuration."""
        try:
            from langchain_openai import AzureChatOpenAI
            
            llm = AzureChatOpenAI(
                deployment_name=model_config["model_id"],
                azure_endpoint=model_config.get("azure_endpoint"),
                api_key=model_config.get("api_key"),
                api_version=model_config.get("api_version", "2024-02-15-preview"),
                temperature=model_config.get("temperature", 0.3),
                max_tokens=model_config.get("max_tokens", 1000)
            )
            
            logger.info(f"✅ Created Azure OpenAI LLM: {model_config['model_id']} (Type: {llm_type})")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create Azure OpenAI LLM: {e}")
            raise 