"""
LangChain Configuration for SwagBot
Platform-agnostic design supporting multiple LLM providers
"""

import os
import logging
from typing import Dict, Any

# Note: Logging configuration is handled in swagbot_app.py for consistency
logger = logging.getLogger(__name__)

class LangChainConfig:
    def __init__(self):
        # Platform Configuration
        self.llm_platform = os.getenv("LLM_PLATFORM", "bedrock").lower()
        
        # Using single optimized LangGraph workflow
        
        # Multi-Agent Model Configuration (platform-agnostic)
        self.planning_model = os.getenv("PLANNING_MODEL", self._get_default_planning_model())
        self.specialist_model = os.getenv("SPECIALIST_MODEL", self._get_default_specialist_model())
        self.synthesizer_model = os.getenv("SYNTHESIZER_MODEL", self._get_default_synthesizer_model())
        
        # Platform-specific Configuration
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_project_id = os.getenv("GOOGLE_PROJECT_ID")
        self.google_location = os.getenv("GOOGLE_LOCATION", "us-central1")
        
        # Azure OpenAI Configuration
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        
        # Flask Configuration
        self.FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
        self.FLASK_PORT = int(os.getenv("FLASK_PORT", 3000))
        self.FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
        
        # Error Simulation
        self.error_simulation_enabled = os.getenv("ERROR_SIMULATION", "false").lower() == "true"
        self.error_simulation_rate = float(os.getenv("ERROR_SIMULATION_RATE", "0.1"))  # 10% default
        self.latency_simulation_enabled = os.getenv("LATENCY_SIMULATION", "true").lower() == "true"
        self.max_latency_ms = int(os.getenv("MAX_LATENCY_MS", "750"))  # 750ms max latency
        
        # Validate and log configuration
        self._validate_config()
        self._log_config()
    
    def _get_default_planning_model(self) -> str:
        """Get default planning model based on platform (fast models for analysis)"""
        defaults = {
            "bedrock": "anthropic.claude-instant-v1",
            "vertex": "gemini-1.5-flash",
            "openai": "gpt-4o-mini",
            "azure": "gpt-4o-mini"
        }
        return defaults.get(self.llm_platform, defaults["bedrock"])
    
    def _get_default_specialist_model(self) -> str:
        """Get default specialist model based on platform (comprehensive models for expertise)"""
        defaults = {
            "bedrock": "anthropic.claude-3-haiku-20240307-v1:0",
            "vertex": "gemini-1.5-pro",
            "openai": "gpt-4o",
            "azure": "gpt-4o"
        }
        return defaults.get(self.llm_platform, defaults["bedrock"])
    
    def _get_default_synthesizer_model(self) -> str:
        """Get default synthesizer model based on platform (balanced models for coordination)"""
        defaults = {
            "bedrock": "anthropic.claude-3-haiku-20240307-v1:0",
            "vertex": "gemini-1.5-pro",
            "openai": "gpt-4o",
            "azure": "gpt-4o"
        }
        return defaults.get(self.llm_platform, defaults["bedrock"])
    

    
    def _validate_config(self):
        """Validate the configuration based on selected platform"""
        if self.llm_platform == "bedrock":
            # AWS Bedrock validation
            if not self.planning_model or not self.specialist_model or not self.synthesizer_model:
                raise ValueError("PLANNING_MODEL, SPECIALIST_MODEL, and SYNTHESIZER_MODEL are required for Bedrock")
                
        elif self.llm_platform == "openai":
            # OpenAI validation
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI platform")
            if not self.planning_model or not self.specialist_model or not self.synthesizer_model:
                raise ValueError("PLANNING_MODEL, SPECIALIST_MODEL, and SYNTHESIZER_MODEL are required for OpenAI")
                
        elif self.llm_platform == "vertex":
            # Google Vertex AI validation
            if not self.google_project_id:
                raise ValueError("GOOGLE_PROJECT_ID is required when using Vertex AI platform")
            if not self.planning_model or not self.specialist_model or not self.synthesizer_model:
                raise ValueError("PLANNING_MODEL, SPECIALIST_MODEL, and SYNTHESIZER_MODEL are required for Vertex AI")
                
        elif self.llm_platform == "azure":
            # Azure OpenAI validation
            if not self.azure_openai_api_key:
                raise ValueError("AZURE_OPENAI_API_KEY is required when using Azure OpenAI platform")
            if not self.azure_openai_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT is required when using Azure OpenAI platform")
            if not self.planning_model or not self.specialist_model or not self.synthesizer_model:
                raise ValueError("PLANNING_MODEL, SPECIALIST_MODEL, and SYNTHESIZER_MODEL are required for Azure OpenAI")
                
        else:
            raise ValueError(f"Unsupported LLM platform: {self.llm_platform}. Supported: bedrock, openai, vertex, azure")
    
    def _log_config(self):
        """Log the current configuration"""
        logger.info("🤖 Multi-Agent LLM Platform Configuration:")
        logger.info(f"   🏗️ Platform: {self.llm_platform.upper()}")
        logger.info(f"   ⚙️ Workflow: LANGGRAPH OPTIMIZED")
        logger.info(f"   🎯 Planning Agent: {self.planning_model}")
        logger.info(f"   🔧 Specialist Agents: {self.specialist_model}")
        logger.info(f"   📝 Synthesizer Agent: {self.synthesizer_model}")
        
        # Platform-specific logging
        if self.llm_platform == "bedrock":
            logger.info(f"   🌍 AWS Region: {self.aws_region}")
        elif self.llm_platform == "vertex":
            logger.info(f"   🌍 Google Project: {self.google_project_id}")
            logger.info(f"   🌍 Google Location: {self.google_location}")
        elif self.llm_platform == "openai":
            logger.info(f"   🔑 OpenAI API Key: {'✅ Set' if self.openai_api_key else '❌ Missing'}")
        elif self.llm_platform == "azure":
            logger.info(f"   🔑 Azure OpenAI API Key: {'✅ Set' if self.azure_openai_api_key else '❌ Missing'}")
            logger.info(f"   🌍 Azure OpenAI Endpoint: {self.azure_openai_endpoint if self.azure_openai_endpoint else '❌ Missing'}")
            logger.info(f"   📋 Azure OpenAI API Version: {self.azure_openai_api_version}")
        
        # Log error simulation configuration
        if self.error_simulation_enabled:
            logger.info("🔥 Error Simulation ENABLED:")
            logger.info(f"   ⚡ Error Simulation Rate: {self.error_simulation_rate*100}%")
            logger.info(f"   🐌 Latency Simulation: {self.latency_simulation_enabled}")
            logger.info(f"   ⏱️ Max Latency: {self.max_latency_ms}ms")
        else:
            logger.info("🔥 Error Simulation DISABLED")
    
    def get_planning_config(self) -> Dict[str, Any]:
        """Get planning agent model configuration for the current platform"""
        base_config = {
            "model_id": self.planning_model,
            "temperature": 0.1,
            "max_tokens": 400  # Increased for JSON planning responses
        }
        
        if self.llm_platform == "bedrock":
            return {
                **base_config,
                "region_name": self.aws_region,
                "model_kwargs": {
                    "temperature": base_config["temperature"],
                    "max_tokens": base_config["max_tokens"]
                }
            }
        elif self.llm_platform == "openai":
            return {
                **base_config,
                "api_key": self.openai_api_key
            }
        elif self.llm_platform == "vertex":
            return {
                **base_config,
                "project_id": self.google_project_id,
                "location": self.google_location
            }
        elif self.llm_platform == "azure":
            return {
                **base_config,
                "api_key": self.azure_openai_api_key,
                "endpoint": self.azure_openai_endpoint,
                "api_version": self.azure_openai_api_version
            }
        else:
            raise ValueError(f"Unsupported platform: {self.llm_platform}")
    
    def get_specialist_config(self) -> Dict[str, Any]:
        """Get specialist agents model configuration for the current platform"""
        base_config = {
            "model_id": self.specialist_model,
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        if self.llm_platform == "bedrock":
            return {
                **base_config,
                "region_name": self.aws_region,
                "model_kwargs": {
                    "temperature": base_config["temperature"],
                    "max_tokens": base_config["max_tokens"]
                }
            }
        elif self.llm_platform == "openai":
            return {
                **base_config,
                "api_key": self.openai_api_key
            }
        elif self.llm_platform == "vertex":
            return {
                **base_config,
                "project_id": self.google_project_id,
                "location": self.google_location
            }
        elif self.llm_platform == "azure":
            return {
                **base_config,
                "api_key": self.azure_openai_api_key,
                "endpoint": self.azure_openai_endpoint,
                "api_version": self.azure_openai_api_version
            }
        else:
            raise ValueError(f"Unsupported platform: {self.llm_platform}")
    
    def get_synthesizer_config(self) -> Dict[str, Any]:
        """Get synthesizer agent model configuration for the current platform"""
        base_config = {
            "model_id": self.synthesizer_model,
            "temperature": 0.2,
            "max_tokens": 2500  # Larger for multi-agent response synthesis
        }
        
        if self.llm_platform == "bedrock":
            return {
                **base_config,
                "region_name": self.aws_region,
                "model_kwargs": {
                    "temperature": base_config["temperature"],
                    "max_tokens": base_config["max_tokens"]
                }
            }
        elif self.llm_platform == "openai":
            return {
                **base_config,
                "api_key": self.openai_api_key
            }
        elif self.llm_platform == "vertex":
            return {
                **base_config,
                "project_id": self.google_project_id,
                "location": self.google_location
            }
        elif self.llm_platform == "azure":
            return {
                **base_config,
                "api_key": self.azure_openai_api_key,
                "endpoint": self.azure_openai_endpoint,
                "api_version": self.azure_openai_api_version
            }
        else:
            raise ValueError(f"Unsupported platform: {self.llm_platform}")
    

    
    def get_error_simulation_config(self) -> Dict[str, Any]:
        """Get error simulation configuration"""
        return {
            "enabled": self.error_simulation_enabled,
            "error_rate": self.error_simulation_rate,
            "latency_enabled": self.latency_simulation_enabled,
            "max_latency_ms": self.max_latency_ms
        }
    
    @property
    def provider(self) -> str:
        """Alias for llm_platform for backward compatibility"""
        return self.llm_platform

# Global configuration instance
config = LangChainConfig()