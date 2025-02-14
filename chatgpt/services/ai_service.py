"""AI service for handling chat functionality."""

import json
from typing import List, Dict, Optional, Any
import mistralai.client
import mistralai.models.chat_completion
from openai import OpenAI
from openai.types.chat import ChatCompletion

class AIService:
    """Service for handling AI chat functionality."""

    def __init__(self, 
                 openai_api_key: str,
                 openai_api_base: Optional[str],
                 mistral_api_key: str,
                 mistral_api_base: Optional[str],
                 allowed_models: List[str],
                 mistral_allowed_models: List[str]):
        """Initialize the AI service.
        
        Args:
            openai_api_key: OpenAI API key
            openai_api_base: Optional custom OpenAI API endpoint
            mistral_api_key: Mistral AI API key
            mistral_api_base: Optional custom Mistral AI API endpoint
            allowed_models: List of allowed OpenAI models
            mistral_allowed_models: List of allowed Mistral AI models
        """
        self.openai_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        self.mistral_client = mistralai.client.MistralClient(
            api_key=mistral_api_key,
            endpoint=mistral_api_base if mistral_api_base else "https://api.mistral.ai"
        )
        self.allowed_models = allowed_models
        self.mistral_allowed_models = mistral_allowed_models

    def is_mistral_model(self, model: str) -> bool:
        """Check if a model is a Mistral AI model.
        
        Args:
            model: The model name to check
            
        Returns:
            True if it's a Mistral model, False otherwise
        """
        return model in self.mistral_allowed_models

    def create_chat_completion(self, 
                             messages: List[Dict[str, str]], 
                             model: str,
                             tools: Optional[List[Dict[str, Any]]] = None,
                             stream: bool = True) -> Any:
        """Create a chat completion using either OpenAI or Mistral AI.
        
        Args:
            messages: List of chat messages
            model: The model to use
            tools: Optional list of tools/functions
            stream: Whether to stream the response
            
        Returns:
            Chat completion response
            
        Raises:
            ValueError: If the model is not allowed
        """
        if self.is_mistral_model(model):
            if model not in self.mistral_allowed_models:
                raise ValueError(f"Invalid Mistral model: {model}")
                
            # Convert messages format for Mistral
            mistral_messages = []
            for msg in messages:
                if msg["role"] == "developer":
                    msg["role"] = "system"
                if "name" in msg:
                    del msg["name"]  # Mistral doesn't support names
                mistral_messages.append(msg)
                
            return self.mistral_client.chat_stream(
                model=model,
                messages=mistral_messages,
            ) if stream else self.mistral_client.chat(
                model=model,
                messages=mistral_messages,
            )
        else:
            if model not in self.allowed_models:
                raise ValueError(f"Invalid OpenAI model: {model}")
                
            return self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                stream=stream,
            )

    def process_chunk(self, chunk: Any, is_mistral: bool) -> Dict[str, Any]:
        """Process a response chunk from either API.
        
        Args:
            chunk: The response chunk
            is_mistral: Whether this is from Mistral AI
            
        Returns:
            Processed chunk with unified format
        """
        if is_mistral:
            return {
                "content": chunk.delta,
                "finish_reason": chunk.finish_reason,
                "tool_calls": None
            }
        else:
            return {
                "content": chunk.choices[0].delta.content,
                "finish_reason": chunk.choices[0].finish_reason,
                "tool_calls": chunk.choices[0].delta.tool_calls
            } 
