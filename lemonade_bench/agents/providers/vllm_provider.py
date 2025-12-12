# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
vLLM provider for local inference with fine-tuned models.

Connects to a vLLM server running locally for fast inference during
online GRPO training. Supports both OpenAI-compatible and native vLLM APIs.

Usage:
    # Start vLLM server first:
    # vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --quantization awq --port 8000
    
    from lemonade_bench.agents.providers import VLLMProvider
    provider = VLLMProvider(base_url="http://localhost:8000/v1")
"""

import json
import os
import uuid
from typing import Any

import requests

from .base import LLMProvider, TokenUsage, ToolResponse, LEMONADE_ACTION_TOOL


class VLLMProvider(LLMProvider):
    """
    Provider for vLLM server inference.
    
    Uses the OpenAI-compatible API that vLLM exposes. This allows
    fast local inference with fine-tuned LoRA adapters.
    
    Attributes:
        base_url: Base URL of the vLLM server (e.g., http://localhost:8000/v1)
        model_id: Model identifier (usually auto-detected from server)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 120,
    ):
        """
        Initialize the vLLM provider.

        Args:
            base_url: vLLM server URL (default: http://localhost:8000/v1)
            model: Model identifier (auto-detected if None)
            temperature: Sampling temperature (default: 0.0 for greedy decoding)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Auto-detect model if not specified
        if self._model is None:
            self._model = self._get_model_from_server()
    
    def _get_model_from_server(self) -> str:
        """Query the vLLM server to get the loaded model name."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                timeout=10,
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            if models:
                return models[0].get("id", "unknown")
        except Exception:
            pass
        return "vllm-model"
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def provider_name(self) -> str:
        return "vllm"
    
    def _build_tool_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Build a tool description to append to the system prompt."""
        tool_descriptions = []
        for tool in tools:
            desc = f"### {tool['name']}\n{tool['description']}\n"
            desc += f"Parameters: {json.dumps(tool['input_schema'], indent=2)}"
            tool_descriptions.append(desc)
        
        return """

## Available Tools

You have access to the following tools. To use a tool, output a JSON object wrapped in <tool_call> tags:

<tool_call>
{"name": "tool_name", "arguments": {...}}
</tool_call>

""" + "\n\n".join(tool_descriptions)
    
    def _parse_tool_call(self, content: str) -> tuple[str, dict[str, Any]] | None:
        """Extract tool call from model output."""
        import re
        
        # Try to find <tool_call> tags
        match = re.search(r'<tool_call>\s*({.*?})\s*</tool_call>', content, re.DOTALL)
        if match:
            try:
                call = json.loads(match.group(1))
                return call.get("name"), call.get("arguments", {})
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to find any JSON object with name/arguments
        match = re.search(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}', content, re.DOTALL)
        if match:
            try:
                args = json.loads(match.group(2))
                return match.group(1), args
            except json.JSONDecodeError:
                pass
        
        # Last resort: try to parse the action tool directly from JSON
        try:
            # Look for a JSON block that might be the action
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                if "price_per_cup" in data:
                    return "take_action", data
        except json.JSONDecodeError:
            pass
        
        return None
    
    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        required_tool: str | None = None,
    ) -> ToolResponse:
        """
        Generate a response with tool calls via vLLM.
        
        Since vLLM doesn't natively support function calling in all cases,
        we use prompt-based tool use with JSON output.
        """
        # Build the full system prompt with tool descriptions
        full_system_prompt = system_prompt + self._build_tool_prompt(tools)
        
        # Convert messages to vLLM/OpenAI format
        vllm_messages = [{"role": "system", "content": full_system_prompt}]
        
        for msg in messages:
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    vllm_messages.append({"role": "user", "content": msg["content"]})
                else:
                    # Handle structured content (extract text)
                    text_parts = []
                    for part in msg["content"]:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part["text"])
                        elif isinstance(part, dict) and part.get("type") == "tool_result":
                            text_parts.append(f"Tool result: {part.get('content', '')}")
                    vllm_messages.append({"role": "user", "content": "\n".join(text_parts)})
            
            elif msg["role"] == "assistant":
                if isinstance(msg["content"], str):
                    vllm_messages.append({"role": "assistant", "content": msg["content"]})
                else:
                    # Extract text from structured content
                    text_parts = []
                    for part in msg["content"]:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part["text"])
                        elif isinstance(part, dict) and part.get("type") == "tool_use":
                            # Format previous tool call
                            tool_call = {
                                "name": part.get("name"),
                                "arguments": part.get("input", {}),
                            }
                            text_parts.append(f"<tool_call>\n{json.dumps(tool_call, indent=2)}\n</tool_call>")
                    vllm_messages.append({"role": "assistant", "content": "\n".join(text_parts)})
        
        # Add instruction to use the required tool if specified
        if required_tool:
            vllm_messages.append({
                "role": "user",
                "content": f"Please use the {required_tool} tool to submit your decision.",
            })
        
        # Make request to vLLM server
        payload = {
            "model": self._model,
            "messages": vllm_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract content and usage
        choice = result["choices"][0]
        content = choice["message"]["content"]
        
        usage = None
        if "usage" in result:
            usage = TokenUsage(
                input_tokens=result["usage"].get("prompt_tokens", 0),
                output_tokens=result["usage"].get("completion_tokens", 0),
            )
        
        # Parse tool call from content
        parsed = self._parse_tool_call(content)
        if parsed is None:
            # If no tool call found, try to extract action directly
            # This handles cases where the model outputs JSON directly
            raise ValueError(f"Could not parse tool call from response: {content[:500]}")
        
        tool_name, tool_args = parsed
        tool_use_id = f"vllm_{uuid.uuid4().hex[:8]}"
        
        return ToolResponse(
            tool_name=tool_name,
            tool_input=tool_args,
            tool_use_id=tool_use_id,
            raw_response=result,
            usage=usage,
            text_content=content,
        )
    
    def format_tool_result(
        self,
        tool_use_id: str,
        result: str,
    ) -> dict[str, Any]:
        """Format a tool result for the conversation."""
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                }
            ],
        }
    
    def format_assistant_message(self, response: ToolResponse) -> dict[str, Any]:
        """Format the assistant's response as a message."""
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response.text_content or "",
                },
                {
                    "type": "tool_use",
                    "id": response.tool_use_id,
                    "name": response.tool_name,
                    "input": response.tool_input,
                },
            ],
        }
    
    def estimate_cost(self, usage: TokenUsage) -> float:
        """Local inference has no API cost."""
        return 0.0
    
    def reload_model(self, lora_path: str | None = None) -> None:
        """
        Request the vLLM server to reload the model with updated LoRA weights.
        
        Note: This requires vLLM to be started with --enable-lora flag and
        may require custom server configuration for hot-reloading.
        
        Args:
            lora_path: Path to the LoRA adapter to load (None to unload)
        """
        # This is a placeholder - actual implementation depends on vLLM version
        # and server configuration. Modern vLLM supports dynamic LoRA loading.
        # For now, we document the expected behavior.
        raise NotImplementedError(
            "Dynamic LoRA reloading requires vLLM >= 0.4.0 with --enable-lora. "
            "Restart the vLLM server with the new LoRA weights, or use the "
            "vLLM Python API directly for dynamic loading."
        )
    
    def health_check(self) -> bool:
        """Check if the vLLM server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

