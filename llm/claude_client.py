"""
Claude API Client
Integration with Anthropic Claude for answer generation.
"""

import os
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

import anthropic

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of LLM generation"""
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    raw_response: Optional[dict] = None


class ClaudeClient:
    """
    Client for interacting with Claude API
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        system_prompt: str = None
    ):
        """
        Initialize Claude client

        Args:
            api_key: Anthropic API key (or use env var ANTHROPIC_API_KEY)
            model: Claude model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            system_prompt: Default system prompt
        """
        # Get API key
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if api_key is None:
            # Try to get from config
            try:
                api_key = Config.get_api_key()
            except ValueError:
                raise ValueError(
                    "API key not provided. Set ANTHROPIC_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        self.client = anthropic.Anthropic(api_key=api_key)

        self.model = model or Config.CLAUDE_MODEL
        self.max_tokens = max_tokens or Config.CLAUDE_MAX_TOKENS
        self.temperature = temperature if temperature is not None else Config.CLAUDE_TEMPERATURE

        # Default system prompt
        self.default_system_prompt = system_prompt or (
            "You are a helpful, accurate, and precise AI assistant specializing in "
            "computer science and technical topics. Provide clear, accurate, and "
            "well-structured answers based on the context provided. "
            "If the context doesn't contain enough information to answer the question, "
            "say so clearly rather than making up information."
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
        stop_sequences: List[str] = None,
        context: str = None
    ) -> GenerationResult:
        """
        Generate response from Claude

        Args:
            prompt: User prompt (question)
            system_prompt: System prompt (overrides default)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            stop_sequences: Sequences that stop generation
            context: Optional context to include in prompt

        Returns:
            GenerationResult object
        """
        # Build full prompt
        if context:
            full_prompt = f"""Context information:
{context}

Question: {prompt}

Based on the context above, please answer the question accurately and precisely.
If the context doesn't contain sufficient information to answer, state that clearly."""
        else:
            full_prompt = prompt

        # Use system prompt
        system = system_prompt or self.default_system_prompt

        # Override parameters if provided
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        logger.info(f"Calling Claude API with model {self.model}")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                stop_sequences=stop_sequences or [anthropic.HUMAN_PROMPT]
            )

            # Extract text from response
            response_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    response_text += block.text

            return GenerationResult(
                text=response_text,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                stop_reason=str(response.stop_reason),
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise

    def generate_with_sources(
        self,
        prompt: str,
        context: str,
        citation_style: str = "inline"
    ) -> GenerationResult:
        """
        Generate response with context and citation instructions

        Args:
            prompt: User question
            context: Retrieved context
            citation_style: How to cite sources

        Returns:
            GenerationResult
        """
        system_prompt = f"""You are a precise AI assistant specializing in computer science.
When answering questions, use ONLY the context provided below.
If the context doesn't contain enough information to answer the question,
clearly state that the information is not available in the provided context.

Cite sources using {citation_style} style when possible."""

        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context
        )

    def generate_rag_response(
        self,
        prompt: str,
        context: str,
        include_sources: bool = True
    ) -> Dict:
        """
        Generate RAG response with structured output

        Args:
            prompt: User question
            context: Retrieved context
            include_sources: Whether to include source info

        Returns:
            Dict with answer, sources, and metadata
        """
        result = self.generate_with_sources(
            prompt=prompt,
            context=context,
            citation_style="numbered"
        )

        response = {
            "answer": result.text,
            "metadata": {
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "stop_reason": result.stop_reason
            }
        }

        if include_sources:
            response["sources"] = {
                "has_context": bool(context),
                "context_length": len(context)
            }

        return response


class ClaudeBatchClient:
    """
    Batch client for processing multiple queries
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.client = ClaudeClient(api_key=api_key, model=model)

    def generate_batch(
        self,
        queries: List[str],
        contexts: List[str] = None,
        show_progress: bool = True
    ) -> List[GenerationResult]:
        """
        Generate responses for multiple queries

        Args:
            queries: List of user questions
            contexts: List of context strings (one per query)
            show_progress: Whether to show progress

        Returns:
            List of GenerationResult objects
        """
        results = []

        contexts = contexts or [None] * len(queries)

        for i, (query, context) in enumerate(zip(queries, contexts)):
            if show_progress:
                logger.info(f"Processing query {i+1}/{len(queries)}")

            result = self.client.generate(prompt=query, context=context)
            results.append(result)

        return results


def create_claude_client(api_key: str = None) -> ClaudeClient:
    """
    Factory function to create Claude client

    Args:
        api_key: Optional API key

    Returns:
        ClaudeClient instance
    """
    return ClaudeClient(api_key=api_key)


if __name__ == "__main__":
    # Test client (requires API key)
    logging.basicConfig(level=logging.INFO)

    try:
        client = ClaudeClient()

        # Test simple generation
        result = client.generate(
            prompt="What is Python?",
            max_tokens=100
        )

        print(f"Model: {result.model}")
        print(f"Input tokens: {result.input_tokens}")
        print(f"Output tokens: {result.output_tokens}")
        print(f"Response: {result.text[:200]}...")

    except ValueError as e:
        print(f"API key not configured: {e}")
