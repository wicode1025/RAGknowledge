"""
Qwen (千问) API Client
Integration with Qwen LLM for answer generation.
"""

import os
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

from openai import OpenAI

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


class QwenClient:
    """
    Client for interacting with Qwen API
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        system_prompt: str = None,
        base_url: str = None
    ):
        """
        Initialize Qwen client

        Args:
            api_key: Qwen API key
            model: Qwen model name (e.g., qwen-turbo, qwen-plus, qwen-max)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            system_prompt: Default system prompt
            base_url: API base URL (for custom endpoints)
        """
        # Get API key
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY")

        if api_key is None:
            # Use the provided API key
            api_key = "sk-27fbf8fcc496431a95e6a313a459d764"

        # Default base URL for DashScope
        base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model = model or "qwen-plus"
        self.max_tokens = max_tokens or Config.CLAUDE_MAX_TOKENS
        self.temperature = temperature if temperature is not None else Config.CLAUDE_TEMPERATURE

        # Default system prompt
        self.default_system_prompt = system_prompt or (
            "你是一个专业、精准的AI助手，专门解答计算机科学和技术相关问题。"
            "请根据提供的上下文信息给出清晰、准确的回答。"
            "如果上下文信息不足以回答问题，请明确说明。"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
        stop: List[str] = None,
        context: str = None
    ) -> GenerationResult:
        """
        Generate response from Qwen

        Args:
            prompt: User prompt (question)
            system_prompt: System prompt (overrides default)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            stop: Sequences that stop generation
            context: Optional context to include in prompt

        Returns:
            GenerationResult object
        """
        # Build full prompt
        if context:
            full_prompt = f"""上下文信息：
{context}

问题：{prompt}

请根据上述上下文信息准确回答问题。如果上下文信息不足，请明确说明。"""
        else:
            full_prompt = prompt

        # Use system prompt
        system = system_prompt or self.default_system_prompt

        # Override parameters if provided
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        logger.info(f"Calling Qwen API with model {self.model}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop
            )

            # Extract text from response
            response_text = response.choices[0].message.content

            # Get usage info
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            return GenerationResult(
                text=response_text or "",
                model=response.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason=response.choices[0].finish_reason,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )

        except Exception as e:
            logger.error(f"Qwen API error: {e}")
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
        system_prompt = f"""你是一个专业的计算机科学助手。
请仅根据下面提供的上下文信息回答问题。
如果上下文信息不足以回答问题，请明确说明。

请使用{citation_style}方式引用来源。"""

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


class QwenBatchClient:
    """
    Batch client for processing multiple queries
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.client = QwenClient(api_key=api_key, model=model)

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


def create_qwen_client(api_key: str = None, model: str = None) -> QwenClient:
    """
    Factory function to create Qwen client

    Args:
        api_key: Optional API key
        model: Model name

    Returns:
        QwenClient instance
    """
    return QwenClient(api_key=api_key, model=model)


if __name__ == "__main__":
    # Test client
    logging.basicConfig(level=logging.INFO)

    try:
        client = QwenClient()

        # Test simple generation
        result = client.generate(
            prompt="什么是机器学习？",
            max_tokens=100
        )

        print(f"Model: {result.model}")
        print(f"Input tokens: {result.input_tokens}")
        print(f"Output tokens: {result.output_tokens}")
        print(f"Response: {result.text[:200]}...")

    except Exception as e:
        print(f"Error: {e}")
