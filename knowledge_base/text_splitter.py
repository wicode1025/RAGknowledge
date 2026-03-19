"""
Recursive Character Text Splitter
Split text into chunks while preserving context and logical flow.
"""

import re
from typing import List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    text: str
    start_index: int
    end_index: int
    chunk_index: int


class RecursiveCharacterTextSplitter:
    """
    Recursively split text using different separators.
    This ensures that related content (like code blocks and paragraphs) stay together.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        keep_separator: bool = False,
        length_function: Callable[[str], int] = None
    ):
        """
        Initialize text splitter

        Args:
            chunk_size: Maximum size of each chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use (in order of priority)
            keep_separator: Whether to keep the separator in the chunk
            length_function: Function to calculate text length
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator

        if length_function is None:
            # Default: count characters
            self.length_function = len
        else:
            self.length_function = length_function

        # Default separators (in order of priority)
        if separators is None:
            self.separators = [
                # First try to split by double newlines (paragraphs)
                "\n\n",
                # Then by single newlines
                "\n",
                # Then by sentences (period + space)
                ". ",
                # Then by commas and spaces
                ", ",
                # Finally by spaces (words)
                " ",
                # And single characters as last resort
                ""
            ]
        else:
            self.separators = separators

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using separators

        Args:
            text: Input text
            separators: List of separators to try

        Returns:
            List of text chunks
        """
        # Handle empty text
        if not text or not text.strip():
            return []

        # Handle empty separators list
        if not separators:
            # Just return the text as is if it's small enough
            if self.length_function(text) <= self.chunk_size:
                return [text]
            # Otherwise, split by characters
            return self._split_text(text, [""])

        # Get the first separator
        separator = separators[0]
        new_separators = separators[1:] if len(separators) > 1 else []

        # Split by current separator
        if separator:
            parts = text.split(separator)
        else:
            parts = list(text)

        # If we have multiple parts, process recursively
        if len(parts) > 1:
            new_parts = []
            for part in parts:
                if part and part.strip():
                    # Recursively split this part
                    sub_parts = self._split_text(part, new_separators)
                    new_parts.extend(sub_parts)
        else:
            new_parts = parts

        # If single part or too large, try smaller separator
        if len(new_parts) == 1 or self.length_function(text) <= self.chunk_size:
            # Base case: return the parts if they're small enough
            if all(self.length_function(p) <= self.chunk_size for p in new_parts):
                return new_parts
            # Otherwise, try with the next separator level
            if new_separators:
                return self._split_text(text, new_separators)
            # If no more separators, force split by characters
            if self.length_function(text) > self.chunk_size:
                # Simple character-based split as fallback
                chunks = []
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                    chunk = text[i:i + self.chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
                return chunks

        # Combine small parts into chunks
        chunks = []
        current_chunk = ""
        current_length = 0

        for part in new_parts:
            part_length = self.length_function(part)

            # Add separator if keeping it
            if self.keep_separator and current_chunk and separator:
                part_to_add = separator + part
            else:
                part_to_add = part

            # Check if adding this part would exceed chunk size
            if current_length + part_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Find overlap text
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + part_to_add
                    current_length = self.length_function(current_chunk)
                else:
                    current_chunk = part_to_add
                    current_length = part_length
            else:
                # Add to current chunk
                current_chunk += part_to_add
                current_length += part_length

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Filter out empty chunks
        chunks = [c for c in chunks if c]

        return chunks

    def create_chunks(
        self,
        text: str,
        source_name: str = None
    ) -> List[TextChunk]:
        """
        Create text chunks with metadata

        Args:
            text: Input text
            source_name: Name of the source document

        Returns:
            List of TextChunk objects
        """
        chunk_texts = self.split_text(text)

        chunks = []
        start_idx = 0

        for i, chunk_text in enumerate(chunk_texts):
            end_idx = start_idx + len(chunk_text)

            chunks.append(TextChunk(
                text=chunk_text,
                start_index=start_idx,
                end_index=end_idx,
                chunk_index=i
            ))

            # Move start index forward with overlap consideration
            start_idx = end_idx - self.chunk_overlap
            if start_idx < 0:
                start_idx = 0

        return chunks


class MarkdownAwareSplitter(RecursiveCharacterTextSplitter):
    """
    Text splitter that understands Markdown structure.
    Keeps code blocks, math formulas, and headers together.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs
    ):
        # Custom separators for Markdown content
        separators = [
            # Code blocks (```)
            "\n```\n",
            # Headers
            "\n## ",
            "\n### ",
            "\n# ",
            # Math formulas ($$)
            "\n$$\n",
            "\n$",
            # Lists
            "\n- ",
            "\n1. ",
            # Paragraphs
            "\n\n",
            # Lines
            "\n",
            # Sentences
            ". ",
            # Words
            " ",
            ""
        ]

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **kwargs
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split Markdown text while preserving structure
        """
        # First, try to keep code blocks and math together
        # by temporarily replacing them with placeholders

        # Handle code blocks
        code_pattern = r'```[\s\S]*?```'
        code_blocks = []
        placeholder_text = text

        def replace_code(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        placeholder_text = re.sub(code_pattern, replace_code, placeholder_text)

        # Handle inline code
        inline_code_pattern = r'`[^`]+`'
        inline_codes = []

        def replace_inline_code(match):
            inline_codes.append(match.group(0))
            return f"__INLINE_CODE_{len(inline_codes) - 1}__"

        placeholder_text = re.sub(inline_code_pattern, replace_inline_code, placeholder_text)

        # Handle math formulas (both $$ and $)
        math_pattern = r'\$\$[\s\S]*?\$|\$[^\$]+\$'
        math_formulas = []

        def replace_math(match):
            math_formulas.append(match.group(0))
            return f"__MATH_{len(math_formulas) - 1}__"

        placeholder_text = re.sub(math_pattern, replace_math, placeholder_text)

        # Split the placeholder text
        chunks = super().split_text(placeholder_text)

        # Restore code blocks and math in chunks
        restored_chunks = []
        for chunk in chunks:
            # Restore math
            for i, formula in enumerate(math_formulas):
                chunk = chunk.replace(f"__MATH_{i}__", formula)

            # Restore inline code
            for i, code in enumerate(inline_codes):
                chunk = chunk.replace(f"__INLINE_CODE_{i}__", code)

            # Restore code blocks
            for i, block in enumerate(code_blocks):
                chunk = chunk.replace(f"__CODE_BLOCK_{i}__", block)

            restored_chunks.append(chunk)

        return restored_chunks


if __name__ == "__main__":
    # Test splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    test_text = """
    This is a test document. It contains multiple sentences.
    Each sentence should be kept together when possible.

    This is a new paragraph. It should be separated from the previous one.

    Here we have some code:
    def hello():
        print("Hello, World!")

    And some math:
    E = mc^2

    The quick brown fox jumps over the lazy dog.
    """ * 5

    chunks = splitter.split_text(test_text)
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
