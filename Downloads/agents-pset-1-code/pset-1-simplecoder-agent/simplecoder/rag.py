# Author: Blessing Ndeh
# Date: 03/02/2026
#  Class: AI Agents: COSC 89.34

"""RAG (Retrieval-Augmented Generation) for semantic code search"""

import os
import glob
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class CodeChunk:
    """
    Represents a chunk of code with metadata.

    A chunk is a piece of code (function, class, or fixed-size block)
    that can be searched and retrieved.
    """
    content: str           # The actual code
    file_path: str        # Which file it's from
    start_line: int       # Starting line number
    end_line: int         # Ending line number
    chunk_type: str       # "function", "class", or "block"
    embedding: Optional[np.ndarray] = None  # Vector representation


class CodeIndexer:
    """
    Indexes a codebase by:
    1. Finding all code files
    2. Chunking them into searchable pieces
    3. Converting to embeddings (vectors)
    4. Storing for fast retrieval
    """

    def __init__(self, embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the code indexer.

        Args:
            embedder_name: Name of the embedding model to use
                          Can be sentence-transformers model or "gemini/gemini-embedding-001"
        """
        self.embedder_name = embedder_name
        self.embedder = None
        self.chunks: List[CodeChunk] = []

        # Initialize embedding model
        self._initialize_embedder()

    def _initialize_embedder(self):
        """Initialize the embedding model"""
        if self.embedder_name.startswith("sentence-transformers/"):
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.embedder_name.replace("sentence-transformers/", "")
                self.embedder = SentenceTransformer(model_name)
                self.embedder_type = "sentence-transformers"
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        elif self.embedder_name.startswith("gemini/"):
            # Use Google Gemini embeddings
            self.embedder_type = "gemini"
            # Will use litellm for embeddings
        else:
            raise ValueError(f"Unknown embedder: {self.embedder_name}")

    def index_codebase(
        self,
        pattern: str = "**/*.py",
        directory: str = ".",
        max_chunk_lines: int = 50
    ) -> int:
        """
        Index all code files matching the pattern.

        Args:
            pattern: Glob pattern for files to index (e.g., "**/*.py")
            directory: Root directory to search
            max_chunk_lines: Maximum lines per chunk

        Returns:
            Number of chunks indexed
        """
        # Find all matching files
        files = self._find_files(pattern, directory)

        if not files:
            print(f"Warning: No files found matching '{pattern}' in '{directory}'")
            return 0

        # Chunk each file
        all_chunks = []
        for file_path in files:
            chunks = self._chunk_file(file_path, max_chunk_lines)
            all_chunks.extend(chunks)

        # Generate embeddings for all chunks
        self._embed_chunks(all_chunks)

        self.chunks = all_chunks
        return len(self.chunks)

    def _find_files(self, pattern: str, directory: str) -> List[str]:
        """
        Find all files matching pattern.

        Args:
            pattern: Glob pattern
            directory: Root directory

        Returns:
            List of file paths
        """
        if pattern.startswith("**"):
            full_pattern = os.path.join(directory, pattern)
        else:
            full_pattern = os.path.join(directory, pattern)

        files = glob.glob(full_pattern, recursive=True)
        # Filter to only files (not directories)
        files = [f for f in files if os.path.isfile(f)]
        return files

    def _chunk_file(self, file_path: str, max_lines: int) -> List[CodeChunk]:
        """
        Chunk a file into searchable pieces.

        Strategy:
        1. Try to detect functions/classes (smart chunking)
        2. Fall back to fixed-size chunks if needed

        Args:
            file_path: Path to the file
            max_lines: Max lines per chunk

        Returns:
            List of CodeChunk objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return []

        chunks = []

        # Try smart chunking (detect functions/classes)
        smart_chunks = self._smart_chunk(lines, file_path)

        if smart_chunks:
            # Smart chunking succeeded
            chunks.extend(smart_chunks)
        else:
            # Fall back to fixed-size chunks
            chunks.extend(self._fixed_chunk(lines, file_path, max_lines))

        return chunks

    def _smart_chunk(self, lines: List[str], file_path: str) -> List[CodeChunk]:
        """
        Smart chunking: detect functions and classes.

        Args:
            lines: File lines
            file_path: Path to file

        Returns:
            List of chunks (one per function/class)
        """
        chunks = []
        current_chunk_lines = []
        current_start = 0
        in_definition = False
        indent_level = 0

        for i, line in enumerate(lines):
            # Detect function or class definition
            if re.match(r'^(def |class )', line):
                # Save previous chunk if exists
                if current_chunk_lines:
                    chunks.append(CodeChunk(
                        content=''.join(current_chunk_lines),
                        file_path=file_path,
                        start_line=current_start,
                        end_line=i,
                        chunk_type="block"
                    ))

                # Start new chunk
                current_chunk_lines = [line]
                current_start = i
                in_definition = True
                indent_level = len(line) - len(line.lstrip())

            elif in_definition:
                # Continue collecting lines for this definition
                line_indent = len(line) - len(line.lstrip())

                # If we're back to same or lower indent, definition ended
                if line.strip() and line_indent <= indent_level:
                    # Save this definition
                    chunks.append(CodeChunk(
                        content=''.join(current_chunk_lines),
                        file_path=file_path,
                        start_line=current_start,
                        end_line=i,
                        chunk_type="function" if "def " in current_chunk_lines[0] else "class"
                    ))
                    current_chunk_lines = [line]
                    current_start = i
                    in_definition = False
                else:
                    current_chunk_lines.append(line)
            else:
                current_chunk_lines.append(line)

        # Add final chunk
        if current_chunk_lines:
            chunks.append(CodeChunk(
                content=''.join(current_chunk_lines),
                file_path=file_path,
                start_line=current_start,
                end_line=len(lines),
                chunk_type="block"
            ))

        return chunks

    def _fixed_chunk(
        self,
        lines: List[str],
        file_path: str,
        max_lines: int
    ) -> List[CodeChunk]:
        """
        Fixed-size chunking (fallback).

        Args:
            lines: File lines
            file_path: Path to file
            max_lines: Max lines per chunk

        Returns:
            List of fixed-size chunks
        """
        chunks = []
        for i in range(0, len(lines), max_lines):
            chunk_lines = lines[i:i + max_lines]
            chunks.append(CodeChunk(
                content=''.join(chunk_lines),
                file_path=file_path,
                start_line=i,
                end_line=min(i + max_lines, len(lines)),
                chunk_type="block"
            ))
        return chunks

    def _embed_chunks(self, chunks: List[CodeChunk]):
        """
        Generate embeddings for all chunks.

        Args:
            chunks: List of CodeChunk objects to embed
        """
        if not chunks:
            return

        # Extract content from chunks
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings based on embedder type
        if self.embedder_type == "sentence-transformers":
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
        elif self.embedder_type == "gemini":
            embeddings = self._embed_with_gemini(texts)
        else:
            raise ValueError(f"Unknown embedder type: {self.embedder_type}")

        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

    def _embed_with_gemini(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using Google Gemini.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of embeddings
        """
        from litellm import embedding

        embeddings = []
        # Process in batches to avoid rate limits
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = embedding(
                    model=self.embedder_name,
                    input=batch
                )

                batch_embeddings = [item['embedding'] for item in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"Warning: Embedding batch {i} failed: {e}")
                # Add zero vectors as fallback
                embeddings.extend([np.zeros(768)] * len(batch))

        return np.array(embeddings)


class CodeRetriever:
    """
    Retrieves relevant code chunks using semantic search.

    Given a query, finds the most similar code chunks
    using cosine similarity between embeddings.
    """

    def __init__(self, indexer: CodeIndexer):
        """
        Initialize retriever with an indexer.

        Args:
            indexer: CodeIndexer with built index
        """
        self.indexer = indexer

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[Tuple[CodeChunk, float]]:
        """
        Search for code chunks similar to the query.

        Args:
            query: Search query (natural language or code)
            top_k: Number of results to return
            min_similarity: Minimum similarity score (0-1)

        Returns:
            List of (CodeChunk, similarity_score) tuples, sorted by relevance
        """
        if not self.indexer.chunks:
            return []

        # Embed the query
        query_embedding = self._embed_query(query)

        # Compute similarities
        similarities = []
        for chunk in self.indexer.chunks:
            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            if similarity >= min_similarity:
                similarities.append((chunk, float(similarity)))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return similarities[:top_k]

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        if self.indexer.embedder_type == "sentence-transformers":
            return self.indexer.embedder.encode([query])[0]
        elif self.indexer.embedder_type == "gemini":
            from litellm import embedding
            response = embedding(model=self.indexer.embedder_name, input=[query])
            return np.array(response.data[0]['embedding'])
        else:
            raise ValueError(f"Unknown embedder type: {self.indexer.embedder_type}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (0-1, where 1 is most similar)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_context_string(
        self,
        query: str,
        top_k: int = 3
    ) -> str:
        """
        Get formatted context string for LLM prompt.

        Args:
            query: Search query
            top_k: Number of results to include

        Returns:
            Formatted string with relevant code snippets
        """
        results = self.search(query, top_k=top_k)

        if not results:
            return "No relevant code found."

        context_parts = ["## Relevant Code Context\n"]

        for i, (chunk, score) in enumerate(results, 1):
            context_parts.append(
                f"### Result {i} (similarity: {score:.2f})\n"
                f"**File:** {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})\n"
                f"```python\n{chunk.content}\n```\n"
            )

        return "\n".join(context_parts)