"""Retriever Module for RAG Pipeline.

Coordinates query embedding and vector similarity search against ChromaDB
to return relevant grounded chunks with distance scores and citation metadata.
"""

from typing import Any, Dict, List, Optional

from src.rag.embeddings import EmbeddingGenerator, get_embedding_generator
from src.rag.vector_store import ChromaVectorStore


class RAGRetriever:
    """Retrieves top-k semantically relevant chunks from the ChromaDB collection."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        default_top_k: int = 5,
    ):
        """Initialize the retriever.

        Args:
            vector_store: ChromaVectorStore instance. Creates default if None.
            embedder: EmbeddingGenerator instance. Uses singleton if None.
            default_top_k: Default number of chunks to return.
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or get_embedding_generator()
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search the vector database for chunks most similar to the user query.

        Args:
            query: The user premise, question, or claim to ground.
            top_k: Number of chunks to retrieve (defaults to self.default_top_k).
            where_filter: Optional ChromaDB metadata filter (e.g. {"source_file": "doc.pdf"}).

        Returns:
            List of dictionaries containing:
                - chunk_id: str
                - content: str
                - metadata: dict
                - distance: float
                - similarity_score: float (1 - distance/2 for cosine)
                - source_file: str
                - page_number: int
        """
        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.default_top_k
        total_chunks = self.vector_store.count()
        if total_chunks == 0:
            return []

        # Don't ask ChromaDB for more items than exist in the collection
        query_k = min(k, total_chunks)

        # Generate dense vector for query
        query_vector = self.embedder.embed_query(query)

        # Execute vector similarity query against ChromaDB
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": query_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        raw_results = self.vector_store.collection.query(**query_kwargs)

        # Parse ChromaDB output structure
        retrieved_chunks: List[Dict[str, Any]] = []

        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for i in range(len(ids)):
            doc_id = ids[i]
            content = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}
            dist = float(distances[i]) if i < len(distances) else 1.0

            # Cosine distance in Chroma ranges [0, 2]. Similarity = 1 - (dist / 2)
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            retrieved_chunks.append(
                {
                    "chunk_id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "distance": round(dist, 4),
                    "similarity_score": round(similarity, 4),
                    "source_file": metadata.get("source_file", "unknown"),
                    "page_number": metadata.get("page_number", 1),
                }
            )

        return retrieved_chunks


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    vector_store: Optional[ChromaVectorStore] = None,
) -> List[Dict[str, Any]]:
    """Convenience helper function to retrieve chunks for a query in one line.

    Args:
        query: Query string.
        top_k: Number of chunks to retrieve.
        vector_store: Optional ChromaVectorStore.

    Returns:
        List of retrieved chunk dictionaries.
    """
    retriever = RAGRetriever(vector_store=vector_store, default_top_k=top_k)
    return retriever.retrieve(query=query, top_k=top_k)
