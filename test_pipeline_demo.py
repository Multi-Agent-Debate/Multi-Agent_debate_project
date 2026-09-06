"""Complete End-to-End RAG Pipeline Demonstration Script.

Demonstrates the 6-stage RAG flow for the Multi-Agent Debate Engine:
1. Load PDF document (document_loader.py)
2. Chunk document into semantic segments (chunker.py)
3. Generate normalized dense vector embeddings (embeddings.py)
4. Store & index in persistent ChromaDB (vector_store.py)
5. Semantic vector similarity retrieval (retriever.py)
6. Grounding context formatting & citation auditing (grounding.py)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.document_loader import load_pdf
from src.rag.chunker import chunk_documents
from src.rag.embeddings import get_embedding_generator
from src.rag.vector_store import ChromaVectorStore
from src.rag.retriever import RAGRetriever
from src.rag.grounding import GroundingManager


def ensure_sample_pdf(pdf_path: str) -> None:
    """Ensure sample debate PDF exists. Creates it with pypdf if missing."""
    if os.path.exists(pdf_path):
        return

    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
    import pypdf

    pdf_content = """%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [5 0 R 8 0 R] /Count 2 >>
endobj
3 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
4 0 obj
<< /Length {len1} >>
stream
{stream1}
endstream
endobj
5 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
6 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
7 0 obj
<< /Length {len2} >>
stream
{stream2}
endstream
endobj
8 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 6 0 R >> >> /MediaBox [0 0 612 792] /Contents 7 0 R >>
endobj
xref
0 9
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000120 00000 n 
0000000190 00000 n 
0000000300 00000 n 
0000000420 00000 n 
0000000490 00000 n 
0000000600 00000 n 
trailer
<< /Size 9 /Root 1 0 R >>
startxref
750
%%EOF"""
    pages_text = [
        [
            "Title: Multi-Agent AI Systems in Deliberative Decision Making",
            "Section 1: The Core Thesis on Adversarial Consensus",
            "Traditional single-prompt LLM reasoning often suffers from confirmation bias and hallucinated facts.",
            "By orchestrating an adversarial debate between a Proponent and an Opponent, critical logical flaws are exposed.",
            "Empirical evaluations demonstrate that multi-agent debate reduces factual errors by over 65 percent compared to standalone LLMs.",
            "A dedicated Moderator agent acts as an impartial judge, auditing evidence against verified document sources.",
        ],
        [
            "Section 2: Empirical Case Studies and Quantitative Benchmarks",
            "In medical diagnostic reviews, adversarial multi-agent debate improved diagnostic accuracy from 71.4 percent to 89.2 percent.",
            "In legal contract analysis, cross-examination by an Opponent agent discovered 84 percent of ambiguous or contradictory clauses.",
            "Section 3: Grounding and Traceability Requirements",
            "To maintain credibility, all agents must provide exact document citations containing the source file and page number.",
            "Claims made without verified citations must be flagged as ungrounded speculations by the Moderator.",
        ],
    ]

    s1 = "BT /F1 12 Tf 50 720 Td " + " ".join([f"({l}) Tj T*" for l in pages_text[0]]) + " ET"
    s2 = "BT /F1 12 Tf 50 720 Td " + " ".join([f"({l}) Tj T*" for l in pages_text[1]]) + " ET"
    raw = pdf_content.format(
        len1=len(s1.encode("latin1")),
        stream1=s1,
        len2=len(s2.encode("latin1")),
        stream2=s2,
    )

    tmp_path = pdf_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(raw.encode("latin1"))

    writer = pypdf.PdfWriter()
    writer.append(tmp_path)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    os.remove(tmp_path)
    print(f"✓ Generated sample document at: {pdf_path}")


def run_rag_demo():
    """Execute complete 6-stage RAG demonstration."""
    print("\n" + "=" * 80)
    print("      MULTI-AGENT DEBATE ENGINE: RAG & GROUNDING PIPELINE DEMO")
    print("=" * 80)

    sample_pdf = os.path.join(PROJECT_ROOT, "data", "documents", "sample_debate_brief.pdf")
    ensure_sample_pdf(sample_pdf)

    # -------------------------------------------------------------------------
    # STAGE 1: PDF Document Loading
    # -------------------------------------------------------------------------
    print("\n[STAGE 1] Loading PDF Document via PyPDFLoader...")
    docs = load_pdf(sample_pdf)
    print(f"✓ Loaded {len(docs)} pages from '{Path(sample_pdf).name}'.")
    for i, doc in enumerate(docs, start=1):
        print(f"  • Page {doc.metadata['page_number']}: {len(doc.page_content)} characters | Source: {doc.metadata['source_file']}")

    # -------------------------------------------------------------------------
    # STAGE 2: Recursive Character Text Chunking
    # -------------------------------------------------------------------------
    print("\n[STAGE 2] Chunking Document into Semantic Segments (800 char, 150 overlap)...")
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=150)
    print(f"✓ Generated {len(chunks)} traceable chunks with metadata:")
    for chunk in chunks:
        meta = chunk.metadata
        print(f"  • Chunk ID: {meta['chunk_id']}")
        print(f"    Page: {meta['page_number']} | Length: {meta['character_count']} chars | Snippet: \"{chunk.page_content[:75]}...\"")

    # -------------------------------------------------------------------------
    # STAGE 3: Dense Vector Embeddings
    # -------------------------------------------------------------------------
    print("\n[STAGE 3] Generating Dense Vector Embeddings (all-MiniLM-L6-v2)...")
    embedder = get_embedding_generator("all-MiniLM-L6-v2")
    sample_vec = embedder.embed_query("Adversarial multi-agent debate reduces hallucinations")
    print(f"✓ Embedding Model Loaded: {embedder.model_name}")
    print(f"✓ Embedding Vector Dimension: {embedder.dimension}-d")
    print(f"✓ Sample Normalized Vector Preview: [{sample_vec[0]:.4f}, {sample_vec[1]:.4f}, {sample_vec[2]:.4f}, ...]")

    # -------------------------------------------------------------------------
    # STAGE 4: Persistent ChromaDB Vector Store
    # -------------------------------------------------------------------------
    chroma_dir = os.path.join(PROJECT_ROOT, "data", "vector_db")
    print(f"\n[STAGE 4] Storing Chunks in ChromaDB at '{chroma_dir}'...")
    vector_store = ChromaVectorStore(
        persist_directory=chroma_dir,
        collection_name="demo_debate_kb",
        embedder=embedder,
    )
    # Clear collection for clean demo run
    vector_store.clear_collection()
    stored_ids = vector_store.add_documents(chunks)
    print(f"✓ Successfully indexed {len(stored_ids)} chunks in collection '{vector_store.collection_name}'.")
    print(f"✓ Verified collection count: {vector_store.count()} chunks.")

    # -------------------------------------------------------------------------
    # STAGE 5: Semantic Retrieval
    # -------------------------------------------------------------------------
    debate_query = "What empirical evidence demonstrates that multi-agent debate reduces hallucinations in AI systems?"
    print(f"\n[STAGE 5] Retrieving Top-K Context for Query:")
    print(f"  Query: \"{debate_query}\"")
    retriever = RAGRetriever(vector_store=vector_store, embedder=embedder, default_top_k=2)
    results = retriever.retrieve(debate_query, top_k=2)

    print(f"✓ Retrieved {len(results)} nearest neighbor chunks:")
    for idx, res in enumerate(results, start=1):
        print(f"\n  [Result #{idx}]")
        print(f"  • Chunk ID        : {res.get('chunk_id')}")
        print(f"  • Source          : {res.get('source_file')}, Page {res.get('page_number')}")
        print(f"  • Similarity Score: {res.get('similarity_score')} (Distance: {res.get('distance')})")
        print(f"  • Content Snippet : \"{res.get('content')[:120]}...\"")

    # -------------------------------------------------------------------------
    # STAGE 6: Grounding & Evidence Formatting
    # -------------------------------------------------------------------------
    print("\n[STAGE 6] Synthesizing Grounded Context & Debate Agent Prompt...")
    evidence_context = GroundingManager.format_evidence_context(results)
    proponent_prompt = GroundingManager.create_grounded_prompt(
        query=debate_query,
        evidence_context=evidence_context,
        role="proponent",
    )

    print("\n--- FORMATTED EVIDENCE CONTEXT PASSED TO DEBATE AGENTS ---")
    print(evidence_context)

    print("\n--- PROMPT INSTRUCTION FOR PROPONENT AGENT ---")
    print(proponent_prompt[:650] + "\n... [Prompt continues with strict citation instructions] ...")

    # Test citation extraction helper
    mock_agent_response = (
        "Adversarial multi-agent debate reduces factual errors by over 65 percent "
        "[Source: sample_debate_brief.pdf, Page: 1]. Furthermore, in diagnostic trials, "
        "accuracy climbed from 71.4% to 89.2% [Source: sample_debate_brief.pdf, Page: 2]."
    )
    citations = GroundingManager.extract_citations(mock_agent_response)
    print(f"\n✓ Citation Extraction Audit Test: Found {len(citations)} citations:")
    for cit in citations:
        print(f"  • Cited File: {cit['source']} | Page: {cit['page']}")

    print("\n" + "=" * 80)
    print("      ✓ COMPLETE RAG & GROUNDING PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_rag_demo()
