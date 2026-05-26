import os
from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent
DB_FOLDER = str(BASE_DIR / "notebooks" / "cbg_chroma_db")  

COLLECTION_NAME = "cbg_documents"
TOP_K = 5
GROQ_MODEL = "llama-3.3-70b-versatile"


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded")
    return model


@lru_cache(maxsize=1)
def get_collection():
    import chromadb

    print("Loading Chroma collection...")
    chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print("Chroma collection loaded")
    return collection


@lru_cache(maxsize=1)
def get_groq_client():
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    return Groq(api_key=api_key, timeout=60.0)


def retrieve_relevant_chunks(question, top_k=TOP_K):
    print("Retrieving chunks...")

    embedding_model = get_embedding_model()
    collection = get_collection()

    question_embedding = embedding_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    retrieved_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, document in enumerate(documents):
        metadata = metadatas[i]

        retrieved_chunks.append({
            "text": document,
            "source_file": metadata.get("source_file", ""),
            "source_folder": metadata.get("source_folder", ""),
            "page": metadata.get("page", ""),
            "content_type": metadata.get("content_type", ""),
            "distance": distances[i]
        })

    print(f"Retrieved {len(retrieved_chunks)} chunks")
    return retrieved_chunks


def build_context(chunks):
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        source_label = (
            f"[Source {i}: "
            f"File: {chunk['source_file']}, "
            f"Page: {chunk['page']}, "
            f"Folder: {chunk['source_folder']}]"
        )

        context_parts.append(f"{source_label}\n{chunk['text']}")

    return "\n\n".join(context_parts)


def generate_answer_with_groq(question, chunks):
    print("Calling Groq...")

    groq_client = get_groq_client()
    context = build_context(chunks)

    system_prompt = """
You are an expert CBG and anaerobic digestion assistant.

Use only the provided context to answer the question.

Guidelines:

Answer clearly, naturally, and in a professional engineering style.
Use layered explanations:
Simple explanation
Technical reasoning
Supporting numbers/examples when available
Keep answers concise, practical, and non-repetitive.
Synthesize information across retrieved sources naturally.
Do not mention "Source 1", "Source 2", or retrieval details.
Support important claims with measurable facts whenever available:
ranges, percentages, yields, methane content, TS/VS values,
retention times, capacities, costs, operational thresholds,
or other engineering metrics.
Explain operational implications where relevant.
Do not invent numerical values, policy details, or technical claims.
If evidence is insufficient or conflicting, clearly say so.
Avoid generic filler and repeated points.
Each sentence should add meaningful new information.
"""

    user_prompt = f"""
Context:
{context}

User question:
{question}
"""

    chat_completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=2000
    )

    print("Groq answer received")
    return chat_completion.choices[0].message.content


def ask_question(question):
    chunks = retrieve_relevant_chunks(question)
    answer = generate_answer_with_groq(question, chunks)

    return answer, chunks
