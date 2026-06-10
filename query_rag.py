"""
Query the MR.AsyncFL RAG database.
Optionally pipe retrieved context into a Qwen2 model for RAG-augmented QA.

Usage:
    # Retrieval only (no LLM)
    python query_rag.py --query "What convergence rate does MR.AsyncFL achieve?"

    # With Qwen2-1.5B-Instruct for answer generation
    python query_rag.py --query "How does MR.AsyncFL differ from FedAsync?" --llm

    # Interactive mode
    python query_rag.py --interactive
"""

import argparse
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "./chroma_db"
COLLECTION_NAME = "mrasyncfl_paper"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

LLM_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"  # or local fine-tuned path
LORA_ADAPTER = "./lora_output"               # set to None to use base model


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=top_k)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "content": doc,
            "section": meta.get("section", "unknown"),
            "distance": round(dist, 4)
        })
    return chunks


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Chunk {i} | Section: {c['section']} | Distance: {c['distance']}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def load_llm(use_lora: bool = False):
    """Load Qwen2-1.5B-Instruct, optionally with LoRA adapter."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"Loading base model: {LLM_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    if use_lora:
        import os
        if os.path.isdir(LORA_ADAPTER):
            print(f"Loading LoRA adapter from {LORA_ADAPTER} ...")
            model = PeftModel.from_pretrained(model, LORA_ADAPTER)
        else:
            print(f"LoRA adapter not found at {LORA_ADAPTER}, using base model.")

    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, query: str, context: str) -> str:
    import torch

    system_msg = (
        "You are an expert AI assistant specializing in federated learning and "
        "the MR.AsyncFL framework. Answer the question using ONLY the provided context. "
        "Be concise and accurate."
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {query}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )
    # Decode only the new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_query(query: str, collection, tokenizer=None, model=None):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print("="*60)

    chunks = retrieve(collection, query)
    context = format_context(chunks)

    print("\n--- Retrieved Context ---")
    for i, c in enumerate(chunks, 1):
        print(f"\n[{i}] Section: {c['section']} | Distance: {c['distance']}")
        print(c["content"][:300] + "..." if len(c["content"]) > 300 else c["content"])

    if tokenizer is not None and model is not None:
        print("\n--- Generated Answer ---")
        answer = generate_answer(tokenizer, model, query, context)
        print(answer)

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single query to answer")
    parser.add_argument("--llm", action="store_true", help="Use LLM for answer generation")
    parser.add_argument("--lora", action="store_true", help="Load LoRA adapter on top of base LLM")
    parser.add_argument("--interactive", action="store_true", help="Interactive query loop")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    collection = get_collection()
    print(f"Loaded collection with {collection.count()} chunks.")

    tokenizer, model = None, None
    if args.llm:
        tokenizer, model = load_llm(use_lora=args.lora)

    if args.interactive:
        print("\nInteractive RAG query. Type 'quit' to exit.\n")
        while True:
            query = input("Query: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if query:
                run_query(query, collection, tokenizer, model)
    elif args.query:
        run_query(args.query, collection, tokenizer, model)
    else:
        # Demo queries
        demo_queries = [
            "What is the model replacement strategy in MR.AsyncFL?",
            "What convergence rate does MR.AsyncFL achieve and what parameter scaling is needed?",
            "How does MR.AsyncFL compare to FedAsync in CIFAR-10 experiments?",
            "What is the recursive weight update rule and why does it preserve normalization?",
        ]
        for q in demo_queries:
            run_query(q, collection, tokenizer, model)


if __name__ == "__main__":
    main()
