"""
Ablation evaluation: compare four conditions on the same 15 held-out questions.

    Condition 1: base        — Qwen2-1.5B-Instruct, no paper knowledge
    Condition 2: rag         — base model + retrieved paper context
    Condition 3: lora        — fine-tuned model, no retrieval
    Condition 4: rag_lora    — fine-tuned model + retrieved paper context

All conditions use the same questions, same generation settings, and same
keyword-coverage scoring, so the numbers are directly comparable.

Usage:
    python evaluate_ablation.py                          # all 4 conditions
    python evaluate_ablation.py --conditions base rag    # subset (e.g. before LoRA is trained)
    python evaluate_ablation.py --top_k 3                # fewer retrieved chunks

Output: comparison table on stdout + ablation_results.json
"""

import argparse
import json
import os
import statistics
import sys

# Windows consoles default to cp1252, which can't encode the Greek/math symbols
# (Δ, ×) used in the report. Switch stdout to UTF-8 so printing never crashes.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import QA_TEST_SET  # the 15 held-out questions

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
LORA_DIR = "./lora_output"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "mrasyncfl_paper"
EMBED_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PLAIN = (
    "You are a helpful AI assistant. Answer the question concisely and accurately."
)
SYSTEM_RAG = (
    "You are a helpful AI assistant. Answer the question using ONLY the provided "
    "context. Be concise and accurate."
)

ALL_CONDITIONS = ["base", "rag", "lora", "rag_lora"]


# ─── Retrieval ────────────────────────────────────────────────────────────────

def get_collection():
    # NOTE: chromadb 1.5.x crashes (Windows access violation in its Rust core)
    # on this machine, so we use an in-process numpy retriever that embeds the
    # SAME chunks with the SAME model (all-MiniLM-L6-v2) under cosine distance.
    # Drop-in: exposes .count() and .query(query_texts=..., n_results=...).
    from simple_retriever import SimpleRetriever

    return SimpleRetriever(embed_model=EMBED_MODEL)


def retrieve_context(collection, question: str, top_k: int) -> str:
    results = collection.query(query_texts=[question], n_results=top_k)
    return "\n\n---\n\n".join(results["documents"][0])


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(with_lora: bool):
    """Load the base model, optionally wrapped with the LoRA adapter.

    Returns (tokenizer, model, has_adapter). If with_lora is True the model is
    a PeftModel whose adapter can be toggled with model.disable_adapter().
    """
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_DIR if with_lora and os.path.isdir(LORA_DIR) else MODEL_ID,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    has_adapter = False
    if with_lora:
        if os.path.isdir(LORA_DIR):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, LORA_DIR)
            has_adapter = True
        else:
            print(f"WARNING: LoRA adapter not found at {LORA_DIR}; "
                  f"lora/rag_lora conditions will be skipped.")
    model.eval()
    return tokenizer, model, has_adapter


# ─── Generation and scoring ───────────────────────────────────────────────────

def generate(tokenizer, model, question: str, context: str | None) -> str:
    if context is not None:
        system = SYSTEM_RAG
        user = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        system = SYSTEM_PLAIN
        user = question

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256, temperature=0.1, do_sample=False
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def keyword_score(answer: str, keywords: list[str]) -> float:
    answer_lower = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in answer_lower) / len(keywords)


# ─── Main ablation loop ───────────────────────────────────────────────────────

def run_ablation(conditions: list[str], top_k: int, save_answers: bool):
    needs_rag = any(c in conditions for c in ("rag", "rag_lora"))
    needs_lora = any(c in conditions for c in ("lora", "rag_lora"))

    collection = None
    if needs_rag:
        collection = get_collection()
        print(f"RAG database loaded: {collection.count()} chunks, top_k={top_k}")

    tokenizer, model, has_adapter = load_model(with_lora=needs_lora)
    if needs_lora and not has_adapter:
        conditions = [c for c in conditions if c in ("base", "rag")]
        print(f"Running reduced condition set: {conditions}")

    # Pre-retrieve contexts once so rag and rag_lora see identical context
    contexts = {}
    if needs_rag:
        for qa in QA_TEST_SET:
            contexts[qa["question"]] = retrieve_context(
                collection, qa["question"], top_k
            )

    scores = {c: [] for c in conditions}
    answers = {c: [] for c in conditions}

    print(f"\nEvaluating {len(QA_TEST_SET)} questions × {len(conditions)} conditions ...\n")

    for idx, qa in enumerate(QA_TEST_SET, 1):
        q = qa["question"]
        print(f"[{idx}/{len(QA_TEST_SET)}] {q[:70]}")

        for cond in conditions:
            use_context = cond in ("rag", "rag_lora")
            use_adapter = cond in ("lora", "rag_lora")
            context = contexts.get(q) if use_context else None

            if has_adapter:
                if use_adapter:
                    answer = generate(tokenizer, model, q, context)
                else:
                    # Temporarily turn the adapter off → pure base model
                    with model.disable_adapter():
                        answer = generate(tokenizer, model, q, context)
            else:
                answer = generate(tokenizer, model, q, context)

            score = keyword_score(answer, qa["expected_keywords"])
            scores[cond].append(score)
            answers[cond].append({"question": q, "answer": answer, "score": score})
            print(f"    {cond:<9} score={score:.2f}")

    # ─── Report ───
    print(f"\n{'='*64}")
    print("ABLATION RESULTS — mean keyword coverage (15 held-out questions)")
    print("="*64)
    print(f"{'Condition':<12} {'Knowledge source':<28} {'Score':>8}")
    print("-"*64)
    labels = {
        "base":     "none",
        "rag":      "retrieval only",
        "lora":     "fine-tuned weights only",
        "rag_lora": "retrieval + fine-tuned",
    }
    summary = {}
    for cond in ALL_CONDITIONS:
        if cond in scores and scores[cond]:
            mean = statistics.mean(scores[cond])
            summary[cond] = round(mean, 4)
            print(f"{cond:<12} {labels[cond]:<28} {mean:>8.3f}")

    if "base" in summary:
        print("-"*64)
        for cond in ("rag", "lora", "rag_lora"):
            if cond in summary:
                print(f"{'Δ vs base':<12} {cond:<28} {summary[cond]-summary['base']:>+8.3f}")

    out = {"summary": summary, "top_k": top_k, "n_questions": len(QA_TEST_SET)}
    if save_answers:
        out["answers"] = answers
    with open("ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to ablation_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions", nargs="+", default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS,
        help="Which conditions to run (default: all four)",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Chunks retrieved for RAG")
    parser.add_argument("--save_answers", action="store_true",
                        help="Include full generated answers in the JSON output")
    args = parser.parse_args()

    run_ablation(args.conditions, args.top_k, args.save_answers)


if __name__ == "__main__":
    main()
