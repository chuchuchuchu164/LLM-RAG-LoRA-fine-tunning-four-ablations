"""
Evaluation suite for the MR.AsyncFL mini-research project.

Covers TWO evaluation tracks:
  1. RAG Evaluation  — retrieval quality metrics + answer quality
  2. LoRA Evaluation — domain QA accuracy, perplexity, base vs fine-tuned comparison

Usage:
    python evaluate.py --rag                    # RAG retrieval + answer quality
    python evaluate.py --lora                   # LoRA model domain QA
    python evaluate.py --rag --lora             # both tracks
    python evaluate.py --rag --llm              # RAG with LLM-generated answers
    python evaluate.py --perplexity             # perplexity on held-out paper text
"""

import argparse
import json
import math
import re
import statistics
from typing import Optional

# ─── Test-set QA for evaluation ───────────────────────────────────────────────
# These are held-out from training — use them to measure domain knowledge.

QA_TEST_SET = [
    {
        "question": "What convergence rate does MR.AsyncFL achieve?",
        "expected_keywords": ["O(T^{-1/4})", "T^{-1/4}", "1/4"],
        "expected_answer": "MR.AsyncFL achieves an O(T^{-1/4}) convergence rate for the minimum expected squared gradient norm.",
        "section": "convergence"
    },
    {
        "question": "What parameter scaling achieves the O(T^{-1/4}) convergence rate?",
        "expected_keywords": ["eta", "1/sqrt(T)", "r = 2/N", "gamma", "T = N^2"],
        "expected_answer": "eta=1/sqrt(T), r=2/N, gamma=1-1/T^{1/4}, T=N^2",
        "section": "convergence"
    },
    {
        "question": "What is the MR.AsyncFL update equation?",
        "expected_keywords": ["gamma", "replacement", "c_{t-1}", "w_stale", "correction"],
        "expected_answer": "w_t^g = gamma*w_{t-1}^g + (1-gamma)*w_t^(i) + gamma*c_{t-1}^(i)*(w_t^(i) - w_{t-delta_t}^(i))",
        "section": "algorithm"
    },
    {
        "question": "What accuracy did MR.AsyncFL achieve on CIFAR-10 IID without a staleness threshold?",
        "expected_keywords": ["80", "80.77"],
        "expected_answer": "80.77 ± 2.33%",
        "section": "experiments"
    },
    {
        "question": "What is the recursive weight update rule for non-participating clients in MR.AsyncFL?",
        "expected_keywords": ["gamma", "c_{t-1}^(j)", "discount"],
        "expected_answer": "For j != i: c_t^(j) = gamma * c_{t-1}^(j). The weight decays by gamma each round they don't participate.",
        "section": "algorithm"
    },
    {
        "question": "Name the three baselines compared in MR.AsyncFL experiments.",
        "expected_keywords": ["FedAsync", "TWAFL", "Rolling FedAvg"],
        "expected_answer": "FedAsync, TWAFL, and Rolling FedAvg",
        "section": "experiments"
    },
    {
        "question": "What datasets are used in MR.AsyncFL experiments?",
        "expected_keywords": ["CIFAR-10", "CIFAR-100"],
        "expected_answer": "CIFAR-10 and CIFAR-100",
        "section": "experiments"
    },
    {
        "question": "What is the staleness delta_t in asynchronous federated learning?",
        "expected_keywords": ["rounds", "outdated", "stale", "trained from"],
        "expected_answer": "Staleness delta_t is the number of aggregation rounds that elapsed while a client was training locally, meaning the client trained from a global model that is delta_t rounds out of date.",
        "section": "algorithm"
    },
    {
        "question": "How many clients are used in MR.AsyncFL experiments?",
        "expected_keywords": ["30", "N=30"],
        "expected_answer": "N = 30 clients",
        "section": "experiments"
    },
    {
        "question": "What is the rho condition (Assumption 6) in MR.AsyncFL?",
        "expected_keywords": ["rho", "tau", "gamma", "r", "contraction", "< 1"],
        "expected_answer": "rho = 4 * tau^2 * [(1-gamma)^2 + gamma^2 * r] < 1. This contraction condition ensures the sum of squared update norms is bounded.",
        "section": "convergence"
    },
    {
        "question": "What model architecture is used in MR.AsyncFL experiments?",
        "expected_keywords": ["ResNet18", "ResNet"],
        "expected_answer": "ResNet18",
        "section": "experiments"
    },
    {
        "question": "What is the key difference between MR.AsyncFL and FedAsync?",
        "expected_keywords": ["replace", "removal", "staleness", "correction term"],
        "expected_answer": "MR.AsyncFL explicitly removes the old stale contribution of the participating client via a correction term gamma*c_{t-1}^(i)*(w_t^(i) - w_stale). FedAsync only attenuates stale contributions through weighting, leaving them in the global model.",
        "section": "algorithm"
    },
    {
        "question": "What Dirichlet concentration parameter is used for non-IID data partitioning?",
        "expected_keywords": ["0.5", "alpha", "Dirichlet"],
        "expected_answer": "Concentration parameter alpha = 0.5",
        "section": "experiments"
    },
    {
        "question": "Does MR.AsyncFL's global model always remain a convex combination of client models?",
        "expected_keywords": ["yes", "convex combination", "sum to 1", "normalization"],
        "expected_answer": "Yes. MR.AsyncFL maintains the invariant that the global model is always a convex combination of the most recent cached local models from all N clients, with weights that sum to 1.",
        "section": "algorithm"
    },
    {
        "question": "What is the extra memory cost of MR.AsyncFL compared to FedAsync?",
        "expected_keywords": ["N", "cached", "N model copies", "one model per client"],
        "expected_answer": "MR.AsyncFL requires storing N additional model copies on the server (one per client), whereas FedAsync only stores the current global model.",
        "section": "practical"
    },
]


# ─── RAG Evaluation ───────────────────────────────────────────────────────────

def evaluate_rag(use_llm: bool = False):
    """
    Evaluate the RAG pipeline:
      - Hit@K: Is a relevant chunk retrieved in the top-K results?
      - MRR: Mean reciprocal rank of first relevant chunk
      - Answer quality (keyword match) if use_llm=True
    """
    import chromadb
    from chromadb.utils import embedding_functions

    DB_PATH = "./chroma_db"
    EMBED_MODEL = "all-MiniLM-L6-v2"
    TOP_K = 5

    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_collection(name="mrasyncfl_paper", embedding_function=embed_fn)

    print(f"\n{'='*60}")
    print("RAG EVALUATION")
    print(f"Collection: {collection.count()} chunks | Top-K: {TOP_K}")
    print("="*60)

    section_to_keywords = {
        "convergence": ["convergence", "rate", "O(T", "T^{-1", "gradient", "assumption", "theorem", "lemma"],
        "algorithm": ["update", "replacement", "weight", "cached", "staleness", "gamma", "client"],
        "experiments": ["CIFAR", "accuracy", "IID", "non-IID", "baseline", "FedAsync", "TWAFL", "result"],
        "practical": ["memory", "storage", "server", "cache", "model"],
    }

    hits_at_1 = []
    hits_at_k = []
    reciprocal_ranks = []
    keyword_scores = []

    tokenizer, model = None, None
    if use_llm:
        from query_rag import load_llm, generate_answer
        tokenizer, model = load_llm(use_lora=False)

    for qa in QA_TEST_SET:
        results = collection.query(query_texts=[qa["question"]], n_results=TOP_K)
        docs = results["documents"][0]

        # Relevance check: does any retrieved chunk contain expected keywords?
        section = qa.get("section", "")
        rel_keywords = section_to_keywords.get(section, []) + qa["expected_keywords"]

        first_relevant = None
        for rank, doc in enumerate(docs, 1):
            doc_lower = doc.lower()
            if any(kw.lower() in doc_lower for kw in rel_keywords):
                first_relevant = rank
                break

        hit1 = 1 if first_relevant == 1 else 0
        hitk = 1 if first_relevant is not None else 0
        rr = 1.0 / first_relevant if first_relevant else 0.0

        hits_at_1.append(hit1)
        hits_at_k.append(hitk)
        reciprocal_ranks.append(rr)

        # Answer keyword matching (with LLM)
        answer_score = None
        if use_llm:
            context = "\n\n".join(docs)
            answer = generate_answer(tokenizer, model, qa["question"], context)
            matched = sum(1 for kw in qa["expected_keywords"] if kw.lower() in answer.lower())
            answer_score = matched / len(qa["expected_keywords"])
            keyword_scores.append(answer_score)

        status = "HIT@K" if hitk else "MISS "
        print(f"[{status}] rank={first_relevant or 'None':>4} | {qa['question'][:65]}")

    print(f"\n--- Retrieval Metrics (n={len(QA_TEST_SET)}) ---")
    print(f"Hit@1:  {statistics.mean(hits_at_1):.3f}")
    print(f"Hit@{TOP_K}:  {statistics.mean(hits_at_k):.3f}")
    print(f"MRR:    {statistics.mean(reciprocal_ranks):.3f}")

    if keyword_scores:
        print(f"\n--- Answer Quality (keyword match) ---")
        print(f"Mean keyword coverage: {statistics.mean(keyword_scores):.3f}")

    return {
        "hit_at_1": statistics.mean(hits_at_1),
        f"hit_at_{TOP_K}": statistics.mean(hits_at_k),
        "mrr": statistics.mean(reciprocal_ranks),
    }


# ─── LoRA Evaluation ──────────────────────────────────────────────────────────

def evaluate_lora(model_dir: str, compare_base: bool = True):
    """
    Evaluate the fine-tuned Qwen2-1.5B-Instruct + LoRA on domain QA.
    Metrics: keyword recall, exact-match rate, side-by-side with base model.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
    SYSTEM = ("You are an expert AI assistant specializing in federated learning "
              "and the MR.AsyncFL framework. Answer concisely and accurately.")

    print(f"\n{'='*60}")
    print("LORA EVALUATION")
    print(f"Model dir: {model_dir} | Base comparison: {compare_base}")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def load_model(lora=True):
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if lora:
            return PeftModel.from_pretrained(base, model_dir)
        return base

    def infer(mdl, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def keyword_score(answer: str, keywords: list[str]) -> float:
        return sum(1 for kw in keywords if kw.lower() in answer.lower()) / len(keywords)

    ft_model = load_model(lora=True)
    ft_model.eval()

    base_model = None
    if compare_base:
        base_model = load_model(lora=False)
        base_model.eval()

    ft_scores, base_scores = [], []

    print(f"\n{'Question':<55} {'FT':>5} {'Base':>5}")
    print("-" * 70)

    for qa in QA_TEST_SET:
        ft_answer = infer(ft_model, qa["question"])
        ft_kw = keyword_score(ft_answer, qa["expected_keywords"])
        ft_scores.append(ft_kw)

        base_kw = 0.0
        if base_model:
            base_answer = infer(base_model, qa["question"])
            base_kw = keyword_score(base_answer, qa["expected_keywords"])
            base_scores.append(base_kw)

        print(f"{qa['question'][:55]:<55} {ft_kw:>5.2f} {base_kw:>5.2f}")

    print(f"\n--- Domain QA Summary ---")
    print(f"Fine-tuned mean keyword coverage:  {statistics.mean(ft_scores):.3f}")
    if base_scores:
        print(f"Base model  mean keyword coverage:  {statistics.mean(base_scores):.3f}")
        delta = statistics.mean(ft_scores) - statistics.mean(base_scores)
        print(f"Improvement from LoRA:              {delta:+.3f}")

    return {"ft_keyword_score": statistics.mean(ft_scores),
            "base_keyword_score": statistics.mean(base_scores) if base_scores else None}


# ─── Perplexity Evaluation ────────────────────────────────────────────────────

def evaluate_perplexity(model_dir: str):
    """
    Compute perplexity of base and fine-tuned models on held-out paper text.
    Lower perplexity = model has better 'knowledge' of the paper's language.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from paper_content import get_full_text

    MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

    # Use a subset of the paper text as held-out evaluation text
    full_text = get_full_text()
    # Take middle portion (avoid beginning/end which overlap with training)
    eval_text = full_text[len(full_text)//3: 2*len(full_text)//3]
    print(f"\nPerplexity evaluation on {len(eval_text)} chars of paper text")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def compute_ppl(mdl) -> float:
        mdl.eval()
        encodings = tokenizer(eval_text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = encodings.input_ids.to(mdl.device)
        with torch.no_grad():
            output = mdl(input_ids, labels=input_ids)
        return math.exp(output.loss.item())

    print("Loading base model ...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                                device_map="auto", trust_remote_code=True)
    base_ppl = compute_ppl(base)
    del base

    print("Loading fine-tuned model ...")
    base2 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                                  device_map="auto", trust_remote_code=True)
    ft_model = PeftModel.from_pretrained(base2, model_dir)
    ft_ppl = compute_ppl(ft_model)

    print(f"\n--- Perplexity Results ---")
    print(f"Base model perplexity:        {base_ppl:.2f}")
    print(f"Fine-tuned model perplexity:  {ft_ppl:.2f}")
    print(f"Reduction:                    {base_ppl - ft_ppl:.2f} ({(base_ppl-ft_ppl)/base_ppl*100:.1f}%)")

    return {"base_ppl": base_ppl, "ft_ppl": ft_ppl}


# ─── Evaluation report ────────────────────────────────────────────────────────

def print_summary(results: dict):
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print("="*60)
    for k, v in results.items():
        if v is not None:
            print(f"  {k:<40}: {v:.4f}" if isinstance(v, float) else f"  {k:<40}: {v}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag", action="store_true", help="Run RAG evaluation")
    parser.add_argument("--lora", action="store_true", help="Run LoRA QA evaluation")
    parser.add_argument("--perplexity", action="store_true", help="Run perplexity evaluation")
    parser.add_argument("--llm", action="store_true", help="Use LLM for RAG answer generation")
    parser.add_argument("--no_base_compare", action="store_true",
                        help="Skip base model comparison in LoRA eval")
    parser.add_argument("--model_dir", type=str, default="./lora_output",
                        help="Path to LoRA adapter directory")
    args = parser.parse_args()

    if not (args.rag or args.lora or args.perplexity):
        parser.print_help()
        print("\nNote: Run with --rag for RAG-only eval (no GPU needed).")
        return

    all_results = {}

    if args.rag:
        rag_results = evaluate_rag(use_llm=args.llm)
        all_results.update({f"rag_{k}": v for k, v in rag_results.items()})

    if args.lora:
        import os
        if not os.path.isdir(args.model_dir):
            print(f"LoRA model not found at {args.model_dir}. Run train_lora.py first.")
        else:
            lora_results = evaluate_lora(args.model_dir, compare_base=not args.no_base_compare)
            all_results.update({f"lora_{k}": v for k, v in lora_results.items()})

    if args.perplexity:
        import os
        if not os.path.isdir(args.model_dir):
            print(f"LoRA model not found at {args.model_dir}. Run train_lora.py first.")
        else:
            ppl_results = evaluate_perplexity(args.model_dir)
            all_results.update({f"ppl_{k}": v for k, v in ppl_results.items()})

    if all_results:
        print_summary(all_results)

        # Save results to JSON
        with open("eval_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\nResults saved to eval_results.json")


if __name__ == "__main__":
    main()
