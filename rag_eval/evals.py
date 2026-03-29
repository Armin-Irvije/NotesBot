import sys
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from ragas import Dataset, experiment
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_chain import LocalRAGChain


def create_notesbot_client(**kwargs) -> LocalRAGChain:
    return LocalRAGChain(
        vector_store_path=str(PROJECT_ROOT / "vector_store"),
        model_name="llama3",
        embedding_model="nomic-embed-text",
        **kwargs,
    )


def query_notesbot_with_contexts(chain: LocalRAGChain, question: str) -> Dict[str, Any]:
    result = chain.query(question, show_sources=False)
    answer = result.get("result", "") or result.get("answer", "")
    source_docs = result.get("source_documents") or []
    retrieved_contexts = [doc.page_content for doc in source_docs]
    return {"answer": answer, "retrieved_contexts": retrieved_contexts}


# Create an OpenAI-compatible client for Ollama (used by RAGAS evaluators, not NotesBot itself)
client = OpenAI(
    api_key="ollama",  # Ollama doesn't require a real key
    base_url="http://localhost:11434/v1",
)
llm = llm_factory("llama3", provider="openai", client=client)
embeddings = embedding_factory("openai", model="nomic-embed-text", client=client)

# NotesBot RAG chain (LocalRAGChain) used for answering questions during evaluation.
# LocalRAGChain does not take a logdir parameter, so we keep this simple.
notesbot_chain = create_notesbot_client()


def load_dataset():
    """
    Load the existing NotesBot evaluation dataset from CSV.

    The dataset lives at:
      evals/datasets/notesbot_eval_dataset.csv
    and has columns:
      document,question,reference_answer,key_claim_1,key_claim_2,key_claim_3,...
    """
    dataset = Dataset.load(
        name="notesbot_eval_dataset",
        backend="local/csv",
        root_dir="evals",
    )
    return dataset


faithfulness_scorer = Faithfulness(llm=llm)
answer_relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)


def extract_key_claims(row) -> list[str]:
    """
    Collect all non-empty key_claim_* fields from a dataset row into a list.
    """
    claims: list[str] = []
    # row may behave like a dict; use .items() for safety
    for col, val in row.items():
        if col.startswith("key_claim_") and val is not None:
            text = str(val).strip()
            if text:
                claims.append(text)
    return claims


def compute_claim_metrics(response: str, key_claims: list[str]) -> dict:
    """
    Simple statistical metrics over key claims:
      - claims_total
      - claims_hit
      - claim_coverage (0-1)
      - all_claims_present (bool)
    """
    normalized_resp = response.lower()
    hits = 0
    for claim in key_claims:
        if claim and claim.lower() in normalized_resp:
            hits += 1
    coverage = hits / len(key_claims) if key_claims else 0.0
    return {
        "claims_total": len(key_claims),
        "claims_hit": hits,
        "claim_coverage": coverage,
        "all_claims_present": coverage == 1.0 if key_claims else False,
    }


@experiment()
async def run_experiment(row):
    """
    Run a single evaluation example:
      - Ask NotesBot (LocalRAGChain) to answer the question.
      - Score with RAGAS Faithfulness and Answer Relevancy.
      - Score with custom claim-level metrics.
    """
    question = row["question"]

    rag_result = query_notesbot_with_contexts(notesbot_chain, question)
    answer = rag_result["answer"]
    retrieved_contexts = rag_result["retrieved_contexts"]

    faith = faithfulness_scorer.score(
        user_input=question,
        response=answer,
        retrieved_contexts=retrieved_contexts,
    )

    relevancy = answer_relevancy_scorer.score(
        user_input=question,
        response=answer,
    )

    key_claims = extract_key_claims(row)
    claim_stats = compute_claim_metrics(answer, key_claims)

    experiment_view = {
        **row,
        "response": answer,
        "faithfulness": faith.value,
        "answer_relevancy": relevancy.value,
        **claim_stats,
    }
    return experiment_view


async def main():
    dataset = load_dataset()
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)

    # Save experiment results to CSV
    experiment_results.save()
    csv_path = Path(".") / "experiments" / f"{experiment_results.name}.csv"
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
