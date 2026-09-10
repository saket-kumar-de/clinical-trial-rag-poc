"""
End-to-end ingestion orchestration:
  fetch trials -> extract entities -> chunk -> embed -> write to Aurora

Run once schema.sql has been applied to the Aurora cluster.
Usage: python -m scripts.run_ingestion --condition "non-small cell lung cancer"
"""
import argparse


def main(condition: str, max_results: int) -> None:
    """
    TODO, in order:
      1. records = fetch_trials(condition, max_results)
      2. For each record: extract Trial/Interventions/Endpoints/Eligibility
      3. Write structured rows to Aurora (trials, interventions, endpoints,
         eligibility_criteria tables)
      4. Chunk each trial's text sections, embed the chunks
      5. Write chunks + embeddings to the chunks table
    """
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--max-results", type=int, default=30)
    args = parser.parse_args()
    main(args.condition, args.max_results)
