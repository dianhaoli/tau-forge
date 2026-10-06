# Gold semantic labels for real agent turns

`turns.jsonl` holds 806 unique assistant text turns that Qwen3-4B-Instruct-2507 wrote in the saved episode audits. The tasks are synthetic and generated from db.json: 196 episodes from `data/trained/episode_transcripts_subset.json` and `episode_foreign_check.json`. Each line carries the task context the labeler saw.

`labels.jsonl` holds what each turn *does*: info requests, consent request, proposed action and details, refusal kind, and so on. The schema is "Semantic NLU labels" in the stage contract. Every batch was labeled by two independent LLM labelers, and an adjudicator settled the 220 turns where they disagreed.

`tests/test_nlu_gold.py` uses these labels to measure the scripted user's understanding of agent replies.
