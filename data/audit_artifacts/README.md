# Audit artifacts (2026-10-04/05, single A10G)

These are the raw outputs behind the reports in `data/reports/`. Every run used:

- model: Qwen/Qwen3-4B-Instruct-2507
- sampling: temperature 1.0, top_p 1.0, top_k disabled
- environment: `VLLM_USE_FLASHINFER_SAMPLER=0`, because the box has no nvcc

The raw JSONs sit in `data/trained/`. That directory is gitignored, so these files were force-added.

## Reports (read these first)

| File | What |
|---|---|
| `data/reports/diagnosis.txt` | Single-step variance audit, `scripts/diagnose.sh` output (n=16, 541 scenarios) |
| `data/reports/recommended_mix.txt`, `dead_scenario_ids.txt` | diagnose.sh side files |
| `data/reports/episode_audit.md` | Episode audit before the fixes (seed 0, 250 tasks x 8) |
| `data/reports/episode_transcript_audit.md` | Transcript audit that found the env bugs |
| `data/reports/episode_audit_postfix.md` | Re-audit after the fixes (`f2cdb02`, seed 1, 250 x 8), plus the foreign_order_refusal transcript check |

## Raw outputs (`data/trained/`)

| File | Run |
|---|---|
| `audit_n16.json` / `.log` | Single-step audit, `zero_shot_baseline --samples-per-scenario 16 --with-shaping --save-completions` |
| `episode_audit.log` | First episode run, 885 x 16. It was killed for low host RAM on turn 1 and produced no results |
| `episode_audit_sub50_n8.json` / `.partial.jsonl` / `.log` | Pre-fix episode audit, 250 x 8 |
| `episode_transcripts_subset.json` / `.partial.jsonl` / `.log` | Pre-fix transcript run: 39 tasks x 4, `--save-transcripts` |
| `episode_audit_s1_sub50_n8.json` / `.partial.jsonl` / `.log` | Post-fix episode audit, 250 x 8 |
| `episode_foreign_check.json` / `.partial.jsonl` / `.log` | Post-fix foreign_order_refusal check: 10 tasks x 4, `--save-transcripts` |

## Task files (`data/episodes/`)

- `episodes_s0.jsonl` and `episodes_s1.jsonl` come from `python -m tau_forge.episodes.generate --per-template 200 --seed {0,1}`.
- Code changes since then alter the generated text, so these are the exact task instances that were audited.
- `*_sub50.jsonl` holds the 50-per-template draws (random.Random(0)).
- `episodes_s0_transcript_subset.jsonl` and `transcript_subset_meta.json` record the transcript-run subset and why each task was picked.
- `episodes_s1_foreign_check.jsonl` is the subset for the post-fix check.

## Readable transcripts (`transcripts_text/`)

One file per episode. The ~6k-token system prompt is stripped. Each file has a header with the task spec, the reward and the reward breakdown, followed by the conversation with message indices. The reports cite these indices.

- `failures/`: 74 failed episodes from the pre-fix transcript run. One subagent audited these.
- `solved/`: 44 successful episodes from always-solved cells, pre-fix. A second subagent audited these.
- `foreign_check_postfix/`: all 40 post-fix foreign_order_refusal episodes. Each one was read for the verdict in `episode_audit_postfix.md`.

The filename format is `<template>__<task_id>__s<sample>__<end_reason>.txt` (the foreign check adds `__r<reward>`).
