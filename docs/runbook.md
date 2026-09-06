# Runbook: SSH login to a measured result

Every command, in order, from the moment the box accepts a connection. Assumes
the instance is already launched -- see `phase7_aws_setup.md` for the sizing
reasoning and the AWS CLI launch sequence.

Two paths run through this doc. **A single GPU** can do everything except
training: the variance audit and the Phase 8 evaluation are both inference-only.
**Full-parameter GRPO on a 4B model does not fit one GPU** (roughly 64 GB of
weights, gradients and Adam state before a single activation), so Step 6 is
marked multi-GPU and there is no honest single-GPU substitute for it short of
LoRA or ZeRO-3 with CPU offload.

---

## Step 1 -- connect, and don't lose the session

```bash
ssh -i ~/.ssh/<your-key>.pem ubuntu@<public-dns>
tmux new -s tauforge
```

Do not skip tmux. Every step past bootstrap runs for hours, and a dropped SSH
connection kills a foreground process and takes the run with it. Detach with
`Ctrl-b d`, reattach later with `tmux attach -t tauforge`.

---

## Step 2 -- get the repo

**Fresh box, nothing there yet.** The bootstrap script does the clone for you --
you do not clone first:

```bash
curl -O https://raw.githubusercontent.com/dianhaoli/tau-forge/main/infra/ec2_bootstrap.sh
chmod +x ec2_bootstrap.sh
./ec2_bootstrap.sh claude/grpo-reward-variance-47cbgo   # or main, once merged
```

**Box that already has `~/tau-forge`** from an earlier session -- the common case
once you have run anything on it before. Do not re-clone; fetch the branch:

```bash
cd ~/tau-forge
git fetch origin claude/grpo-reward-variance-47cbgo
git checkout claude/grpo-reward-variance-47cbgo
git submodule update --init --recursive
```

`git pull` alone is not enough: it updates the branch you are on, and the work
this runbook depends on is on a different one. No dependency changes are needed
for the audit -- the only `pyproject.toml` change is a floor bump on `trl`,
which lives in the `train` extra and only matters at Step 6.

The branch argument matters: the fixes this runbook depends on (prompt
truncation, prompt parity, shaping, curriculum, the scorecard) live on that
branch until it merges.

The script checks the driver and disk, installs `tmux` and `uv`, clones the repo
and the tau2 submodule, installs both dependency sets, copies `.env.example` to
`.env`, runs the test suite, and pre-downloads the model weights. That last one
matters: an 8 GB download inside your first timed run makes that run's wall-clock
meaningless.

If it finishes clean, `uv run pytest -q` passed and the GPU is visible. If the
test suite fails, stop here -- nothing downstream is worth starting.

---

## Step 3 -- the API key for the user simulator

Only Phase 8 evaluation needs this. The policy model is served locally by vLLM
and needs no key; the *user* in every conversation is a hosted model.

```bash
cd ~/tau-forge
nano .env      # set OPENAI_API_KEY=sk-...
```

tau2 calls `load_dotenv()` with no path, which searches upward from the working
directory, so a `.env` at the repo root is found whenever you run from there.
It is gitignored.

**Anthropic instead of OpenAI:** set `ANTHROPIC_API_KEY` in the same file and
pass `--user-llm anthropic/claude-sonnet-5` (or `anthropic/claude-haiku-4-5`)
in Step 5. tau2 routes the user simulator through litellm, so any provider it
supports works unchanged.

Two constraints on that choice, both about comparability rather than
correctness. Hold the simulator **fixed** across the baseline and every trained
run: it speaks half of every conversation, so swapping it changes the benchmark
rather than the policy, and a before/after measured across two different
simulators measures nothing. And tau2's published leaderboard numbers use the
gpt-4.1 default, so a run on any other simulator is internally valid but cannot
be lined up against those.

Cost scales as tasks x trials x turns. The default Step 5 run is 40 tasks x 4
trials, and each task is a multi-turn conversation, so budget accordingly and
start with `--num-trials 1` if you want to see the shape of the bill first.

---

## A note on `uv run` before Step 4

Every GPU command below is `uv run --extra train ...`, not plain `uv run`.

`uv run` syncs the environment to match the lockfile before executing, and an
extra you are not asking for is not part of that match -- so a plain `uv run`
in a project where you previously ran `uv sync --extra train` can *uninstall*
torch, vLLM and transformers on its way to running your command. You would then
watch it re-download several gigabytes on the next `--extra train` invocation.

Commands that touch only tau2 (`data_scorecard.py`, `bucket_analysis.py`,
`run_tau2`, `pytest`) need no extra and are written without one.

---

## Step 4 -- variance audit (single GPU, ~1-2 h)

```bash
cd ~/tau-forge
uv run --extra train python -m tau_forge.train.zero_shot_baseline --use-vllm \
    --samples-per-scenario 16 \
    --temperature 1.0 --top-p 1.0 --top-k 0 \
    --max-new-tokens 256 --max-model-len 8192 \
    --with-shaping --save-completions \
    --output data/trained/audit_n16.json
```

The sampling flags are set explicitly so this measures the same distribution the
trainer will sample from. Left unset, the plain-HF path inherits Qwen3's shipped
`generation_config` (`top_p=0.8`, `top_k=20`), which narrows exactly the
variance this run exists to measure.

`--with-shaping` scores every completion twice, under `reward()` alone and under
`reward()` plus the shaping term, so one pass reports both zero-variance
fractions. `--save-completions` makes the output large but is what lets you read
the actual failures afterward.

Generation is the GPU part and finishes in minutes. Scoring is the CPU part and
runs *after* vLLM tears its engine down and prints its shutdown banner, which is
why a run that is working can look hung. It now prints progress, and
`--score-workers` splits it across processes -- unset, it takes one per core,
capped at 16. Pass `--score-workers 1` to score in-process if you need a
deterministic single-threaded run to profile.

Then diagnose it:

```bash
uv run python scripts/data_scorecard.py data/trained/audit_n16.json
uv run python scripts/bucket_analysis.py data/trained/audit_n16.json
```

The scorecard ranks all 30 cells by yield x headroom and prints a recommended
`--category-mix`. The bucket analysis prints the flat/varying split before and
after shaping, plus how much of the corpus is structurally binary and therefore
beyond any sampling fix.

### Two different things this command can measure

`--split all` (the default above) scores every scenario. That is the right
setting for a variance audit: you want the whole corpus characterized.

It is the wrong setting for a **before/after on synthetic data**. Train on the
corpus, re-score the same corpus, and the improvement you measure is partly the
policy having memorized those exact prompts. For that comparison use the
held-out slice:

```bash
uv run --extra train python -m tau_forge.train.zero_shot_baseline --use-vllm \
    --split val --val-fraction 0.1 --category-mix real --curriculum-seed 0 \
    --samples-per-scenario 16 \
    --temperature 1.0 --top-p 1.0 --top-k 0 \
    --max-new-tokens 256 --max-model-len 8192 --with-shaping \
    --output data/trained/synth_baseline_val.json
```

`--split val` reproduces exactly the slice `grpo_train` holds out, so pass the
**same** `--val-fraction`, `--category-mix` and `--curriculum-seed` to both
commands. Different values compute different splits and the comparison is void;
a test asserts the two code paths agree given equal flags, and the settings are
recorded into the output JSON so a later run can be checked rather than assumed.

After training, re-run that exact command against the checkpoint and compare
`mean_score`.

**What that number does and does not tell you.** It measures whether training
moved the reward it was optimized against, on prompts it did not see. That is a
real sanity check and it is cheap. It is not evidence of tau2-bench improvement:
reward can rise while benchmark performance does not, because each scenario is
one decision graded in isolation while a real task chains roughly five and fails
whole. Step 5 is the number that answers the actual question.

---

## Step 5 -- baseline evaluation (single GPU)

Two processes. In one tmux pane, serve the policy:

```bash
uv run --extra train vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --served-model-name tau-forge-policy \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 16384 --port 8000
```

`--tool-call-parser hermes` parses Qwen's `<tool_call>{...}</tool_call>`
convention, the same one training grades. Without it vLLM returns the call as
plain assistant text, tau2 sees an agent that never calls a tool, and every task
fails for a reason unrelated to the policy.

In another pane (`Ctrl-b c`), run the benchmark:

```bash
cd ~/tau-forge
uv run python -m tau_forge.eval.run_tau2 --label baseline \
    --task-split-name test --num-trials 4
```

Add `--dry-run` first to print the resolved config without spending anything.
Results land in `data/simulations/tau_forge_baseline_retail_test_<timestamp>/`.
Score them with `uv run tau2 view`.

**This is the number everything else is measured against.** It is also the only
source of real failure trajectories, which is what turns the scorecard's
relevance prior from an argument into evidence.

---

## Step 6 -- training (multi-GPU)

Preflight first. It imports no torch and touches no GPU:

```bash
MIX=$(uv run python scripts/data_scorecard.py data/trained/audit_n16.json --emit-mix)
uv run --extra train python -m tau_forge.train.grpo_train --dry-run \
    --category-mix "$MIX" \
    --exclude-zero-variance-from data/trained/audit_n16.json
```

Read the printed mixture and prompt-token distribution before spending a
GPU-hour. Then the smoke test, then the full run:

```bash
uv run --extra train accelerate launch --config_file infra/accelerate_zero2.yaml \
    -m tau_forge.train.grpo_train --smoke-test \
    --category-mix "$MIX" \
    --exclude-zero-variance-from data/trained/audit_n16.json

uv run --extra train accelerate launch --config_file infra/accelerate_zero2.yaml \
    -m tau_forge.train.grpo_train \
    --category-mix "$MIX" \
    --exclude-zero-variance-from data/trained/audit_n16.json
```

The smoke test caps at 100 steps and writes to a separate directory. Its job is
to prove the generation to reward to update loop works end to end and to produce
a real seconds-per-rollout number, so run it even when you are confident.

---

## Step 7 -- evaluate the trained checkpoint

Restart the vLLM server pointed at the checkpoint, then re-run Step 5 changing
only the label:

```bash
uv run --extra train vllm serve data/trained/phase7_run/final \
    --served-model-name tau-forge-policy \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 16384 --port 8000

uv run python -m tau_forge.eval.run_tau2 --label step-final \
    --task-split-name test --num-trials 4
```

Every other flag stays identical to the baseline run. That is the whole point of
them being flags with recorded defaults.

---

## Step 8 -- get the results off the box before it dies

The instance is ephemeral relative to your work. Checkpoints and simulation
outputs are gitignored because they are gigabytes, so they will not leave via a
commit.

```bash
# summary numbers: commit them
cd ~/tau-forge && git add README.md && git commit && git push

# artifacts: copy them down
scp -i ~/.ssh/<key>.pem \
    ubuntu@<public-dns>:~/tau-forge/data/trained/audit_n16.json ./
scp -r -i ~/.ssh/<key>.pem \
    ubuntu@<public-dns>:~/tau-forge/data/simulations ./
```

Then stop the instance rather than terminating it, so the model cache and
checkpoints survive to the next session:

```bash
aws ec2 stop-instances --instance-ids <InstanceId>
```

---

## If something goes wrong

| Symptom | Cause |
|---|---|
| Every task fails with the agent never calling a tool | vLLM served without `--tool-call-parser hermes` |
| `ValueError: ... would truncate ... prompts` | A `--max-prompt-length` below the real ~5-7k. Drop the flag |
| `--max-model-len N is below the M tokens ...` | Raise `--max-model-len`, or lower `--max-new-tokens`. Never shorten the prompt |
| `AssertionError: Training and evaluation system prompts differ` | Prompt parity regression. See `tau_forge/eval/prompt_parity.py` |
| `No .env file found` warning from tau2 | Harmless unless you are running Step 5, which needs the key |
| CUDA OOM during training | `--per-device-train-batch-size` down, `--gradient-accumulation-steps` up by the same factor. The effective batch is unchanged |
| Run died when SSH dropped | tmux |
