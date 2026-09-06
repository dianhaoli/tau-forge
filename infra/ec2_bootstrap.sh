#!/usr/bin/env bash
# Phase 7/8 GPU box bootstrap. Run on a fresh AWS Deep Learning AMI (Ubuntu,
# GPU PyTorch variant) after SSH'ing in. It does not launch or configure the
# instance itself -- see docs/phase7_aws_setup.md for that, and docs/runbook.md
# for the complete command sequence this script is only the first step of.
#
# Usage: ./ec2_bootstrap.sh [git-ref]
#   git-ref defaults to "main". Pass a feature branch to test unmerged work.
set -euo pipefail

GIT_REF="${1:-main}"
REPO_URL="https://github.com/dianhaoli/tau-forge.git"
REPO_DIR="$HOME/tau-forge"

echo "== GPU and driver =="
nvidia-smi
GPU_COUNT="$(nvidia-smi --list-gpus | wc -l)"
echo "GPUs visible: $GPU_COUNT"

echo "== Disk =="
# Model weights (~8GB bf16) plus a HF cache plus checkpoints. A 4B
# full-parameter checkpoint is ~16GB on disk per save, and --save-total-limit
# defaults to 3 in grpo_train, so budget accordingly.
df -h "$HOME" | tail -1

echo "== tmux =="
# Every command in the runbook past this point runs for hours. A dropped SSH
# connection kills a foreground process and takes the run with it, so the
# runbook assumes tmux; install it here rather than discovering it is missing
# three hours in.
command -v tmux >/dev/null 2>&1 || sudo apt-get install -y -qq tmux
tmux -V

echo "== uv =="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi
uv --version

echo "== Cloning tau-forge (ref: $GIT_REF) =="
if [ -d "$REPO_DIR" ]; then
    git -C "$REPO_DIR" fetch origin "$GIT_REF"
    git -C "$REPO_DIR" checkout "$GIT_REF"
    git -C "$REPO_DIR" pull origin "$GIT_REF"
else
    git clone --branch "$GIT_REF" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "== tau2-bench submodule =="
git submodule update --init --recursive

echo "== tau2-bench extras =="
(cd third_party/tau2-bench && uv sync --extra gym)

echo "== tau-forge, with the GPU training extra =="
uv sync --extra train

echo "== Secrets scaffold =="
# tau2 calls dotenv's load_dotenv() with no path, which searches upward from the
# working directory -- so a .env at the repo root is found when running from
# here. It is gitignored; the example file is not.
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example -- fill in OPENAI_API_KEY before Phase 8 eval."
else
    echo ".env already present, left alone."
fi

echo "== Sanity checks =="
uv run python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU visible')"
# The whole torch-free half of the pipeline, against all 541 real scenarios.
# If this fails, nothing downstream is worth starting.
uv run pytest -q

echo "== Pre-warming the model cache =="
# Downloading 8GB inside the first timed run makes that run's wall-clock
# meaningless and stalls it behind the network. Do it now, explicitly.
uv run python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3-4B-Instruct-2507')
print('cached at', p)
"

cat <<EOF

== Bootstrap done ==
Repo:   $REPO_DIR   (branch: $GIT_REF)
GPUs:   $GPU_COUNT

Nothing has been trained or evaluated yet. Follow docs/runbook.md from
"Step 3" -- it has the exact commands in order, with the single-GPU and
multi-GPU paths marked.

Start a tmux session first; every step past here runs for hours:
    tmux new -s tauforge
EOF
