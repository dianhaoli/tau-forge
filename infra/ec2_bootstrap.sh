#!/usr/bin/env bash
# Phase 7 GPU box bootstrap. Run this on a fresh AWS Deep Learning AMI
# (Ubuntu, GPU PyTorch variant) after SSH'ing in — it does not launch or
# configure the instance itself; see docs/phase7_aws_setup.md for that.
#
# Usage: ./ec2_bootstrap.sh [git-ref]
#   git-ref defaults to "main".
set -euo pipefail

GIT_REF="${1:-main}"
REPO_URL="https://github.com/dianhaoli/tau-forge.git"
REPO_DIR="$HOME/tau-forge"

echo "== Checking for NVIDIA driver / CUDA =="
nvidia-smi

echo "== Installing uv =="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
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

echo "== Initializing tau2-bench submodule =="
git submodule update --init --recursive

echo "== Installing tau2-bench extras =="
(cd third_party/tau2-bench && uv sync --extra gym)

echo "== Installing tau-forge, including the GPU training dependency group =="
uv sync --extra train

echo "== Sanity check: torch sees the GPU =="
uv run python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU visible')"

cat <<'EOF'

== Bootstrap done ==
Repo:        ~/tau-forge
Training deps: installed via `uv sync --extra train`

Nothing has been trained yet. This box is ready for Phase 7's training script
once that's written and explicitly greenlit — see docs/phase7_aws_setup.md.

Remember to stop/terminate this instance when you're done for the session.
EOF
