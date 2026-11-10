#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
abc_root="${ABC_ROOT:-${repo_dir}/third_party/abc}"
revision=d0832d12651d1b260a652861a14648dc5f3660c7
export TMPDIR="${TMPDIR:-/data-vol2/jaehyeong/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR}/uv-cache}"
export ABC_CACHE="${ABC_CACHE:-${TMPDIR}/abc-cache}"
mkdir -p "${TMPDIR}" "${UV_CACHE_DIR}" "${ABC_CACHE}"
if [[ ! -d "${abc_root}" ]]; then
  git clone https://github.com/amazon-far/abc.git "${abc_root}"
fi
if [[ -n "$(git -C "${abc_root}" status --porcelain --untracked-files=no)" ]]; then
  echo "ABC checkout has tracked changes; refusing to change its revision." >&2
  exit 1
fi
git -C "${abc_root}" fetch origin "${revision}"
git -C "${abc_root}" checkout --detach "${revision}"
cd "${abc_root}"
export UV_PROJECT_ENVIRONMENT="${abc_root}/.venv"
uv sync --python 3.12
uv run --no-sync python prepare.py --sim
