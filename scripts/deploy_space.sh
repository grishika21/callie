#!/usr/bin/env bash
# Upload the phone-line demo to a Hugging Face Docker Space.
#
#   scripts/deploy_space.sh <hf-username>/callie
#
# Needs `hf auth login` first. Copies only what the Space needs (files git tracks or would
# track: never .env, the users file, the audit log or caches) into a clean folder, puts
# space/Dockerfile and space/README.md at its root, creates the Space if needed (Docker,
# public) and uploads it. Secrets are set on the Space's settings page, not here.
set -euo pipefail
REPO="${1:?usage: $0 <hf-username>/<space-name>}"
cd "$(dirname "$0")/.."
HF=".venv/bin/hf"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

git ls-files -z --cached --others --exclude-standard -- \
    talktable callie data mcp_server.py requirements.txt requirements-callie.txt space LICENSE \
  | while IFS= read -r -d '' path; do
      [[ -f "$path" ]] && mkdir -p "$STAGE/$(dirname "$path")" && cp "$path" "$STAGE/$path"
    done
cp space/Dockerfile "$STAGE/Dockerfile"
cp space/README.md "$STAGE/README.md"
if grep -rqE "gsk_[A-Za-z0-9]{10,}|hf_[A-Za-z0-9]{20,}" "$STAGE"; then
  echo "refusing to upload: something that looks like an API key is in the files" >&2; exit 1
fi
if [[ -n "${STAGE_ONLY:-}" ]]; then  # build the same files locally instead: STAGE_ONLY=dir scripts/deploy_space.sh x/y
  rm -rf "$STAGE_ONLY" && cp -r "$STAGE" "$STAGE_ONLY" && echo "staged $(find "$STAGE_ONLY" -type f | wc -l) files in $STAGE_ONLY"
  exit 0
fi
echo "uploading $(find "$STAGE" -type f | wc -l) files to spaces/$REPO"
"$HF" repo create "$REPO" --repo-type space --space-sdk docker --exist-ok >/dev/null
"$HF" upload "$REPO" "$STAGE" . --repo-type space --commit-message "Deploy the Callie phone demo"
echo "https://huggingface.co/spaces/$REPO  ->  https://$(echo "$REPO" | tr '/' '-' | tr '[:upper:]' '[:lower:]').hf.space"
