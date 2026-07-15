#!/usr/bin/env bash
# Build, push, and roll out GPT-SoVITS RunPod image.
#
# Usage:
#   ./scripts/build_and_push.sh                  # auto-increment vN (queries ECR)
#   ./scripts/build_and_push.sh v6               # explicit tag
#   ./scripts/build_and_push.sh --no-push        # build only
#   ./scripts/build_and_push.sh --no-runpod      # build + push, skip template update
#
# Required env (or ~/.aws/credentials, RunPod console):
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY      (or FORGE_AWS_* alternatives)
#   RUNPOD_API_KEY                                (only if RunPod update enabled)

set -euo pipefail

ECR_REPO="public.ecr.aws/r2p3x7v0/forge-gpt-sovits"
RUNPOD_TEMPLATES=(
  "jhmmb6yc60:inference"
  "hona1gacbj:training"
)

PUSH=1
UPDATE_RUNPOD=1
TAG=""
for arg in "$@"; do
  case "$arg" in
    --no-push)   PUSH=0; UPDATE_RUNPOD=0 ;;
    --no-runpod) UPDATE_RUNPOD=0 ;;
    *)           TAG="$arg" ;;
  esac
done

: "${AWS_ACCESS_KEY_ID:=${FORGE_AWS_ACCESS_KEY_ID:-}}"
: "${AWS_SECRET_ACCESS_KEY:=${FORGE_AWS_SECRET_ACCESS_KEY:-}}"
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
  echo "ERROR: AWS credentials missing." >&2
  echo "  export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=..." >&2
  echo "  (or source envs/<env>/voice_service.env which sets FORGE_AWS_*)" >&2
  exit 1
fi
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

# Auto-increment vN tag by inspecting existing ECR tags.
if [ -z "$TAG" ]; then
  echo ">> Querying ECR for latest vN tag..."
  last=$(aws ecr-public describe-images \
      --repository-name forge-gpt-sovits \
      --region us-east-1 \
      --query 'imageDetails[].imageTags[]' \
      --output text 2>/dev/null \
    | tr '\t' '\n' \
    | grep -E '^v[0-9]{1,4}$' \
    | sed 's/^v//' \
    | sort -n | tail -1 || true)
  next=$((${last:-0} + 1))
  TAG="v${next}"
  echo ">> Auto-selected tag: $TAG (previous: ${last:+v$last}${last:-none})"
fi
IMAGE="${ECR_REPO}:${TAG}"

if command -v podman >/dev/null 2>&1; then
  CONTAINER_TOOL=podman
elif command -v docker >/dev/null 2>&1; then
  CONTAINER_TOOL=docker
else
  echo "ERROR: neither podman nor docker found in PATH." >&2
  exit 1
fi

cd "$(dirname "$0")/.."

echo ">> Building $IMAGE with $CONTAINER_TOOL"
"$CONTAINER_TOOL" build --progress=plain -f Dockerfile.runpod -t "$IMAGE" .

if [ "$PUSH" -eq 0 ]; then
  echo ">> Skipping push (--no-push). Local image: $IMAGE"
  exit 0
fi

# Isolate docker config in a temp dir to bypass wincred / desktop credential
# helpers that often choke on the long ECR token (esp. on Docker Desktop +
# WSL). The ephemeral config has no credsStore, so login writes auth to a
# plain JSON file and push reads from it.
DOCKER_CONFIG=$(mktemp -d -t docker-cfg-XXXXXX)
export DOCKER_CONFIG
trap 'rm -rf "$DOCKER_CONFIG"' EXIT
echo '{}' > "$DOCKER_CONFIG/config.json"

echo ">> Logging in to public ECR (config: $DOCKER_CONFIG)"
aws ecr-public get-login-password --region us-east-1 | \
  "$CONTAINER_TOOL" --config "$DOCKER_CONFIG" login \
    --username AWS --password-stdin public.ecr.aws

echo ">> Pushing $IMAGE"
"$CONTAINER_TOOL" --config "$DOCKER_CONFIG" push "$IMAGE"

echo ""
echo "✓ Pushed: $IMAGE"

if [ "$UPDATE_RUNPOD" -eq 0 ]; then
  echo ">> Skipping RunPod template update (--no-runpod)."
  exit 0
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
  echo "WARN: RUNPOD_API_KEY not set, skipping RunPod template update." >&2
  echo "      Update template imageName manually:" >&2
  for entry in "${RUNPOD_TEMPLATES[@]}"; do
    echo "        ${entry##*:}: ${entry%%:*} → $IMAGE" >&2
  done
  exit 0
fi

echo ""
echo ">> Updating RunPod templates"
for entry in "${RUNPOD_TEMPLATES[@]}"; do
  tid="${entry%%:*}"
  role="${entry##*:}"
  resp=$(curl -fsS -X PATCH \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"imageName\": \"${IMAGE}\"}" \
    "https://rest.runpod.io/v1/templates/${tid}" 2>&1) || {
      echo "  ✗ ${role} (${tid}): update failed" >&2
      echo "$resp" >&2
      exit 1
    }
  echo "  ✓ ${role} (${tid}) → $IMAGE"
done

echo ""
echo "✓ Done. Trigger worker recycle on RunPod console (or wait for idle timeout)."
