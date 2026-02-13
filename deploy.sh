#!/usr/bin/env bash
set -euo pipefail

REGION="europe-west1"
PROJECT="aicamp-13-02-2026"
REPO="containers"
SERVICE="pet-app"

if [ $# -lt 1 ]; then
  echo "Usage: ./deploy.sh <version-tag>"
  echo "Example: ./deploy.sh 2"
  exit 1
fi

TAG="$1"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:${TAG}"

echo "Building: $IMAGE"
gcloud builds submit --tag "$IMAGE"

echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated

echo "Done."
