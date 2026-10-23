#!/usr/bin/env bash
# Deploy to Cloud Run with Gemini + embeddings served by Vertex AI.
# Usage: PROJECT_ID=my-project ./deploy/deploy.sh [region]
set -euo pipefail
PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${1:-europe-west4}"
SERVICE=agentic-rag
SA="${SERVICE}-sa@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com aiplatform.googleapis.com \
  cloudbuild.googleapis.com artifactregistry.googleapis.com

# Dedicated service account with only the Vertex AI role: no API keys in the image.
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${SERVICE}-sa" --display-name "Agentic RAG runtime"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SA}" --role roles/aiplatform.user --condition=None >/dev/null

# Source deploy: Cloud Build uses the Dockerfile; bundled docs are embedded with Vertex AI on cold start.
gcloud run deploy "$SERVICE" --source . --region "$REGION" \
  --service-account "$SA" --allow-unauthenticated \
  --set-env-vars "LLM_PROVIDER=vertex,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}" \
  --memory 1Gi --min-instances 0 --max-instances 3
