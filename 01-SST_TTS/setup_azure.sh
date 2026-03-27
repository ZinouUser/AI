#!/usr/bin/env bash
set -e

# ─── Config ───────────────────────────────────
RESOURCE_GROUP="rg-robot-ai"
LOCATION="qatarcentral"
SPEECH_NAME="speech-robot-amal"
SKU="F0"
# ──────────────────────────────────────────────

echo "🔐 Connexion Azure..."
az login --use-device-code

echo "📦 Création du groupe de ressources..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

echo "🎙️  Création du service Speech (${SKU})..."
az cognitiveservices account create \
    --name "$SPEECH_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --kind "SpeechServices" \
    --sku "$SKU" \
    --location "$LOCATION" \
    --yes \
    --output none

echo "🔑 Récupération des clés..."
SPEECH_KEY=$(az cognitiveservices account keys list \
    --name "$SPEECH_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "key1" --output tsv)

SPEECH_REGION=$(az cognitiveservices account show \
    --name "$SPEECH_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "location" --output tsv)

echo "📝 Génération du fichier .env..."
cat > "$(dirname "$0")/.env" << EOF
AZURE_SPEECH_KEY=${SPEECH_KEY}
AZURE_SPEECH_REGION=${SPEECH_REGION}
EOF

echo ""
echo "✅ Provisionnement terminé !"
echo "   Région : ${SPEECH_REGION}"
echo "   Clé    : ${SPEECH_KEY:0:6}********************"
echo ""
echo "👉 Lance maintenant : python voice_agent.py"

