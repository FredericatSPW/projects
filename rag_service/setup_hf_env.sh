#!/bin/bash

# Chemin du modèle sur l'EC2
HF_MODEL_PATH="/home/ubuntu/hf_models/intfloat-multilingual-e5-base"

# Vérifier si le dossier existe
if [ ! -d "$HF_MODEL_PATH" ]; then
    echo "❌ Le dossier $HF_MODEL_PATH n'existe pas. Vérifie la copie du modèle."
    exit 1
fi

# Ajouter la variable d'environnement dans ~/.bashrc
echo "export LOCAL_HF_EMB_PATH=\"$HF_MODEL_PATH\"" >> ~/.bashrc
echo "export EMB_DEVICE=\"cpu\"" >> ~/.bashrc

# Recharger ~/.bashrc
source ~/.bashrc

echo "✅ Variables d'environnement configurées :"
echo "LOCAL_HF_EMB_PATH=$LOCAL_HF_EMB_PATH"
echo "EMB_DEVICE=$EMB_DEVICE"
