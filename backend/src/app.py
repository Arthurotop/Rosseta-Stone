import os
import json
import torch
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from loguru import logger
from huggingface_hub import InferenceClient

from ml.Seq2seq import Encoder, Decoder, Seq2Seq
from ml.traduction import translate_sentence
from ml.utils import use_api_

# ==============================
# Configuration Flask
# ==============================
app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
# Chargement données & modèle
# ==============================
data = torch.load("./data/preprocessed_data/processed_data.pt")

en_word2id = data["en_word2id"]
fr_word2id = data["fr_word2id"]
en_id2word = data["en_id2word"]
fr_id2word = data["fr_id2word"]

INPUT_DIM = len(en_word2id)
OUTPUT_DIM = len(fr_word2id)

df = pd.read_csv("backend/data/reporting/gridsearch_res/gridsearch_results.csv")
model_parameters = df[df["model_name"] == "seq2seq_lstm_v28.pth"]

EMB_SIZE = int(model_parameters["EMB_SIZE"])
HID_SIZE = int(model_parameters["HID_SIZE"])

encoder = Encoder(INPUT_DIM, EMB_SIZE, HID_SIZE).to(DEVICE)
decoder = Decoder(OUTPUT_DIM, EMB_SIZE, HID_SIZE * 2, HID_SIZE).to(DEVICE)
model = Seq2Seq(encoder, decoder).to(DEVICE)

model.load_state_dict(
    torch.load("backend/data/reporting/model/seq2seq_lstm_v28.pth", map_location=DEVICE)
)
model.eval()

logger.info("✅ Modèle local chargé avec succès")


# ==============================
# Hugging Face Client
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN")  # ⚠️ Définir via export HF_TOKEN="xxx"
if not HF_TOKEN:
    logger.warning("⚠️ Aucun token Hugging Face trouvé dans les variables d'environnement")

client = InferenceClient("Helsinki-NLP/opus-mt-en-fr", token=HF_TOKEN)


# ==============================
# API Endpoint
# ==============================
@app.route("/translate", methods=["POST"])
def translate():
    """Traduit une phrase EN → FR avec modèle local ou Hugging Face API."""
    data_req = request.get_json()

    if not data_req:
        return jsonify({"error": "Requête invalide, JSON manquant"}), 400

    sentence = data_req.get("text", "").strip()
    if not sentence:
        return jsonify({"error": "Texte vide"}), 400

    from_lang = data_req.get("from")
    to_lang = data_req.get("to")

    # ==============================
    # Traduction
    # ==============================
    try:
        if from_lang == "en" and to_lang == "fr":
            if use_api_(sentence, en_word2id):
                # Hugging Face API
                if not HF_TOKEN:
                    return jsonify({"error": "API Hugging Face indisponible (token manquant)"}), 503

                response = client.post(json={"inputs": sentence})
                data = json.loads(response)
                translation = data[0]["translation_text"]
            else:
                # Modèle local
                translation = translate_sentence(
                    sentence, model, en_word2id, fr_word2id, fr_id2word, device=DEVICE
                )
        else:
            # Fallback : toujours modèle local
            translation = translate_sentence(
                sentence, model, en_word2id, fr_word2id, fr_id2word, device=DEVICE
            )

        return jsonify({"translation_text": translation})

    except Exception as e:
        logger.error(f"Erreur pendant la traduction: {e}")
        return jsonify({"error": "Erreur interne serveur"}), 500


# ==============================
# Lancement serveur
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
