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

DIM_EN = len(en_word2id)
DIM_FR = len(fr_word2id)

df = pd.read_csv("backend/data/reporting/gridsearch_res/gridsearch_en-fr_train.csv")
model_parameters = df[df["model_name"] == "seq2seq_en-fr_v8.pth"]

EMB_SIZE = int(model_parameters["EMB_SIZE"])
HID_SIZE = int(model_parameters["HID_SIZE"])

# Modèles EN→FR
encoder_en = Encoder(DIM_EN, EMB_SIZE, HID_SIZE).to(DEVICE)
decoder_en = Decoder(DIM_FR, EMB_SIZE, HID_SIZE * 2, HID_SIZE).to(DEVICE)
model_en_fr = Seq2Seq(encoder_en, decoder_en).to(DEVICE)
model_en_fr.load_state_dict(
    torch.load("backend/data/reporting/model/seq2seq_en-fr_v8.pth", map_location=DEVICE)
)
model_en_fr.eval()

# Modèles FR→EN
encoder_fr = Encoder(DIM_FR, EMB_SIZE, HID_SIZE).to(DEVICE)
decoder_fr = Decoder(DIM_EN, EMB_SIZE, HID_SIZE * 2, HID_SIZE).to(DEVICE)
model_fr_en = Seq2Seq(encoder_fr, decoder_fr).to(DEVICE)
model_fr_en.load_state_dict(
    torch.load("backend/data/reporting/model/seq2seq_fr-en_v8.pth", map_location=DEVICE)
)
model_fr_en.eval()

logger.info("Modèles locaux chargés avec succès")

# ==============================
# Hugging Face Client
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN")  # ⚠️ Définir via export HF_TOKEN="xxx"
if not HF_TOKEN:
    logger.warning("Aucun token Hugging Face trouvé dans les variables d'environnement")

client = InferenceClient("Helsinki-NLP/opus-mt-en-fr", token=HF_TOKEN)

# ==============================
# API Endpoint
# ==============================
@app.route("/translate", methods=["POST"])
def translate():
    """Traduit une phrase EN ↔ FR avec modèle local ou Hugging Face API."""
    data_req = request.get_json()

    if not data_req:
        return jsonify({"error": "Requête invalide, JSON manquant"}), 400

    sentence = data_req.get("text", "").strip()
    if not sentence:
        return jsonify({"error": "Texte vide"}), 400

    from_lang = data_req.get("from")
    to_lang = data_req.get("to")

    try:
        translation = None
        source = None  

        # -----------------------------
        # EN → FR
        # -----------------------------
        if from_lang == "en" and to_lang == "fr":
            if use_api_(sentence, en_word2id):
                if not HF_TOKEN:
                    return jsonify({
                        "error": "API Hugging Face indisponible (token manquant)"
                    }), 503
                response = client.post(json={"inputs": sentence})
                data_hf = json.loads(response)
                translation = data_hf[0]["translation_text"]
                source = "huggingface"
            else:
                translation = translate_sentence(
                    sentence, model_en_fr, en_word2id, fr_word2id, fr_id2word, device=DEVICE
                )
                source = "local"

        # -----------------------------
        # FR → EN
        # -----------------------------
        elif from_lang == "fr" and to_lang == "en":
            if use_api_(sentence, fr_word2id):
                if not HF_TOKEN:
                    return jsonify({
                        "error": "API Hugging Face indisponible (token manquant)"
                    }), 503
                # Utiliser modèle Hugging Face FR→EN
                client_fr_en = InferenceClient("Helsinki-NLP/opus-mt-fr-en", token=HF_TOKEN)
                response = client_fr_en.post(json={"inputs": sentence})
                data_hf = json.loads(response)
                translation = data_hf[0]["translation_text"]
                source = "huggingface"
            else:
                translation = translate_sentence(
                    sentence, model_fr_en, fr_word2id, en_word2id, en_id2word, device=DEVICE
                )
                source = "local"

        else:
            return jsonify({"error": "Paires de langues non supportées"}), 400

        return jsonify({
            "translation_text": translation,
            "model_source": source
        })

    except Exception as e:
        logger.error(f"Erreur pendant la traduction: {e}")
        return jsonify({"error": "Erreur interne serveur"}), 500


# ==============================
# Lancement serveur
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
