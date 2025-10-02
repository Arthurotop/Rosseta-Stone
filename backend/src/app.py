import os
import json
import torch
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from loguru import logger
from huggingface_hub import InferenceClient
import requests
import zipfile

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
# Télécharger modèles depuis Google Drive si manquants
# ==============================
MODEL_DIR = "backend/data/reporting/model"
MODEL_ZIP_ID = "1xtCwYVjxkaKplQoWXCNjd84NL5ms8kHd"
MODEL_ZIP_PATH = "models.zip"
model_files = ["seq2seq_en-fr_v8.pth", "seq2seq_fr-en_v8.pth"]

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

if not all(os.path.exists(f"{MODEL_DIR}/{mf}") for mf in model_files):
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Téléchargement des modèles depuis Google Drive...")
    download_file_from_google_drive(MODEL_ZIP_ID, MODEL_ZIP_PATH)
    logger.info("Extraction du ZIP des modèles...")
    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    os.remove(MODEL_ZIP_PATH)

# ==============================
# Chargement données & modèles
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
    torch.load(f"{MODEL_DIR}/seq2seq_en-fr_v8.pth", map_location=DEVICE)
)
model_en_fr.eval()

# Modèles FR→EN
encoder_fr = Encoder(DIM_FR, EMB_SIZE, HID_SIZE).to(DEVICE)
decoder_fr = Decoder(DIM_EN, EMB_SIZE, HID_SIZE * 2, HID_SIZE).to(DEVICE)
model_fr_en = Seq2Seq(encoder_fr, decoder_fr).to(DEVICE)
model_fr_en.load_state_dict(
    torch.load(f"{MODEL_DIR}/seq2seq_fr-en_v8.pth", map_location=DEVICE)
)
model_fr_en.eval()

logger.info("Modèles locaux chargés avec succès")

# ==============================
# Hugging Face Client
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("Aucun token Hugging Face trouvé dans les variables d'environnement")

client = InferenceClient("Helsinki-NLP/opus-mt-en-fr", token=HF_TOKEN)

# ==============================
# API Endpoint
# ==============================
@app.route("/translate", methods=["POST"])
def translate():
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

        # EN → FR
        if from_lang == "en" and to_lang == "fr":
            if use_api_(sentence, en_word2id):
                if not HF_TOKEN:
                    return jsonify({"error": "API Hugging Face indisponible (token manquant)"}), 503
                response = client.post(json={"inputs": sentence})
                data_hf = json.loads(response)
                translation = data_hf[0]["translation_text"]
                source = "huggingface"
            else:
                translation = translate_sentence(
                    sentence, model_en_fr, en_word2id, fr_word2id, fr_id2word, device=DEVICE
                )
                source = "local"

        # FR → EN
        elif from_lang == "fr" and to_lang == "en":
            if use_api_(sentence, fr_word2id):
                if not HF_TOKEN:
                    return jsonify({"error": "API Hugging Face indisponible (token manquant)"}), 503
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

        return jsonify({"translation_text": translation, "model_source": source})

    except Exception as e:
        logger.error(f"Erreur pendant la traduction: {e}")
        return jsonify({"error": "Erreur interne serveur"}), 500

# ==============================
# Lancement serveur
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
