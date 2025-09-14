import re
from pathlib import Path

import torch
import pandas as pd

from ml.Seq2seq import Encoder, Decoder, Seq2Seq


# ==============================
# Configuration
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 50

DATA_PATH = Path("./data/preprocessed_data/processed_data.pt")
GRIDSEARCH_FILE = Path("backend/data/reporting/gridsearch_res/gridsearch_results.csv")
MODEL_DIR = Path("backend/data/reporting/model")
MODEL_NAME = "seq2seq_lstm_v28.pth"
MODEL_PATH = MODEL_DIR / MODEL_NAME


# ==============================
# Chargement des données
# ==============================
data = torch.load(DATA_PATH)

train_pairs = data["train"]
val_pairs = data["val"]

en_word2id = data["en_word2id"]
fr_word2id = data["fr_word2id"]
en_id2word = data["en_id2word"]
fr_id2word = data["fr_id2word"]

INPUT_DIM = len(en_word2id)
OUTPUT_DIM = len(fr_word2id)


# ==============================
# Chargement des hyperparamètres
# ==============================
df = pd.read_csv(GRIDSEARCH_FILE)
params = df[df["model_name"] == MODEL_NAME].iloc[0]

EMB_SIZE = int(params["EMB_SIZE"])
HID_SIZE = int(params["HID_SIZE"])

print(f"EMB_SIZE={EMB_SIZE}, HID_SIZE={HID_SIZE}")


# ==============================
# Construction du modèle
# ==============================
encoder = Encoder(INPUT_DIM, EMB_SIZE, HID_SIZE).to(DEVICE)
decoder = Decoder(OUTPUT_DIM, EMB_SIZE, HID_SIZE * 2, HID_SIZE).to(DEVICE)

model = Seq2Seq(encoder, decoder).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ==============================
# Fonction de traduction
# ==============================
def translate_sentence(
    sentence,
    model,
    en_word2id,
    fr_word2id,
    fr_id2word,
    max_len: int = MAX_LEN,
    device: str = DEVICE,
):
    """Traduit une phrase en anglais vers le français avec un modèle Seq2Seq."""

    model.eval()

    # --- 1. Prétraitement ---
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    tokens = sentence.split()

    src_ids = [en_word2id.get(tok, en_word2id["<unk>"]) for tok in tokens]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_len = torch.tensor([len(src_ids)]).to(device)

    # --- 2. Encoder ---
    with torch.no_grad():
        encoder_outputs, (h, c) = model.encoder(src_tensor, src_len)
        mask = model.create_mask(src_tensor)

    # --- 3. Décodage ---
    tgt_ids = [fr_word2id["<sos>"]]
    for _ in range(max_len):
        input_tok = torch.tensor([tgt_ids[-1]], dtype=torch.long).to(device)

        with torch.no_grad():
            logits, h, c = model.decoder.forward_step(
                input_tok, h, c, encoder_outputs, mask
            )

        pred_token = logits.argmax(1).item()
        if pred_token == fr_word2id["<eos>"]:
            break
        tgt_ids.append(pred_token)

    # --- 4. Conversion en texte ---
    translation = [fr_id2word[i] for i in tgt_ids[1:]]  # enlever <sos>
    return " ".join(translation)


# ==============================
# Exemple d’utilisation
# ==============================
# sentence = "I love cats"
# print(translate_sentence(sentence, model, en_word2id, fr_word2id, fr_id2word))
