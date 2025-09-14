import os
import time
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import TranslationDataset, collate_fn
from Seq2seq import Encoder, Decoder, Seq2Seq
from utils import compute_accuracy, compute_bleu, train_epoch, test_epoch


# ==============================
# Configuration
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 64

DATA_DIR = Path("./data/preprocessed_data")
REPORTING_DIR = Path("backend/data/reporting")
MODEL_DIR = REPORTING_DIR / "model"

GRIDSEARCH_FILE = REPORTING_DIR / "gridsearch_res" / "gridsearch_results.csv"
OUTPUT_FILE = REPORTING_DIR / "gridsearch_results_val.csv"


# ==============================
# Chargement des données
# ==============================
data = torch.load(DATA_DIR / "processed_data.pt")

test_pairs = data["test"]
fr_word2id = data["fr_word2id"]
fr_id2word = data["fr_id2word"]
en_word2id = data["en_word2id"]

test_loader = DataLoader(
    TranslationDataset(test_pairs),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)

INPUT_DIM = len(en_word2id)
OUTPUT_DIM = len(fr_word2id)


# ==============================
# Chargement du CSV Gridsearch
# ==============================
df = pd.read_csv(GRIDSEARCH_FILE)


# ==============================
# Évaluation des modèles
# ==============================
results = []

for _, row in tqdm(df.iterrows(), desc="Models"):
    model_name = row["model_name"]
    model_path = MODEL_DIR / model_name

    # Construction du modèle
    encoder = Encoder(INPUT_DIM, row["EMB_SIZE"], row["HID_SIZE"]).to(DEVICE)
    decoder = Decoder(
        OUTPUT_DIM,
        row["EMB_SIZE"],
        row["HID_SIZE"] * 2,
        row["HID_SIZE"],
    ).to(DEVICE)

    model = Seq2Seq(encoder, decoder).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr=row["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Tracking des résultats
    epoch_losses, epoch_accs, epoch_bleus = [], [], []

    # Début du timing
    start_time = time.time()

    for epoch in trange(EPOCHS, desc="Testing Epoch"):
        loss, acc, bleu = test_epoch(
            model,
            test_loader,
            optimizer,
            criterion,
            row["teacher_forcing_ratio"],
            fr_id2word,
        )
        epoch_losses.append(loss)
        epoch_accs.append(acc)
        epoch_bleus.append(bleu)

    test_time = time.time() - start_time

    # Ajout des résultats
    results.append({
        "model_name": model_name,
        "losses": epoch_losses,
        "accuracy": epoch_accs,
        "BLEU": epoch_bleus,
        "EMB_SIZE": row["EMB_SIZE"],
        "HID_SIZE": row["HID_SIZE"],
        "learning_rate": row["learning_rate"],
        "teacher_forcing_ratio": row["teacher_forcing_ratio"],
        "training_time_sec": test_time,
    })


# ==============================
# Sauvegarde des résultats
# ==============================
df_test = pd.DataFrame(results)
df_test.to_csv(OUTPUT_FILE, index=False)

print(f"Résultats sauvegardés dans {OUTPUT_FILE}")
