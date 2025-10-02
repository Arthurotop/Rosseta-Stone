import time
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import TranslationDataset, collate_fn
from Seq2seq import Encoder, Decoder, Seq2Seq
from utils import test_epoch

# ==============================
# Configuration
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 64

DATA_DIR = Path("./data/preprocessed_data")
REPORTING_DIR = Path("backend/data/reporting/")
MODEL_DIR = REPORTING_DIR / "model"

# Directions à traiter
DIRECTIONS = ["en-fr", "fr-en"]

# Chargement des données
data = torch.load(DATA_DIR / "processed_data.pt")

for DIRECTION in DIRECTIONS:
    print(f"\n🚀 Évaluation des modèles pour la direction : {DIRECTION}")

    # Définir les CSV d'entraînement selon la direction
    TRAIN_CSV = REPORTING_DIR / "gridsearch_res" / f"gridsearch_{DIRECTION}_train.csv"
    df_train = pd.read_csv(TRAIN_CSV)

    # Préparer les paires de test et les vocabulaires
    test_pairs = data["test"]
    fr_word2id = data["fr_word2id"]
    fr_id2word = data["fr_id2word"]
    en_word2id = data["en_word2id"]
    en_id2word = data["en_id2word"]

    if DIRECTION == "en-fr":
        src_word2id = en_word2id
        tgt_word2id = fr_word2id
        tgt_id2word = fr_id2word
    else:  # fr-en
        src_word2id = fr_word2id
        tgt_word2id = en_word2id
        tgt_id2word = en_id2word
        test_pairs = [(fr, en) for (en, fr) in test_pairs]

    test_loader = DataLoader(
        TranslationDataset(test_pairs),
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
    )

    INPUT_DIM = len(src_word2id)
    OUTPUT_DIM = len(tgt_word2id)

    results = []

    # Évaluation des modèles listés dans le CSV d'entraînement
    for _, row in tqdm(df_train.iterrows(), desc=f"Models {DIRECTION}"):
        model_name = row["model_name"]
        model_path = MODEL_DIR / model_name

        # Construction du modèle
        encoder = Encoder(INPUT_DIM, int(row["EMB_SIZE"]), int(row["HID_SIZE"])).to(DEVICE)
        decoder = Decoder(
            OUTPUT_DIM,
            int(row["EMB_SIZE"]),
            int(row["HID_SIZE"]) * 2,
            int(row["HID_SIZE"]),
        ).to(DEVICE)

        model = Seq2Seq(encoder, decoder).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        optimizer = torch.optim.Adam(model.parameters(), lr=float(row["learning_rate"]))
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Tracking des résultats
        epoch_losses, epoch_accs, epoch_bleus, epoch_rouges = [], [], [], []

        start_time = time.time()
        for epoch in trange(EPOCHS, desc="Testing Epoch"):
            loss, acc, bleu, rouge = test_epoch(
                model,
                test_loader,
                optimizer,
                criterion,
                float(row["teacher_forcing_ratio"]),
                tgt_id2word,
            )
            epoch_losses.append(loss)
            epoch_accs.append(acc)
            epoch_bleus.append(bleu)
            epoch_rouges.append(rouge)

        test_time = time.time() - start_time

        results.append({
            "model_name": model_name,
            "losses": epoch_losses,
            "accuracy": epoch_accs,
            "BLEU": epoch_bleus,
            "ROUGE": epoch_rouges,
            "EMB_SIZE": int(row["EMB_SIZE"]),
            "HID_SIZE": int(row["HID_SIZE"]),
            "learning_rate": float(row["learning_rate"]),
            "teacher_forcing_ratio": float(row["teacher_forcing_ratio"]),
            "testing_time_sec": test_time,
        })

    # Sauvegarde dans un CSV spécifique à la direction
    OUTPUT_FILE = REPORTING_DIR / f"gridsearch_results_test_{DIRECTION}.csv"
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"\n Résultats sauvegardés dans {OUTPUT_FILE}")
