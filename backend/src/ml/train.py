import time
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import TranslationDataset, collate_fn
from Seq2seq import Encoder, Decoder, Seq2Seq
from utils import train_epoch, test_epoch


# ==============================
# Configuration
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 64

DATA_PATH = Path("./data/preprocessed_data/processed_data.pt")
REPORTING_DIR = Path("backend/data/reporting")

TRAIN_FILE = REPORTING_DIR / "gridsearch_results_train.csv"
VAL_FILE = REPORTING_DIR / "gridsearch_results_val.csv"

# Grille d’hyperparamètres
PARAM_GRID = {
    "EMB_SIZE": [128, 256],
    "HID_SIZE": [64, 128],
    "learning_rate": [1e-3, 5e-4],
    "teacher_forcing_ratio": [1],
}


# ==============================
# Chargement des données
# ==============================
data = torch.load(DATA_PATH)

train_pairs, val_pairs = data["train"], data["val"]
fr_word2id, fr_id2word = data["fr_word2id"], data["fr_id2word"]
en_word2id = data["en_word2id"]

train_loader = DataLoader(
    TranslationDataset(train_pairs),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=True,
)
val_loader = DataLoader(
    TranslationDataset(val_pairs),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)

INPUT_DIM = len(en_word2id)
OUTPUT_DIM = len(fr_word2id)


# ==============================
# Grid Search - Entraînement + Validation
# ==============================
train_results = []
val_results = []

combinations = list(itertools.product(*PARAM_GRID.values()))

for i, (emb_size, hid_size, lr, tf_ratio) in enumerate(combinations, 1):
    print(
        f"\n🔹 Training model {i}/{len(combinations)} "
        f"(EMB={emb_size}, HID={hid_size}, lr={lr}, tf_ratio={tf_ratio})"
    )

    # Création du modèle
    encoder = Encoder(INPUT_DIM, emb_size, hid_size).to(DEVICE)
    decoder = Decoder(OUTPUT_DIM, emb_size, hid_size * 2, hid_size).to(DEVICE)
    model = Seq2Seq(encoder, decoder).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Tracking
    train_losses, train_accs, train_bleus = [], [], []
    val_losses, val_accs, val_bleus = [], [], []

    # Début du timing
    start_time = time.time()

    for epoch in trange(EPOCHS, desc="Training Epoch"):
        # --- Training ---
        tr_loss, tr_acc, tr_bleu = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            tf_ratio,
            fr_id2word,
        )

        # --- Validation ---
        val_loss, val_acc, val_bleu = test_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            tf_ratio,
            fr_id2word,
        )

        # Stockage
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        train_bleus.append(tr_bleu)

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_bleus.append(val_bleu)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train: loss={tr_loss:.4f}, acc={tr_acc:.4f}, BLEU={tr_bleu:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, BLEU={val_bleu:.4f}"
        )

    training_time = time.time() - start_time
    print(f"⏱ Training time: {training_time:.2f} sec")

    # Sauvegarde du modèle
    model_name = f"seq2seq_lstm_v{i}.pth"
    model_path = REPORTING_DIR / model_name
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved at: {model_path}")

    # Résultats d'entraînement
    train_results.append({
        "model_name": model_name,
        "losses": train_losses,
        "accuracy": train_accs,
        "BLEU": train_bleus,
        "EMB_SIZE": emb_size,
        "HID_SIZE": hid_size,
        "learning_rate": lr,
        "teacher_forcing_ratio": tf_ratio,
        "training_time_sec": training_time,
    })

    # Résultats de validation
    val_results.append({
        "model_name": model_name,
        "losses": val_losses,
        "accuracy": val_accs,
        "BLEU": val_bleus,
        "EMB_SIZE": emb_size,
        "HID_SIZE": hid_size,
        "learning_rate": lr,
        "teacher_forcing_ratio": tf_ratio,
        "training_time_sec": training_time,
    })


# ==============================
# Sauvegarde des résultats
# ==============================
pd.DataFrame(train_results).to_csv(TRAIN_FILE, index=False)
pd.DataFrame(val_results).to_csv(VAL_FILE, index=False)

print(f"Résultats entraînement sauvegardés dans {TRAIN_FILE}")
print(f"Résultats validation sauvegardés dans {VAL_FILE}")
