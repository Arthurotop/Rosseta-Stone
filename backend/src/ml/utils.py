import re
import torch
import torch.nn as nn
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Télécharger les ressources nécessaires
nltk.download("punkt")

# ==============================
# Configuration
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
# Métriques
# ==============================
def compute_bleu(output, target, id2word):
    """
    Calcule le score BLEU moyen d'un batch.

    Args:
        output (Tensor): [batch_size, seq_len, vocab_size] (logits du modèle)
        target (Tensor): [batch_size, seq_len] (indices cibles)
        id2word (dict): mapping {id: mot}

    Returns:
        float: BLEU moyen du batch
    """
    preds = output.argmax(dim=-1).cpu().tolist()
    refs = target.cpu().tolist()

    smoothie = SmoothingFunction().method4
    bleu_scores = []

    for pred_seq, ref_seq in zip(preds, refs):
        # Supprimer padding (0), <sos>=1 et <eos>=2
        pred_tokens = [id2word[idx] for idx in pred_seq if idx not in [0, 1, 2]]
        ref_tokens = [id2word[idx] for idx in ref_seq if idx not in [0, 1, 2]]

        if pred_tokens and ref_tokens:
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu)

    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


def compute_accuracy(output, target):
    """
    Calcule l'accuracy en ignorant le padding.

    Args:
        output (Tensor): [batch_size, seq_len, vocab_size]
        target (Tensor): [batch_size, seq_len]

    Returns:
        float: accuracy du batch
    """
    preds = output.argmax(dim=-1)  # [batch_size, seq_len]
    mask = target != 0  # ignorer padding

    correct = (preds == target) & mask
    return correct.sum().item() / mask.sum().item()


# ==============================
# Boucles d'entraînement / test
# ==============================
def run_epoch(model, loader, optimizer, criterion, teacher_forcing_ratio, id2word, training=True):
    """
    Exécute une époque (train ou validation).

    Args:
        model (nn.Module): modèle Seq2Seq
        loader (DataLoader): batches source/target
        optimizer (torch.optim.Optimizer): optimiseur
        criterion (nn.Module): fonction de perte
        teacher_forcing_ratio (float): taux de teacher forcing
        id2word (dict): mapping {id: mot}
        training (bool): True = entraînement, False = validation

    Returns:
        tuple: (loss, accuracy, bleu) moyens sur l'époque
    """
    model.train() if training else model.eval()
    total_loss, total_acc, total_bleu = 0, 0, 0

    for src, src_lens, tgt, tgt_lens in tqdm(loader, desc="Batch", leave=False):
        src, tgt, src_lens = src.to(DEVICE), tgt.to(DEVICE), src_lens.to(DEVICE)

        if training:
            optimizer.zero_grad()

        output = model(src, src_lens, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        output_dim = output.shape[-1]

        loss = criterion(
            output[:, 1:, :].reshape(-1, output_dim),
            tgt[:, 1:].reshape(-1),
        )

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(output[:, 1:, :], tgt[:, 1:])
        total_bleu += compute_bleu(output[:, 1:, :], tgt[:, 1:], id2word)

    n_batches = len(loader)
    return (
        total_loss / n_batches,
        total_acc / n_batches,
        total_bleu / n_batches,
    )


def train_epoch(model, loader, optimizer, criterion, teacher_forcing_ratio, id2word):
    """Entraînement (1 époque)."""
    return run_epoch(model, loader, optimizer, criterion, teacher_forcing_ratio, id2word, training=True)


def test_epoch(model, loader, optimizer, criterion, teacher_forcing_ratio, id2word):
    """Validation (1 époque)."""
    return run_epoch(model, loader, optimizer, criterion, teacher_forcing_ratio, id2word, training=False)


# ==============================
# Utilitaire
# ==============================
def use_api_(sentence, vocab):
    """
    Détecte si une phrase contient un mot absent du vocabulaire.

    Args:
        sentence (str): phrase à vérifier
        vocab (set ou dict): vocabulaire connu

    Returns:
        bool: True si un mot n'est pas dans le vocab
    """
    tokens = re.findall(r"\w+", sentence)
    return any(token not in vocab for token in tokens)
