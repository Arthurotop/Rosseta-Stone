import os
import ast
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Configuration des chemins
# ==============================
script_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "reporting"))

# Dictionnaire des fichiers CSV
csv_files = {
    "fr-en": {
        "train": os.path.join(data_dir, "gridsearch_res", "gridsearch_fr-en_train.csv"),
        "val":   os.path.join(data_dir, "gridsearch_res", "gridsearch_fr-en_val.csv"),
        "test":  os.path.join(data_dir, "gridsearch_res", "gridsearch_fr-en_test.csv"),
    },
    "en-fr": {
        "train": os.path.join(data_dir, "gridsearch_res", "gridsearch_en-fr_train.csv"),
        "val":   os.path.join(data_dir, "gridsearch_res", "gridsearch_en-fr_val.csv"),
        "test":  os.path.join(data_dir, "gridsearch_res", "gridsearch_en-fr_test.csv"),
    }
}

# Lecture des CSV
dfs = {
    lang: {split: pd.read_csv(path) for split, path in files.items()}
    for lang, files in csv_files.items()
}

# Dossier de sauvegarde pour les plots
plot_dir = os.path.join(data_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)


# ==============================
# Fonction de tracé
# ==============================
def plot_figure(df, variable, lang, split):
    plt.figure(figsize=(10, 6))
    selected_rows = df

    # Cas particulier : temps (training/test)
    if variable in ["time_sec"]:
        col = "training_time_sec" if split in ["train", "val"] else "testing_time_sec"

        if col not in df.columns:
            print(f"⚠️ Column {col} not found in {lang} {split}, skipping.")
            return

        values = selected_rows[col].tolist()
        labels = selected_rows["model_name"].tolist()
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Time (sec)")
        plt.title(f"{col.replace('_',' ').title()} ({lang}, {split})", fontsize=14, fontweight="bold")

    elif split == "test":
        # === Barplot pour test ===
        values, labels = [], []
        for _, row in selected_rows.iterrows():
            model = row["model_name"]
            y = ast.literal_eval(row[variable])
            score = y[0] if isinstance(y, list) else y
            values.append(score)
            labels.append(model)

        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(variable.capitalize())
        plt.title(f"{variable.capitalize()} (Test - {lang})", fontsize=14, fontweight="bold")

    else:
        # === Courbes pour train/val ===
        for _, row in selected_rows.iterrows():
            model = row["model_name"]
            y = ast.literal_eval(row[variable])
            plt.plot(range(1, len(y) + 1), y, label=model, linewidth=2)

        plt.xlabel("Epochs")
        plt.ylabel(variable.capitalize())
        plt.title(f"{variable.capitalize()} over Epochs ({lang}, {split})", fontsize=14, fontweight="bold")
        plt.legend(loc="best", fontsize=8)

    plt.grid(True, linestyle="--", alpha=0.6)

    # Sauvegarde
    save_path = os.path.join(plot_dir, f"{lang}_{split}_{variable}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Plot saved: {save_path}")


# ==============================
# Fonction pour modèles spécifiques
# ==============================
def plot_selected_models(df, variable, lang, split, model_ids=[7, 19, 22]):
    plt.figure(figsize=(10, 6))
    selected_rows = df[df.index.isin(model_ids)]

    # Cas particulier : temps
    if variable in ["time_sec"]:
        col = "training_time_sec" if split in ["train", "val"] else "testing_time_sec"

        if col not in df.columns:
            print(f"⚠️ Column {col} not found in {lang} {split}, skipping.")
            return

        values = selected_rows[col].tolist()
        labels = selected_rows["model_name"].tolist()
        plt.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Time (sec)")
        plt.title(f"{col.replace('_',' ').title()} ({lang}, {split}, Models {model_ids})", fontsize=14, fontweight="bold")

    elif split == "test":
        values, labels = [], []
        for _, row in selected_rows.iterrows():
            model = row["model_name"]
            y = ast.literal_eval(row[variable])
            score = y[0] if isinstance(y, list) else y
            values.append(score)
            labels.append(model)

        plt.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(variable.capitalize())
        plt.title(f"{variable.capitalize()} (Test - {lang}, Models {model_ids})", fontsize=14, fontweight="bold")

    else:
        for _, row in selected_rows.iterrows():
            model = row["model_name"]
            y = ast.literal_eval(row[variable])
            plt.plot(range(1, len(y) + 1), y, label=f"{model} (idx {row.name})", linewidth=2)

        plt.xlabel("Epochs")
        plt.ylabel(variable.capitalize())
        plt.title(f"{variable.capitalize()} over Epochs ({lang}, {split}, Models {model_ids})", fontsize=14, fontweight="bold")
        plt.legend(loc="best", fontsize=8)

    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(plot_dir, f"{lang}_{split}_{variable}_models_{'-'.join(map(str, model_ids))}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Plot saved (selected models): {save_path}")


# ==============================
# Génération des courbes
# ==============================
variables = ["losses", "accuracy", "BLEU", "ROUGE", "time_sec"]

for lang, splits in dfs.items():
    for split, df in splits.items():
        for variable in variables:
            plot_figure(df, variable, lang, split)
            plot_selected_models(df, variable, lang, split, model_ids=[7, 19, 22])
