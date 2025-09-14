import os
import ast
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Configuration des chemins
# ==============================
script_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "reporting"))

csv_path_train = os.path.join(data_dir, "gridsearch_res", "gridsearch_results.csv")
csv_path_test = os.path.join(data_dir, "gridsearch_res", "gridsearch_results_test.csv")

df_train = pd.read_csv(csv_path_train)
df_test = pd.read_csv(csv_path_test)

# Dossier de sauvegarde pour les plots
plot_dir = os.path.join(data_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)


# ==============================
# Fonction de tracé
# ==============================
def plot_figure(df, variable, methode):
    """
    Trace et sauvegarde l'évolution d'une métrique pour plusieurs modèles.

    Args:
        df (pd.DataFrame): résultats de la gridsearch
        variable (str): nom de la colonne (ex: "losses", "accuracy", "BLEU")
        methode (str): "training" ou "test"
    """
    plt.figure(figsize=(10, 6))

    # Sélection : 10 premiers modèles + le 28e (meilleur choisi)
    selected_rows = pd.concat([df.head(10), df.iloc[[27]]])

    for _, row in selected_rows.iterrows():
        model = row["model_name"]
        y = ast.literal_eval(row[variable])  # convertir la string en liste
        plt.plot(range(1, len(y) + 1), y, label=model, linewidth=2)

    # Mise en forme
    plt.xlabel("Epochs")
    plt.ylabel(variable.capitalize())
    plt.title(f"{variable.capitalize()} over Epochs ({methode})", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Sauvegarde
    save_path = os.path.join(plot_dir, f"{variable}_{methode}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Plot saved: {save_path}")


# ==============================
# Génération des courbes
# ==============================
variables = ["losses", "accuracy", "BLEU"]

for variable in variables:
    plot_figure(df_train, variable, "training")
    plot_figure(df_test, variable, "test")
