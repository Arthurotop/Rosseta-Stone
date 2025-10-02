// Variables pour les langues
let fromLang = "fr";
let toLang = "en";

// Fonction de traduction
async function traduire() {
  const texte = document.getElementById("source").value;
  const output = document.querySelector(".output");
  const modelInfo = document.getElementById("modelSource"); // <- zone ajoutée dans ton HTML

  if (!texte.trim()) {
    output.textContent = ""; 
    modelInfo.textContent = "";
    return;
  }

  try {
    const res = await fetch("http://127.0.0.1:5000/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        text: texte,
        from: fromLang,
        to: toLang
      })
    });

    if (!res.ok) {
      const errText = await res.text();
      console.error("Erreur API:", errText);
      output.textContent = "Erreur lors de la traduction";
      modelInfo.textContent = "";
      return;
    }

    const data = await res.json();
    output.textContent = data.translation_text || "";

    // Affiche la source du modèle
    if (data.model_source) {
      modelInfo.textContent = 
        data.model_source === "local"
          ? "Traduit avec le modèle local"
          : "Traduit via API Hugging Face";
    } else {
      modelInfo.textContent = "";
    }
  } catch (err) {
    console.error(err);
    output.textContent = "Erreur réseau ou API";
    modelInfo.textContent = "";
  }
}

// Gestion des boutons de langue
document.addEventListener("DOMContentLoaded", () => {
  const textarea = document.getElementById("source");
  let timeout;

  // Traduction en temps réel quand on tape
  textarea.addEventListener("input", () => {
    clearTimeout(timeout);
    timeout = setTimeout(traduire, 500); // 500ms après la dernière frappe
  });

  const langButtons = document.querySelectorAll(".lang-btn");

  langButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      // Met à jour le style actif
      langButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      // Met à jour les langues
      fromLang = btn.dataset.from;
      toLang = btn.dataset.to;

      // Relance la traduction si du texte est présent
      if (textarea.value.trim()) {
        traduire();
      }
    });
  });
});
