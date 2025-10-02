# Rosseta Stone - Traducteur EN↔FR

## Description

Rosseta Stone est un traducteur automatique anglais-français et français-anglais basé sur des modèles **LSTM**.  
Le projet est structuré en trois parties principales :  

1. **Backend** : API Flask (`backend/src/app.py`) servant les modèles de traduction et gérant les requêtes du frontend.  
2. **Frontend** : Interface web (`frontend/index.html`, `frontend/styles.css` et  `frontend/translate.js`) pour saisir du texte et afficher la traduction en temps réel.  
3. **Analyse et préparation des données** : Scripts pour nettoyer et prétraiter les données nécessaires à l'entraînement des modèles.

---

## Modèles

- Deux modèles LSTM distincts ont été entraînés :  
  - **EN→FR**  
  - **FR→EN**  
- Plusieurs configurations ont été testées via **Grid Search** pour choisir les meilleurs hyperparamètres (`EMB_SIZE`, `HID_SIZE`, etc.).  
- Les modèles ne sont pas bidirectionnels : chaque sens de traduction possède son propre modèle.  

---

## Données

Les données étant volumineuses, elles ne sont pas incluses dans le dépôt.  
Téléchargez-les depuis Google Drive :  
[Google Drive - Données pour Rosseta Stone](https://drive.google.com/file/d/1Ecbj5CZT4BgkRjvV7MxZ8VJmZBK-4oRb/view?usp=drive_link)  

**Instructions :**  

1. Télécharger et extraire le fichier ZIP.  
2. Placer le contenu dans le dossier `backend/data`.  

---

## Lancement en local

Pour exécuter le projet en local :  

1. Installer les dépendances via `requirements.txt` :  
   ```bash
   pip install -r backend/requirements.txt

2. Lancer le serveur Flask depuis le backend :
   python backend/src/app.py

3. L’API sera accessible à l’URL affichée dans la console, généralement http://127.0.0.1:5000/.
4. Ouvrir frontend/index.html sinon dans un navigateur pour accéder à l’interface utilisateur.

## Déploiement en ligne

Le projet a été déployé sur Render.com, offrant un accès direct depuis un navigateur :

* URL publique de l’interface web : https://rosseta-stone.onrender.com/

* Le backend Flask tourne en continu et expose l’API /translate pour le frontend.

# Points importants :

 * Render peut mettre en pause les services gratuits après une période d’inactivité. L’URL reste la même, mais le serveur peut mettre quelques secondes à se réveiller.

 * Les données volumineuses et modèles ne sont pas inclus dans le déploiement. Le script app.py propose de télécharger automatiquement les modèles depuis Google Drive si nécessaire.

 * Les dépendances sont installées via le fichier requirements.txt lors du déploiement.

## Flux de fonctionnement 

Frontend (index.html + styles.css + translate.js)
           │
           │ Requête HTTP POST /translate
           ▼
Backend Flask (app.py)
           │
           │ Vérifie si la phrase peut être traduite localement
           │
   ┌─────────────┴─────────────┐
   │                           │
Local LSTM EN↔FR         API Hugging Face (optionnel)
   │                           │
   └─────────────┬─────────────┘
                 │
             Traduction
                 │
                 ▼
        Frontend (affichage)
        
 * Le frontend envoie la phrase à traduire via l’API Flask.

 * L’API décide si le modèle local peut traduire ou si l’API Hugging Face doit être utilisée.

* La traduction est renvoyée au frontend, qui l’affiche et indique la source du modèle.

## Fonctionnalités

* Traduction texte EN↔FR et FR↔EN via les modèles LSTM locaux.

* Utilisation optionnelle de l'API Hugging Face pour les phrases hors vocabulaire local.

* Interface web simple et réactive pour saisir du texte et afficher la traduction en temps réel.

* Affichage de la source du modèle (local ou Hugging Face).

## Structure du projet

rosetta/
│
├─ backend/
│   ├─ data/                 # Données téléchargées depuis Google Drive
│   └─ src/
│       ├─ app.py            # API Flask
│       ├─ ml/               # Scripts et modèles LSTM
│       └─ reporting/        # Scripts et résultats de reporting
│
├─ frontend/
│   ├─ index.html            # Interface utilisateur
│   ├─ translate.js          # Script de traduction
│   └─ styles.css
│
├─ analyze_text.ipynb        # Notebook d'analyse des données
├─ processing_data/          # Scripts de traitement et prétraitement des données
├─ .gitattributes
├─ .gitignore
└─ requirements.txt



## Remarques

Configurez la variable d’environnement HF_TOKEN pour utiliser l’API Hugging Face.

Le modèle local est prioritaire ; l’API Hugging Face est utilisée uniquement si nécessaire.

Le projet peut être exécuté en local ou via le déploiement en ligne sur Render.


