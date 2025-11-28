# 🩺 Synevola

**Transcription & Résumé aidé par IA en local**

Synevola est une application de transcription audio médicale avec diarisation (identification des locuteurs) et génération automatique de résumés, le tout fonctionnant **100% en local** pour garantir la confidentialité des données patients.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Optional-orange.svg)

---

## ✨ Fonctionnalités

### 🎤 Transcription Audio
- **Multi-format** : MP3, WAV, OGG, FLAC, M4A, WebM
- **Modèles Whisper** : tiny, base, small, medium, large, faster-whisper
- **Diarisation** : Identification automatique des locuteurs (pyannote.audio)
- **Renommage** : Possibilité de renommer les locuteurs (SPEAKER_00 → Dr. Martin)

### 🎙️ Enregistrement Audio
- **Sélection du microphone** : Choix parmi les périphériques disponibles
- **Contrôle du gain** : Ajustement du niveau d'entrée (0-200%)
- **VU-mètre temps réel** : Visualisation du niveau audio
- **Pause/Reprise** : Contrôle complet de l'enregistrement

### 🧠 Résumé Intelligent
- **LLM local** : Intégration avec LM Studio (Mistral, Qwen, Llama, etc.)
- **Personnalisable** : Prompts système et utilisateur modifiables
- **Chunking intelligent** : Gestion des longs documents par blocs
- **Multi-modes** : Résumé direct ou par blocs + synthèse

### 📤 Export
- **TXT** : Transcription et résumé en texte brut
- **DOCX** : Compte rendu médical formaté (Word)

---

## 🚀 Installation

### Prérequis

- Python 3.9 ou supérieur
- [LM Studio](https://lmstudio.ai/) pour les résumés IA
- ffmpeg pour le traitement audio
- (Optionnel) GPU NVIDIA avec CUDA pour accélération

### 1. Cloner le repository

```bash
git clone https://github.com/votre-username/synevola.git
cd synevola
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Installer ffmpeg

```bash
# Windows (avec Chocolatey)
choco install ffmpeg

# macOS (avec Homebrew)
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt install ffmpeg
```

### 5. (Optionnel) Support GPU CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 6. Token HuggingFace (pour pyannote)

La diarisation nécessite un token HuggingFace :

1. Créez un compte sur [huggingface.co](https://huggingface.co)
2. Acceptez les conditions sur [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Créez un token dans vos paramètres
4. Configurez la variable d'environnement :

```bash
# Windows
set HF_TOKEN=votre_token_ici

# macOS/Linux
export HF_TOKEN=votre_token_ici
```

---

## 📖 Utilisation

### 1. Démarrer LM Studio

1. Téléchargez et installez [LM Studio](https://lmstudio.ai/)
2. Chargez un modèle (recommandés : Mistral-7B, Qwen2.5-7B, Llama-3.1-8B)
3. Configurez le serveur :
   - **Context Length** : 8192-32768 tokens (important !)
   - **GPU Offload** : Maximum possible
4. Démarrez le serveur local (port 1234 par défaut)

### 2. Lancer Synevola

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur à `http://localhost:8501`

### 3. Workflow typique

1. **Vérifiez les indicateurs** : CUDA ✅ et LM Studio ✅ dans la sidebar
2. **Importez un audio** ou **enregistrez en direct**
3. **Configurez** les paramètres (modèle STT, diarisation, prompts)
4. **Cliquez** sur "🚀 Transcrire + Résumer"
5. **Exportez** le résultat en TXT ou DOCX

---

## ⚙️ Configuration

### Paramètres STT (Speech-to-Text)

| Paramètre | Description | Valeurs |
|-----------|-------------|---------|
| Modèle STT | Taille du modèle Whisper | tiny, base, small, medium, large |
| Diarisation | Identification des locuteurs | On/Off |
| Normalisation | Conversion mono + normalisation | On/Off |

### Paramètres LLM

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| Température | Créativité du modèle | 0.2 |
| Max tokens | Longueur max de la réponse | 1024 |
| Taille bloc | Tokens par chunk | 6000 |
| Chevauchement | Overlap entre chunks | 200 |

### Configuration LM Studio recommandée

| Modèle | Context Length | GPU Layers |
|--------|---------------|------------|
| Mistral-7B | 8192-16384 | Max |
| Qwen2.5-7B | 32768 | Max |
| Llama-3.1-8B | 8192-16384 | Max |

---

## 📁 Structure du projet

```
synevola/
├── app.py                 # Application principale Streamlit
├── audio_processing.py    # Pipeline de traitement audio
├── requirements.txt       # Dépendances Python
├── README.md             # Ce fichier
├── LICENSE               # Licence MIT
├── .gitignore           # Fichiers à ignorer
├── .env.example         # Exemple de configuration
└── docs/
    └── CONFIGURATION.md  # Guide de configuration détaillé
```

---

## 🔧 Dépannage

### Erreur "Channel Error" ou "prediction-error" avec LM Studio

**Cause** : Context Length insuffisant

**Solution** :
1. Dans LM Studio, augmentez le Context Length (8192 → 16384 ou plus)
2. Activez le GPU Offload au maximum
3. Rechargez le modèle

### Erreur "CUDA out of memory"

**Solutions** :
1. Utilisez un modèle STT plus petit (small au lieu de large)
2. Réduisez le Context Length dans LM Studio
3. Fermez les autres applications GPU

### La diarisation ne fonctionne pas

**Vérifiez** :
1. Le token HuggingFace est configuré (`HF_TOKEN`)
2. Vous avez accepté les conditions sur HuggingFace
3. pyannote.audio est correctement installé

### L'enregistrement audio ne fonctionne pas

**Vérifiez** :
1. ffmpeg est installé (`ffmpeg -version`)
2. Le navigateur a accès au microphone
3. streamlit-audiorecorder est installé

---

## 🤝 Contribution

Les contributions sont les bienvenues ! 

1. Forkez le projet
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Committez vos changements (`git commit -m 'Ajout de fonctionnalité'`)
4. Pushez la branche (`git push origin feature/amelioration`)
5. Ouvrez une Pull Request

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- [OpenAI Whisper](https://github.com/openai/whisper) - Modèle de transcription
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - Implémentation optimisée
- [Pyannote](https://github.com/pyannote/pyannote-audio) - Diarisation
- [LM Studio](https://lmstudio.ai/) - Inférence LLM locale
- [Streamlit](https://streamlit.io/) - Framework UI

---

## 📬 Contact

Pour toute question ou suggestion, ouvrez une [issue](https://github.com/votre-username/synevola/issues) sur GitHub.

---

<p align="center">
  <b>Synevola</b> - Transcription médicale intelligente, 100% locale et confidentielle 🔒
</p>
