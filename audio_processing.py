"""
Synevola - Audio Processing Module
==================================

Ce module gère le traitement audio : transcription et diarisation.

Vous devez adapter ce fichier selon votre configuration WhisperX/Pyannote.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

# --- Configuration ---
TEMP_DIR = tempfile.gettempdir()
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# --- Imports conditionnels ---
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

try:
    from faster_whisper import WhisperModel
    HAVE_FASTER_WHISPER = True
except ImportError:
    HAVE_FASTER_WHISPER = False

try:
    import whisper
    HAVE_WHISPER = True
except ImportError:
    HAVE_WHISPER = False

try:
    from pyannote.audio import Pipeline
    HAVE_PYANNOTE = True
except ImportError:
    HAVE_PYANNOTE = False


def process_audio(
    audio_path: str,
    diarization_enabled: bool = True,
    model_name: str = "small",
    language: str = "fr"
) -> Tuple[List, Dict[str, str]]:
    """
    Traite un fichier audio : transcription + diarisation optionnelle.

    Args:
        audio_path: Chemin vers le fichier audio
        diarization_enabled: Activer l'identification des locuteurs
        model_name: Modèle Whisper (tiny, base, small, medium, large)
        language: Langue de transcription

    Returns:
        Tuple contenant:
        - Liste de transcriptions (format dépend de diarization_enabled)
        - Dictionnaire des fichiers audio par locuteur (vide si pas de diarisation)
    """
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
    
    # Transcription
    if model_name == "faster-whisper" and HAVE_FASTER_WHISPER:
        transcription = _transcribe_faster_whisper(audio_path, language)
    elif HAVE_WHISPER:
        transcription = _transcribe_whisper(audio_path, model_name, language)
    else:
        raise RuntimeError("Aucun moteur de transcription disponible. Installez whisper ou faster-whisper.")
    
    # Diarisation
    if diarization_enabled and HAVE_PYANNOTE:
        try:
            segments = _diarize_audio(audio_path, transcription)
            return segments, {}
        except Exception as e:
            print(f"Erreur diarisation: {e}. Retour à la transcription simple.")
            return [transcription], {}
    else:
        return [transcription], {}


def _transcribe_whisper(
    audio_path: str, 
    model_name: str = "small",
    language: str = "fr"
) -> str:
    """Transcription avec OpenAI Whisper."""
    model = whisper.load_model(model_name, device=DEVICE)
    result = model.transcribe(audio_path, language=language)
    return result["text"]


def _transcribe_faster_whisper(
    audio_path: str,
    language: str = "fr"
) -> str:
    """Transcription avec Faster Whisper."""
    model = WhisperModel("small", device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")
    segments, info = model.transcribe(audio_path, language=language)
    return " ".join([segment.text for segment in segments])


def _diarize_audio(
    audio_path: str,
    transcription: str
) -> List[Tuple[float, float, str, str]]:
    """
    Diarisation avec Pyannote.
    
    Returns:
        Liste de tuples (start, end, speaker, text)
    """
    if not HF_TOKEN:
        raise ValueError("Token HuggingFace requis pour la diarisation. Définissez HF_TOKEN.")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    
    if DEVICE == "cuda":
        import torch
        pipeline.to(torch.device("cuda"))
    
    diarization = pipeline(audio_path)
    
    # Combiner diarisation et transcription
    # Note: Ceci est une implémentation simplifiée
    # Pour une meilleure précision, utilisez whisperx avec alignement
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((
            turn.start,
            turn.end,
            speaker,
            ""  # Le texte serait assigné via alignement
        ))
    
    # Si pas de segments, retourner la transcription complète
    if not segments:
        return [(0.0, 0.0, "SPEAKER_00", transcription)]
    
    # Assigner le texte au premier segment (simplification)
    segments[0] = (segments[0][0], segments[0][1], segments[0][2], transcription)
    
    return segments


def clean_temp_files() -> None:
    """Nettoie les fichiers temporaires créés par le traitement audio."""
    temp_patterns = ["*.wav", "*.tmp", "*.temp"]
    temp_path = Path(TEMP_DIR)
    
    for pattern in temp_patterns:
        for file in temp_path.glob(f"synevola_{pattern}"):
            try:
                file.unlink()
            except Exception:
                pass


# --- Point d'entrée pour tests ---
if __name__ == "__main__":
    print("=== Synevola Audio Processing ===")
    print(f"Device: {DEVICE}")
    print(f"Faster Whisper: {'✓' if HAVE_FASTER_WHISPER else '✗'}")
    print(f"OpenAI Whisper: {'✓' if HAVE_WHISPER else '✗'}")
    print(f"Pyannote: {'✓' if HAVE_PYANNOTE else '✗'}")
    print(f"HF Token: {'✓ configuré' if HF_TOKEN else '✗ non configuré'}")
