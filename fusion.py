"""
fusion.py - late (decision-level) fusion of audio and face probabilities.

    fused = w_audio * audio_probs + (1 - w_audio) * face_probs,   0.5 <= w_audio <= 1

Why late fusion?  The two models are trained separately on different datasets
(RAVDESS audio, FER-2013 faces), so we combine their OUTPUT probabilities rather
than their features. Audio is the primary modality, so its weight is never below 0.5;
if there is no face (or face is disabled) the result is simply the audio prediction.
"""
import numpy as np

import config


def fuse(audio_probs, face_probs=None, w_audio=config.AUDIO_WEIGHT):
    """Return dict: probs (8,), emotion, confidence, mode, w_audio."""
    if not (config.MIN_AUDIO_WEIGHT <= w_audio <= 1.0):
        raise ValueError(f"w_audio must be in [{config.MIN_AUDIO_WEIGHT}, 1.0] (audio must dominate), got {w_audio}")
    audio_probs = np.asarray(audio_probs, dtype=np.float32)
    if audio_probs.shape != (config.NUM_CLASSES,):
        raise ValueError(f"audio_probs must have shape ({config.NUM_CLASSES},), got {audio_probs.shape}")

    if face_probs is None:
        probs, mode, w = audio_probs, "audio-only", 1.0
    else:
        face_probs = np.asarray(face_probs, dtype=np.float32)
        if face_probs.shape != audio_probs.shape:
            raise ValueError("face_probs must have the same shape as audio_probs")
        probs, mode, w = w_audio * audio_probs + (1.0 - w_audio) * face_probs, "audio+face", w_audio
    probs = probs / probs.sum()
    idx = int(np.argmax(probs))
    return {"probs": probs, "emotion": config.EMOTIONS[idx], "confidence": float(probs[idx]),
            "mode": mode, "w_audio": w}


def format_table(audio_probs, face_probs, result):
    """Pretty text table: emotion | audio | face | fused."""
    lines = [f"{'emotion':<10} {'audio':>7} {'face':>7} {'fused':>7}", "-" * 34]
    for i, name in enumerate(config.EMOTIONS):
        face = f"{face_probs[i]:7.3f}" if face_probs is not None else f"{'-':>7}"
        mark = "  <==" if name == result["emotion"] else ""
        lines.append(f"{name:<10} {audio_probs[i]:7.3f} {face} {result['probs'][i]:7.3f}{mark}")
    lines.append("-" * 34)
    lines.append(f"FINAL: {result['emotion'].upper()} ({result['confidence'] * 100:.1f}%)  "
                 f"mode={result['mode']}  w_audio={result['w_audio']:.2f}")
    return "\n".join(lines)
