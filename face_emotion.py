"""
face_emotion.py - OPTIONAL facial emotion recogniser used to support the audio model.

Backends
    "deepface" : pretrained DeepFace emotion model (pip install deepface tf-keras)
    "custom"   : our own CNN trained on FER-2013 (train_face_cnn.py) + OpenCV Haar face detector
    "auto"     : try deepface, fall back to custom

Every backend outputs probabilities over the 7 FER emotions, which are then MAPPED
to the 8 RAVDESS audio classes (config.FACE_TO_AUDIO; "neutral" is split 50/50
into neutral + calm, because faces cannot distinguish neutral from calm).
If no face is found, predict_* returns None and fusion falls back to audio-only.
"""
import os
import json

import numpy as np
import cv2

import config

FACE_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
_ALIASES = {"angry": "angry", "anger": "angry", "disgust": "disgust", "disgusted": "disgust",
            "fear": "fear", "fearful": "fear", "happy": "happy", "happiness": "happy",
            "sad": "sad", "sadness": "sad", "surprise": "surprise", "surprised": "surprise",
            "neutral": "neutral"}
CUSTOM_DIR = os.path.join(config.MODELS_DIR, "face_cnn")


def map_face_to_audio(face_probs):
    """dict {fer_emotion: prob} -> np.array (8,) in config.EMOTIONS order, summing to 1."""
    out = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    for name, p in face_probs.items():
        key = _ALIASES.get(name.lower())
        if key is None:
            continue
        for audio_name, share in config.FACE_TO_AUDIO[key].items():
            out[config.EMOTIONS.index(audio_name)] += share * float(p)
    total = out.sum()
    return out / total if total > 0 else out


class FaceEmotionPredictor:
    def __init__(self, backend="auto", custom_dir=CUSTOM_DIR):
        self.backend = None
        errors = []
        if backend in ("auto", "deepface"):
            try:
                from deepface import DeepFace
                self.DeepFace = DeepFace
                self.backend = "deepface"
            except Exception as e:                        # not installed / broken install
                errors.append(f"deepface: {e}")
                if backend == "deepface":
                    raise ImportError("DeepFace backend requested but unavailable: "
                                      "pip install deepface tf-keras") from e
        if self.backend is None and backend in ("auto", "custom"):
            self._load_custom(custom_dir)
            self.backend = "custom"
        if self.backend is None:
            raise ValueError(f"Unknown face backend '{backend}'. Errors: {errors}")
        print(f"[face] using backend: {self.backend}")

    def _load_custom(self, custom_dir):
        model_path = os.path.join(custom_dir, "model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Custom face model not found at {model_path}. "
                                    "Run train_face_cnn.py or install deepface.")
        import keras                      # Keras 3 (installed with TensorFlow >= 2.16)
        self.model = keras.models.load_model(model_path, compile=False)
        with open(os.path.join(custom_dir, "class_names.json")) as f:
            self.class_names = json.load(f)                # order = FER folder order used in training
        cascade = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.detector = cv2.CascadeClassifier(cascade)

    # ------------------------------------------------------------------
    def predict_frame(self, frame_bgr):
        """One BGR frame -> (8 audio-space probs or None, face box (x, y, w, h) or None)."""
        if frame_bgr is None:
            return None, None
        if self.backend == "deepface":
            return self._predict_deepface(frame_bgr)
        return self._predict_custom(frame_bgr)

    def _predict_deepface(self, frame):
        try:
            res = self.DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)
        except Exception:
            return None, None
        res = res[0] if isinstance(res, list) else res
        region = res.get("region", {})
        h, w = frame.shape[:2]
        # With enforce_detection=False DeepFace returns the WHOLE image when no face is found.
        whole_image = region.get("w", w) >= w - 1 and region.get("h", h) >= h - 1
        if res.get("face_confidence", 1.0) == 0 or whole_image:
            return None, None
        probs = map_face_to_audio(res["emotion"])
        box = (region["x"], region["y"], region["w"], region["h"])
        return probs, box

    def _predict_custom(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return None, None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])          # largest face
        face = cv2.resize(gray[y:y + h, x:x + w], (config.FACE_IMG_SIZE, config.FACE_IMG_SIZE))
        x_in = face.astype(np.float32)[None, :, :, None]            # model rescales 0..255 itself
        p = np.asarray(self.model(x_in, training=False))[0]
        probs = map_face_to_audio({name: float(v) for name, v in zip(self.class_names, p)})
        return probs, (int(x), int(y), int(w), int(h))

    def predict_frames(self, frames, n=config.FACE_N_FRAMES):
        """Analyse ~n evenly spaced frames and average the probabilities of frames with a face.
        Returns (avg probs or None, number of frames with a face, last face box)."""
        if not frames:
            return None, 0, None
        idx = np.unique(np.linspace(0, len(frames) - 1, num=min(n, len(frames))).astype(int))
        found, last_box = [], None
        for i in idx:
            probs, box = self.predict_frame(frames[i])
            if probs is not None:
                found.append(probs)
                last_box = box
        if not found:
            return None, 0, None
        return np.mean(found, axis=0).astype(np.float32), len(found), last_box


def annotate_frame(frame, lines, box=None):
    """Draw a face box and text lines on a copy of the frame (used by live_demo / predict)."""
    img = frame.copy()
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for i, text in enumerate(lines):
        org = (10, 30 + 28 * i)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img
