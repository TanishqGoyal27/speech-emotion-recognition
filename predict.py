"""
predict.py - predict emotion for an audio FILE, optionally fused with a face IMAGE.

    python predict.py --audio clip.wav
    python predict.py --audio clip.wav --image face.jpg --face-backend auto --audio-weight 0.7

Prints the audio | face | fused table and saves to results/:
    prediction_<name>.png   annotated face image (with --image) or a probability bar chart (audio only)
"""
import os
import argparse

import numpy as np

import config
from audio_predictor import AudioEmotionPredictor
from fusion import fuse, format_table


def save_bar_chart(audio_probs, face_probs, result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(config.NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.27, audio_probs, 0.27, label="audio")
    if face_probs is not None:
        ax.bar(x, face_probs, 0.27, label="face")
    ax.bar(x + 0.27, result["probs"], 0.27, label="fused")
    ax.set_xticks(x, config.EMOTIONS, rotation=30)
    ax.set_ylim(0, 1)
    ax.set_title(f"Prediction: {result['emotion']} ({result['confidence'] * 100:.1f}%) - {result['mode']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from an audio file (+ optional face image).")
    parser.add_argument("--audio", required=True, help="path to a .wav/.mp3 file")
    parser.add_argument("--image", help="optional face image (jpg/png)")
    parser.add_argument("--audio-model", default="cnn_bilstm", choices=config.AUDIO_MODELS)
    parser.add_argument("--face-backend", default="auto", choices=["auto", "deepface", "custom"])
    parser.add_argument("--audio-weight", type=float, default=config.AUDIO_WEIGHT)
    args = parser.parse_args()
    if not config.MIN_AUDIO_WEIGHT <= args.audio_weight <= 1.0:
        parser.error(f"--audio-weight must be between {config.MIN_AUDIO_WEIGHT} and 1.0")

    audio_probs = AudioEmotionPredictor(args.audio_model).predict_file(args.audio)

    face_probs, box, frame = None, None, None
    if args.image:
        import cv2
        from face_emotion import FaceEmotionPredictor
        frame = cv2.imread(args.image)
        if frame is None:
            parser.error(f"cannot read image {args.image}")
        face_probs, box = FaceEmotionPredictor(args.face_backend).predict_frame(frame)
        if face_probs is None:
            print("No face detected in the image -> audio-only result.")

    result = fuse(audio_probs, face_probs, args.audio_weight)
    print(format_table(audio_probs, face_probs, result))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.audio))[0]
    out_path = os.path.join(config.RESULTS_DIR, f"prediction_{stem}.png")
    if frame is not None:
        import cv2
        from face_emotion import annotate_frame
        img = annotate_frame(frame, [f"{result['emotion'].upper()} {result['confidence'] * 100:.0f}%",
                                     f"mode: {result['mode']}"], box)
        cv2.imwrite(out_path, img)
    else:
        save_bar_chart(audio_probs, face_probs, result, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
