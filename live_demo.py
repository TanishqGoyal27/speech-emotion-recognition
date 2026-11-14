"""
live_demo.py - real-time demo from the microphone (+ optional webcam).

    python live_demo.py                                   # audio only, CNN+BiLSTM
    python live_demo.py --face --face-backend auto        # audio + face fusion
    python live_demo.py --audio-model wav2vec2 --seconds 4 --audio-weight 0.8

Press Enter to record N seconds; type q + Enter to quit.
With --face, webcam frames are captured DURING the recording (live preview window),
~6 evenly spaced frames are analysed and their face probabilities averaged.
"""
import time
import argparse

import numpy as np
import sounddevice as sd

import config
from audio_predictor import AudioEmotionPredictor
from fusion import fuse, format_table


def record(seconds, sr, cap=None):
    """Record audio (non-blocking) while grabbing webcam frames for the same duration."""
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    frames, start = [], time.time()
    if cap is not None:
        import cv2
        while time.time() - start < seconds:
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
                preview = frame.copy()
                cv2.putText(preview, f"REC {seconds - (time.time() - start):.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("live_demo", preview)
                cv2.waitKey(1)
    sd.wait()                                   # make sure the recording has finished
    return audio[:, 0], frames


def main():
    parser = argparse.ArgumentParser(description="Live speech emotion recognition (+ optional face fusion).")
    parser.add_argument("--audio-model", default="cnn_bilstm", choices=config.AUDIO_MODELS)
    parser.add_argument("--face", action="store_true", help="also use the webcam")
    parser.add_argument("--face-backend", default="auto", choices=["auto", "deepface", "custom"])
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--audio-weight", type=float, default=config.AUDIO_WEIGHT)
    args = parser.parse_args()
    if not config.MIN_AUDIO_WEIGHT <= args.audio_weight <= 1.0:
        parser.error(f"--audio-weight must be between {config.MIN_AUDIO_WEIGHT} and 1.0")

    audio_model = AudioEmotionPredictor(args.audio_model)
    sr = int(sd.query_devices(kind="input")["default_samplerate"])   # mic's native rate; predictor resamples
    print(f"audio model: {args.audio_model} | mic sample rate: {sr} Hz")

    cap = face_model = None
    if args.face:
        import cv2
        from face_emotion import FaceEmotionPredictor, annotate_frame
        face_model = FaceEmotionPredictor(args.face_backend)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("WARNING: webcam not available -> audio-only mode")
            cap = face_model = None

    try:
        while True:
            cmd = input(f"\nPress Enter to record {args.seconds:.0f} s (q + Enter to quit): ").strip().lower()
            if cmd == "q":
                break
            print("Recording... speak now!")
            audio, frames = record(args.seconds, sr, cap)
            if np.max(np.abs(audio)) < 1e-3:
                print("Recording is (almost) silent - check your microphone.")

            audio_probs = audio_model.predict_waveform(audio, sr)
            face_probs, n_faces, box = (None, 0, None)
            if face_model is not None:
                face_probs, n_faces, box = face_model.predict_frames(frames, config.FACE_N_FRAMES)
                print(f"face detected in {n_faces} of {min(config.FACE_N_FRAMES, len(frames))} analysed frames")

            result = fuse(audio_probs, face_probs, args.audio_weight)
            print(format_table(audio_probs, face_probs, result))

            if frames:
                import cv2
                img = annotate_frame(frames[-1], [f"{result['emotion'].upper()} {result['confidence'] * 100:.0f}%",
                                                  f"mode: {result['mode']}"], box)
                cv2.imshow("live_demo", img)
                print("(press any key in the image window to continue)")
                cv2.waitKey(0)
    finally:
        if cap is not None:
            import cv2
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
