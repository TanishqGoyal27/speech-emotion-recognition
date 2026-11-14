import os
import json
import shutil
import tempfile

import numpy as np
import soundfile as sf

import config
import data

rng = np.random.default_rng(0)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  [ok] {msg}")


def make_fake_ravdess(root):
    sr = config.SR
    for actor in [1, 2, 3, 19, 21]:
        for emo in range(1, 9):
            t = np.arange(int(sr * rng.uniform(2.5, 4.0))) / sr
            y = 0.3 * np.sin(2 * np.pi * (150 + 40 * emo) * t) + 0.02 * rng.standard_normal(len(t))
            y[: sr // 4] = 0.0                                          # leading silence to be trimmed
            name = f"03-01-{emo:02d}-01-01-01-{actor:02d}.wav"
            for folder in [f"Actor_{actor:02d}", f"audio_speech_actors_01-24/Actor_{actor:02d}"]:  # duplicate copy
                os.makedirs(os.path.join(root, folder), exist_ok=True)
                sf.write(os.path.join(root, folder, name), y.astype(np.float32), sr)
        # files that must be IGNORED: video modality (01) and song channel (02)
        sf.write(os.path.join(root, f"Actor_{actor:02d}", f"01-01-03-01-01-01-{actor:02d}.wav"), np.zeros(sr), sr)
        sf.write(os.path.join(root, f"Actor_{actor:02d}", f"03-02-03-01-01-01-{actor:02d}.wav"), np.zeros(sr), sr)


def main():
    tmp = tempfile.mkdtemp(prefix="ser_smoke_")
    try:
        data_dir, models_dir, results_dir = (os.path.join(tmp, d) for d in ("RAVDESS", "models", "results"))
        make_fake_ravdess(data_dir)

        print("\n1) file listing / split")
        items = data.list_ravdess_files(data_dir)
        check(len(items) == 5 * 8, f"40 unique speech files after filtering + de-dup (got {len(items)})")
        splits = data.split_by_actor(items)
        check({a for _, _, a in splits["train"]} == {1, 2, 3} and {a for _, _, a in splits["val"]} == {19}
              and {a for _, _, a in splits["test"]} == {21}, "actors land in the right split, no overlap")

        print("\n2) preprocessing / augmentation / features")
        y = data.load_audio(items[0][0])
        check(y.shape == (config.N_SAMPLES,) and abs(np.abs(y).max() - 1) < 1e-4, "prepare_waveform -> 3 s, peak 1")
        y_aug = data.augment_waveform(y, rng)
        check(y_aug.shape == y.shape, "augmented waveform keeps fixed length")
        mel = data.extract_logmel(y)
        check(mel.shape == (config.N_MELS, config.N_FRAMES), f"log-mel shape {mel.shape}")
        check(data.spec_augment(mel, rng).shape == mel.shape, "SpecAugment keeps shape")
        check(data.extract_handcrafted(y).shape == (data.HANDCRAFTED_DIM,), f"handcrafted dim {data.HANDCRAFTED_DIM}")

        ds = data.build_dataset(data_dir, os.path.join(tmp, "cache", "features.npz"), rebuild=True, n_jobs=2)
        n_train_files = len(splits["train"])
        check((ds["split"] == 0).sum() == n_train_files * (1 + config.N_AUGMENT), "train = originals + 2 aug copies")
        check(ds["aug"][ds["split"] > 0].sum() == 0, "NO augmented samples in val/test (no leakage)")
        ds2 = data.build_dataset(data_dir, os.path.join(tmp, "cache", "features.npz"))
        check(np.array_equal(ds2["y"], ds["y"]), "feature cache reloads")

        print("\n3) Keras models: build, train 1 epoch, save")
        import train
        for name in train.KERAS_MODELS:
            res = train.train_one(name, ds, epochs=1, batch=8, models_dir=models_dir,
                                  results_dir=results_dir, verbose=0)
            check(os.path.exists(os.path.join(models_dir, name, "model.keras")), f"{name} trained + saved "
                  f"({res['params']:,} params)")
            check(os.path.exists(os.path.join(results_dir, f"{name}_confusion.png")) and
                  os.path.exists(os.path.join(results_dir, f"{name}_curves.png")), f"{name} plots written")

        print("\n4) AudioEmotionPredictor on 44.1 kHz input")
        from audio_predictor import AudioEmotionPredictor
        wave_44k = rng.standard_normal(44100 * 4).astype(np.float32) * 0.1
        audio_probs = None
        for name in train.KERAS_MODELS:
            p = AudioEmotionPredictor(name, models_dir=models_dir).predict_waveform(wave_44k, 44100)
            check(p.shape == (8,) and abs(p.sum() - 1) < 1e-4, f"{name} predictor -> 8 probs summing to 1")
            audio_probs = p

        print("\n5) face mapping + custom backend + fusion")
        from face_emotion import map_face_to_audio, FaceEmotionPredictor
        m = map_face_to_audio({"neutral": 1.0})
        check(np.isclose(m[config.EMOTIONS.index("neutral")], 0.5) and np.isclose(m[config.EMOTIONS.index("calm")], 0.5),
              "face neutral -> 50% neutral + 50% calm")
        m = map_face_to_audio({"fear": 30, "surprise": 70})            # DeepFace gives percentages
        check(np.isclose(m.sum(), 1) and np.isclose(m[config.EMOTIONS.index("surprised")], 0.7), "fear/surprise mapping")

        from train_face_cnn import build_face_cnn
        face_dir = os.path.join(models_dir, "face_cnn")
        os.makedirs(face_dir)
        fer_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        face_model = build_face_cnn(len(fer_names))
        check(face_model.predict(rng.uniform(0, 255, (2, 48, 48, 1)).astype("float32"), verbose=0).shape == (2, 7),
              "face CNN builds, output (batch, 7)")
        face_model.save(os.path.join(face_dir, "model.keras"))
        with open(os.path.join(face_dir, "class_names.json"), "w") as f:
            json.dump(fer_names, f)
        fp = FaceEmotionPredictor("custom", custom_dir=face_dir)
        noise_frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        probs, n_found, _ = fp.predict_frames([noise_frame] * 10)
        check(probs is None and n_found == 0, "no face in a noise image -> None (audio-only fallback)")

        from fusion import fuse, format_table
        face_probs = map_face_to_audio({"happy": 1.0})
        r = fuse(audio_probs, face_probs, 0.7)
        check(r["probs"].shape == (8,) and np.isclose(r["probs"].sum(), 1) and r["mode"] == "audio+face",
              "fusion output is a valid distribution")
        check(np.allclose(r["probs"], 0.7 * audio_probs + 0.3 * face_probs, atol=1e-5), "fused = 0.7*audio + 0.3*face")
        check(fuse(audio_probs, None)["mode"] == "audio-only", "no face -> audio-only")
        try:
            fuse(audio_probs, face_probs, 0.4)
            check(False, "w_audio < 0.5 must be rejected")
        except ValueError:
            check(True, "w_audio < 0.5 rejected (audio must dominate)")
        print(format_table(audio_probs, face_probs, r))

        print("\n6) compare.py + optional wav2vec2")
        import compare
        rows = compare.load_results(results_dir)
        check(len(rows) == 3, "compare.py reads the 3 result JSON files")
        compare.plot(rows, os.path.join(results_dir, "comparison.png"))
        print(compare.make_table(rows))
        try:
            import torch
            from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification
            cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                 conv_dim=(32, 32), conv_stride=(5, 2), conv_kernel=(10, 3),
                                 num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, num_labels=8)
            w2v = Wav2Vec2ForSequenceClassification(cfg)
            w2v.freeze_feature_encoder()
            logits = w2v(torch.randn(2, config.W2V_SR * 3)).logits
            check(tuple(logits.shape) == (2, 8), "tiny wav2vec2 classifier forward -> (2, 8)")
        except ImportError:
            print("  [skip] torch/transformers not installed (optional) - wav2vec2 check skipped")

        print("\nALL SMOKE TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
