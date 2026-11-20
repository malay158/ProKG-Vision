# ProKG-Vision

**Progressive Knowledge Graph Fusion for Difficult Visual Recognition**

This repository presents a progressive knowledge-guided visual recognition framework that learns to incorporate spoken observations based on their relevance and reliability.

## 1. What is included

| Location                              | Contents                                                                                                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `prokg/model.py`                      | Image encoder; predicted object–attribute–relation graph; spoken encoding; progressive alignment; region refinement; evidence verification; reliability gate |
| `prokg/losses.py`                     | Recognition, graph, grounding, three-way verification, counterfactual and multi-positive retrieval objectives                                                |
| `prokg/train.py`                      | Training with seed, split manifests and checkpoint saving                                                                                                    |
| `prokg/evaluate.py`                   | Separate attribute macro-F1, relation macro-F1, multi-label mAP, and speech-to-image Recall@10                                                               |
| `scripts/prepare_visual_genome.py`    | Official scene-graph annotation conversion, supported observations and **unverified** contradiction candidates                                               |
| `scripts/prepare_spokencoco.py`       | COCO-class multi-label annotations joined to verified image/audio pairs                                                                                      |
| `scripts/prepare_spoken_objectnet.py` | Zero-shot _test-only_ retrieval manifest                                                                                                                     |
| `scripts/transcribe_asr.py`           | Optional faster-whisper transcription to avoid reference-transcript leakage at inference                                                                     |
| `scripts/synthesize_vg_tts.py`        | Optional text-to-speech generation from verified observations                                                                                                |
| `scripts/run_baselines.py`            | Six proposed-system/baseline configurations with identical split and seed                                                                                    |
| `scripts/audit_splits.py`             | Source-image overlap checks across VG, COCO and test sets                                                                                                    |
| `scripts/audit_caption_leakage.py`    | Flags captions containing target class names                                                                                                                 |
| `scripts/export_stagewise.py`         | Exports actual predicted stagewise scores for a results plot                                                                                                 |
| `scripts/export_qualitative.py`       | Exports actual evidence probabilities, gates and selected regions                                                                                            |

## 2. Install

Python 3.10 or newer is recommended; Python 3.11 was used for the packaging checks.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

If using an NVIDIA GPU, install matching PyTorch and Torchaudio CUDA wheels from [the official PyTorch selector](https://pytorch.org/get-started/locally/) before installing remaining packages. PyTorch, Torchaudio and their CUDA versions must match.

## 3. Check the code immediately (demo)

The demo is already included: 20 training images, five validation images and five test images. Speech was generated locally with `espeak`. It is small enough for an ordinary CPU. From the repository root:

```bash
python scripts/validate_manifest.py data/demo/train.jsonl data/demo/val.jsonl data/demo/test.jsonl --require-audio
python scripts/audit_splits.py --train data/demo/train.jsonl --val data/demo/val.jsonl --test data/demo/test.jsonl
python -m prokg.train --train data/demo/train.jsonl --val data/demo/val.jsonl --vocab configs/vocab_demo.json --method full --epochs 1 --batch-size 4 --dim 32 --output checkpoints/demo_full.pt
python -m prokg.evaluate --checkpoint checkpoints/demo_full.pt --manifest data/demo/test.jsonl --output results/demo_metrics.json
python scripts/export_stagewise.py --checkpoint checkpoints/demo_full.pt --manifest data/demo/test.jsonl --output results/demo_stagewise.csv
python scripts/export_qualitative.py --checkpoint checkpoints/demo_full.pt --manifest data/demo/test.jsonl --index 0
python -m pytest -q
```

## 4. Dataset downloads and preparation

Download datasets **directly from their creators**. The exact, verified source pages and access notes are in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

**Visual Genome.** Download the scene-graph annotations, image metadata and image files from its official portal. Run:

```bash
python scripts/prepare_visual_genome.py \
  --scene-graphs data/raw/vg/scene_graphs.json.zip \
  --image-metadata data/raw/vg/image_data.json \
  --image-root data/raw/vg/images \
  --output-dir data/processed/vg
```

VG objects/attributes/edges become training labels. The converter generates supported statements and **candidate** conflicts: a different colour is _not necessarily_ contradicted for a multicoloured object. **Visually verify candidates before adding `"evidence": "contradicted"`.** Absence of an annotation must never be used to label a claim as false.

Generate audio from the reviewed statements, one split at a time, or use your recorded audio:

```bash
python scripts/synthesize_vg_tts.py \
  --input data/processed/vg/train.jsonl \
  --output data/processed/vg/train_audio.jsonl \
  --audio-dir data/processed/vg/audio/train
```

**SpokenCOCO.** Download the MIT spoken captions and separately download the matching MS COCO images and instance annotations. Create a CSV **from official audio–image pairing metadata** with columns:

`image_id,image_path,audio_path,transcript,split`

Do not infer audio–image pairs from filenames or free-form captions. The MS COCO annotation year must match the intended image IDs and experimental split. Then run:

```bash
python scripts/prepare_spokencoco.py \
  --pairs data/raw/spokencoco/pairs.csv \
  --instances data/raw/coco/instances_train2017.json \
  --output-dir data/processed/spokencoco
python scripts/audit_caption_leakage.py \
  --manifest data/processed/spokencoco/test.jsonl
```

**Spoken ObjectNet.** Download MIT's SON audio and separately obtain ObjectNet images. Use the original audio–image association metadata to create `image_id,image_path,audio_path,transcript`, then:

```bash
python scripts/prepare_spoken_objectnet.py \
  --pairs data/raw/son/pairs.csv \
  --output data/processed/son/test.jsonl
```

For **zero-shot target-domain** evaluation, no SON image/audio pairs may be used in any training, selection or fine-tuning stage. Follow ObjectNet's image-use instructions.

## 5. Train on real processed manifests

Combine only the relevant training and validation files; never mix test records:

```bash
python scripts/merge_manifests.py \
  --inputs data/processed/vg/train_audio.jsonl data/processed/spokencoco/train.jsonl \
  --output data/processed/joint_train.jsonl
python scripts/merge_manifests.py \
  --inputs data/processed/vg/val_audio.jsonl data/processed/spokencoco/val.jsonl \
  --output data/processed/joint_val.jsonl
python scripts/audit_splits.py \
  --train data/processed/joint_train.jsonl \
  --val data/processed/joint_val.jsonl \
  --test data/processed/vg/test_audio.jsonl data/processed/spokencoco/test.jsonl data/processed/son/test.jsonl
python scripts/build_vocab.py --train data/processed/joint_train.jsonl --output configs/vocab_real.json
python -m prokg.train \
  --train data/processed/joint_train.jsonl \
  --val data/processed/joint_val.jsonl \
  --vocab configs/vocab_real.json \
  --method full --epochs 30 --batch-size 4 \
  --max-audio-seconds 6 --output checkpoints/prokg_full.pt
```

For a **controlled baseline run**, train the six configurations with the same split and fixed seed:

```bash
python scripts/run_baselines.py \
  --train data/processed/joint_train.jsonl \
  --val data/processed/joint_val.jsonl \
  --test data/processed/vg/test_audio.jsonl \
  --vocab configs/vocab_real.json --epochs 30 --batch-size 4 --seed 42
```

## 6. Six comparison methods

| Flag                 | What the model does                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| `image_only`         | Images only; no spoken update                                                                          |
| `direct_fusion`      | One combined speech representation applied globally                                                    |
| `progressive_fusion` | Successive observations, no graph-based selection                                                      |
| `progressive_kg`     | Successive observations aligned to the predicted image graph                                           |
| `kg_refinement`      | Knowledge alignment and spatial refinement, with no reliability gate                                   |
| `full`               | Image-specific graph, progressive alignment, selected-region refinement, verification and gated update |

## 7. Metrics

- **Visual Genome:** attribute macro-F1 and relation macro-F1, reported separately.
- **SpokenCOCO:** multi-label mean average precision (mAP), over categories where both positive and negative test examples exist.
- **Spoken ObjectNet:** speech-to-image Recall@10 against a **deduplicated image gallery**, with image-level correspondence and no adaptation.
- **Additional:** per-observation evidence probabilities, alignment, gate values and stagewise predictions.
