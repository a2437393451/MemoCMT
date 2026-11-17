# Beyond Explicit Cues: Uncertainty-Aware Rationale Learning for Multimodal Emotion Recognition

URM-ER classifies emotions from speech, transcripts, and text-derived rationales. It uses frozen HuBERT and BERT encoders, estimates the reliability of each input branch, and uses those estimates in fusion and supervised contrastive learning. The audio-text baseline is based on MemoCMT. See `LICENSE` and the source headers for upstream notices.

## Environment

Run the following commands from the code directory. The repository's dependency specifications target Python 3.8 and PyTorch 2.0.1. The example uses CUDA 11.8.

```bash
conda create -n urmer python=3.8 -y
conda activate urmer
conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install -r requirements.txt
```

Make the pretrained BERT Base Uncased and HuBERT Base weights available to the configured encoders. Check the inherited configuration values if you use a local model cache.

## Data and rationales

Obtain IEMOCAP and ESD from their dataset providers. The paper uses the four shared categories of anger, happiness, sadness, and neutral. IEMOCAP merges excited into happiness.

| Dataset | Five-fold protocol | Train / validation / test per fold |
| --- | --- | --- |
| IEMOCAP | Session-independent | 3 / 1 / 1 sessions |
| ESD | Speaker-independent, ten English speakers | 6 / 2 / 2 speakers |

For each fold, prepare `train.pkl`, `val.pkl`, and `test.pkl` using the exact session or speaker assignments from the experiment. Preserve the fixed validation rotation and keep test data out of model selection.

The prompts in `prompts/` describe transcript-only rationale generation. The decoding settings are temperature 0.2 and top-p 0.9. The annotation utility accepts a JSON object mapping transcript strings to rationale strings, or a list of objects with `text` and `llm_explanation` fields. It matches by transcript text. Check for missing rationales and conflicting annotations for identical transcript strings before training.

```bash
python scripts/prepare_llm_annotations.py \
  --input_dir data/iemocap/fold1 \
  --output_dir data/iemocap/fold1_annotated \
  --annotation_json data/iemocap/rationales.json
```

Repeat for each fold and dataset. The annotated directory contains the three split pickles consumed at training and evaluation time.

## Training and evaluation

| Dataset | Full-model configuration |
| --- | --- |
| IEMOCAP | `src/configs/hubert_llm_racmt_contrastive_v3.py` |
| ESD | `src/configs/hubert_llm_racmt_contrastive_v3_esd.py` |

Train one run per fold. The path overrides below replace the machine-specific defaults in the configuration.

```bash
python scripts/train.py \
  --config src/configs/hubert_llm_racmt_contrastive_v3.py \
  --override data_root=data/iemocap/fold1_annotated \
  --override checkpoint_dir=checkpoints/iemocap/fold1
```

Replace `<run-directory>` with the actual training output directory containing `cfg.log` and the checkpoints. Evaluation uses the dataset path saved in the run configuration.

```bash
python scripts/eval.py --checkpoint_path "<run-directory>" --test_set test.pkl
```

The evaluator reports balanced accuracy (BACC or UW-Acc), overall accuracy (ACC or W-Acc), Macro-F1, and weighted F1. Evaluate the checkpoint selected on the validation split and summarize the five held-out runs using the paper's aggregation protocol.

## Code layout

| Path | Purpose |
| --- | --- |
| `src/models/` | Model architectures and losses |
| `src/configs/` | Dataset and model configurations |
| `src/data/` | Dataset loading and batch construction |
| `scripts/train.py`, `scripts/eval.py` | Training and held-out evaluation |
| `scripts/prepare_llm_annotations.py`, `prompts/` | Rationale inputs and annotation prompts |
| `scripts/prepare_section_4_8_controls.py`, `scripts/train_section_4_8.py` | Rationale controls, including masking, shuffling, and single-input conditions |

Code will be made publicly available.
