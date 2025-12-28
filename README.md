# NLP Course Project — Chinese–English Neural Machine Translation

This repository contains my course project for Neural Machine Translation (NMT).
I implemented and compared three families of models:

- RNN-based seq2seq (2-layer GRU) with attention, training strategy, decoding, and data scaling
- Transformer (from scratch) with ablations, hyperparameter sensitivity, and scaling
- Pretrained model fine-tuning using T5-small

All experiments are evaluated with SacreBLEU.

---

## Repository Structure
.
├── prepare_data.py
├── dataset.py
├── train_rnn.py
├── model_rnn.py
├── decode_rnn.py
├── decode_rnn_beam.py
├── train_transformer.py
├── train_transformer_patched.py
├── model_transformer.py
├── decode_transformer.py
├── model_transformer_abs_ln.py
├── model_transformer_abs_rms.py
├── model_transformer_rel_ln.py
├── model_transformer_rel_rms.py
├── train_t5.py
├── decode_t5.py
├── Data/
│   ├── raw/        # train_10k.jsonl, train_100k.jsonl, valid.jsonl, test.jsonl
│   └── processed/  # SentencePiece model + *.pt files
└── runs/           # checkpoints & logs

---

## Environment

Recommended environment:

- Python 3.9+
- PyTorch 2.x
- transformers
- sentencepiece
- sacrebleu
- tqdm

Install dependencies: pip install torch transformers sentencepiece sacrebleu tqdm

## Dataset Format

Raw data is stored in JSONL format. Each line:

{ "zh": "...", "en": "..." }

Files used: train_10k.jsonl、train_100k.jsonl、valid.jsonl、test.jsonl


## Data Preprocessing (Unified Pipeline)
All models share the same preprocessing pipeline implemented in prepare_data.py.

Steps:

Train a shared SentencePiece Unigram tokenizer (vocab size = 16k)

Tokenize Chinese and English sentences

Truncate to max length = 100

Serialize splits into PyTorch .pt files

Run preprocessing:

python prepare_data.py \
  --train_jsonl Data/raw/train_10k.jsonl \
  --valid_jsonl Data/raw/valid.jsonl \
  --test_jsonl  Data/raw/test.jsonl \
  --out_dir     Data/processed/spm16k


## RNN-based NMT (GRU)
Baseline (10k, no attention)
Train:
python train_rnn.py \
  --data_dir Data/processed/spm16k \
  --train_pt train_10k.pt \
  --valid_pt valid.pt \
  --run_dir runs/rnn_gru_baseline \
  --emb 256 --hid 512 --layers 2 --dropout 0.2 \
  --batch 32 --lr 1e-3 --epochs 10 \
  --tf_ratio 1.0

Decode:

python decode_rnn.py \
  --run_dir runs/rnn_gru_baseline \
  --ckpt best.pt \
  --data_dir Data/processed/spm16k \
  --split test

Results:
Valid BLEU: 0.40
Test BLEU: 0.46

## All BLEU scores are computed using sacreBLEU with default settings.

Attention Ablation (10k, greedy)
Attention	Valid BLEU	Test BLEU
None	       0.40	      0.46
Bahdanau	   0.55	      0.43
Dot	           0.22	      0.29
General	       0.39	      0.35

Training Strategy (Teacher Forcing)
TF Ratio	Valid BLEU	Test BLEU
1.0	           0.55	      0.43
0.0	           0.06	      0.13

Decoding Strategy (Bahdanau, 10k)
Decoding	       Valid BLEU	Test BLEU
Greedy	              0.55	       0.43
Beam (k=5, α=0.6)	  0.82	       0.50

Scaling to 100k
Decoding	      Valid BLEU	Test BLEU
Greedy	             3.46	       2.13
Beam (k=5, α=0.6)	 3.97	       2.49

Transformer (from scratch)
Baseline (10k)
Setting	              Valid BLEU	Test BLEU
Greedy (10 epochs)	      0.24	       0.39
Beam-3 (α=0.6)	          0.30	        —

Baseline (10k, 20 epochs)
Decoding	Valid BLEU	Test BLEU
Greedy	        0.21	  0.30
Beam-3	        0.27	  0.37

Scaling to 100k
Decoding	   Valid BLEU	Test BLEU
Greedy	          10.14	       7.39
Beam-3 (α=0.6)	  11.09	       7.40

Transformer Ablation (PosEnc × Norm, 100k)
Greedy
PosEnc	Norm	          Valid	    Test
Absolute	LayerNorm	   9.43	    6.46
Absolute	RMSNorm	       7.61	    5.01
RoPE	LayerNorm	       9.04	    6.65
RoPE	RMSNorm	           9.06	    7.27

Beam-3 (α=0.6)
PosEnc	Norm	        Valid	Test
Absolute	LayerNorm	10.06	7.64
Absolute	RMSNorm	     8.25	5.51
RoPE	LayerNorm	     9.42	7.38
RoPE	RMSNorm	         9.59	7.05

Hyperparameter Sensitivity (100k, Abs + LayerNorm)
Batch Size
Batch	lr_factor	Valid	Test
32	       1.0	     0.09	0.08
32	       0.5	     8.04	6.11
64	       1.0	     10.14	7.39
128	       1.0	     9.48	7.65

Learning Rate Scaling (batch=64)
lr_factor	Valid	Test
1.0	        10.14	7.39
0.5	         9.97	7.66

Model Scale
Model	Valid (Greedy)	Test (Greedy)	Test (Beam-3)
Base	10.14	            7.39	        7.40
Small	9.85	            7.18	        7.51

Pretrained Fine-tuning (T5-small, 100k)
Train:

python train_t5.py \
  --train_file Data/raw/train_100k.jsonl \
  --valid_file Data/raw/valid.jsonl \
  --run_dir runs/t5_small_100k \
  --model_name t5-small \
  --batch 32 --epochs 5 --lr 3e-4

Results:

Split	Greedy (beam=1)	Beam-3
Valid	    29.53	    15.02
Test	     8.91	    14.47

## Quick Inference (TA-friendly)

To quickly train and evaluate a trained model, use the corresponding decode scripts:

- RNN:
  python decode_rnn.py --run_dir <RUN_DIR> --ckpt best.pt --split test

- Transformer:
  python decode_transformer.py --run_dir <RUN_DIR> --split test --beam 3

- T5:
  python decode_t5.py --run_dir <RUN_DIR> --data_file Data/raw/test.jsonl --beam 3

## Notes on Checkpoints and Data

Due to GitHub size limits, trained checkpoints under `runs/` and processed data under `Data/processed/` are not included. All results reported in this README can be reproduced by running the provided training commands.

## Summary
RNN models benefit from attention and beam search but remain limited at scale.

Transformer models are highly sensitive to data size and optimization hyperparameters.

Pretrained T5 fine-tuning provides the strongest performance in this project.