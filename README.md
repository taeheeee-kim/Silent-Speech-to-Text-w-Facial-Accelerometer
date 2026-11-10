# Silent Speech-to-Text Using Facial Accelerometers

Code for viseme-based silent speech recognition from facial accelerometer signals.
A 2D CNN + bidirectional LSTM model is trained with CTC loss to predict viseme sequences.
Evaluation is leave-one-subject-out (LOSO): the model is pretrained on 9 subjects and fine-tuned on the held-out target subject.



## Repository structure

```
.
├─ sweep_config.py                        # subjects and LR grid (single source for all scripts)
├─ main_iteration_LOSO_ft.py              # runs the full LR sweep
├─ main_I2T_leave_one_out_finetune.py     # LOSO pretrain + fine-tuning (one LR setting)
├─ train.py                               # CTC forward, decoding, result saving
├─ utils.py                               # datasets, augmentation, collate, LR scheduler
├─ model_2d_crnn.py                       # IMU_2D_to_RNN (2D CNN + BiLSTM)
├─ per.py                                 # phoneme error rate
├─ ft_inference_seen_sentence.py          # word-in-sentence inference with fine-tuned models
├─ ft_sentence_inference_iteration.py     # runs sentence inference over the sweep
├─ ft_sentence_inference_result_check_iteration.py   # converts sentence results to CSV
├─ ft_word_inference_result_check_iteration.py       # converts word test results to CSV
├─ ft_summary_metrics_from_converted_csv_3fold.py    # accuracy / PER summary tables
├─ result/
│  ├─ ft_result_eng_viseme_convert_index_with_nbest_seen.py
│  └─ ft_result_eng_viseme_convert_index_with_nbest_seen_sentence.py
├─ preprocessing/  (MATLAB)
│  ├─ S00_10ch_IMU_Preprocessing_new_norm.m            # word training data
│  ├─ S00_10ch_IMU_Preprocessing_Seendata_new_norm.m   # word test data
│  ├─ S00_10ch_IMU_sentence_preprocessing.m            # word on/offsets in sentences
│  ├─ MakingTriggers.m                                 # collects on/offsets into Triggers.mat
│  └─ EpochingSentence_10ch_new_norm.m                 # sentence data
├─ utils/
│  ├─ eng_viseme_vocab_npho15.json
│  └─ ver3/600_word_TIMIT_mono.csv
└─ data/feat/sub{X}/   # not included, see "Data"
```

All scripts (Python and MATLAB) use paths relative to the repository root. **Run every script with the repository root as the current folder.**

## Requirements

- Python 3.9
- PyTorch 2.1.1 and torchaudio 2.1.1 with CUDA 11.8
- MATLAB (preprocessing only)

```bash
conda create -n ssi python=3.9.18
conda activate ssi
pip install torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

`flashlight-text` is required by `torchaudio.models.decoder.ctc_decoder` (beam search decoding).

### Environment used for the paper

| | |
|---|---|
| OS | Windows 10 |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| NVIDIA driver | 610.88 |
| Python | 3.9.18 |
| PyTorch / torchaudio | 2.1.1+cu118 |
| CUDA (PyTorch build) | 11.8 |
| cuDNN (PyTorch build) | 8.7.0 |

## Data

Subjects: `10, 12, 13, 14, 17, 18, 21, 22, 23, 24`.

Each subject needs three files in `data/feat/sub{X}/`:

| File | Used for |
|---|---|
| `sig_flip_padd_trial_soft_zscore.mat` | pretraining and fine-tuning (600 words) |
| `sig_flip_padd_seen_1_trial_soft_zscore.mat` | word test during fine-tuning |
| `sub{X}_sentence_trial_soft_zscore.mat` | word-in-sentence inference |

Download: _TBD (link)_

### Preprocessing

The `.mat` files are produced by the MATLAB scripts with `preprocess_mode = "trial_soft_zscore"`.
Each trial is normalized per channel:

```
x_out = (x - mean(x)) / sqrt(std(x)^2 + sig_floor^2),   sig_floor = 0.01
```

For sentence data, the statistics are computed only on the valid (non-padded) samples (`use_valid_stats_only = true`).

The scripts read raw recordings from `data/orig/sub{X}/` and write features to `data/feat/sub{X}/`.
Set the subject number at the top of each script (`sub_num`, `SUB_ID`, `sub_id` or `i_sub`).
Then run the script in MATLAB with the repository root as the current folder:

```matlab
run('preprocessing/S00_10ch_IMU_Preprocessing_new_norm.m')
```

| Step | Script | Output |
|---|---|---|
| 1 | `S00_10ch_IMU_Preprocessing_new_norm.m` | `sig_flip_padd_trial_soft_zscore.mat` |
| 2 | `S00_10ch_IMU_Preprocessing_Seendata_new_norm.m` | `sig_flip_padd_seen_1_trial_soft_zscore.mat` |
| 3 | `S00_10ch_IMU_sentence_preprocessing.m` | `data/preproc/sub{X}/sentence/IMU/session_1_multi_pick/sentence*_onoff.mat` |
| 4 | `MakingTriggers.m` | `data/orig/sub{X}/Triggers.mat` (word onsets/offsets) |
| 5 | `EpochingSentence_10ch_new_norm.m` | `sub{X}_sentence_trial_soft_zscore.mat` |

## Usage

### 1. Train (pretrain + fine-tuning)

A single LR setting (the defaults are pretrain LR 1e-4 / 8e-4, fine-tuning LR 3e-4 / 1e-3, batch size 128):

```bash
python main_I2T_leave_one_out_finetune.py
```

Other settings:

```bash
python main_I2T_leave_one_out_finetune.py \
  --pretrain_min_lr 0.0001 --pretrain_max_lr 0.0008 \
  --save_folder_name P2_f10_bs128_lr_0.0001_0.0008_c4 \
  --finetune_min_lr 0.0003 --finetune_max_lr 0.001
```

The script loops over all 10 target subjects. For each target, it pretrains on the other 9 subjects (500 epochs).
It then fine-tunes on the target subject (300 epochs) and saves a checkpoint and a word test result every 20 epochs.
If `G_0500` already exists in the pretrain folder, pretraining is skipped.

The full grid from the paper (11 pretrain × 11 fine-tuning LR settings, defined in `sweep_config.py`):

```bash
python main_iteration_LOSO_ft.py
```

Output layout:

```
result/P2_f10_bs128_lr_{pre_min}_{pre_max}_c4/leave_out_sub{X}/
├─ pretrain_except_sub{X}_train_sub.../save_model/G_0100 ... G_0500
└─ ft_a_{ft_min}_{ft_max}/
   ├─ save_model/sub{X}/CV0/G_0020 ... G_0300
   └─ save_mat/sub{X}/CV0/test0.mat ... test300.mat
```

### 2. Sentence inference

```bash
python ft_sentence_inference_iteration.py
```

One run only:

```bash
python ft_inference_seen_sentence.py --sub_num 10 \
  --model_folder_name ./result/P2_f10_bs128_lr_0.0001_0.0008_c4 \
  --finetune_exp_name ft_a_0.0003_0.001 \
  --data_mat_override sub10_sentence_trial_soft_zscore.mat
```

### 3. Convert results to CSV

```bash
python ft_sentence_inference_result_check_iteration.py   # sentence
python ft_word_inference_result_check_iteration.py       # word test
```

### 4. Summary tables

```bash
python ft_summary_metrics_from_converted_csv_3fold.py --eval_type sentence
python ft_summary_metrics_from_converted_csv_3fold.py --eval_type word
```

The tables are written to `_ft_summary_tables/{eval_type}/`.
For sentences, the 192 words are split into three sets (set1: words 1–66, set2: 67–130, set3: 131–192), and results are reported per set and per pair of sets.
Use `--subs`, `--folders` and `--finetune_exp_names` to summarize a subset.

## Citation

_TBD_
