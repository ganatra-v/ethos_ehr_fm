# ethos_ehr_fm
A repository for simulating patient health timelines using tranformers - we implement an [ETHOS](https://www.nature.com/articles/s41746-024-01235-0) style model for Electronic Health Records. We experiment with two versions - 
1. A small decoder-only transformer is pretrained on tokenized patient timelines (next-token prediction) from the Synthea EHR records, then fine-tuned to predict which conditions will be **newly diagnosed in the five years after a patient's anchor date**, using only events before the anchor.

2. A small BERT encoder trained **jointly** from random initialization on two objectives at once: masked language modelling (MLM) over the timeline tokens, and multi-label classification of the same future-condition targets from a `<CLS>` summary token. There is no separate pretraining stage.

## Setup Instructions

From the project root:

```
python -m venv src/venv
source ./venv/bin/activate
pip install torch transformers pandas numpy scikit-learn tqdm matplotlib umap-learn wandb
```

Choose the appropriate version of `torch` from [here](https://pytorch.org/get-started/locally/). `wandb` is only needed if you want experiment tracking (every training script has `--no-wandb`). Log in once with `wandb login`.

Expected data layout in the data directory:

```
data/patient_splits.csv      # Id -> split (train | val | test)
data/target_conditions.csv   # CODE, DESCRIPTION of the scored conditions
data/test_anchors.csv        # Id, anchor_date for test patients
data/train_val/              # full CSV tables for train + val patients
data/test/                   # CSV tables for test patients, already truncated at the anchor
```

## Running Scripts

Run these in order:

```powershell
python tokenizer.py --data-dir <PATH to data/train_val> --output-dir <OUT>   
# to build the vocabulary from train patients only. The default implementation is for GPT-2; the BERT version includes <CLS> and <MASK> tokens which can be specified with the --include-cls flag


python pht.py --data-dir <PATH to data/train_val> --tokenizer-path <PATH to tokenizer.pkl>
# inspect a tokenized timelines for patients

python _prep_data.py --data-dir <PATH to data/train_val> --test-dir <PATH to data/test> --tokenizer-path <PATH to tokenizer.pkl> --splits-path <PATH to patient_splits.csv> --targets-path <PATH to target_conditions.csv> --test-anchors-path <PATH to test_anchors.csv> --output-path <PATH to DOWNSTREAM.pkl file>
# build downstream dataset (timelines + future labels for target conditions) -> ../downstream.pkl

## to train GPT2 variants 

python pretrain_gpt2.py --data-dir <PATH to data/train_val> --tokenizer-path <PATH to tokenizer.pkl> --splits-path <PATH to patient_splits.csv> --cache-path <PATH to cache.pkl file> --rebuild-cache --out-dir <OUT>
# runs pretraining via next-token prediction -- cache-path is the path where the pipeline stores the patient timelines: this is implemented separately for pretraining in order to use more timelines for pretraining than for the downstream finetuning.
# pretraining results in checkpoints output:
1. OUT/best/ -- # best performing model on validation data
2. OUT/last/ -- # final checkpoint from pretraining

python finetune_gpt2.py --data-path <PATH to DOWNSTREAM.pkl file> --pretrained <PATH to /best or /last checkpoint from pretraining> --out-dir <OUT> --pos-weight
# finetune the model for predicting future conditions


## to train BERT variants
python train_bert.py --mode joint --data-path  <PATH to DOWNSTREAM.pkl> --tokenizer-path <PATH to tokenizer.pkl> --out-dir <OUT>
# make sure the tokenizer for BERT is generated with --include-cls in the tokenizer.py script.
```
For all training runs, either specificy --wandb-project <> --wandb-run-name <> if you want to track your training with wandb or --no-wandb

If you only want to run inference and reproduce the results, run the following commands:
```powershell
python tokenizer.py --data-dir <PATH to data/train_val> --output-dir <OUT>   
# to build the vocabulary from train patients only. The default implementation is for GPT-2

python _prep_data.py --data-dir <PATH to data/train_val> --test-dir <PATH to data/test> --tokenizer-path <PATH to tokenizer.pkl> --splits-path <PATH to patient_splits.csv> --targets-path <PATH to target_conditions.csv> --test-anchors-path <PATH to test_anchors.csv> --output-path <PATH to DOWNSTREAM.pkl file>
# build downstream dataset (timelines + future labels for target conditions) -> ../downstream.pkl

python infer.py --data-path <PATH to DOWNSTREAM.pkl file> --out-dir <OUT> --split <val/test>
# this uses the models available on huggingface to reproduce the reported results - running on val computes the results as well, test only produces predictions since labels are hidden.
```

## Results

Validation set metrics, macro-averaged across the target conditions.

| Description | Model | AUROC | mAP |
|---|---|---|---|
| w/o pretraining | GPT-2 | 0.717 | 0.149 |
| w/ pretraining | GPT-2 | 0.736 | 0.210 |
| Joint Training | BERT | 0.732 | 0.212 |
| 2 heads - w/ pretraining | GPT-2 | 0.747 | 0.217

Notes:
- AUROC and mAP are threshold-free.
- Test labels are withheld, so metrics come from the validation set.
- individual condition level performance can be found at `<OUT>/val_per_condition.csv`
