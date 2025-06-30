# Certificate NER Project

This repository contains utilities for training a NER model to extract
information from certificates such as the recipient name, course name,
date, issuer and verification URL.

## Generating Synthetic Training Data

Use `xlm_roberta_training/synthetic_data_label.py` to create labelled
training examples. The script now includes additional layouts which
produce text that is closer to typical certificate wording in Thai and
English. Run the script to regenerate the dataset:

```bash
python xlm_roberta_training/synthetic_data_label.py
```

The command produces three files in the `datasets` folder:
`train_data_label.json`, `val_data_label.json` and `test_data_label.json`.

## Validating Datasets

A helper script `validate_dataset.py` verifies that the labelled entity
spans match the text in each dataset. Run it on any dataset file:

```bash
python validate_dataset.py datasets/train_data_label.json
```

The script reports any mismatched entities.
