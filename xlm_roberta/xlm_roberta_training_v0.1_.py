import os
import json
from seqeval.metrics import classification_report
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification, AutoTokenizer
)
from datasets import load_from_disk

# กำหนด path
TOKENIZED_DATA_DIR = "./tokenized_data"

# โหลดข้อมูลที่บันทึกไว้


def load_tokenized_data():
    # โหลด tokenized dataset
    tokenized_datasets = load_from_disk(
        os.path.join(TOKENIZED_DATA_DIR, "tokenized_dataset"))

    # โหลด metadata
    with open(os.path.join(TOKENIZED_DATA_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    label_list = metadata["label_list"]
    # แปลง key เป็น int
    id2label = {int(k): v for k, v in metadata["id2label"].items()}
    label2id = metadata["label2id"]

    # โหลด tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZED_DATA_DIR)

    return tokenized_datasets, label_list, id2label, label2id, tokenizer


# โหลดข้อมูล
tokenized_datasets, label_list, id2label, label2id, tokenizer = load_tokenized_data()


######################### โหลดโมเดลและตั้งค่าการฝึก ################################


# โหลดโมเดล
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# ตั้งค่าการฝึก
training_args = TrainingArguments(
    output_dir="xlm-roberta-certificate-ner",
    eval_strategy="epoch",  # ประเมินทุก epoch
    learning_rate=2e-5,           # อัตราการเรียนรู้ที่เหมาะสมสำหรับ fine-tuning
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,          # จำนวน epoch
    weight_decay=0.01,            # regularization
    save_strategy="epoch",         # บันทึกโมเดลทุก epoch
    load_best_model_at_end=True,   # โหลดโมเดลที่ดีที่สุดเมื่อจบการฝึก
    metric_for_best_model="f1",    # ใช้ f1 เป็นตัวตัดสินโมเดลที่ดีที่สุด
    greater_is_better=True,
    fp16=True,                    # ใช้ mixed precision training
    logging_dir='./logs',
    report_to="none",             # ไม่รายงานไปยังบริการภายนอก
    logging_steps=50,             # บันทึก log ทุก 50 steps
    save_total_limit=3            # เก็บ checkpoint สุดท้าย 3 อันเท่านั้น
)

# ตั้งค่า Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# ฟังก์ชันประเมินผล (แก้ไขให้ปลอดภัยจาก KeyError)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # เอาเฉพาะตำแหน่งที่ไม่ใช่ -100
    true_predictions = []
    true_labels = []

    for i in range(len(predictions)):
        preds = []
        lbls = []
        for j in range(len(predictions[i])):
            if labels[i][j] != -100:
                preds.append(label_list[predictions[i][j]])
                lbls.append(label_list[labels[i][j]])
        true_predictions.append(preds)
        true_labels.append(lbls)

    # ถ้าไม่มีข้อมูลให้คืนค่า default
    if len(true_labels) == 0 or all(len(x) == 0 for x in true_labels):
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0
        }

    # คำนวณ metrics
    results = classification_report(
        true_labels, true_predictions,
        output_dict=True,
        zero_division=0  # ตั้งค่าเป็น 0 เมื่อหารด้วยศูนย์
    )

    # ตรวจสอบค่าต่างๆ ก่อนคืนค่า
    return {
        "precision": results.get('micro avg', {}).get("precision", 0.0),
        "recall": results.get('micro avg', {}).get("recall", 0.0),
        "f1": results.get('micro avg', {}).get("f1-score", 0.0),
        "accuracy": results.get('accuracy', 0.0)
    }


# สร้าง Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# เริ่มการฝึก
print("\nStarting training...")
trainer.train()

# บันทึกโมเดล
print("\nSaving model...")
model.save_pretrained("trained_xlm_roberta_ner")
tokenizer.save_pretrained("trained_xlm_roberta_ner")

# ประเมินผลโมเดล
print("\nEvaluating model...")
results = trainer.evaluate()
print(f"Final evaluation results: {results}")
