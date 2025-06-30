from transformers import AutoTokenizer
import json
from datasets import Dataset, Features, Sequence, ClassLabel, Value
import os
from datasets import load_from_disk


def load_and_prepare_data(file_path):
    # โหลดข้อมูลจากไฟล์ JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # กำหนดรายการ label ทั้งหมด (BIO format)
    label_list = [
        "O",
        "B-PERSON", "I-PERSON",
        "B-COURSE", "I-COURSE",
        "B-DATE", "I-DATE",
        "B-ORG", "I-ORG",
        "B-URL", "I-URL"
    ]

    # สร้าง mapping ระหว่าง label และ ID
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    processed_data = []

    for item in data:
        text = item['text']
        # แบ่งข้อความเป็น character-level tokens
        tokens = list(text)

        # เริ่มต้นด้วย label "O" (non-entity) ทั้งหมด
        ner_tags = ["O"] * len(tokens)

        # กำหนด label ให้กับ entities
        for entity in item['entities']:
            start = entity['start']
            end = entity['end']
            entity_type = entity['label']

            # แปลง entity type ให้ตรงกับรูปแบบ BIO
            if entity_type == "PERSON":
                prefix = "PERSON"
            elif entity_type == "COURSE":
                prefix = "COURSE"
            elif entity_type == "DATE":
                prefix = "DATE"
            elif entity_type == "ISSUER":  # ISSUER จะถูกแปลงเป็น ORG
                prefix = "ORG"
            elif entity_type == "URL":
                prefix = "URL"
            else:
                continue  # ข้าม entity type อื่นๆ

            # ตั้งค่า B-tag สำหรับ token แรกของ entity
            ner_tags[start] = f"B-{prefix}"

            # ตั้งค่า I-tags สำหรับ token ที่เหลือของ entity
            for i in range(start + 1, end):
                ner_tags[i] = f"I-{prefix}"

        # แปลง labels เป็น IDs
        ner_ids = [label2id[tag] for tag in ner_tags]

        # เพิ่มข้อมูลที่ประมวลผลแล้ว
        processed_data.append({
            "id": str(len(processed_data)),
            "tokens": tokens,
            "ner_tags": ner_ids
        })

    # กำหนดโครงสร้างของ dataset
    features = Features({
        'id': Value('string'),
        'tokens': Sequence(Value('string')),
        'ner_tags': Sequence(ClassLabel(names=label_list))
    })

    # สร้าง dataset
    dataset = Dataset.from_list(processed_data, features=features)

    return dataset, label_list, id2label, label2id


# โหลดข้อมูล
file_path = "./datasets/train_data_label.json"
dataset, label_list, id2label, label2id = load_and_prepare_data(file_path)

# แสดงตัวอย่างข้อมูล
print(f"Total examples: {len(dataset)}")
print(f"Label list: {label_list}")

# # แสดงตัวอย่างที่ 1
# example_idx = 0
# example = dataset[example_idx]
# print("\nExample 1:")
# print(f"Text: {''.join(example['tokens'])}")
# print(f"Tokens: {example['tokens'][:30]}...")  # แสดง 30 tokens แรก
# # แสดง 30 tags แรก
# print(f"NER tags: {[id2label[id] for id in example['ner_tags'][:30]]}...")

# # แสดงตัวอย่างที่ 2
# example_idx = 1
# example = dataset[example_idx]
# print("\nExample 2:")
# print(f"Text: {''.join(example['tokens'])}")
# print(f"Tokens: {example['tokens'][:30]}...")
# print(f"NER tags: {[id2label[id] for id in example['ner_tags'][:30]]}...")

# # แสดงสถิติของ entities
# entity_counts = {label: 0 for label in label_list if label != "O"}
# for example in dataset:
#     for tag_id in example['ner_tags']:
#         label = id2label[tag_id]
#         if label != "O":
#             entity_counts[label] += 1

# print("\nEntity counts:")
# for label, count in entity_counts.items():
#     print(f"{label}: {count}")

########################  Tokenize และ Align Labels ################################


# แบ่งข้อมูลเป็น train/validation (80/20)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Train examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['test'])}")

# โหลด Tokenizer
model_checkpoint = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint,
    add_prefix_space=True  # จำเป็นสำหรับการทำงานกับภาษาไทย
)

# ฟังก์ชันสำหรับ tokenize และจัด alignment ของ labels


def tokenize_and_align_labels(examples):
    # Tokenize ข้อความ (ใช้ tokens ที่แบ่งเป็น character แล้ว)
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,  # ใช้ข้อมูลที่แบ่ง tokens แล้ว
        truncation=True,
        max_length=256,
        padding="max_length"
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # กำหนดเป็น -100 สำหรับ special tokens
            if word_idx is None:
                label_ids.append(-100)
            # ใช้ label เดิมถ้าเป็นคำใหม่
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # ถ้าเป็นส่วนเดียวกันของ entity
            else:
                # ใช้ I- tag สำหรับส่วนต่อของ entity
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# กำหนด path สำหรับบันทึกข้อมูล
TOKENIZED_DATA_DIR = "./tokenized_data"

# ฟังก์ชันเตรียมและบันทึกข้อมูล


def prepare_and_save_data(file_path):
    # โหลดและเตรียมข้อมูล
    dataset, label_list, id2label, label2id = load_and_prepare_data(file_path)

    # แบ่งข้อมูล
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # สร้างโฟลเดอร์หากไม่มี
    os.makedirs(TOKENIZED_DATA_DIR, exist_ok=True)

    # บันทึกข้อมูลดิบ (เพื่อใช้อ้างอิง)
    dataset.save_to_disk(os.path.join(TOKENIZED_DATA_DIR, "raw_dataset"))

    # โหลด Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "xlm-roberta-base",
        add_prefix_space=True
    )

    # Tokenize ข้อมูล
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # ตรวจสอบผลลัพธ์
    print("\nTokenized dataset structure:")
    print(tokenized_datasets)

    # แสดงตัวอย่าง tokenized data
    example_idx = 0
    example = tokenized_datasets["train"][example_idx]
    input_ids = example["input_ids"]
    attention_mask = example["attention_mask"]
    labels = example["labels"]

    # แปลงกลับเป็น tokens และ labels ที่อ่านได้
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    valid_labels = [id2label.get(l, "O") for l in labels if l != -100]
    valid_tokens = [t for t, l in zip(tokens, labels) if l != -100]

    print("\nExample tokenized input (first 70 tokens):")
    print(valid_tokens[30:70])
    print(valid_labels[30:70])

    # บันทึก tokenized data
    tokenized_datasets.save_to_disk(os.path.join(
        TOKENIZED_DATA_DIR, "tokenized_dataset"))

    # บันทึก metadata
    metadata = {
        "label_list": label_list,
        "id2label": id2label,
        "label2id": label2id
    }

    with open(os.path.join(TOKENIZED_DATA_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    # บันทึก tokenizer
    tokenizer.save_pretrained(TOKENIZED_DATA_DIR)

    return tokenized_datasets, label_list, id2label, label2id


# โหลดข้อมูล
file_path = "./datasets/train_data_label.json"
prepare_and_save_data(file_path)
