import json
from datasets import Dataset, Features, Sequence, ClassLabel, Value


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

# แสดงตัวอย่างที่ 1
example_idx = 0
example = dataset[example_idx]
print("\nExample 1:")
print(f"Text: {''.join(example['tokens'])}")
print(f"Tokens: {example['tokens'][:30]}...")  # แสดง 30 tokens แรก
# แสดง 30 tags แรก
print(f"NER tags: {[id2label[id] for id in example['ner_tags'][:30]]}...")

# แสดงตัวอย่างที่ 2
example_idx = 1
example = dataset[example_idx]
print("\nExample 2:")
print(f"Text: {''.join(example['tokens'])}")
print(f"Tokens: {example['tokens'][:30]}...")
print(f"NER tags: {[id2label[id] for id in example['ner_tags'][:30]]}...")

# แสดงสถิติของ entities
entity_counts = {label: 0 for label in label_list if label != "O"}
for example in dataset:
    for tag_id in example['ner_tags']:
        label = id2label[tag_id]
        if label != "O":
            entity_counts[label] += 1

print("\nEntity counts:")
for label, count in entity_counts.items():
    print(f"{label}: {count}")
