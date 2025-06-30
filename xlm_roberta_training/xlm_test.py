import os
import re
import json
import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer
)

# 1. ฟังก์ชันแบ่งคำ (word tokenization)


def split_to_words(text):
    """แบ่งข้อความเป็นคำ (รองรับทั้งภาษาไทยและอังกฤษ)"""
    # แบ่งคำไทย/อังกฤษ และเก็บเครื่องหมายพิเศษ
    return re.findall(r'[\wก-๙]+|[^\w\s]', text)

# 2. โหลดโมเดลและ tokenizer


def load_model(model_path):
    """โหลดโมเดลและ tokenizer"""
    # ตรวจสอบและเลือกอุปกรณ์
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'} for inference")

    # โหลดโมเดลและ tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path).to(device)

    return model, tokenizer, device

# 3. ทำนาย entities


def predict_entities(text, model, tokenizer, device):
    """ทำนาย entities ด้วยโมเดล"""
    # แบ่งข้อความเป็นคำ
    words = split_to_words(text)

    # Tokenize และจัดเตรียมข้อมูล
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # ทำนาย
    with torch.no_grad():
        outputs = model(**encoding)

    # ดึง predictions
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # แปลง predictions เป็น labels
    label_ids = predictions[0]  # ใช้ตัวอย่างแรก (batch size=1)
    tokens = encoding.tokens()
    word_ids = encoding.word_ids()

    # แปลง label IDs เป็น label names
    id2label = model.config.id2label
    labels = [id2label[id] for id in label_ids]

    # รวบรวมผลลัพธ์
    current_word = None
    current_label = None
    start_idx = None
    end_idx = None
    entities = []

    # เก็บตำแหน่งของคำในข้อความเดิม
    word_positions = []
    current_position = 0
    for word in words:
        start = text.find(word, current_position)
        end = start + len(word)
        word_positions.append((start, end))
        current_position = end

    # ประมวลผลผลลัพธ์
    for i, (token, label, word_id) in enumerate(zip(tokens, labels, word_ids)):
        # ข้าม special tokens
        if word_id is None:
            continue

        # ดึงตำแหน่งของคำ
        word_start, word_end = word_positions[word_id]

        # เริ่ม entity ใหม่
        if label.startswith("B-"):
            # บันทึก entity ก่อนหน้า
            if current_label:
                entities.append({
                    "text": text[start_idx:end_idx],
                    "label": current_label,
                    "start": start_idx,
                    "end": end_idx
                })

            # เริ่ม entity ใหม่
            current_label = label[2:]
            start_idx = word_start
            end_idx = word_end

        # ต่อ entity เดิม
        elif label.startswith("I-") and current_label == label[2:]:
            end_idx = word_end

        # ไม่ใช่ entity
        else:
            # บันทึก entity ก่อนหน้า
            if current_label:
                entities.append({
                    "text": text[start_idx:end_idx],
                    "label": current_label,
                    "start": start_idx,
                    "end": end_idx
                })
                current_label = None

    # บันทึก entity สุดท้าย
    if current_label:
        entities.append({
            "text": text[start_idx:end_idx],
            "label": current_label,
            "start": start_idx,
            "end": end_idx
        })

    return entities

# 4. ฟังก์ชันแสดงผลลัพธ์


def print_results(text, entities):
    """แสดงผลลัพธ์อย่างสวยงาม"""
    print("=" * 80)
    print("Input Text:")
    print(text)
    print("\n" + "-" * 80)
    print("Extracted Entities:")

    if not entities:
        print("No entities found")
        return

    for ent in entities:
        print(f"- {ent['text']} ({ent['label']}) [{ent['start']}:{ent['end']}]")

    # สร้างข้อความไฮไลท์ entities
    highlighted = list(text)
    for ent in entities:
        start, end = ent['start'], ent['end']
        # หลีกเลี่ยง index out of range
        if start < len(highlighted):
            highlighted[start] = f"【{highlighted[start]}"
        if end-1 < len(highlighted):
            highlighted[end-1] = f"{highlighted[end-1]}】"

    print("\n" + "-" * 80)
    print("Highlighted Text:")
    print(''.join(highlighted))


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # โหลดโมเดล
    model_path = os.path.join("..", "trained_xlm_roberta_ner")
    model, tokenizer, device = load_model(model_path)

    # ตัวอย่างข้อความ (5 ตัวอย่าง)
    sample_texts = [
        # ตัวอย่างที่ 1
        "OFFICIAL CERTIFICATE\n\nThis document certifies that\nสุรนัย เนื่องนนท์\nhas successfully completed the training program\n'Sharable clear-thinking groupware'\non 2547-05-16.\n\nIssued by: ห้างหุ้นส่วนจำกัด นานายนเซอร์วิส\nAuthorized by: ใกล้รุ่ง ทองอยู่, President\nCertificate ID: kIV5490\n\nVerify at: https://www.haanghunswncchamkad.org/",

        # ตัวอย่างที่ 2
        "CERTIFICATE OF COMPLETION\n\nAwarded to: จิราพร ศรีสุข\nFor completing the course: Advanced Data Analytics\nDate: 2023-08-15\nOrganization: บริษัท เทคโนโลยีไทย จำกัด\n\nCertificate ID: XYZ789\nVerify: http://verify-techthai.com/cert/XYZ789",

        # ตัวอย่างที่ 3
        "ประกาศนียบัตร\n\nชื่อผู้รับ: วีรวัฒน์ อินทร์เทพ\nหลักสูตร: การเขียนโปรแกรม Python ขั้นสูง\nวันที่: 10 มกราคม 2566\nออกโดย: สถาบันพัฒนาทักษะดิจิทัล\n\nตรวจสอบได้ที่: https://digital-skills-institute.org/cert/12345",

        # ตัวอย่างที่ 4
        "DIPLOMA\n\nRecipient: Somchai Jaidee\nCourse: Machine Learning Fundamentals\nDate: 2024-02-28\nIssuer: Thai AI Academy\n\nID: ML-2024-0028\nURL: https://thai-ai-academy.edu/diploma/ML-2024-0028",

        # ตัวอย่างที่ 5 (ง่ายขึ้น)
        "ใบรับรอง\n\nชื่อ: สมชาย ใจดี\nคอร์ส: การใช้ AI ขั้นพื้นฐาน\nวันที่: 1 มกราคม 2567"
    ]

    # ทำนายและแสดงผล
    for i, text in enumerate(sample_texts, 1):
        print(f"\n\n{'#' * 30} EXAMPLE {i} {'#' * 30}")
        entities = predict_entities(text, model, tokenizer, device)
        print_results(text, entities)

    # บันทึกผลลัพธ์ทั้งหมดเป็น JSON
    all_results = []
    for text in sample_texts:
        entities = predict_entities(text, model, tokenizer, device)
        all_results.append({
            "text": text,
            "entities": entities
        })

    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nPredictions saved to predictions.json")
