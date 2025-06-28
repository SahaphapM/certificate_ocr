import random
import faker
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ตั้งค่า Faker สำหรับภาษาไทย
fake = faker.Faker("th_TH")

# ========================
# ส่วนที่ 1: สร้างข้อมูลพื้นฐาน
# ========================


def generate_base_data():
    """สร้างข้อมูลพื้นฐานสำหรับใบรับรอง"""
    company = fake.company()
    company_id = fake.bothify(text='???####')
    recipient = fake.name()

    # สร้างชื่อคอร์สทั้งไทยและอังกฤษ
    course_en = fake.catch_phrase()
    course_th = fake.bs()

    # สร้างวันที่ทั้งสองรูปแบบ
    date_th = fake.date(pattern='%d/%m/%Y')
    date_en = fake.date(pattern='%Y-%m-%d')

    director = fake.name()
    title_th = random.choice(
        ['ผู้อำนวยการ', 'ผู้จัดการทั่วไป', 'ประธานกรรมการ'])
    title_en = random.choice(['Director', 'CEO', 'President'])
    url = fake.url()

    return {
        'company': company,
        'company_id': company_id,
        'recipient': recipient,
        'course_en': course_en,
        'course_th': course_th,
        'date_th': date_th,
        'date_en': date_en,
        'director': director,
        'title_th': title_th,
        'title_en': title_en,
        'url': url
    }

# ========================
# ส่วนที่ 2: กำหนด Layout ต่างๆ
# ========================


def layout_thai_traditional(data):
    """รูปแบบไทยดั้งเดิม"""
    text = (
        f"{data['company']} {data['company_id']}\n"
        f"ใบรับรอง\n"
        f"มอบให้แก่\n\n"
        f"{data['recipient']}\n\n"
        f"ได้ผ่านการอบรม\n"
        f"หลักสูตร: {data['course_th']}\n\n"
        f"วันที่ {data['date_th']}\n\n"
        f"{data['director']}\n"
        f"({data['title_th']})\n\n"
        f"{data['url']}"
    )

    # กำหนดตำแหน่ง entities
    entities = [
        {'text': data['recipient'], 'label': 'PERSON', 'start': text.find(
            data['recipient']), 'end': text.find(data['recipient']) + len(data['recipient'])},
        {'text': data['course_th'], 'label': 'COURSE', 'start': text.find(
            data['course_th']), 'end': text.find(data['course_th']) + len(data['course_th'])},
        {'text': data['date_th'], 'label': 'DATE', 'start': text.find(
            data['date_th']), 'end': text.find(data['date_th']) + len(data['date_th'])},
        {'text': data['company'], 'label': 'ISSUER', 'start': text.find(
            data['company']), 'end': text.find(data['company']) + len(data['company'])},
        {'text': data['url'], 'label': 'URL', 'start': text.find(
            data['url']), 'end': text.find(data['url']) + len(data['url'])}
    ]

    return text, entities


def layout_english_formal(data):
    """รูปแบบภาษาอังกฤษทางการ"""
    text = (
        f"OFFICIAL CERTIFICATE\n\n"
        f"This document certifies that\n"
        f"{data['recipient'].upper()}\n"
        f"has successfully completed the training program\n"
        f"'{data['course_en']}'\n"
        f"on {data['date_en']}.\n\n"
        f"Issued by: {data['company']}\n"
        f"Authorized by: {data['director']}, {data['title_en']}\n"
        f"Certificate ID: {data['company_id']}\n\n"
        f"Verify at: {data['url']}"
    )

    entities = [
        {'text': data['recipient'], 'label': 'PERSON', 'start': text.find(
            data['recipient']), 'end': text.find(data['recipient']) + len(data['recipient'])},
        {'text': data['course_en'], 'label': 'COURSE', 'start': text.find(
            data['course_en']), 'end': text.find(data['course_en']) + len(data['course_en'])},
        {'text': data['date_en'], 'label': 'DATE', 'start': text.find(
            data['date_en']), 'end': text.find(data['date_en']) + len(data['date_en'])},
        {'text': data['company'], 'label': 'ISSUER', 'start': text.find(
            data['company']), 'end': text.find(data['company']) + len(data['company'])},
        {'text': data['url'], 'label': 'URL', 'start': text.find(
            data['url']), 'end': text.find(data['url']) + len(data['url'])}
    ]

    return text, entities


def layout_mixed_modern(data):
    """รูปแบบผสมสไตล์โมเดิร์น"""
    text = (
        f"=== CERTIFICATE OF ACHIEVEMENT ===\n\n"
        f"RECIPIENT: {data['recipient']}\n"
        f"COURSE: {data['course_en']} / {data['course_th']}\n"
        f"DATE: {data['date_en']} ({data['date_th']})\n"
        f"ORGANIZATION: {data['company']}\n\n"
        f"ID: {data['company_id']} | URL: {data['url']}\n"
        f"VALIDATED BY: {data['director']}, {data['title_en']}"
    )

    entities = [
        {'text': data['recipient'], 'label': 'PERSON', 'start': text.find(
            data['recipient']), 'end': text.find(data['recipient']) + len(data['recipient'])},
        {'text': data['course_en'], 'label': 'COURSE', 'start': text.find(
            data['course_en']), 'end': text.find(data['course_en']) + len(data['course_en'])},
        {'text': data['course_th'], 'label': 'COURSE', 'start': text.find(
            data['course_th']), 'end': text.find(data['course_th']) + len(data['course_th'])},
        {'text': data['date_en'], 'label': 'DATE', 'start': text.find(
            data['date_en']), 'end': text.find(data['date_en']) + len(data['date_en'])},
        {'text': data['date_th'], 'label': 'DATE', 'start': text.find(
            data['date_th']), 'end': text.find(data['date_th']) + len(data['date_th'])},
        {'text': data['company'], 'label': 'ISSUER', 'start': text.find(
            data['company']), 'end': text.find(data['company']) + len(data['company'])},
        {'text': data['url'], 'label': 'URL', 'start': text.find(
            data['url']), 'end': text.find(data['url']) + len(data['url'])}
    ]

    return text, entities

# เพิ่ม layout อื่นๆ ตามต้องการ...


# รายการ layout ทั้งหมด
ALL_LAYOUTS = [
    layout_thai_traditional,
    layout_english_formal,
    layout_mixed_modern
    # เพิ่ม layout อื่นๆ ที่นี่
]

# ========================
# ส่วนที่ 3: สร้าง Synthetic Data
# ========================


def generate_labeled_certificate():
    """สร้างใบรับรองพร้อม label"""
    # สร้างข้อมูลพื้นฐาน
    base_data = generate_base_data()

    # สุ่มเลือก layout
    layout_func = random.choice(ALL_LAYOUTS)

    # สร้างข้อความและ entities
    text, entities = layout_func(base_data)

    return {
        'text': text,
        'entities': entities,
        'layout': layout_func.__name__,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'language': 'th' if 'thai' in layout_func.__name__.lower() else
                        'en' if 'english' in layout_func.__name__.lower() else
                        'mixed'
        }
    }

# สร้าง dataset


def generate_dataset(num_samples=1000):
    """สร้าง dataset ทั้งหมด"""
    dataset = []
    for _ in tqdm(range(num_samples), desc="Generating Certificates"):
        dataset.append(generate_labeled_certificate())
    return dataset

# ========================
# ส่วนที่ 4: แปลงรูปแบบสำหรับการฝึกโมเดล
# ========================


def convert_to_spacy_format(labeled_data):
    """แปลงเป็นรูปแบบที่ใช้กับ spaCy"""
    training_data = []
    for example in labeled_data:
        entities = []
        for ent in example['entities']:
            entities.append((ent['start'], ent['end'], ent['label']))

        training_data.append((example['text'], {'entities': entities}))
    return training_data


def convert_to_conll_format(labeled_data):
    """แปลงเป็นรูปแบบ CONLL"""
    conll_data = []
    for example in labeled_data:
        text = example['text']
        # สร้าง tags (เริ่มต้นด้วย 'O')
        tags = ['O'] * len(text)

        # กำหนด tags สำหรับแต่ละ entity
        for ent in example['entities']:
            # ตั้งค่าเริ่มต้น (B-)
            tags[ent['start']] = f"B-{ent['label']}"
            # ตั้งค่าส่วนที่เหลือ (I-)
            for i in range(ent['start']+1, ent['end']):
                tags[i] = f"I-{ent['label']}"

        # สร้างรูปแบบ CONLL
        conll_entry = []
        for char, tag in zip(text, tags):
            conll_entry.append(f"{char} {tag}")
        conll_data.append("\n".join(conll_entry) + "\n")

    return conll_data


def convert_to_labelstudio_format(labeled_data):
    """แปลงเป็นรูปแบบ Label Studio"""
    tasks = []
    for example in labeled_data:
        task = {
            "data": {
                "text": example['text']
            },
            "annotations": [{
                "result": [
                    {
                        "value": {
                            "text": ent['text'],
                            "start": ent['start'],
                            "end": ent['end']
                        },
                        "id": f"id_{i}",
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "origin": "manual"
                    } for i, ent in enumerate(example['entities'])
                ]
            }]
        }
        tasks.append(task)
    return tasks

# ========================
# ส่วนที่ 5: บันทึกไฟล์
# ========================


def save_dataset(dataset, filename, format='json'):
    """บันทึก dataset เป็นไฟล์"""
    if format == 'json':
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
    elif format == 'spacy':
        spacy_data = convert_to_spacy_format(dataset)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(spacy_data, f, ensure_ascii=False, indent=2)
    elif format == 'conll':
        conll_data = convert_to_conll_format(dataset)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(conll_data))
    elif format == 'labelstudio':
        ls_data = convert_to_labelstudio_format(dataset)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(ls_data, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

# ========================
# ส่วนที่ 6: ใช้งานจริง
# ========================


if __name__ == "__main__":
    # สร้าง dataset
    print("กำลังสร้าง dataset...")
    dataset = generate_dataset(num_samples=1000)

    # แบ่ง train/val/test
    train_data, temp_data = train_test_split(
        dataset, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42)

    print(f"\nสรุปข้อมูล:")
    # print(f"- ข้อมูลทั้งหมด: {len(dataset)} ตัวอย่าง")
    # print(f"- ข้อมูลฝึกอบรม: {len(train_data)} ตัวอย่าง")
    # print(f"- ข้อมูลตรวจสอบ: {len(val_data)} ตัวอย่าง")
    # print(f"- ข้อมูลทดสอบ: {len(test_data)} ตัวอย่าง")

    # บันทึกไฟล์
    print("\nกำลังบันทึกไฟล์...")
    # คือข้อมูลที่ใช้ฝึกโมเดล
    save_dataset(train_data, "train_data_label.json")
    # คือข้อมูลที่ใช้ตรวจสอบความถูกต้องของโมเดล
    save_dataset(val_data, "val_data_label.json")
    # คือข้อมูลที่ใช้ทดสอบความถูกต้องของโมเดล
    save_dataset(test_data, "test_data_label.json")

    # บันทึกในรูปแบบอื่นๆ
    # save_dataset(dataset, "all_data_spacy.json", format='spacy')
    # save_dataset(dataset, "all_data.conll", format='conll')
    # save_dataset(dataset, "all_data_labelstudio.json", format='labelstudio')

    print("\nบันทึกไฟล์เรียบร้อยแล้ว!")
    print("- train_data.json, val_data.json, test_data.json")
    print("- all_data_spacy.json (รูปแบบ spaCy)")
    print("- all_data.conll (รูปแบบ CONLL)")
    print("- all_data_labelstudio.json (รูปแบบ Label Studio)")
