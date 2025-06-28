import random
import faker
import json

fake = faker.Faker("th_TH")

# สร้างข้อมูลจำลอง (Data Augmentation)


def generate_fake_certificate():
    layouts = [
        f"{fake.company()} {fake.bothify(text='???####')}\nCERTIFICATE\n{random.choice(['มอบให้แก่', 'This is awarded to'])}\n\n{fake.name()}\n\nได้ผ่านการอบรม\n{random.choice(['หลักสูตร', 'Course'])}: {fake.bs()}\n\nวันที่ {fake.date(pattern='%d/%m/%Y')}\n\n{fake.name()}\n({fake.name()})\n{random.choice(['ผู้อำนวยการ', 'Director'])}\n\n{fake.url()}",
        f"ประกาศนียบัตร\n{random.choice(['ชื่อ', 'Name'])}: {fake.name()}\n{random.choice(['คอร์ส', 'Course'])}: {fake.catch_phrase()}\n{random.choice(['วันที่สำเร็จ', 'Completed on'])}: {fake.date(pattern='%Y-%m-%d')}\n{random.choice(['ออกโดย', 'Issued by'])}: {fake.company()}\n\n{fake.url()}",
        f"{random.choice(['CERTIFICATE', 'ใบรับรอง'])} ID: {fake.bothify(text='??###')}\n\n{random.choice(['สำหรับ', 'Awarded to'])}: {fake.name()}\n\n{random.choice(['ในหลักสูตร', 'For the course'])}: {fake.text(max_nb_chars=40)}\n\n{random.choice(['วันที่', 'Date'])}: {fake.date(pattern='%d %b %Y')}\n{random.choice(['ผู้ลงนาม', 'Instructor'])}: {fake.name()}\n\n{fake.url()}"
    ]
    return random.choice(layouts)


# สร้าง dataset 200 ตัวอย่าง
data = [{"text": generate_fake_certificate()} for _ in range(200)]

# สร้างข้อมูลในรูปแบบ Label Studio
# สร้างข้อมูลในรูปแบบที่ถูกต้อง
tasks = [
    {
        "data": {
            "text": cert["text"],
            # ต้องเพิ่ม key "custom_field" เพื่อแก้ปัญหาการเพิ่ม file_upload_id โดยอัตโนมัติ
            "custom_field": "dummy_value"
        }
    }
    for cert in data
]

# บันทึกเป็น JSON
with open("certificate_data.json", "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)
