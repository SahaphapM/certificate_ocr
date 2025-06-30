import torch
import numpy as np
from tokenization import split_to_words


def merge_entities(entities):
    """รวม entities ที่ติดกันและเป็นประเภทเดียวกัน"""
    if not entities:
        return []

    merged = []
    current = entities[0]

    for i in range(1, len(entities)):
        next_ent = entities[i]

        # ตรวจสอบว่าเป็น entity เดียวกันและติดกัน
        if (current['end'] == next_ent['start'] and
                current['label'] == next_ent['label']):

            # รวม entity
            current['text'] += next_ent['text']
            current['end'] = next_ent['end']
        else:
            merged.append(current)
            current = next_ent

    merged.append(current)
    return merged


def predict_entities(text, model, tokenizer, device):
    words = split_to_words(text)

    # หาตำแหน่งคำ
    word_positions = []
    current_position = 0
    for word in words:
        start = text.find(word, current_position)
        end = start + len(word)
        word_positions.append((start, end))
        current_position = end

    # Tokenize และทำนาย
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    label_ids = predictions[0]
    word_ids = encoding.word_ids()

    id2label = model.config.id2label

    # รวบรวมผลลัพธ์
    entities = []
    current_entity = None

    for i, (token, word_id) in enumerate(zip(encoding.tokens(), word_ids)):
        if word_id is None:
            continue

        label = id2label.get(label_ids[i], "O")
        word_start, word_end = word_positions[word_id]

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": text[word_start:word_end],
                "label": label[2:],
                "start": word_start,
                "end": word_end
            }
        elif label.startswith("I-") and current_entity:
            # ตรวจสอบว่าเป็น entity เดิมและต่อเนื่อง
            if current_entity['end'] == word_start and current_entity['label'] == label[2:]:
                current_entity['text'] += text[word_start:word_end]
                current_entity['end'] = word_end
            else:
                entities.append(current_entity)
                current_entity = {
                    "text": text[word_start:word_end],
                    "label": label[2:],
                    "start": word_start,
                    "end": word_end
                }
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # รวม entities ที่ติดกัน
    return merge_entities(entities)
